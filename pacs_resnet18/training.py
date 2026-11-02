import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _u(n, lo, hi, dev):
    return torch.empty(n, device=dev).uniform_(lo, hi)


def _gray(x):
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def crop_flip(x, out, scale, rot_deg=0.0, rot_p=0.0, ratio=(3 / 4, 4 / 3)):
    B, dev = x.size(0), x.device
    area = _u(B, scale[0], scale[1], dev)
    r = _u(B, math.log(ratio[0]), math.log(ratio[1]), dev).exp()
    w, h = (area * r).sqrt().clamp(max=1.0), (area / r).sqrt().clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * (1 - w) + w / 2) * 2 - 1
    cy = (torch.rand(B, device=dev) * (1 - h) + h / 2) * 2 - 1
    flip = 1 - 2 * (torch.rand(B, device=dev) < 0.5).float()
    ang = torch.zeros(B, device=dev)
    if rot_deg > 0:
        ang = _u(B, -rot_deg, rot_deg, dev) * (math.pi / 180) * (torch.rand(B, device=dev) < rot_p).float()
    cs, sn = ang.cos(), ang.sin()
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = w * flip * cs, -h * sn, cx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = w * flip * sn, h * cs, cy
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_jitter(x, b, c, s, h, p):
    B, dev = x.size(0), x.device
    v = lambda t: t.view(B, 1, 1, 1)
    y = (x * v(_u(B, max(0.0, 1 - b), 1 + b, dev))).clamp(0, 1)
    m = _gray(y).mean(dim=(1, 2, 3), keepdim=True)
    y = ((y - m) * v(_u(B, max(0.0, 1 - c), 1 + c, dev)) + m).clamp(0, 1)
    g = _gray(y)
    y = ((y - g) * v(_u(B, max(0.0, 1 - s), 1 + s, dev)) + g).clamp(0, 1)
    if h > 0:
        M = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
        yiq = torch.einsum("ij,bjhw->bihw", M, y)
        a = _u(B, -h, h, dev) * 2 * math.pi
        cs, sn = a.cos().view(B, 1, 1), a.sin().view(B, 1, 1)
        yiq = torch.stack([yiq[:, 0], cs * yiq[:, 1] - sn * yiq[:, 2], sn * yiq[:, 1] + cs * yiq[:, 2]], 1)
        y = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(M), yiq).clamp(0, 1)
    return torch.where(v(torch.rand(B, device=dev) < p), y, x)


def random_gray(x, p):
    sel = (torch.rand(x.size(0), device=x.device) < p).view(-1, 1, 1, 1)
    return torch.where(sel, _gray(x).expand_as(x), x)


def edge_map(x, p):
    sel = torch.rand(x.size(0), device=x.device) < p
    if not bool(sel.any()):
        return x
    pad = lambda t: F.pad(t, (1, 1, 1, 1), mode="reflect")
    g = F.avg_pool2d(pad(_gray(x[sel])), 3, stride=1)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    mag = (F.conv2d(pad(g), kx) ** 2 + F.conv2d(pad(g), kx.transpose(2, 3)) ** 2).sqrt()
    q = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1) + 1e-6
    out = x.clone()
    out[sel] = (1 - (mag / q).clamp(0, 1)).expand(-1, 3, -1, -1)
    return out


def amplitude_mix(x, p=0.5, eta=1.0):
    sel = torch.rand(x.size(0), device=x.device) < p
    if not bool(sel.any()):
        return x
    X = torch.fft.fft2(x.float(), dim=(-2, -1))
    amp, pha = X.abs(), X.angle()
    lam = torch.rand(x.size(0), 1, 1, 1, device=x.device) * eta
    amp = (1 - lam) * amp + lam * amp[torch.randperm(x.size(0), device=x.device)]
    xm = torch.fft.ifft2(torch.polar(amp, pha), dim=(-2, -1)).real.to(x.dtype)
    return torch.where(sel[:, None, None, None], xm, x)


def augment(xu8, img, mode, level=0):
    x = xu8.float().div_(255)
    if mode == "test":
        x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True, align_corners=False)
    elif mode == "view":
        x = crop_flip(x, img, (0.5, 1.0))
    else:
        rot, gray, edge = (0.0, 0.25, 0.5)[level], (0.2, 0.35, 0.5)[level], (0.0, 0.1, 0.2)[level]
        x = crop_flip(x, img, (0.5, 1.0), 30.0 if rot else 0.0, rot)
        x = random_gray(color_jitter(x, 0.4, 0.4, 0.4, 0.1, 0.8), gray)
        if edge:
            x = edge_map(x, edge)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last) if x.is_cuda else x


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        dt, x = x.dtype, x.float()
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt().detach()
        lam = self.beta.sample((x.size(0), 1, 1, 1)).to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        return ((x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + lam * mu + (1 - lam) * mu[perm]).to(dt)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        pairs = list(zip(self.model.state_dict().values(), model.state_dict().values()))
        self.ef = [e for e, v in pairs if e.dtype.is_floating_point]
        self.vf = [v for e, v in pairs if e.dtype.is_floating_point]
        self.other = [(e, v) for e, v in pairs if not e.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_mul_(self.ef, d)
        torch._foreach_add_(self.ef, self.vf, alpha=1 - d)
        for e, v in self.other:
            e.copy_(v)


class SWA:
    def __init__(self, model):
        self.model, self.n = copy.deepcopy(model).eval(), 0
        self.params = list(self.model.parameters())

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        src = list(model.parameters())
        if self.n == 1:
            for a, b in zip(self.params, src):
                a.copy_(b)
        else:
            torch._foreach_mul_(self.params, (self.n - 1) / self.n)
            torch._foreach_add_(self.params, src, alpha=1 / self.n)

    @torch.no_grad()
    def recompute_bn(self, batches, dev, amp):
        m = self.model
        bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        ms = [x for x in m.modules() if isinstance(x, MixStyle)]
        for x in ms:
            x.p = 0.0
        for b in bns:
            b.reset_running_stats()
            b.momentum = None
        m.train()
        for x in batches:
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                m(x)
        for b in bns:
            b.momentum = 0.1
        for x in ms:
            x.p = 0.5
        m.eval()


def fit(store, model, loss_fn, score_fn, args, dev, items, seed, level, epochs, prog):
    amp = dev.type == "cuda"
    idx = store.idx(items)
    g = torch.Generator().manual_seed(seed)
    bs = args.batch
    steps = len(idx) // bs
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    opt = torch.optim.SGD([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                          momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema, swa = EMA(model), SWA(model)
    swa_start = int(0.3 * epochs * steps)
    best, best_state, step = -float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
        for i in range(steps):
            b = perm[i * bs:(i + 1) * bs]
            x = amplitude_mix(augment(store.x[b], args.img, "train", level))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            loss = loss_fn(out, store.y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(step)
            if step >= swa_start:
                swa.update(model)
            step += 1
        score = score_fn(ema.model)
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in ema.model.state_dict().items()}
        prog.step()
    model.load_state_dict(best_state)
    if swa.n > 0:
        perm = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
        swa.recompute_bn((augment(store.x[perm[i * bs:(i + 1) * bs]], args.img, "train", level)
                          for i in range(min(steps, 200))), dev, amp)
        if score_fn(swa.model) >= score_fn(model):
            model.load_state_dict(swa.model.state_dict())
    return model.eval()


@torch.no_grad()
def extract(store, model, items, mode, args, dev, views=1, seed=0):
    amp = dev.type == "cuda"
    model.eval()
    idx = store.idx(items)
    out = []
    for v in range(views):
        torch.manual_seed(seed + v)
        zs = []
        for b in idx.split(256):
            x = augment(store.x[b], args.img, mode)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                z = 0.5 * (model(x).float() + model(torch.flip(x, dims=[3])).float())
                H, W = x.shape[-2:]
                h, w = int(round(H * 0.875)), int(round(W * 0.875))
                t, l = (H - h) // 2, (W - w) // 2
                xc = F.interpolate(x[..., t:t + h, l:l + w].float(), size=(H, W), mode="bilinear",
                                   align_corners=False).to(x.dtype)
                z = 0.5 * (z + 0.5 * (model(xc).float() + model(torch.flip(xc, dims=[3])).float()))
            zs.append(z)
        out.append(torch.cat(zs))
    return torch.cat(out), store.y[idx].repeat(views)
