import copy
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def crop_flip(x, out):
    B, dev = x.size(0), x.device
    area = torch.empty(B, device=dev).uniform_(0.5, 1.0)
    r = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)).exp()
    w, h = (area * r).sqrt().clamp(max=1.0), (area / r).sqrt().clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * (1 - w) + w / 2) * 2 - 1
    cy = (torch.rand(B, device=dev) * (1 - h) + h / 2) * 2 - 1
    flip = 1 - 2 * (torch.rand(B, device=dev) < 0.5).float()
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w * flip, cx, h, cy
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def translate_flip(x, out):
    B, dev = x.size(0), x.device
    s = torch.where(torch.rand(B, device=dev) < 0.5, torch.full((B,), 1 / 1.125, device=dev), torch.ones(B, device=dev))
    cx, cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - s), (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    flip = 1 - 2 * (torch.rand(B, device=dev) < 0.5).float()
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = s * flip, cx, s, cy
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def amplitude_mix(x, p=0.5):
    sel = torch.rand(x.size(0), device=x.device) < p
    if not bool(sel.any()):
        return x
    X = torch.fft.fft2(x.float(), dim=(-2, -1))
    amp, pha = X.abs(), X.angle()
    lam = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    amp = (1 - lam) * amp + lam * amp[torch.randperm(x.size(0), device=x.device)]
    xm = torch.fft.ifft2(torch.polar(amp, pha), dim=(-2, -1)).real.to(x.dtype)
    return torch.where(sel[:, None, None, None], xm, x)


def augment(xu8, img, mode):
    x = xu8.float().div_(255)
    if mode == "test":
        x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True, align_corners=False)
    elif mode == "view":
        x = crop_flip(x, img)
    else:
        x = translate_flip(x, img)
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


class SWAD:
    def __init__(self, model, n_converge=3, n_tolerance=6, ratio=0.3):
        self.model = copy.deepcopy(model).eval()
        self.params = list(self.model.parameters())
        self.nc, self.ratio = n_converge, ratio
        self.hist, self.losses = deque(maxlen=n_tolerance), []
        self.seg, self.seg_n = [torch.zeros_like(p) for p in self.params], 0
        self.avg, self.k, self.th, self.dead = None, 0, None, False

    @torch.no_grad()
    def update(self, model):
        self.seg_n += 1
        torch._foreach_add_(self.seg, list(model.parameters()))

    def _add(self, seg):
        self.k += 1
        if self.avg is None:
            self.avg = [t.clone() for t in seg]
        else:
            torch._foreach_mul_(self.avg, (self.k - 1) / self.k)
            torch._foreach_add_(self.avg, seg, alpha=1 / self.k)

    @torch.no_grad()
    def end_segment(self, loss):
        seg = [t / self.seg_n for t in self.seg]
        self.seg, self.seg_n = [torch.zeros_like(p) for p in self.params], 0
        if self.dead:
            return
        j = len(self.losses)
        loss = loss if math.isfinite(loss) else float("inf")
        self.losses.append(loss)
        self.hist.append((j, loss, seg))
        if self.th is None:
            if j < self.nc - 1:
                return
            c = j - self.nc + 1
            window = self.losses[c:]
            if int(np.argmin(window)) != 0:
                return
            self.th = float(np.mean(window)) * (1 + self.ratio)
            before = [h for h in self.hist if h[0] <= c]
            start = 0
            for i in reversed(range(len(before))):
                if before[i][1] > self.th:
                    start = i + 1
                    break
            for _, _, sg in before[start:]:
                self._add(sg)
            for _, ls, sg in [h for h in self.hist if h[0] > c]:
                if ls > self.th:
                    self.dead = True
                    return
                self._add(sg)
            return
        if loss > self.th:
            self.dead = True
        else:
            self._add(seg)

    @torch.no_grad()
    def finalize(self, batches, dev, amp):
        for a, b in zip(self.params, self.avg if self.avg is not None else self.hist[-1][2]):
            a.copy_(b)
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


def fit(store, model, loss_fn, score_fn, args, dev, items):
    amp = dev.type == "cuda"
    idx = store.idx(items)
    g = torch.Generator().manual_seed(0)
    bs = args.batch
    steps = len(idx) // bs
    if amp:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema, swad = EMA(model), SWAD(model)
    best, best_state, step = -float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
        for i in range(steps):
            b = perm[i * bs:(i + 1) * bs]
            x = amplitude_mix(augment(store.x[b], args.img, "train"))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            loss = loss_fn(out, store.y[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(step)
            swad.update(model)
            step += 1
        swad.end_segment(-score_fn(model, tta=False))
        score = score_fn(ema.model)
        score = score if math.isfinite(score) else -float("inf")
        if best_state is None or score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in ema.model.state_dict().items()}
    model.load_state_dict(best_state)
    perm = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
    swad.finalize((augment(store.x[perm[i * bs:(i + 1) * bs]], args.img, "train") for i in range(min(steps, 200))),
                  dev, amp)
    s_swad, s_ema = score_fn(swad.model), score_fn(model)
    if math.isfinite(s_swad) and (s_swad >= s_ema or not math.isfinite(s_ema)):
        model.load_state_dict(swad.model.state_dict())
    return model.eval()


@torch.no_grad()
def extract(store, model, items, mode, args, dev, views=1, seed=0, tta=True):
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
                if not tta:
                    zs.append(model(x).float())
                    continue
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
