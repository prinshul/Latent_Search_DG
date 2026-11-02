import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from training import MixStyle, fit, extract

RULES = ["elbow", "last", "ens", "recon", "enc_elbow", "enc_last"]
ARCHS = {"resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512, 256),
         "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 1024, 1024)}


class Backbone(nn.Module):
    def __init__(self, arch="resnet18", weights=None):
        super().__init__()
        build, pretrained = ARCHS[arch][:2]
        net = build(weights=None)
        if weights:
            net.load_state_dict(torch.load(weights, map_location="cpu"))
        else:
            net.load_state_dict(pretrained.get_state_dict(progress=False))
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net, self.ms = net, MixStyle()

    def forward(self, x):
        b = self.net
        x = self.ms(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
        x = self.ms(b.layer2(x))
        return torch.flatten(b.avgpool(b.layer4(b.layer3(x))), 1)


class FNet(nn.Module):
    def __init__(self, arch="resnet18", weights=None):
        super().__init__()
        hidden, feat = ARCHS[arch][2:]
        self.backbone = Backbone(arch, weights)
        self.fc1, self.fc2 = nn.Linear(self.backbone.out_dim, hidden), nn.Linear(hidden, feat)

    def forward(self, x):
        return self.fc2(F.leaky_relu(self.fc1(self.backbone(x)), 0.2))


def pair_loss(z, y, tau):
    z = F.normalize(z.float(), dim=1)
    return F.binary_cross_entropy_with_logits(z @ z.T / tau, (y[:, None] == y[None, :]).float())


def train_fnet(store, args, dev, tr, va, seed, level, epochs, prog):
    torch.manual_seed(seed)
    model = FNet(args.arch, args.weights).to(dev)

    def loss_fn(z, y):
        return pair_loss(z, y, args.tau)

    def score_fn(m):
        z, y = extract(store, m, va, "test", args, dev)
        return -pair_loss(z, y, args.tau).item()

    return fit(store, model, loss_fn, score_fn, args, dev, tr, seed, level, epochs, prog)


class VAE(nn.Module):
    def __init__(self, d, latent, hidden):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def G(self, u):
        return self.dec(u) * self.sd + self.mu

    def encode(self, z):
        return self.enc((z - self.mu) / self.sd).chunk(2, dim=1)

    def reconstruct(self, z, sample=False):
        m, lv = self.encode(z)
        return self.G(m + torch.randn_like(m) * (0.5 * lv).exp() if sample else m)


def train_vae(Z, args, dev, seed):
    torch.manual_seed(seed)
    vae = VAE(Z.shape[1], args.latent, args.vae_hidden).to(dev)
    vae.mu.copy_(Z.mean(0))
    vae.sd.copy_(Z.std(0) + 1e-6)
    Zs = (Z - vae.mu) / vae.sd
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(args.vae_epochs):
        for b in torch.randperm(len(Zs), device=dev).split(256):
            m, lv = vae.enc(Zs[b]).chunk(2, 1)
            r = vae.dec(m + torch.randn_like(m) * (0.5 * lv).exp())
            rec = ((r - Zs[b]) ** 2).sum(1) + (r - Zs[b]).abs().sum(1)
            kl = -0.5 * (1 + lv - m ** 2 - lv.exp()).sum(1)
            loss = (rec + kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.eval()


def train_clf(Z, y, vae, args, dev, seed):
    torch.manual_seed(seed)
    C = nn.Linear(Z.shape[1], args.num_classes).to(dev)
    opt = torch.optim.Adam(C.parameters(), lr=3e-3)
    for _ in range(args.clf_epochs):
        for b in torch.randperm(len(Z), device=dev).split(64):
            zb, yb = Z[b], y[b]
            if vae is not None:
                with torch.no_grad():
                    zb, yb = torch.cat([zb, vae.reconstruct(zb, sample=True)]), torch.cat([yb, yb])
            loss = F.cross_entropy(C(zb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in C.parameters():
        p.requires_grad_(False)
    return C.eval()


def elbow_index(L, W):
    k = torch.ones(1, 1, W, device=L.device) / W
    s = F.conv1d(L.T.unsqueeze(1), k).squeeze(1)
    d2 = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
    return d2.argmax(1) + 1 + (W - 1) // 2


def project(vae, C, zt, args, init, seed, prog=None, weight=1):
    dev, M = zt.device, args.proj_iters
    g = torch.Generator().manual_seed(seed)
    if init == "random":
        u = torch.randn(len(zt), vae.latent, generator=g).to(dev)
    else:
        u = vae.encode(zt)[0].detach().clone()
    u.requires_grad_(True)
    zn = F.normalize(zt, dim=1)
    L = torch.empty(M, len(zt), device=dev)
    P = torch.empty(M, len(zt), dtype=torch.int8, device=dev)
    for t in range(M):
        z = vae.G(u)
        ls = 1 - (F.normalize(z, dim=1) * zn).sum(1)
        with torch.no_grad():
            L[t] = ls
            P[t] = C(z).argmax(1).to(torch.int8)
        u.grad = None
        ls.sum().backward()
        with torch.no_grad():
            u.sub_(args.beta * u.grad)
        if prog is not None and (t + 1) % 2000 == 0:
            prog.step(weight)
    with torch.no_grad():
        zf = vae.G(u)
    return P, L, elbow_index(L, args.window), zf


def acc(pred, y):
    return float((pred.long() == y).float().mean() * 100)


def evaluate(vae, C, z, y, args, seed, prog=None):
    r = {}
    ar = torch.arange(len(z), device=z.device)
    with torch.no_grad():
        p0 = C(z).softmax(1)
        r["recon"] = acc(C(vae.reconstruct(z)).argmax(1), y)
    P, L, n, zf = project(vae, C, z, args, "random", seed, prog, 2)
    r["elbow"], r["last"] = acc(P[n, ar], y), acc(P[-1], y)
    with torch.no_grad():
        r["ens"] = acc((p0 + C(zf).softmax(1)).argmax(1), y)
    P, L, n, _ = project(vae, C, z, args, "encoder", seed, prog, 2)
    r["enc_elbow"] = acc(torch.where(L[-1] > 0.5, p0.argmax(1), P[n, ar].long()), y)
    r["enc_last"] = acc(P[-1], y)
    return r


def predict(vae, C, z, y, args, seed, rule, prog):
    ar = torch.arange(len(z), device=z.device)
    with torch.no_grad():
        p0 = C(z).softmax(1)
        if rule == "recon":
            return acc(C(vae.reconstruct(z)).argmax(1), y)
    if rule in ("elbow", "last", "ens"):
        P, L, n, zf = project(vae, C, z, args, "random", seed, prog, 3)
        if rule == "elbow":
            return acc(P[n, ar], y)
        if rule == "last":
            return acc(P[-1], y)
        with torch.no_grad():
            return acc((p0 + C(zf).softmax(1)).argmax(1), y)
    P, L, n, _ = project(vae, C, z, args, "encoder", seed, prog, 3)
    if rule == "enc_elbow":
        return acc(torch.where(L[-1] > 0.5, p0.argmax(1), P[n, ar].long()), y)
    return acc(P[-1], y)


def select_rule(feats, args, dev, seed, prog):
    doms = list(feats)
    scores = {k: [] for k in RULES}
    for i, s in enumerate(doms):
        rest = [d for d in doms if d != s]
        Ztr = torch.cat([feats[d]["train"][0] for d in rest])
        ytr = torch.cat([feats[d]["train"][1] for d in rest])
        vae = train_vae(Ztr, args, dev, seed + 10 + i)
        C = train_clf(Ztr, ytr, vae, args, dev, seed + 10 + i)
        Ze, ye = feats[s]["clean"]
        if len(Ze) > args.select_max:
            pick = torch.randperm(len(Ze), generator=torch.Generator().manual_seed(seed + i))[: args.select_max]
            Ze, ye = Ze[pick.to(dev)], ye[pick.to(dev)]
        r = evaluate(vae, C, Ze, ye, args, seed + 20 + i, prog)
        for k in scores:
            scores[k].append(r[k])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(RULES, key=lambda k: mean[k]), mean
