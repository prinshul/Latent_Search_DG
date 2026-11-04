import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from training import MixStyle, extract, fit


class Backbone(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        net = models.resnet18(weights=None)
        if weights:
            net.load_state_dict(torch.load(weights, map_location="cpu"))
        else:
            net.load_state_dict(models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False))
        net.fc = nn.Identity()
        self.net, self.ms = net, MixStyle()

    def forward(self, x):
        b = self.net
        x = self.ms(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
        x = self.ms(b.layer2(x))
        return torch.flatten(b.avgpool(b.layer4(b.layer3(x))), 1)


class FNet(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.backbone = Backbone(weights)
        self.fc1, self.fc2 = nn.Linear(512, 512), nn.Linear(512, 512)
        for fc in (self.fc1, self.fc2):
            nn.init.eye_(fc.weight)
            nn.init.zeros_(fc.bias)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc2(F.leaky_relu(self.fc1(self.backbone(x)), 0.2))


def train_fnet(store, args, dev, tr, va):
    torch.manual_seed(0)
    model = FNet(args.num_classes, args.weights).to(dev)

    def loss_fn(z, y):
        return F.cross_entropy(model.fc3(z.float()), y, label_smoothing=0.1)

    def score_fn(m, tta=True):
        z, y = extract(store, m, va, "test", args, dev, tta=tta)
        with torch.no_grad():
            return -F.cross_entropy(m.fc3(z.float()), y, label_smoothing=0.1).item()

    return fit(store, model, loss_fn, score_fn, args, dev, tr)


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


def train_vae(Z, latent, args, dev):
    torch.manual_seed(0)
    vae = VAE(Z.shape[1], latent, args.vae_hidden).to(dev)
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


def project(vae, zt, prior, init, args):
    if init == "random":
        u = torch.randn(len(zt), vae.latent, generator=torch.Generator().manual_seed(0)).to(zt.device)
    else:
        u = vae.enc((zt - vae.mu) / vae.sd).chunk(2, dim=1)[0].detach().clone()
    u.requires_grad_(True)
    zn = F.normalize(zt, dim=1)
    for _ in range(args.iters):
        z = vae.G(u)
        loss = 1 - (F.normalize(z, dim=1) * zn).sum(1)
        if prior:
            loss = loss + prior * (u ** 2).mean(1)
        u.grad = None
        loss.sum().backward()
        with torch.no_grad():
            u.sub_(args.beta * u.grad)
    with torch.no_grad():
        return z.detach(), vae.G(u)
