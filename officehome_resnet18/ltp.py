import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import time
import warnings

import numpy as np
import torch

from data import DOMAINS, Store, list_domain, split_domain
from method import project, train_fnet, train_vae
from training import extract

warnings.filterwarnings("ignore")

SETTINGS = {
    "Art": (128, 0.0, "random", 0.5),
    "Clipart": (128, 0.1, "random", None),
    "Product": (32, 0.1, "random", 0.5),
    "Real World": (128, 0.0, "encoder", 0.25),
}


def run_target(store, args, dev, target):
    latent, prior, init, alpha = SETTINGS[target]
    sources = [d for d in DOMAINS if d != target]
    splits = {d: split_domain(args.data, d, args.classes) for d in sources}
    tr = [it for d in sources for it in splits[d][0]]
    va = [it for d in sources for it in splits[d][1]]
    f = train_fnet(store, args, dev, tr, va)

    Zs = []
    for i, d in enumerate(sources):
        part = splits[d][0] + splits[d][1]
        Zs.append(extract(store, f, part, "test", args, dev)[0])
        Zs.append(extract(store, f, part, "view", args, dev, views=args.views, seed=100 * i)[0])
    Zt, yt = extract(store, f, list_domain(args.data, target, args.classes), "test", args, dev)

    vae = train_vae(torch.cat(Zs), latent, args, dev)
    z_last, zf = project(vae, Zt, prior, init, args)
    with torch.no_grad():
        if alpha is None:
            logits = f.fc3(z_last)
        else:
            scale = Zt.norm(dim=1, keepdim=True) / zf.norm(dim=1, keepdim=True).clamp_min(1e-6)
            logits = f.fc3(alpha * Zt + (1 - alpha) * zf * scale)
    return float((logits.argmax(1) == yt).float().mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/OfficeHomeDataset_10072016")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--vae_hidden", type=int, default=512)
    ap.add_argument("--vae_epochs", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()
    if not all(os.path.isdir(os.path.join(args.data, d)) for d in DOMAINS):
        sys.exit(f"Office-Home not found at {args.data}")
    first = os.path.join(args.data, DOMAINS[0])
    args.classes = sorted(c for c in os.listdir(first) if os.path.isdir(os.path.join(first, c)))
    args.num_classes = len(args.classes)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    store = Store(args.data, args.classes, dev)
    print(f"loaded {len(store.y)} images  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    results = {}
    for t in DOMAINS:
        results[t] = run_target(store, args, dev, t)
        print(f"{t:<14}{results[t]:>12.2f}  [{(time.time() - t0) / 60:.0f} min]", flush=True)

    print()
    print("Office-Home, ResNet-18")
    print(f"{'Target':<14}{'Accuracy':>12}")
    print("-" * 26)
    for t, a in results.items():
        print(f"{t:<14}{a:>12.2f}")
    print("-" * 26)
    print(f"{'Average':<14}{np.mean(list(results.values())):>12.2f}")


if __name__ == "__main__":
    main()
