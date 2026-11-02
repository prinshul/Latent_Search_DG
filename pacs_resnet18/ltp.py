import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import random
import sys
import time
import warnings

import numpy as np
import torch

from data import DOMAINS, Store, list_domain, split_domain
from method import acc, predict, select_rule, train_clf, train_fnet, train_vae
from training import extract

warnings.filterwarnings("ignore")
T0 = time.time()


def say(msg, end=False):
    line = f"{msg}  [{(time.time() - T0) / 60:.0f} min]"
    sys.stdout.write("\r" + line.ljust(60) + ("\n" if end else ""))
    sys.stdout.flush()


class Progress:
    def __init__(self, name, total):
        self.name, self.total, self.done = name, max(1, total), 0

    def step(self, n=1):
        self.done += n
        say(f"{self.name:<13} {min(99, int(100 * self.done / self.total)):3d}%")


def select_strength(store, args, dev, target, splits, seed, prog):
    path = os.path.join(args.out, f"config_{target}.json")
    if os.path.isfile(path):
        return json.load(open(path))
    sources = list(splits)

    def score(level):
        accs = []
        for i, h in enumerate(sources):
            rest = [d for d in sources if d != h]
            tr = [it for d in rest for it in splits[d][0]]
            va = [it for d in rest for it in splits[d][1]]
            f = train_fnet(store, args, dev, tr, va, seed + 500 + i, level, args.select_epochs, prog)
            Z, y = extract(store, f, tr, "test", args, dev)
            C = train_clf(Z, y, None, args, dev, seed)
            Zh, yh = extract(store, f, splits[h][0] + splits[h][1], "test", args, dev)
            with torch.no_grad():
                accs.append(acc(C(Zh).argmax(1), yh))
            del f
        return float(np.mean(accs))

    levels = {lv: score(lv) for lv in (0, 1, 2)}
    cfg = {"level": max(levels, key=lambda lv: levels[lv]), "level_scores": levels}
    json.dump(cfg, open(path, "w"), indent=1)
    return cfg


def run_target(store, args, dev, target, seed):
    sources = [d for d in DOMAINS if d != target]
    splits = {d: split_domain(args.data, d, seed) for d in sources}
    tr = [it for d in sources for it in splits[d][0]]
    va = [it for d in sources for it in splits[d][1]]
    te = list_domain(args.data, target)
    units = math.ceil(args.proj_iters / 2000)
    cached = os.path.isfile(os.path.join(args.out, f"config_{target}.json"))
    prog = Progress(target, (0 if cached else 3 * len(sources) * args.select_epochs) + args.epochs + 5
                    + len(sources) * 2 * units * 2 + 3 * units)

    cfg = select_strength(store, args, dev, target, splits, seed, prog)
    f = train_fnet(store, args, dev, tr, va, seed, cfg["level"], args.epochs, prog)

    feats = {}
    for i, d in enumerate(sources):
        a, b = splits[d]
        Zc, yc = extract(store, f, a, "test", args, dev)
        Zv, yv = extract(store, f, b, "test", args, dev)
        Za, ya = extract(store, f, a, "view", args, dev, views=args.feat_views, seed=seed + 100 * i)
        feats[d] = {"train": (torch.cat([Zc, Za]), torch.cat([yc, ya])), "val": (Zv, yv),
                    "clean": (torch.cat([Zc, Zv]), torch.cat([yc, yv]))}
    Zs = torch.cat([v["train"][0] for v in feats.values()])
    ys = torch.cat([v["train"][1] for v in feats.values()])
    Zt, yt = extract(store, f, te, "test", args, dev)
    prog.step(3)

    vae = train_vae(Zs, args, dev, seed)
    C = train_clf(Zs, ys, vae, args, dev, seed)
    prog.step(2)

    rule, _ = select_rule(feats, args, dev, seed, prog)
    result = predict(vae, C, Zt, yt, args, seed, rule, prog)

    del f, vae, C
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    say(f"{target:<13} {result:6.2f}", end=True)
    return result


def table(results):
    lines = ["PACS, ResNet-18", f"{'Target':<14}{'Accuracy':>12}", "-" * 26]
    means = []
    for t, runs in results.items():
        vals = list(runs.values())
        means.append(float(np.mean(vals)))
        cell = f"{means[-1]:.2f}" if len(vals) == 1 else f"{means[-1]:.2f} ± {np.std(vals):.2f}"
        lines.append(f"{t:<14}{cell:>12}")
    lines += ["-" * 26, f"{'Average':<14}{np.mean(means):>12.2f}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Homework3-PACS/PACS")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--targets", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--select_epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--feat_views", type=int, default=5)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--vae_hidden", type=int, default=512)
    ap.add_argument("--vae_epochs", type=int, default=100)
    ap.add_argument("--clf_epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--proj_iters", type=int, default=20000)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--select_max", type=int, default=1000)
    args = ap.parse_args()

    for d in DOMAINS:
        if not os.path.isdir(os.path.join(args.data, d)):
            sys.exit(f"PACS not found at {args.data}")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    store = Store(args.data, dev, say)
    results = {t: {} for t in args.targets}
    for seed in args.seeds:
        for target in args.targets:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            results[target][str(seed)] = run_target(store, args, dev, target, seed)
            json.dump({"args": vars(args), "results": results}, open(os.path.join(args.out, "results.json"), "w"),
                      indent=1)
    text = table(results)
    open(os.path.join(args.out, "results.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
