import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import math
import random
import sys
import time
import warnings

import numpy as np
import torch

from data import DATASETS, Store, find_classes, list_domain, resolve_domains, split_domain
from method import ARCHS, acc, predict, select_rule, train_clf, train_fnet, train_vae
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


def config_path(args, target):
    return os.path.join(args.out, f"config_{args.arch}_{target}{'_lr' if args.select_lr else ''}.json")


def select_config(store, args, dev, target, splits, seed, prog):
    path = config_path(args, target)
    if os.path.isfile(path):
        return json.load(open(path))
    sources = list(splits)

    def score(level, lr):
        a = copy.copy(args)
        a.lr = lr
        accs = []
        for i, h in enumerate(sources):
            rest = [d for d in sources if d != h]
            tr = [it for d in rest for it in splits[d][0]]
            va = [it for d in rest for it in splits[d][1]]
            f = train_fnet(store, a, dev, tr, va, seed + 500 + i, level, args.select_epochs, prog)
            Z, y = extract(store, f, tr, "test", a, dev)
            C = train_clf(Z, y, None, a, dev, seed)
            Zh, yh = extract(store, f, splits[h][0] + splits[h][1], "test", a, dev)
            with torch.no_grad():
                accs.append(acc(C(Zh).argmax(1), yh))
            del f
        return float(np.mean(accs))

    levels = {lv: score(lv, args.lr) for lv in (0, 1, 2)}
    level = max(levels, key=lambda lv: levels[lv])
    cfg = {"level": level, "lr": args.lr, "level_scores": levels}
    if args.select_lr:
        lrs = {args.lr: levels[level]}
        for lr in (args.lr / 2, args.lr * 2):
            lrs[lr] = score(level, lr)
        cfg["lr"] = max(lrs, key=lambda k: lrs[k])
        cfg["lr_scores"] = {str(k): v for k, v in lrs.items()}
    json.dump(cfg, open(path, "w"), indent=1)
    return cfg


def run_target(store, args, dev, target, seed):
    sources = [d for d in args.domains if d != target]
    splits = {d: split_domain(args.data, d, args.classes, seed) for d in sources}
    tr = [it for d in sources for it in splits[d][0]]
    va = [it for d in sources for it in splits[d][1]]
    te = list_domain(args.data, target, args.classes)
    units = math.ceil(args.proj_iters / 2000)
    cached = os.path.isfile(config_path(args, target))
    trials = 0 if cached else (5 if args.select_lr else 3) * len(sources)
    prog = Progress(target, trials * args.select_epochs + args.epochs + 5 + len(sources) * 2 * units * 2 + 3 * units)

    cfg = select_config(store, args, dev, target, splits, seed, prog)
    final = copy.copy(args)
    final.lr = cfg.get("lr", args.lr)
    f = train_fnet(store, final, dev, tr, va, seed, cfg["level"], args.epochs, prog)

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


def table(results, args):
    lines = [f"{DATASETS[args.dataset][0]}, ResNet-{args.arch[6:]}", f"{'Target':<14}{'Accuracy':>12}", "-" * 26]
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
    ap.add_argument("--dataset", default="pacs", choices=list(DATASETS))
    ap.add_argument("--data", default="data/Homework3-PACS/PACS")
    ap.add_argument("--out", default="results")
    ap.add_argument("--arch", default="resnet18", choices=list(ARCHS))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--select_epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--select_lr", action="store_true")
    ap.add_argument("--sampler", default="auto", choices=["auto", "random", "balanced"])
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

    args.domains = resolve_domains(args.data, args.dataset)
    for d in args.domains:
        if not os.path.isdir(os.path.join(args.data, d)):
            sys.exit(f"{DATASETS[args.dataset][0]} not found at {args.data}")
    args.classes = find_classes(args.data, args.domains)
    args.num_classes = len(args.classes)
    args.balanced = args.sampler == "balanced" or (args.sampler == "auto" and args.num_classes > 20)
    args.targets = args.targets or args.domains
    if any(t not in args.domains for t in args.targets):
        sys.exit(f"targets must be among {args.domains}")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    store = Store(args.data, args.domains, args.classes, dev, say, args.dataset)
    results = {t: {} for t in args.targets}
    for seed in args.seeds:
        for target in args.targets:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            results[target][str(seed)] = run_target(store, args, dev, target, seed)
            json.dump({"args": {k: v for k, v in vars(args).items() if k != "classes"}, "results": results},
                      open(os.path.join(args.out, "results.json"), "w"), indent=1)
    text = table(results, args)
    open(os.path.join(args.out, "results.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
