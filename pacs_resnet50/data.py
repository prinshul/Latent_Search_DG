import os
import random

import numpy as np
import torch
from PIL import Image

DATASETS = {
    "pacs": ("PACS", ["art_painting", "cartoon", "photo", "sketch"]),
    "officehome": ("Office-Home", ["Art", "Clipart", "Product", "Real World"]),
}
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_domains(root, dataset):
    domains = []
    for d in DATASETS[dataset][1]:
        alt = d.replace(" ", "_")
        domains.append(d if os.path.isdir(os.path.join(root, d)) or not os.path.isdir(os.path.join(root, alt)) else alt)
    return domains


def find_classes(root, domains):
    return sorted(c for c in os.listdir(os.path.join(root, domains[0])) if os.path.isdir(os.path.join(root, domains[0], c)))


def list_domain(root, domain, classes):
    items = []
    for c, name in enumerate(classes):
        folder = os.path.join(root, domain, name)
        items += [(os.path.join(folder, f), c) for f in sorted(os.listdir(folder)) if f.lower().endswith(EXT)]
    return items


def split_domain(root, domain, classes, seed):
    items = list_domain(root, domain, classes)
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    nv = int(round(0.1 * len(items)))
    return [items[i] for i in idx[nv:]], [items[i] for i in idx[:nv]]


class Store:
    def __init__(self, root, domains, classes, dev, log, name="pacs", size=227):
        root = os.path.abspath(root)
        items = [it for d in domains for it in list_domain(root, d, classes)]
        rel = [os.path.relpath(os.path.abspath(p), root) for p, _ in items]
        parent = os.path.dirname(root)
        cache = os.path.join(parent if os.access(parent, os.W_OK) else ".", f".{name}_cache_{size}.pt")
        x = None
        if os.path.isfile(cache):
            c = torch.load(cache, weights_only=False)
            if c.get("paths") == rel:
                x = c["x"]
        if x is None:
            x = torch.empty(len(items), 3, size, size, dtype=torch.uint8)
            for i, (p, _) in enumerate(items):
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if im.size != (size, size):
                        im = im.resize((size, size), Image.BILINEAR)
                    x[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
                if i % 250 == 0:
                    log(f"loading images {i + 1}/{len(items)}")
            tmp = f"{cache}.{os.getpid()}"
            torch.save({"paths": rel, "x": x}, tmp)
            os.replace(tmp, cache)
        log(f"loaded {len(items)} images", end=True)
        self.root, self.dev = root, dev
        self.index = {p: i for i, p in enumerate(rel)}
        self.x = x.to(dev)
        self.y = torch.tensor([c for _, c in items], device=dev)

    def idx(self, items):
        return torch.tensor([self.index[os.path.relpath(os.path.abspath(p), self.root)] for p, _ in items],
                            device=self.dev)
