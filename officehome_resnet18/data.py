import os
import random

import numpy as np
import torch
from PIL import Image

DOMAINS = ["Art", "Clipart", "Product", "Real World"]


def list_domain(root, domain, classes):
    items = []
    for c, name in enumerate(classes):
        folder = os.path.join(root, domain, name)
        items += [(os.path.join(folder, f), c) for f in sorted(os.listdir(folder))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return items


def split_domain(root, domain, classes):
    items = list_domain(root, domain, classes)
    idx = list(range(len(items)))
    random.Random(0).shuffle(idx)
    nv = int(round(0.1 * len(items)))
    return [items[i] for i in idx[nv:]], [items[i] for i in idx[:nv]]


class Store:
    def __init__(self, root, classes, dev, size=227):
        root = os.path.abspath(root)
        items = [it for d in DOMAINS for it in list_domain(root, d, classes)]
        rel = [os.path.relpath(p, root) for p, _ in items]
        cache = os.path.join(os.path.dirname(root), f".officehome_cache_{size}.pt")
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
            torch.save({"paths": rel, "x": x}, cache)
        self.root, self.dev = root, dev
        self.index = {p: i for i, p in enumerate(rel)}
        self.x = x.to(dev)
        self.y = torch.tensor([c for _, c in items], device=dev)

    def idx(self, items):
        return torch.tensor([self.index[os.path.relpath(p, self.root)] for p, _ in items], device=self.dev)
