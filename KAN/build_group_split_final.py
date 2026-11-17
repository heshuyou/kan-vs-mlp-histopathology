# -*- coding: utf-8 -*-
"""Rebuild the grouped split from the saved pair/NCC table at a chosen threshold,
and export a few pairs inside 0.995-0.999 for visual inspection."""
import os, json, shutil
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(BASE, "KAN", "lc25000_dup_pairs_v5.npz")
THR = 0.995

d = np.load(PAIRS)
pairs, ncc, labels = d["pairs"], d["ncc"], d["labels"]
n = len(labels)

keep = pairs[ncc >= THR]
A = coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
ncomp, roots = connected_components(A, directed=False)
u, cnt = np.unique(roots, return_counts=True)

tr_rows = [ln.split() for ln in open(os.path.join(BASE, "train_clean.txt"), encoding="utf-8") if ln.strip()]
va_rows = [ln.split() for ln in open(os.path.join(BASE, "val_clean.txt"), encoding="utf-8") if ln.strip()]
paths = [r[0] for r in tr_rows] + [r[0] for r in va_rows]
y = np.array([int(r[1]) for r in tr_rows] + [int(r[1]) for r in va_rows])

rng = np.random.RandomState(0)
is_val = np.zeros(n, dtype=bool)
for c in np.unique(y):
    g = np.unique(roots[y == c])
    rng.shuffle(g)
    is_val |= np.isin(roots, g[: int(round(0.20 * len(g)))])

with open(os.path.join(BASE, "train_group2.txt"), "w", encoding="utf-8") as fo:
    fo.write("\n".join(paths[i] for i in range(n) if not is_val[i]) + "\n")
with open(os.path.join(BASE, "val_group2.txt"), "w", encoding="utf-8") as fo:
    fo.write("\n".join(paths[i] for i in range(n) if is_val[i]) + "\n")
np.save(os.path.join(BASE, "KAN", "lc25000_group2_labels.npy"), roots)

# cross-partition check: no group may appear on both sides
bad = len(set(roots[is_val]) & set(roots[~is_val]))
stats = {"threshold": float(THR), "n_images": int(n), "n_pairs": int(len(keep)),
         "n_components": int(ncomp), "n_multi": int((cnt > 1).sum()),
         "images_in_groups": int(cnt[cnt > 1].sum()),
         "frac_in_groups": float(cnt[cnt > 1].sum() / n),
         "max_cluster": int(cnt.max()), "size_hist": np.bincount(cnt)[:10].tolist(),
         "n_train": int((~is_val).sum()), "n_val": int(is_val.sum()),
         "groups_spanning_split": int(bad),
         "median_ncc": float(np.median(ncc[ncc >= THR]))}
json.dump(stats, open(os.path.join(BASE, "KAN", "lc25000_group2_stats.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(stats, indent=2))

# export pairs in the ambiguous band for visual inspection
out = os.path.join(BASE, "KAN", "_pair_preview2")
os.makedirs(out, exist_ok=True)
band = np.where((ncc >= THR) & (ncc < 0.999))[0]
pick = np.random.RandomState(3).choice(band, min(5, len(band)), replace=False)
for k, p in enumerate(pick):
    a, b = int(pairs[p, 0]), int(pairs[p, 1])
    print("pair %d ncc %.4f" % (k, ncc[p]), paths[a], paths[b])
    for tag, m in (("a", a), ("b", b)):
        shutil.copy(os.path.join(BASE, paths[m]), os.path.join(out, f"q{k}_{tag}_{os.path.basename(paths[m])}"))
