# -*- coding: utf-8 -*-
"""
CRITICAL-1 follow-up: the B-spline head with the linear base weight removed from every edge.

The question
-----------
In the upper block of Table 4 the B-spline KAN beats the parameter-matched MLP by +0.364 pp (p<1e-4). Each B-spline
edge, however, carries a linear base weight and 8 spline coefficients, whereas the Chebyshev edge has no base weight,
so the 0.36 cannot be attributed to the spline basis rather than to that extra linear path.

This script drops the base weight (keeping only the B-spline coefficients) and changes nothing else:

  * data: lc25000_feats_resnet50.npz (18,976 train / 4,744 validation)
  * architecture: 2048->128->64->32->5, grid=5, order=3, BN+SiLU+Dropout(0.2) per block
  * protocol: batch 512, AdamW lr 1e-3 wd 1e-4, cosine annealing, 15 epochs,
          label smoothing 0.1, epoch chosen on validation accuracy (identical to Table 4)
  * seeds: the first 20 by default, the same seeds as Table 4

The MLP column is not re-run: it is read from bspline_kanresnet50_b512_result.json, so the
99.312 ± 0.070 it reports is the same set of numbers.

How to read the result
---------
  difference ≈ +0.36 (as in Table 4) -> the gain comes from the spline basis; CRITICAL-1 is resolved
  difference ≈ 0 (significance gone)  -> the gain comes from that linear path, and the conclusion becomes
                                "what matters is the linear term on each edge, not the spline basis"
  anything in between                 -> both contribute, and the shared contribution is reported

Usage (from the KAN directory; a GPU is required, CPU is slow)
    conda activate gpt2
    python bspline_kan_nobasew.py --batch 512 --seeds 20
    python bspline_kan_nobasew.py --batch 256 --seeds 20    # optional: re-check at another batch
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from bspline_kan_experiment import (BsplineKANLayer, BsplineKANBlock,
                                    build_bspline_kan_head, build_mlp_matched,
                                    set_seed, n_params, DEFAULT_SEEDS,
                                    FEATS, DEVICE, N_CLS)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoBaseBsplineLayer(BsplineKANLayer):
    """Same shape as BsplineKANLayer but without the linear base weight (spline terms only)."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__(in_features, out_features, grid_size, spline_order)
        del self.base_weight          # removed from _parameters, so the parameter count drops accordingly

    def forward(self, x):
        spline = F.linear(self.b_splines(x).reshape(x.size(0), -1), self.spline_weight)
        return spline


class NoBaseBsplineBlock(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, dropout=0.2):
        super().__init__()
        self.kan = NoBaseBsplineLayer(in_features, out_features, grid_size, spline_order)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.kan(x))))


def build_nobase_head(d_in, n_cls=N_CLS, grid_size=5, spline_order=3):
    return nn.Sequential(
        NoBaseBsplineBlock(d_in, 128, grid_size, spline_order, 0.2),
        NoBaseBsplineBlock(128, 64, grid_size, spline_order, 0.2),
        NoBaseBsplineBlock(64, 32, grid_size, spline_order, 0.2),
        NoBaseBsplineLayer(32, n_cls, grid_size, spline_order),
    )


def train_acc(head, Xtr, Ytr, Xva, Yva, epochs=15, batch_size=512):
    """Exactly the protocol of bspline_kan_experiment.train_head: epoch chosen on validation accuracy."""
    head = head.to(DEVICE)
    tr_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=batch_size, shuffle=True)
    va_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        head.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        sched.step()
        head.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                corr += (head(xb).argmax(1) == yb).sum().item()
                tot += yb.size(0)
        acc = corr / tot
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return best_acc * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = np.load(FEATS)
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xva, Yva = d["Xva"].astype(np.float32), d["Yva"]
    dim = Xtr.shape[1]
    seeds = DEFAULT_SEEDS[:args.seeds]

    # reference column: the 20 MLP values of the upper block of Table 4, read as is, not re-run
    ref_path = os.path.join(BASE, "KAN", "bspline_kanresnet50_b512_result.json")
    if not os.path.exists(ref_path):
        raise SystemExit("bspline_kanresnet50_b512_result.json is missing, so the MLP column of Table 4 cannot be reused")
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    mlp_ref = ref["mlp_matched"]
    mlp_H = ref.get("H")
    if len(mlp_ref) < len(seeds):
        raise SystemExit("the reference MLP has only %d seeds, fewer than the %d requested" % (len(mlp_ref), len(seeds)))

    head0 = build_nobase_head(dim)
    print(f"[B-spline without base weight] params {n_params(head0):,}")
    print(f"[B-spline of Table 4]          params {ref['bspline_params']:,}")
    print(f"  the two differ by {ref['bspline_params'] - n_params(head0):,} parameters, the removed base weights")
    print(f"train {Xtr.shape} | val {Xva.shape} | batch {args.batch} | seeds {len(seeds)}")

    partial = os.path.join(BASE, "KAN", f"bspline_nobasew_partial_b{args.batch}.json")
    res = {"nobase": []}
    if os.path.exists(partial) and not args.force:
        with open(partial, encoding="utf-8") as f:
            res = json.load(f)

    for i, seed in enumerate(seeds):
        if len(res["nobase"]) > i:
            continue
        set_seed(seed)
        a = train_acc(build_nobase_head(dim), Xtr, Ytr, Xva, Yva, batch_size=args.batch)
        res["nobase"].append(a)
        print(f"  seed {seed:>4d}: no-basew {a:6.3f}% | Table 4 MLP {mlp_ref[i]:6.3f}%", flush=True)
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(res, f)

    nb = np.array(res["nobase"])
    mm = np.array(mlp_ref[:len(nb)])
    diff = nb - mm
    t, p = stats.ttest_rel(nb, mm)
    F = max(nb.var(ddof=1), mm.var(ddof=1)) / min(nb.var(ddof=1), mm.var(ddof=1))
    p_f = 2 * (1 - stats.f.cdf(F, len(nb) - 1, len(mm) - 1))

    print("\n=========== B-spline without base weight vs parameter-matched MLP ===========")
    print(f"no-basew B-spline ({n_params(head0):,}): {nb.mean():.3f}% ± {nb.std(ddof=1):.3f}%")
    print(f"wide MLP (H={mlp_H})   ({n_params(build_mlp_matched(dim, n_params(head0))[0]):,}): "
          f"{mm.mean():.3f}% ± {mm.std(ddof=1):.3f}%")
    print(f"paired diff {diff.mean():+.3f} pp | t={t:.2f} p={p:.4g} | F={F:.2f} p_f={p_f:.4g}")
    print("reference: the B-spline delta of Table 4 is +0.364 pp (p<1e-4)")

    out = {"setting": "LC25000 + ResNet50, B-spline without base weight",
           "batch": args.batch, "seeds": len(nb), "seed_list": seeds[:len(nb)],
           "select": "acc, early stopping on the reported partition",
           "nobase_params": n_params(head0),
           "bspline_with_base_params": ref["bspline_params"],
           "mlp_ref_from": "bspline_kan_result_b512.json",
           "nobase": nb.tolist(), "mlp": mm.tolist(),
           "nobase_acc_mean": float(nb.mean()), "nobase_acc_sd": float(nb.std(ddof=1)),
           "mlp_acc_mean": float(mm.mean()), "mlp_acc_sd": float(mm.std(ddof=1)),
           "diff_mean": float(diff.mean()), "t": float(t), "p": float(p),
           "F": float(F), "p_f": float(p_f)}
    out_path = os.path.join(BASE, "KAN", "bspline_nobasew_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"results saved to {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
