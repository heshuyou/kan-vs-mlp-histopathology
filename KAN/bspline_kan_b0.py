# -*- coding: utf-8 -*-
"""
Experiment A-B0: B-spline KAN control on the EfficientNet-B0 branch (1,280-d features).

Same layer definitions, training loop and statistics as bspline_kan_experiment.py, with three differences:
  1. it reads the B0 feature cache (by default it searches KAN/lc25000_feats_*b0*.npz / *efficientnet*.npz);
  2. the B-spline head and the parameter-matched wide MLP use the same batch size (512 by default),
     which removes the 256 vs 512 asymmetry of the ResNet50 branch (disclosed in the table note of the paper);
  3. results go to bspline_kan_b0_result.json and leave existing result files untouched.

Expected parameter counts (1,280-d input): B-spline KAN 1,568,608; matched wide MLP at width H=620 (about 1,567,365).

Usage:
    conda activate gpt2
    cd code
    python bspline_kan_b0.py --seeds 20                # default batch 512, identical for both heads
    python bspline_kan_b0.py --seeds 20 --batch 256    # drop both heads to 256 if 4 GB of VRAM is not enough

To also run the batch-consistency check for the ResNet50 branch (answers the 0.43 pp question):
    python bspline_kan_b0.py --feats ../KAN/lc25000_feats_resnet50.npz \
        --tag resnet50_b256 --batch 256 --seeds 10

Note: a different --tag produces different partial and result files; bspline_kan_result.json is never overwritten.
"""
import os
import glob
import math
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 5
DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 100, 200, 300, 400, 500,
                 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def resolve_feats(path_arg):
    """When --feats is not given, locate the EfficientNet-B0 feature cache automatically."""
    if path_arg:
        if not os.path.exists(path_arg):
            raise FileNotFoundError(f"feature file not found: {path_arg}")
        return path_arg

    kandir = os.path.join(BASE, "KAN")
    cands = glob.glob(os.path.join(kandir, "lc25000_feats_*.npz"))
    if not cands:
        raise FileNotFoundError(
            f"no lc25000_feats_*.npz under {kandir}; run the feature extraction once\n"
            f"  python clean_lc25000_heads.py --backbone efficientnet_b0 --seeds 1\n"
            f"to create the cache, or pass --feats explicitly.")
    # prefer b0 / efficientnet
    keyed = [p for p in cands if ("b0" in os.path.basename(p).lower()
                                or "efficient" in os.path.basename(p).lower())]
    if len(keyed) == 1:
        return keyed[0]
    print("[!] several candidate feature files found; pick one with --feats:")
    for p in (keyed or cands):
        print("    " + p)
    raise SystemExit(1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class BsplineKANLayer(nn.Module):
    """B-spline KAN layer (shared grid), base (linear) + spline (B-spline). Same as the main experiment."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_size + spline_order) * in_features
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, h))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        h_g = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.cat([
            torch.linspace(grid_range[0] - h_g * spline_order, grid_range[0] - h_g, spline_order),
            torch.linspace(grid_range[0], grid_range[1], grid_size + 1),
            torch.linspace(grid_range[1] + h_g, grid_range[1] + h_g * spline_order, spline_order),
        ])
        self.register_buffer("grid", grid)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:(-k)]) * bases[:, :, 1:]
            )
        return bases

    def forward(self, x):
        base = F.linear(x, self.base_weight)
        spline = F.linear(self.b_splines(x).reshape(x.size(0), -1), self.spline_weight)
        return base + spline


class BsplineKANBlock(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, dropout=0.2):
        super().__init__()
        self.kan = BsplineKANLayer(in_features, out_features, grid_size, spline_order)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.kan(x))))


def build_bspline_kan_head(d_in, n_cls=N_CLS, grid_size=5, spline_order=3):
    return nn.Sequential(
        BsplineKANBlock(d_in, 128, grid_size, spline_order, 0.2),
        BsplineKANBlock(128, 64, grid_size, spline_order, 0.2),
        BsplineKANBlock(64, 32, grid_size, spline_order, 0.2),
        BsplineKANLayer(32, n_cls, grid_size, spline_order),
    )


def build_mlp_matched(d_in, target, n_cls=N_CLS):
    """Same width search as the main experiment: one width H for all three layers, step 4."""
    best_H, best_diff = None, None
    for H in range(64, 2049, 4):
        p = 2 * H * H + (d_in + n_cls + 2) * H + n_cls
        diff = abs(p - target)
        if best_diff is None or diff < best_diff:
            best_H, best_diff = H, diff
    H = best_H
    return nn.Sequential(
        nn.Linear(d_in, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H, n_cls),
    ), H


def train_head(head, Xtr, Ytr, Xva, Yva, epochs=15, batch_size=512):
    head = head.to(DEVICE)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va, batch_size=max(batch_size, 512))  # affects evaluation speed only, not the result
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        head.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
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
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds, at most 20 (the first N of the fixed seed list used in the paper)")
    ap.add_argument("--batch", type=int, default=512, help="batch size shared by both heads")
    ap.add_argument("--bspline-batch", type=int, default=None,
                    help="only needed when memory is short and the B-spline head must run at a smaller batch; defaults to --batch")
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--spline-order", type=int, default=3)
    ap.add_argument("--feats", type=str, default=None, help="path to the feature npz; the B0 cache is searched by default")
    ap.add_argument("--tag", type=str, default=None, help="suffix for the result file; inferred from the feature file name by default")
    args = ap.parse_args()

    feats = resolve_feats(args.feats)
    tag = args.tag or ("_b0" if ("b0" in feats.lower() or "efficient" in feats.lower())
                       else "_" + os.path.basename(feats).replace("lc25000_feats_", "").replace(".npz", ""))
    kdir = os.path.join(BASE, "KAN")
    partial_path = os.path.join(kdir, f"bspline_kan{tag}_partial.json")
    result_path = os.path.join(kdir, f"bspline_kan{tag}_result.json")

    d = np.load(feats)
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xva, Yva = d["Xva"].astype(np.float32), d["Yva"]
    dim = int(Xtr.shape[1])
    seeds = DEFAULT_SEEDS[:args.seeds]
    print(f"feature file: {feats}")
    print(f"train {Xtr.shape} | val {Xva.shape} | feature dim {dim} | tag {tag}")
    if dim != 1280:
        print(f"[!] feature dim is {dim}, not the 1280 expected for EfficientNet-B0; check that --feats points at the right file.")

    bkan = build_bspline_kan_head(dim, grid_size=args.grid_size, spline_order=args.spline_order)
    mm, H = build_mlp_matched(dim, n_params(bkan))
    bs_batch = args.bspline_batch or args.batch
    print(f"B-spline KAN params {n_params(bkan):,} | parameter-matched wide MLP(H={H}) {n_params(mm):,}")
    print(f"batch: B-spline {bs_batch} | wide MLP {args.batch}")
    if bs_batch != args.batch:
        print("[!] the two heads use different batch sizes; the B0 comparison is then not fully controlled.")

    res = {"bspline_kan": [], "mlp_matched": [], "epoch_sec": []}
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            res = json.load(f)
        print(f"partial file found, {len(res['bspline_kan'])} seeds already done, resuming.")

    for i, seed in enumerate(seeds):
        if len(res["bspline_kan"]) > i:
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        try:
            set_seed(seed)
            a_b = train_head(build_bspline_kan_head(dim, grid_size=args.grid_size
                                                    , spline_order=args.spline_order),
                             Xtr, Ytr, Xva, Yva, batch_size=bs_batch)
            set_seed(seed)
            a_m = train_head(build_mlp_matched(dim, n_params(bkan))[0],
                             Xtr, Ytr, Xva, Yva, batch_size=args.batch)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] out of memory at seed {seed}. Re-run with a smaller batch:")
            print(f"    python bspline_kan_b0.py --seeds {args.seeds} --batch 256")
            torch.cuda.empty_cache()
            raise SystemExit(1)
        res["bspline_kan"].append(float(a_b))
        res["mlp_matched"].append(float(a_m))
        res["epoch_sec"].append(round((time.time() - t0) / 2, 1))
        print(f"  seed {seed:>4d}: B-spline KAN {a_b:6.3f}% | wide MLP {a_m:6.3f}%"
              f" ({res['epoch_sec'][-1]}s/head)", flush=True)
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(res, f)

    b = np.array(res["bspline_kan"])
    m = np.array(res["mlp_matched"])
    print("\n============== B-spline KAN control (B0 / %s) ==========" % tag)
    print(f"B-spline KAN ({n_params(bkan):,}): {b.mean():.3f}% ± {b.std(ddof=1):.3f}%  (n={len(b)})")
    print(f"wide MLP       ({n_params(mm):,}): {m.mean():.3f}% ± {m.std(ddof=1):.3f}%  (n={len(m)})")
    print(f"paired diff: {(b - m).mean():+.3f}%")
    if len(b) >= 2:
        from scipy import stats
        t, p = stats.ttest_rel(b, m)
        F = max(b.var(ddof=1), m.var(ddof=1)) / min(b.var(ddof=1), m.var(ddof=1))
        p_f = 2 * (1 - stats.f.cdf(F, len(b) - 1, len(m) - 1))
        print(f"t={t:.3f} p={p:.4f} | F={F:.2f} p={p_f:.4f}")

    out = {"feats": os.path.basename(feats), "dim": dim, "tag": tag,
           "bspline_batch": bs_batch, "mlp_batch": args.batch,
           "bspline_params": n_params(bkan), "mlp_params": n_params(mm), "H": H,
           "seeds": seeds[:len(b)],
           "bspline_kan": b.tolist(), "mlp_matched": m.tolist()}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"results saved to {os.path.basename(result_path)}")


if __name__ == "__main__":
    main()
