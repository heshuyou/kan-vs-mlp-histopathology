# -*- coding: utf-8 -*-
"""
Experiment A: B-spline KAN control, answering the objection that only Chebyshev was tested.
Implements a B-spline KAN layer (shared grid, grid_size=5, spline_order=3),
trains it on cached ResNet50 features and compares it with a parameter-matched wide MLP over 10 seeds.

Usage: conda activate gpt2 && python bspline_kan_experiment.py --seeds 10
"""
import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = os.path.join(BASE, "KAN", "lc25000_feats_resnet50.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 5
DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 100, 200, 300, 400, 500,
                 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class BsplineKANLayer(nn.Module):
    """B-spline KAN layer (shared grid), base (linear) + spline (B-spline)."""

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
        # Shared grid (with boundary extension)
        h_g = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.cat([
            torch.linspace(grid_range[0] - h_g * spline_order, grid_range[0] - h_g, spline_order),
            torch.linspace(grid_range[0], grid_range[1], grid_size + 1),
            torch.linspace(grid_range[1] + h_g, grid_range[1] + h_g * spline_order, spline_order),
        ])
        self.register_buffer("grid", grid)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)                      # [B, in, 1]
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:(-k)]) * bases[:, :, 1:]
            )
        return bases                            # [B, in, grid_size + spline_order]

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
    va_dl = DataLoader(va, batch_size=batch_size)
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
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--spline-order", type=int, default=3)
    # 2026-11-11: the original version trained the B-spline head at batch 256 and the matched MLP at the default 512,
    # so a single comparison mixed two batch sizes; both heads now share --batch.
    ap.add_argument("--batch", type=int, default=512,
                    help="batch size shared by both heads (use 256 if memory is short)")
    args = ap.parse_args()

    d = np.load(FEATS)
    Xtr, Ytr, Xva, Yva = d["Xtr"].astype(np.float32), d["Ytr"], d["Xva"].astype(np.float32), d["Yva"]
    dim = Xtr.shape[1]
    seeds = DEFAULT_SEEDS[:args.seeds]
    print(f"train {Xtr.shape} | val {Xva.shape} | feature dim {dim}")

    bkan = build_bspline_kan_head(dim, grid_size=args.grid_size, spline_order=args.spline_order)
    mm, H = build_mlp_matched(dim, n_params(bkan))
    print(f"B-spline KAN params {n_params(bkan):,} | parameter-matched wide MLP(H={H}) {n_params(mm):,}")

    partial = os.path.join(BASE, "KAN", f"bspline_kan_partial_b{args.batch}.json")
    res = {"bspline_kan": [], "mlp_matched": []}
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            res = json.load(f)

    for i, seed in enumerate(seeds):
        if len(res["bspline_kan"]) > i:
            continue
        set_seed(seed)
        a_b = train_head(build_bspline_kan_head(dim, grid_size=args.grid_size, spline_order=args.spline_order),
                         Xtr, Ytr, Xva, Yva, batch_size=args.batch)
        set_seed(seed)
        a_m = train_head(build_mlp_matched(dim, n_params(bkan))[0], Xtr, Ytr, Xva, Yva,
                         batch_size=args.batch)
        res["bspline_kan"].append(a_b)
        res["mlp_matched"].append(a_m)
        print(f"  seed {seed:>4d}: B-spline KAN {a_b:6.3f}% | wide MLP {a_m:6.3f}%", flush=True)
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(res, f)

    b = np.array(res["bspline_kan"])
    m = np.array(res["mlp_matched"])
    print("\n================ B-spline KAN control ================")
    print(f"B-spline KAN ({n_params(bkan):,}): {b.mean():.3f}% ± {b.std(ddof=1):.3f}%")
    print(f"wide MLP       ({n_params(mm):,}): {m.mean():.3f}% ± {m.std(ddof=1):.3f}%")
    print(f"paired diff: {(b - m).mean():+.3f}%")
    from scipy import stats
    t, p = stats.ttest_rel(b, m)
    F = max(b.var(ddof=1), m.var(ddof=1)) / min(b.var(ddof=1), m.var(ddof=1))
    p_f = 2 * (1 - stats.f.cdf(F, len(b) - 1, len(m) - 1))
    print(f"t={t:.3f} p={p:.4f} | F={F:.2f} p={p_f:.4f}")

    out = {"batch": args.batch, "seeds": len(b),
           "bspline_params": n_params(bkan), "mlp_params": n_params(mm), "H": H,
           "bspline_kan": b.tolist(), "mlp_matched": m.tolist()}
    with open(os.path.join(BASE, "KAN", f"bspline_kan_result_b{args.batch}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"results saved to bspline_kan_result_b{args.batch}.json")


if __name__ == "__main__":
    main()
