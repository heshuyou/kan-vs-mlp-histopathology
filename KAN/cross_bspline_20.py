# -*- coding: utf-8 -*-
"""
Extends the cross-domain comparison (LC25000 colon -> CRC TUM/NORM) to 20 seeds so that Table 3 uses one seed count.

Background: Table 3 used to come in two blocks, a Chebyshev block with 20 seeds (including the narrow MLP) and a B-spline block with 10 seeds (without it).
Different seed counts and different row counts made the blocks hard to compare. This script adds
seeds 11-20 for the B-spline head and its matched MLP, reuses the first 10 from the cross block of bspline_extra_result_b256.json,
and writes a new file, cross_bspline_20_result.json, without overwriting any existing result.

Protocol identical to the original cross-domain run: batch 256, 15 epochs, cosine annealing, label smoothing 0.1, no early stopping
(the cross-domain set is a test set and never takes part in epoch selection).

Usage:
    conda activate gpt2
    python cross_bspline_20.py
"""
import os
import json
import numpy as np
from scipy import stats

import bspline_kan_experiment as bke
import bspline_kan_extra as bkx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS_FEATS = os.path.join(BASE, "KAN", "colon_gen_feats.npz")
OLD = os.path.join(BASE, "KAN", "bspline_extra_result_b256.json")
OUT = os.path.join(BASE, "KAN", "cross_bspline_20_result.json")
N_CLS = 2
BATCH = 256


def evaluate_acc(head, X, Y, batch_size=512):
    acc, _ = bkx.evaluate(head, X, Y, N_CLS, batch_size=batch_size)
    return acc


def main():
    with open(OLD, encoding="utf-8") as f:
        old = json.load(f)["cross"]
    seeds_old = list(old["seeds"])
    acc_b = list(old["bspline"])
    acc_m = list(old["mlp"])
    print(f"reusing {len(seeds_old)} previous seeds: KAN {np.mean(acc_b):.3f} | MLP {np.mean(acc_m):.3f}")

    d = np.load(CROSS_FEATS)
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xte, Yte = d["Xte"].astype(np.float32), d["Yte"]
    dim = Xtr.shape[1]

    bkan = bke.build_bspline_kan_head(dim, n_cls=N_CLS)
    mm, H = bke.build_mlp_matched(dim, bke.n_params(bkan), n_cls=N_CLS)
    print(f"B-spline {bke.n_params(bkan):,} | MLP(H={H}) {bke.n_params(mm):,}")

    rest = [s for s in bke.DEFAULT_SEEDS if s not in seeds_old][:20 - len(seeds_old)]
    print(f"seeds to run: {rest}")
    for seed in rest:
        bke.set_seed(seed)
        head = bke.build_bspline_kan_head(dim, n_cls=N_CLS).to(bkx.DEVICE)
        bkx.train_model(head, Xtr, Ytr, Xte, Yte, N_CLS, batch_size=BATCH,
                        select="acc", early_stop=False)
        acc_b.append(evaluate_acc(head, Xte, Yte))

        bke.set_seed(seed)
        head2 = bke.build_mlp_matched(dim, bke.n_params(bkan), n_cls=N_CLS)[0].to(bkx.DEVICE)
        bkx.train_model(head2, Xtr, Ytr, Xte, Yte, N_CLS, batch_size=BATCH,
                        select="acc", early_stop=False)
        acc_m.append(evaluate_acc(head2, Xte, Yte))
        print(f"  seed {seed:>4d}: KAN {acc_b[-1]:6.3f}% | MLP {acc_m[-1]:6.3f}%", flush=True)

    b, m = np.array(acc_b), np.array(acc_m)
    t, p = stats.ttest_rel(b, m)
    F = max(b.var(ddof=1), m.var(ddof=1)) / min(b.var(ddof=1), m.var(ddof=1))
    p_f = 2 * (1 - stats.f.cdf(F, len(b) - 1, len(m) - 1))
    print(f"\nB-spline {b.mean():.3f}±{b.std(ddof=1):.3f} | MLP(H={H}) {m.mean():.3f}±{m.std(ddof=1):.3f}")
    print(f"paired diff {(b - m).mean():+.3f} pp | t={t:.2f} p={p:.4g} | F={F:.2f} p_f={p_f:.4g}")

    out = {"setting": "LC25000 colon -> CRC-VAL-HE-7K TUM/NORM",
           "batch": BATCH, "seeds": len(b), "seed_list": seeds_old + rest,
           "early_stop": False, "bspline_params": bke.n_params(bkan),
           "mlp_params": bke.n_params(mm), "H": H,
           "bspline": b.tolist(), "mlp": m.tolist(),
           "bspline_mean": float(b.mean()), "bspline_sd": float(b.std(ddof=1)),
           "mlp_mean": float(m.mean()), "mlp_sd": float(m.std(ddof=1)),
           "diff_mean": float((b - m).mean()), "t": float(t), "p": float(p),
           "F": float(F), "p_f": float(p_f),
           "first10_from": "bspline_extra_result_b256.json cross block"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved cross_bspline_20_result.json")


if __name__ == "__main__":
    main()
