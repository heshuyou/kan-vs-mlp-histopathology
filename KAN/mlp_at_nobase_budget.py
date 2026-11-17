# -*- coding: utf-8 -*-
"""
Retrains the MLP to the parameter count of the "B-spline head without base weight" (2,180,800),
so that the control becomes a strictly parameter-aligned comparison.

Protocol identical to the upper block of Table 4: batch 512, 15 epochs, cosine annealing, label smoothing 0.1,
epoch chosen on validation accuracy, first 20 seeds.
Results are written to mlp_at_nobase_budget_result.json.
"""
import os, json
import numpy as np
from scipy import stats

import bspline_kan_experiment as bke

BASE = bke.BASE
TARGET = 2180800


def main():
    d = np.load(bke.FEATS)
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xva, Yva = d["Xva"].astype(np.float32), d["Yva"]
    dim = Xtr.shape[1]
    mlp, H = bke.build_mlp_matched(dim, TARGET)
    p = bke.n_params(mlp)
    print(f"target {TARGET:,} | actual H={H} params {p:,} | diff {p - TARGET:+,}")

    accs = []
    for seed in bke.DEFAULT_SEEDS:
        bke.set_seed(seed)
        a = bke.train_head(bke.build_mlp_matched(dim, TARGET)[0], Xtr, Ytr, Xva, Yva,
                           batch_size=512)
        accs.append(a)
        print(f"  seed {seed:>4d}: {a:6.3f}%", flush=True)
    m = np.array(accs)

    ref = os.path.join(BASE, "KAN", "bspline_nobasew_result.json")
    with open(ref, encoding="utf-8") as f:
        nb = np.array(json.load(f)["nobase"])
    diff = nb - m
    t, pv = stats.ttest_rel(nb, m)
    print(f"\nno-basew B-spline {nb.mean():.3f}±{nb.std(ddof=1):.3f} (2,180,800)")
    print(f"strictly matched MLP(H={H}) {m.mean():.3f}±{m.std(ddof=1):.3f} ({p:,})")
    print(f"paired diff {diff.mean():+.3f} pp | t={t:.2f} p={pv:.4g}")

    out = {"H": H, "mlp_params": p, "target_params": TARGET,
           "seeds": bke.DEFAULT_SEEDS, "mlp_acc": accs,
           "mlp_mean": float(m.mean()), "mlp_sd": float(m.std(ddof=1)),
           "nobase_mean": float(nb.mean()), "diff_mean": float(diff.mean()),
           "t": float(t), "p": float(pv)}
    with open(os.path.join(BASE, "KAN", "mlp_at_nobase_budget_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved mlp_at_nobase_budget_result.json")


if __name__ == "__main__":
    main()
