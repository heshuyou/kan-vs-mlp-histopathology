# -*- coding: utf-8 -*-
"""
Follow-up runs of the B-spline KAN control on the nine-class CRC task and the cross-domain task (fills the gap under Fig. 3).
Reuses cached features and can resume after an interruption.

Revision of 2026-11-11:

  1) both heads are forced to share one batch size (256 by default).
     The original ran the B-spline head at 256 and the matched MLP at 512, so a single comparison
     mixed two batch sizes, contradicting the statement that every head within one dataset uses
     the same batch size, and making +0.34 pp / -0.63 pp impossible to attribute.

  2) the nine-class task now selects the epoch on validation macro-F1 (the original used accuracy),
     matching crc_9class_heads.py and the selection rule in Methods of the paper.
     The cross-domain setting does not early-stop (that would leak the test set), so the option has no effect there.

  3) macro-F1 is recorded per seed (the original kept accuracy only), which supplies the F1 column of the table.

  4) results go to bspline_extra_result_b{batch}.json and never overwrite earlier ones;
     the partial files are likewise separated by batch.

Usage (from the KAN directory):
    conda activate gpt2
    python bspline_kan_extra.py --seeds 20 --batch 256
    python bspline_kan_extra.py --batch 256 --only cross --cross-seeds 10
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from bspline_kan_experiment import (build_bspline_kan_head, build_mlp_matched,
                                     set_seed, n_params, DEFAULT_SEEDS, DEVICE)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRC_FEATS = os.path.join(BASE, "KAN", "crc9_resnet50_feats.npz")
CROSS_FEATS = os.path.join(BASE, "KAN", "colon_gen_feats.npz")


def macro_f1(y_true, y_pred, n_cls):
    """Macro-averaged F1 (each class weighted equally), the same definition as sklearn average='macro'."""
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    f1s = []
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def evaluate(head, X, Y, n_cls, batch_size=512):
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                    batch_size=batch_size)
    ys, ps = [], []
    head.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            ps.append(head(xb).argmax(1).cpu().numpy())
            ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float((y == p).mean() * 100), macro_f1(y, p, n_cls)


def train_model(head, Xtr, Ytr, Xev, Yev, n_cls, epochs=15, batch_size=256,
                select="acc", early_stop=True):
    """Train one head and return (accuracy, macro_f1).

    select: 'acc' or 'f1', the metric that decides the epoch (ignored when early_stop=False).
    """
    head = head.to(DEVICE)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                       batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_key, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if early_stop:
            acc, f1 = evaluate(head, Xev, Yev, n_cls, batch_size=512)
            key = acc if select == "acc" else f1
            if key > best_key:
                best_key = key
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if early_stop and best_state is not None:
        head.load_state_dict(best_state)
    return evaluate(head, Xev, Yev, n_cls, batch_size=512)


def run_setting(name, feats_path, n_cls, xev_key, yev_key, seeds, batch,
                select="acc", early_stop=True):
    d = np.load(feats_path)
    Xtr = d["Xtr"].astype(np.float32)
    Ytr = d["Ytr"]
    Xev = d[xev_key].astype(np.float32)
    Yev = d[yev_key]
    dim = Xtr.shape[1]
    print(f"[{name}] train {Xtr.shape} | eval {Xev.shape} | n_cls {n_cls} | "
          f"batch {batch} | select {select if early_stop else 'none'} | seeds {len(seeds)}",
          flush=True)

    bkan = build_bspline_kan_head(dim, n_cls=n_cls)
    mm, H = build_mlp_matched(dim, n_params(bkan), n_cls=n_cls)
    print(f"  B-spline KAN {n_params(bkan):,} | matched MLP(H={H}) {n_params(mm):,}", flush=True)

    acc_b, acc_m, f1_b, f1_m = [], [], [], []
    for seed in seeds:
        set_seed(seed)
        a, f = train_model(build_bspline_kan_head(dim, n_cls=n_cls),
                           Xtr, Ytr, Xev, Yev, n_cls,
                           batch_size=batch, select=select, early_stop=early_stop)
        set_seed(seed)
        a2, f2 = train_model(build_mlp_matched(dim, n_params(bkan), n_cls=n_cls)[0],
                             Xtr, Ytr, Xev, Yev, n_cls,
                             batch_size=batch, select=select, early_stop=early_stop)
        acc_b.append(a); f1_b.append(f)
        acc_m.append(a2); f1_m.append(f2)
        print(f"  seed {seed:>4d}: KAN {a:6.3f}% / F1 {f:.4f} | "
              f"MLP {a2:6.3f}% / F1 {f2:.4f}", flush=True)

    b, m = np.array(acc_b), np.array(acc_m)
    fb, fm = np.array(f1_b), np.array(f1_m)
    t, p = stats.ttest_rel(b, m)
    F = max(b.var(ddof=1), m.var(ddof=1)) / min(b.var(ddof=1), m.var(ddof=1))
    p_f = 2 * (1 - stats.f.cdf(F, len(b) - 1, len(m) - 1))
    print(f"  => acc  KAN {b.mean():.3f}% ± {b.std(ddof=1):.3f}% | "
          f"MLP {m.mean():.3f}% ± {m.std(ddof=1):.3f}% | diff {(b - m).mean():+.3f}% | "
          f"t={t:.2f} p={p:.4g} | F={F:.2f} p_f={p_f:.4g}", flush=True)
    print(f"  => F1   KAN {fb.mean():.4f} ± {fb.std(ddof=1):.4f} | "
          f"MLP {fm.mean():.4f} ± {fm.std(ddof=1):.4f} | diff {(fb - fm).mean():+.4f}",
          flush=True)

    return {"name": name, "n_cls": n_cls, "batch": batch,
            "seeds": list(seeds), "select": select if early_stop else "none",
            "bspline_params": n_params(bkan), "mlp_params": n_params(mm), "H": H,
            "bspline": b.tolist(), "mlp": m.tolist(),
            "bspline_f1": fb.tolist(), "mlp_f1": fm.tolist(),
            "bspline_acc_mean": float(b.mean()), "bspline_acc_sd": float(b.std(ddof=1)),
            "mlp_acc_mean": float(m.mean()), "mlp_acc_sd": float(m.std(ddof=1)),
            "bspline_f1_mean": float(fb.mean()), "bspline_f1_sd": float(fb.std(ddof=1)),
            "mlp_f1_mean": float(fm.mean()), "mlp_f1_sd": float(fm.std(ddof=1)),
            "diff_mean": float((b - m).mean()), "t": float(t), "p": float(p),
            "F": float(F), "p_f": float(p_f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds for the nine-class task (the paper uses 20)")
    ap.add_argument("--cross-seeds", type=int, default=10, help="number of seeds for the cross-domain task (the paper uses 10)")
    ap.add_argument("--batch", type=int, default=256, help="batch size shared by both heads")
    ap.add_argument("--only", choices=["both", "crc9", "cross"], default="both")
    ap.add_argument("--force", action="store_true", help="ignore the partial file and rerun everything")
    args = ap.parse_args()

    partial = os.path.join(BASE, "KAN", f"bspline_extra_partial_b{args.batch}.json")
    results = {}
    if os.path.exists(partial) and not args.force:
        with open(partial, encoding="utf-8") as f:
            results = json.load(f)
        print(f"loading partial {os.path.basename(partial)}: {list(results)} already present")

    if args.only in ("both", "crc9") and "crc9" not in results:
        results["crc9"] = run_setting("CRC 9-class", CRC_FEATS, 9, "Xva", "Yva",
                                      DEFAULT_SEEDS[:args.seeds], args.batch, select="f1")
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"partial saved {os.path.basename(partial)}")

    if args.only in ("both", "cross") and "cross" not in results:
        results["cross"] = run_setting("Cross-domain", CROSS_FEATS, 2, "Xte", "Yte",
                                       DEFAULT_SEEDS[:args.cross_seeds], args.batch,
                                       select="acc", early_stop=False)
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"partial saved {os.path.basename(partial)}")

    out = os.path.join(BASE, "KAN", f"bspline_extra_result_b{args.batch}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {os.path.basename(out)} (the old bspline_extra_result.json is untouched)")


if __name__ == "__main__":
    main()
