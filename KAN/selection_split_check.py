# -*- coding: utf-8 -*-
"""
Follow-up run that separates the partition used to choose the epoch from the partition used to report the result.

The problem
----
Every script today (clean_lc25000_heads, crc_9class_heads, bspline_kan_experiment,
bspline_kan_extra, linear_probe_eval) follows the same pattern:

        for epoch in 1..15:
            acc = evaluate(head, Xva, Yva)        # evaluate on the validation set
            if acc > best: best_state = snapshot  # choose the epoch on the validation set
        return evaluate(head, Xva, Yva)           # report on that same validation set

In other words, the reported number is the best of 15 epochs on that validation set,
so it is optimistic relative to an independent test set. A reviewer asks this in the first round.

What this script does
-----------
It carves a further 10% out of the training pool (stratified by class, fixed across seeds) as a selection set:

    train(81%) / selection(9%) / validation(10% report set)

* training uses train;
* the epoch is chosen on selection, so the model never sees the report set;
* validation is evaluated once, at the end, and that number is reported.

Three key comparisons are each run once:

    (a) LC25000    Chebyshev KAN vs H=528 wide MLP   batch 512, select acc
    (b) CRC 9-class Chebyshev KAN vs H=528 wide MLP   batch 256, select macro-F1
    (c) LC25000    B-spline KAN vs H=708 wide MLP   batch 512, select acc

The resulting delta is then compared with the delta already in the paper: if the ordering and the significance
hold, one sentence in Methods (re-checked on an independent selection split, the conclusion is unchanged) closes the point.

Usage (from the KAN directory; a GPU is required)
    conda activate gpt2
    python selection_split_check.py --seeds 10
    python selection_split_check.py --seeds 10 --only a     # only (a)
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import bspline_kan_experiment as bke
import clean_lc25000_heads as clh
import crc_9class_heads as c9

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = bke.DEVICE
SEEDS = bke.DEFAULT_SEEDS


def macro_f1(y_true, y_pred, n_cls):
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
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=batch_size)
    ys, ps = [], []
    head.eval()
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(head(xb.to(DEVICE)).argmax(1).cpu().numpy())
            ys.append(yb.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean() * 100), macro_f1(y, p, n_cls)


def stratified_split(Xtr, Ytr, frac=0.10):
    """Stratified selection split; the order is fixed and does not depend on the seed."""
    idx_sel, idx_tr = [], []
    for c in np.unique(Ytr):
        idx = np.where(Ytr == c)[0]
        n_sel = max(1, int(round(len(idx) * frac)))
        idx_sel.append(idx[:n_sel])
        idx_tr.append(idx[n_sel:])
    sel = np.concatenate(idx_sel)
    tr = np.concatenate(idx_tr)
    return Xtr[tr], Ytr[tr], Xtr[sel], Ytr[sel]


def train_with_selection(head, Xtr, Ytr, Xsel, Ysel, Xva, Yva, n_cls,
                         epochs=15, batch_size=512, select="acc"):
    head = head.to(DEVICE)
    tr_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_key, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        sched.step()
        acc, f1 = evaluate(head, Xsel, Ysel, n_cls, batch_size=512)   # selection set only
        key = acc if select == "acc" else f1
        if key > best_key:
            best_key, best_epoch = key, ep
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    acc_va, f1_va = evaluate(head, Xva, Yva, n_cls, batch_size=512)     # evaluated once only
    return acc_va, f1_va, best_epoch


CONFIGS = {
    "a": dict(name="LC25000 Chebyshev vs H=528", feats="lc25000_feats_resnet50.npz",
              n_cls=5, kind="cheb", batch=512, select="acc"),
    "b": dict(name="CRC 9-class Chebyshev vs H=528", feats="crc9_resnet50_feats.npz",
              n_cls=9, kind="cheb", batch=256, select="f1"),
    "c": dict(name="LC25000 B-spline vs H=708", feats="lc25000_feats_resnet50.npz",
              n_cls=5, kind="bspline", batch=512, select="acc"),
}


def build_pair(kind, dim, n_cls):
    if kind == "bspline":
        kan = bke.build_bspline_kan_head(dim, n_cls=n_cls)
        mlp, H = bke.build_mlp_matched(dim, bke.n_params(kan), n_cls=n_cls)
        return kan, mlp, H, bke.n_params(kan), bke.n_params(mlp)
    if n_cls == 9:
        kan = c9.build_kan_head(dim, n_cls=n_cls)
        mlp, H = c9.build_mlp_matched(dim, n_cls=n_cls, target=bke.n_params(kan))
    else:
        kan = clh.build_kan_head(dim, n_cls=n_cls)
        mlp, H = clh.build_mlp_matched(dim, n_cls=n_cls, target=bke.n_params(kan))
    return kan, mlp, H, bke.n_params(kan), bke.n_params(mlp)


def run_one(key, cfg, seeds):
    d = np.load(os.path.join(BASE, "KAN", cfg["feats"]))
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xva, Yva = d["Xva"].astype(np.float32), d["Yva"]
    dim = Xtr.shape[1]
    Xt, Yt, Xs, Ys = stratified_split(Xtr, Ytr, 0.10)
    print(f"[{key}] {cfg['name']} | train {Xt.shape} | select {Xs.shape} | "
          f"report {Xva.shape} | batch {cfg['batch']} | select={cfg['select']} | "
          f"seeds {len(seeds)}", flush=True)

    kan, mlp, H, p_kan, p_mlp = build_pair(cfg["kind"], dim, cfg["n_cls"])
    print(f"    KAN {p_kan:,} | MLP(H={H}) {p_mlp:,}", flush=True)

    a_kan, a_mlp, f_kan, f_mlp, ep_kan, ep_mlp = [], [], [], [], [], []
    for seed in seeds:
        bke.set_seed(seed)
        acc, f1, ep = train_with_selection(build_pair(cfg["kind"], dim, cfg["n_cls"])[0],
                                           Xt, Yt, Xs, Ys, Xva, Yva, cfg["n_cls"],
                                           batch_size=cfg["batch"], select=cfg["select"])
        bke.set_seed(seed)
        acc2, f12, ep2 = train_with_selection(build_pair(cfg["kind"], dim, cfg["n_cls"])[1],
                                              Xt, Yt, Xs, Ys, Xva, Yva, cfg["n_cls"],
                                              batch_size=cfg["batch"], select=cfg["select"])
        a_kan.append(acc); f_kan.append(f1); ep_kan.append(ep)
        a_mlp.append(acc2); f_mlp.append(f12); ep_mlp.append(ep2)
        print(f"  seed {seed:>4d}: KAN {acc:6.3f}%(ep {ep:2d}) | "
              f"MLP {acc2:6.3f}%(ep {ep2:2d})", flush=True)

    k, m = np.array(a_kan), np.array(a_mlp)
    t, p = stats.ttest_rel(k, m)
    print(f"  => KAN {k.mean():.3f}±{k.std(ddof=1):.3f} | MLP {m.mean():.3f}±{m.std(ddof=1):.3f} "
          f"| Δ {(k - m).mean():+.3f} pp | t={t:.2f} p={p:.4g} | "
          f"best epoch KAN {np.mean(ep_kan):.1f} MLP {np.mean(ep_mlp):.1f}", flush=True)

    return {"name": cfg["name"], "batch": cfg["batch"], "select": cfg["select"],
            "seeds": len(seeds), "n_train": int(len(Yt)), "n_select": int(len(Ys)),
            "n_report": int(len(Yva)), "kan_params": p_kan, "mlp_params": p_mlp, "H": H,
            "kan_acc": k.tolist(), "mlp_acc": m.tolist(),
            "kan_f1": f_kan, "mlp_f1": f_mlp,
            "kan_best_epoch": ep_kan, "mlp_best_epoch": ep_mlp,
            "kan_mean": float(k.mean()), "kan_sd": float(k.std(ddof=1)),
            "mlp_mean": float(m.mean()), "mlp_sd": float(m.std(ddof=1)),
            "diff_mean": float((k - m).mean()), "t": float(t), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", choices=["all", "a", "b", "c"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    out_path = os.path.join(BASE, "KAN", "selection_split_result.json")
    res = {}
    if os.path.exists(out_path) and not args.force:
        with open(out_path, encoding="utf-8") as f:
            res = json.load(f)

    for key in (["a", "b", "c"] if args.only == "all" else [args.only]):
        if key in res:
            print(f"[{key}] result already present, skipping (use --force to rerun)")
            continue
        res[key] = run_one(key, CONFIGS[key], seeds)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n============ comparison with the delta reported in the paper (old rule: epoch chosen on the report set) ============")
    print("(a) LC25000 Cheb  vs H=528   paper delta = -0.226 pp (p<1e-4)")
    print("(b) CRC9    Cheb  vs H=528   paper delta = -0.251 pp (p<1e-4)")
    print("(c) LC25000 BSpl  vs H=708   paper delta = +0.364 pp (p<1e-4)")
    print(f"results saved to {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
