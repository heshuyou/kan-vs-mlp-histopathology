# -*- coding: utf-8 -*-
"""Re-run classification heads under a clean protocol.

Modes
  threeway : original image-level split; 10% of the training pool is carved out for
             epoch selection, the reported partition is evaluated exactly once.
  group3   : prototype-grouped split (train_group.txt / val_group.txt); selection set
             is carved out at GROUP level so that no group spans train/select/report.

Tasks
  lc5   : cleaned LC25000 5-class, cached features lc25000_feats_{backbone}.npz
  crc9  : CRC-VAL-HE-7K 9-class, cached features crc9_resnet50_feats.npz

Outputs go to a NEW json (existing result files are never overwritten).
"""
import os, sys, json, argparse, time
import numpy as np

KAN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KAN)
import torch
import torch.nn as nn
from clean_lc25000_heads import (build_kan_head, build_mlp_head, build_mlp_matched,
                                 set_seed, n_params, DEVICE)
from bspline_kan_experiment import build_bspline_kan_head

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 100, 200, 300, 400, 500,
                 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def gpu_free_check(max_mem_mib=800, max_util=30):
    """Refuse to start if another process already holds the GPU: on a 4 GB laptop
    card a second context turns a 3-minute epoch into an indefinite hang."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=15).stdout.strip().splitlines()[0]
        mem, util = [float(x) for x in out.split(",")[:2]]
    except Exception:
        print("  [warn] nvidia-smi unavailable, GPU check skipped", flush=True)
        return
    if mem > max_mem_mib or util > max_util:
        pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                               "--format=csv,noheader,nounits"], capture_output=True,
                              text=True, timeout=15).stdout.strip()
        print("  [error] GPU busy: %d MiB in use, %d%% utilisation (pids: %s)"
              % (mem, util, pids.replace("\n", " ") or "n/a"), flush=True)
        print("  Stop the other process first, e.g.  Stop-Process -Id <pid> -Force", flush=True)
        sys.exit(2)


def train_with_retry(builder, seed, *args, **kwargs):
    """Build a fresh model per attempt so a OOM retry restarts from the same seed."""
    batch = kwargs.pop("batch", 512)
    for attempt in range(4):
        set_seed(seed)
        try:
            return train_head(builder(), *args, batch=batch, **kwargs)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch <= 32:
                raise
            torch.cuda.empty_cache()
            batch = max(32, batch // 2)
            print("  [warn] CUDA OOM, retrying with batch %d" % batch, flush=True)


def read_list(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return [ln.strip().split() for ln in f if ln.strip()]


def stratified_indices(y, frac, rng):
    out = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        out.extend(idx[: int(round(frac * len(idx)))])
    return np.array(sorted(out))


def train_head(head, Xtr, Ytr, Xsel, Ysel, Xrep, Yrep, select="acc", batch=512, epochs=15):
    head = head.to(DEVICE)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    dl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_key, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        sched.step()
        head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xsel), 1024):
                preds.append(head(torch.from_numpy(Xsel[s:s + 1024]).to(DEVICE)).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        acc = float((p == Ysel).mean())
        if select == "f1":
            from sklearn.metrics import f1_score
            key = float(f1_score(Ysel, p, average="macro"))
        else:
            key = acc
        if key > best_key:
            best_key, best_state = key, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(Xrep), 1024):
            preds.append(head(torch.from_numpy(Xrep[s:s + 1024]).to(DEVICE)).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    from sklearn.metrics import f1_score
    return float((p == Yrep).mean() * 100), float(f1_score(Yrep, p, average="macro")), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lc5", choices=["lc5", "crc9"])
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--mode", default="group3", choices=["threeway", "group3"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0, help="index offset into DEFAULT_SEEDS")
    ap.add_argument("--resume", action="store_true", help="continue from the seeds already in --out")
    ap.add_argument("--heads", default="probe,narrow,cheb,wide,bspline")
    ap.add_argument("--groupset", default="group2", help="prefix of train_/val_ list files and label array")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    heads = [h for h in args.heads.split(",") if h]
    seeds = DEFAULT_SEEDS[args.seed_start:args.seed_start + args.seeds]

    if args.task == "lc5":
        npz = os.path.join(KAN, f"lc25000_feats_{args.backbone}.npz")
        d = np.load(npz)
        X = np.vstack([d["Xtr"], d["Xva"]])
        Y = np.concatenate([d["Ytr"], d["Yva"]])
        n_cls = int(Y.max()) + 1
        paths = [r[0] for r in read_list("train_clean.txt")] + [r[0] for r in read_list("val_clean.txt")]
        batch, select = 512, "acc"
        if args.mode == "threeway":
            tr_idx = np.arange(len(d["Xtr"]))
            rep_idx = np.arange(len(d["Xtr"]), len(X))
            rng = np.random.RandomState(0)
            sel_idx = tr_idx[stratified_indices(Y[tr_idx], 0.10, rng)]
            tr_idx = np.array([i for i in tr_idx if i not in set(sel_idx)])
        else:
            roots = np.load(os.path.join(KAN, f"lc25000_{args.groupset}_labels.npy"))
            tr_paths = set(r[0] for r in read_list(f"train_{args.groupset}.txt"))
            rep_paths = set(r[0] for r in read_list(f"val_{args.groupset}.txt"))
            tr_idx = np.array([i for i, p in enumerate(paths) if p in tr_paths])
            rep_idx = np.array([i for i, p in enumerate(paths) if p in rep_paths])
            # selection carved at group level
            rng = np.random.RandomState(0)
            g = np.unique(roots[tr_idx])
            rng.shuffle(g)
            n_sel = int(0.10 * len(g))
            sel_groups = set(g[:n_sel].tolist())
            sel_idx = np.array([i for i in tr_idx if roots[i] in sel_groups])
            tr_idx = np.array([i for i in tr_idx if roots[i] not in sel_groups])
    else:
        d = np.load(os.path.join(KAN, "crc9_resnet50_feats.npz"))
        X, Y = np.vstack([d["Xtr"], d["Xva"]]), np.concatenate([d["Ytr"], d["Yva"]])
        n_cls = 9
        batch, select = 256, "f1"
        tr_idx = np.arange(len(d["Xtr"]))
        rep_idx = np.arange(len(d["Xtr"]), len(X))
        rng = np.random.RandomState(0)
        sel_idx = tr_idx[stratified_indices(Y[tr_idx], 0.10, rng)]
        tr_idx = np.array([i for i in tr_idx if i not in set(sel_idx)])

    print(f"{args.task}/{args.backbone}/{args.mode}: train {len(tr_idx)} select {len(sel_idx)} report {len(rep_idx)}")

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xsel, Ysel = X[sel_idx], Y[sel_idx]
    Xrep, Yrep = X[rep_idx], Y[rep_idx]

    ref_cheb = build_kan_head(X.shape[1], n_cls)
    ref_bs = build_bspline_kan_head(X.shape[1], n_cls, 5, 3)
    builders = {
        "probe": lambda: nn.Sequential(nn.Linear(X.shape[1], n_cls)),
        "narrow": lambda: build_mlp_head(X.shape[1], n_cls),
        "cheb": lambda: build_kan_head(X.shape[1], n_cls),
        "wide": lambda: build_mlp_matched(X.shape[1], n_cls, n_params(ref_cheb))[0],
        "bspline": lambda: build_bspline_kan_head(X.shape[1], n_cls, 5, 3),
        "wide_bs": lambda: build_mlp_matched(X.shape[1], n_cls, n_params(ref_bs))[0],
    }
    if "bspline" in heads and "wide_bs" not in heads:
        heads = heads + ["wide_bs"]

    params = {h: n_params(builders[h]()) for h in heads}
    H_wide = build_mlp_matched(X.shape[1], n_cls, n_params(ref_cheb))[1]
    H_wide_bs = build_mlp_matched(X.shape[1], n_cls, n_params(ref_bs))[1]
    print("params", params, "H_wide", H_wide, "H_wide_bs", H_wide_bs)

    res = {h: {"acc": [], "f1": []} for h in heads}
    preds0 = {}
    seeds_done = 0
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8"))
        if "res" in prev and all(h in prev["res"] for h in heads):
            res = {h: {"acc": list(prev["res"][h]["acc"]), "f1": list(prev["res"][h]["f1"])} for h in heads}
            seeds_done = len(res[heads[0]]["acc"])
            print("resume: %d seeds already done" % seeds_done, flush=True)
    seeds = seeds[seeds_done:]
    if seeds:
        gpu_free_check()
    t0 = time.time()
    for k, seed in enumerate(seeds):
        for h in heads:
            acc, f1, p = train_with_retry(builders[h], seed, Xtr, Ytr, Xsel, Ysel, Xrep, Yrep,
                                          select=select, batch=batch, epochs=15)
            res[h]["acc"].append(acc)
            res[h]["f1"].append(f1)
            if seeds_done == 0 and k == 0:
                preds0[h] = p
            print("  seed %4d | %-8s acc %.3f f1 %.4f | %.0fs" % (seed, h, acc, f1, time.time() - t0), flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": {"task": args.task, "backbone": args.backbone, "mode": args.mode,
                                "seeds": DEFAULT_SEEDS[:args.seed_start + seeds_done + k + 1],
                                "batch": batch, "select": select,
                                "n_train": int(len(tr_idx)), "n_select": int(len(sel_idx)),
                                "n_report": int(len(rep_idx)), "params": params,
                                "H_wide": H_wide, "H_wide_bs": H_wide_bs},
                       "res": res}, f, indent=2)

    # stats
    from scipy import stats
    all_seeds = DEFAULT_SEEDS[:args.seed_start + len(res[heads[0]]["acc"])]
    out = {"meta": {"task": args.task, "backbone": args.backbone, "mode": args.mode, "seeds": all_seeds,
                    "batch": batch, "select": select, "n_train": int(len(tr_idx)),
                    "n_select": int(len(sel_idx)), "n_report": int(len(rep_idx)),
                    "params": params, "H_wide": H_wide, "H_wide_bs": H_wide_bs},
           "res": res, "summary": {}}
    ref = "wide" if "wide" in heads else None
    for h in heads:
        a = np.array(res[h]["acc"])
        out.setdefault("summary", {})[h] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
        if ref and h != ref:
            b = np.array(res[ref]["acc"])
            t, p = stats.ttest_rel(a, b)
            f = float(a.var(ddof=1) / b.var(ddof=1))
            pf = float(2 * min(stats.f.cdf(f, len(a) - 1, len(b) - 1), 1 - stats.f.cdf(f, len(a) - 1, len(b) - 1)))
            out["summary"][h].update({"diff_vs_wide": float(a.mean() - b.mean()), "t": float(t), "p": float(p),
                                      "F_vs_wide": f, "p_F": pf})
    if preds0:
        np.savez(args.out.replace(".json", "_preds.npz"),
                 ytrue=Yrep, **{f"pred_{h}": preds0[h] for h in preds0})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
