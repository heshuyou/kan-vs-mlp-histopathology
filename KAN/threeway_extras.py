# -*- coding: utf-8 -*-
"""Three-way-protocol reruns for the paper.

Why this file exists
--------------------
Tables 1/2/4(outer)/probes were produced with rerun_protocol.py: a 10% selection
slice is carved out of the training pool and used for epoch selection, so the
reported partition is touched exactly once.  The parameter sweep, the degree /
depth / L1 ablations, the spline-only block, the batch-256 check and the error
analysis were all produced by older scripts that select their best epoch ON the
reported partition.  v29 cited the whole of Section 4 as "three-way", which the
code does not support.  This script re-runs those blocks under the same three-way
split, reusing the identical builders and identical hyper-parameters, so that the
paper can report one protocol only.

Split
-----
Identical to rerun_protocol.py:  X = concat(Xtr, Xva) of lc25000_feats_resnet50.npz,
selection = per-class 10% of the training half drawn with RandomState(0),
i.e. n_train 17078 / n_select 1898 / n_report 4744.

Jobs
----
  nobase  : B-spline head without the linear base weight (2,180,800 params)
            + MLP matched to that budget (H=652, 2,190,725 params)   20 seeds
  b256    : complete B-spline head + its matched MLP, batch 256      10 seeds
  sweep   : Chebyshev KAN point + 10 equal-width MLP widths          10 seeds
  ablation: degree in {3,5,8}; depth in {1,2,3} KAN and MLP          10 seeds
  l1      : L1 on the Chebyshev coefficients, lambda in {0,1e-5,1e-4,1e-3} 10 seeds
  error   : Chebyshev head, 3 seeds, confusion matrix on the report partition

Usage (KAN directory, GPU):
  python threeway_extras.py --job nobase
  python threeway_extras.py --job b256
  python threeway_extras.py --job sweep
  python threeway_extras.py --job ablation
  python threeway_extras.py --job l1
  python threeway_extras.py --job error

Every job writes <job>_partial.json after each seed (resume by simply re-running)
and a final <job>_threeway.json.  Existing result files are never overwritten.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

KAN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KAN)

from rerun_protocol import train_head, stratified_indices, gpu_free_check, DEFAULT_SEEDS
from clean_lc25000_heads import set_seed, n_params
from bspline_kan_experiment import (build_bspline_kan_head, build_mlp_matched)
from bspline_kan_nobasew import build_nobase_head
from baselines import KANBlock, KANLayer

FEATS = os.path.join(KAN, "lc25000_feats_resnet50.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 5
NOBASE_TARGET = 2180800
SWEEP_H = [64, 96, 128, 192, 256, 384, 528, 640, 768, 1024]


# ---------------------------------------------------------------- heads
def build_kan_head(d_in, degree=5, depth=3, n_cls=N_CLS):
    """Same construction as param_sweep.py / kan_ablation_clean.py."""
    if depth == 1:
        return nn.Sequential(KANBlock(d_in, 64, degree, 0.2),
                             KANLayer(64, n_cls, degree, 0.0))
    if depth == 2:
        return nn.Sequential(KANBlock(d_in, 64, degree, 0.2),
                             KANBlock(64, 32, degree, 0.2),
                             KANLayer(32, n_cls, degree, 0.0))
    return nn.Sequential(KANBlock(d_in, 128, degree, 0.2),
                         KANBlock(128, 64, degree, 0.2),
                         KANBlock(64, 32, degree, 0.2),
                         KANLayer(32, n_cls, degree, 0.0))


def build_mlp_head(d_in, depth=3, n_cls=N_CLS):
    if depth == 1:
        return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(64, n_cls))
    if depth == 2:
        return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(32, n_cls))
    return nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(32, n_cls))


def build_mlp_width(d_in, H, n_cls=N_CLS):
    return nn.Sequential(nn.Linear(d_in, H), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(H, n_cls))


# ---------------------------------------------------------------- split
def load_threeway():
    d = np.load(FEATS)
    X = np.vstack([d["Xtr"], d["Xva"]])
    Y = np.concatenate([d["Ytr"], d["Yva"]])
    tr_all = np.arange(len(d["Xtr"]))
    rep_idx = np.arange(len(d["Xtr"]), len(X))
    rng = np.random.RandomState(0)
    sel_idx = tr_all[stratified_indices(Y[tr_all], 0.10, rng)]
    tr_idx = np.array([i for i in tr_all if i not in set(sel_idx)])
    return X, Y, tr_idx, sel_idx, rep_idx


def stats_block(res, ref):
    from scipy import stats
    out = {}
    for h, a in res.items():
        a = np.array(a)
        d = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": len(a)}
        if ref and h != ref:
            b = np.array(res[ref])
            t, p = stats.ttest_rel(a, b)
            f = float(a.var(ddof=1) / b.var(ddof=1))
            pf = float(2 * min(stats.f.cdf(f, len(a) - 1, len(b) - 1),
                               1 - stats.f.cdf(f, len(a) - 1, len(b) - 1)))
            d.update({"diff": float(a.mean() - b.mean()), "t": float(t), "p": float(p),
                      "F": f, "p_F": pf, "ref": ref})
        out[h] = d
    return out


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_heads(partial, key_of, builders, seeds, Xtr, Ytr, Xsel, Ysel, Xrep, Yrep,
              batch=512, select="acc"):
    """Run each builder over all seeds, resuming from the partial file."""
    res = {}
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            res = json.load(f)
        print("  resume: %s" % {k: len(v) for k, v in res.items()}, flush=True)
    t0 = time.time()
    for seed in seeds:
        for name, b in builders.items():
            res.setdefault(name, [])
            if len(res[name]) > seeds.index(seed):
                continue
            set_seed(seed)
            acc, f1, _ = train_head(b(), Xtr, Ytr, Xsel, Ysel, Xrep, Yrep,
                                    select=select, batch=batch, epochs=15)
            res[name].append(acc)
            print("  seed %4d | %-14s acc %.4f | %.0fs" % (seed, name, acc, time.time() - t0),
                  flush=True)
            dump(partial, res)
    return res


# ---------------------------------------------------------------- jobs
def job_nobase(args):
    X, Y, tr, sel, rep = load_threeway()
    nb0 = build_nobase_head(X.shape[1], N_CLS)
    p_nb = n_params(nb0)
    mm, H = build_mlp_matched(X.shape[1], p_nb, N_CLS)
    p_mm = n_params(mm)
    assert abs(p_nb - NOBASE_TARGET) < 1, p_nb
    print("nobase %d params | matched MLP H=%d %d params" % (p_nb, H, p_mm), flush=True)
    seeds = DEFAULT_SEEDS[:args.seeds]
    builders = {"spline_only": lambda: build_nobase_head(X.shape[1], N_CLS),
                "mlp_matched": lambda: build_mlp_matched(X.shape[1], p_nb, N_CLS)[0]}
    partial = os.path.join(KAN, "threeway_nobase_partial.json")
    res = run_heads(partial, None, builders, seeds,
                    X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=512)
    out = {"setting": "LC25000 + ResNet50, B-spline without base weight, three-way protocol",
           "protocol": "three-way", "select": "acc on the selection slice",
           "n_train": int(len(tr)), "n_select": int(len(sel)), "n_report": int(len(rep)),
           "batch": 512, "seeds": len(seeds),
           "spline_only_params": p_nb, "mlp_params": p_mm, "H": H,
           "res": res, "summary": stats_block(res, "mlp_matched")}
    dump(os.path.join(KAN, "threeway_nobase_result.json"), out)
    print(json.dumps(out["summary"], indent=2))


def job_b256(args):
    X, Y, tr, sel, rep = load_threeway()
    ref_bs = build_bspline_kan_head(X.shape[1], N_CLS, 5, 3)
    p_bs = n_params(ref_bs)
    mm, H = build_mlp_matched(X.shape[1], p_bs, N_CLS)
    print("bspline %d params | matched MLP H=%d %d params" % (p_bs, H, n_params(mm)), flush=True)
    seeds = DEFAULT_SEEDS[:args.seeds]
    builders = {"bspline": lambda: build_bspline_kan_head(X.shape[1], N_CLS, 5, 3),
                "wide_bs": lambda: build_mlp_matched(X.shape[1], p_bs, N_CLS)[0]}
    partial = os.path.join(KAN, "threeway_b256_partial.json")
    res = run_heads(partial, None, builders, seeds,
                    X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=256)
    out = {"setting": "LC25000 + ResNet50, complete B-spline head, three-way protocol",
           "protocol": "three-way", "select": "acc on the selection slice",
           "n_train": int(len(tr)), "n_select": int(len(sel)), "n_report": int(len(rep)),
           "batch": 256, "seeds": len(seeds),
           "bspline_params": p_bs, "mlp_params": n_params(mm), "H": H,
           "res": res, "summary": stats_block(res, "wide_bs")}
    dump(os.path.join(KAN, "threeway_b256_result.json"), out)
    print(json.dumps(out["summary"], indent=2))


def job_sweep(args):
    X, Y, tr, sel, rep = load_threeway()
    seeds = DEFAULT_SEEDS[:args.seeds]
    dim = X.shape[1]
    builders = {"kan": lambda: build_kan_head(dim, 5, 3)}
    params = {"kan": n_params(build_kan_head(dim, 5, 3))}
    for H in SWEEP_H:
        builders["H%d" % H] = (lambda h: (lambda: build_mlp_width(dim, h, N_CLS)))(H)
        params["H%d" % H] = n_params(build_mlp_width(dim, H, N_CLS))
    print("params", params, flush=True)
    partial = os.path.join(KAN, "threeway_sweep_partial.json")
    res = run_heads(partial, None, builders, seeds,
                    X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=512)
    out = {"setting": "LC25000 + ResNet50, parameter sweep, three-way protocol",
           "protocol": "three-way", "select": "acc on the selection slice",
           "n_train": int(len(tr)), "n_select": int(len(sel)), "n_report": int(len(rep)),
           "batch": 512, "seeds": len(seeds), "params": params,
           "res": res, "summary": stats_block(res, "H528")}
    dump(os.path.join(KAN, "threeway_sweep_result.json"), out)
    for k, v in out["summary"].items():
        print("  %-6s %12d  %.4f +- %.4f" % (k, params[k], v["mean"], v["sd"]), flush=True)


def _sweep_kan_accs():
    """The depth-3, degree-5 Chebyshev head is one and the same configuration in the
    sweep, in ablation block (a) at degree 5, in ablation block (b) at depth 3 and in
    the lambda=0 row of block (c). Running it four times with the same seeds would
    recompute bit-identical numbers, so the sweep's copy is reused here."""
    path = os.path.join(KAN, "threeway_sweep_result.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["res"]["kan"]


def job_ablation(args):
    X, Y, tr, sel, rep = load_threeway()
    dim = X.shape[1]
    seeds = DEFAULT_SEEDS[:args.seeds]
    builders, params = {}, {}
    for deg in (3, 5, 8):
        builders["degree_%d" % deg] = (lambda d: (lambda: build_kan_head(dim, d, 3)))(deg)
        params["degree_%d" % deg] = n_params(build_kan_head(dim, deg, 3))
    for dep in (1, 2, 3):
        builders["depth_%d_kan" % dep] = (lambda d: (lambda: build_kan_head(dim, 5, d)))(dep)
        builders["depth_%d_mlp" % dep] = (lambda d: (lambda: build_mlp_head(dim, d)))(dep)
        params["depth_%d_kan" % dep] = n_params(build_kan_head(dim, 5, dep))
        params["depth_%d_mlp" % dep] = n_params(build_mlp_head(dim, dep))
    print("params", params, flush=True)
    partial = os.path.join(KAN, "threeway_ablation_partial.json")
    kan = _sweep_kan_accs()
    if kan is not None:
        res = json.load(open(partial, encoding="utf-8")) if os.path.exists(partial) else {}
        res["degree_5"] = list(kan)
        res["depth_3_kan"] = list(kan)
        dump(partial, res)
        print("  reused the sweep's depth-3/degree-5 Chebyshev run for degree_5 and depth_3_kan",
              flush=True)
    # degree 8 needs the smaller batch used in the original ablation
    res = {}
    res.update(run_heads(partial, None, {"degree_3": builders["degree_3"],
                                         "degree_5": builders["degree_5"]}, seeds,
                         X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=512))
    res = json.load(open(partial, encoding="utf-8"))
    for name in ("depth_1_kan", "depth_1_mlp", "depth_2_kan", "depth_2_mlp",
                 "depth_3_kan", "depth_3_mlp"):
        res.update(run_heads(partial, None, {name: builders[name]}, seeds,
                             X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=512))
        res = json.load(open(partial, encoding="utf-8"))
    res.update(run_heads(partial, None, {"degree_8": builders["degree_8"]}, seeds,
                         X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], batch=128))
    res = json.load(open(partial, encoding="utf-8"))
    out = {"setting": "LC25000 + ResNet50, degree/depth ablations, three-way protocol",
           "protocol": "three-way", "select": "acc on the selection slice",
           "n_train": int(len(tr)), "n_select": int(len(sel)), "n_report": int(len(rep)),
           "batch": {"degree_8": 128, "other": 512}, "seeds": len(seeds), "params": params,
           "res": res, "summary": stats_block(res, "degree_5")}
    dump(os.path.join(KAN, "threeway_ablation_result.json"), out)
    for k, v in out["summary"].items():
        print("  %-14s %10d  %.4f +- %.4f" % (k, params[k], v["mean"], v["sd"]), flush=True)


def train_head_l1(head, Xtr, Ytr, Xsel, Ysel, Xrep, Yrep, l1_lambda,
                  batch=512, epochs=15):
    """train_head of rerun_protocol.py plus the L1 penalty used in kan_regularization.py."""
    head = head.to(DEVICE)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_key, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            if l1_lambda > 0:
                l1 = sum(m.coefficients.abs().sum()
                         for m in head.modules() if isinstance(m, KANLayer))
                loss = loss + l1_lambda * l1
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(Xsel), 1024):
                preds.append(head(torch.from_numpy(Xsel[s:s + 1024]).to(DEVICE)).argmax(1).cpu().numpy())
        acc = float((np.concatenate(preds) == Ysel).mean())
        if acc > best_key:
            best_key, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(Xrep), 1024):
            preds.append(head(torch.from_numpy(Xrep[s:s + 1024]).to(DEVICE)).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return float((p == Yrep).mean() * 100), p


def job_l1(args):
    X, Y, tr, sel, rep = load_threeway()
    dim = X.shape[1]
    seeds = DEFAULT_SEEDS[:args.seeds]
    lams = [0.0, 1e-5, 1e-4, 1e-3]
    partial = os.path.join(KAN, "threeway_l1_partial.json")
    res = {}
    if os.path.exists(partial):
        res = json.load(open(partial, encoding="utf-8"))
    kan = _sweep_kan_accs()
    if kan is not None and "l1_0" not in res:
        res["l1_0"] = list(kan)      # lambda=0 is the unregularized head, unchanged by L1
        dump(partial, res)
        print("  reused the sweep's depth-3/degree-5 run for the lambda=0 row", flush=True)
    t0 = time.time()
    for lam in lams:
        key = "l1_%g" % lam
        res.setdefault(key, [])
        for i, seed in enumerate(seeds):
            if len(res[key]) > i:
                continue
            set_seed(seed)
            acc, _ = train_head_l1(build_kan_head(dim, 5, 3),
                                   X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep], lam)
            res[key].append(acc)
            print("  L1 %-6g seed %4d acc %.4f | %.0fs" % (lam, seed, acc, time.time() - t0),
                  flush=True)
            dump(partial, res)
    out = {"setting": "LC25000 + ResNet50, L1 on Chebyshev coefficients, three-way protocol",
           "protocol": "three-way", "select": "acc on the selection slice",
           "n_train": int(len(tr)), "n_select": int(len(sel)), "n_report": int(len(rep)),
           "batch": 512, "seeds": len(seeds), "l1_list": lams, "res": res}
    for k, v in res.items():
        a = np.array(v)
        out.setdefault("summary", {})[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
        print("  %-8s %.4f +- %.4f" % (k, a.mean(), a.std(ddof=1)), flush=True)
    dump(os.path.join(KAN, "threeway_l1_result.json"), out)


def job_error(args):
    X, Y, tr, sel, rep = load_threeway()
    seeds = DEFAULT_SEEDS[:args.seeds]
    res = {"seeds": seeds, "runs": []}
    for seed in seeds:
        set_seed(seed)
        acc, f1, p = train_head(build_kan_head(X.shape[1], 5, 3),
                                X[tr], Y[tr], X[sel], Y[sel], X[rep], Y[rep],
                                select="acc", batch=512, epochs=15)
        cm = np.zeros((N_CLS, N_CLS), dtype=int)
        for t, q in zip(Y[rep], p):
            cm[t, q] += 1
        res["runs"].append({"seed": seed, "acc": acc, "f1": f1,
                            "n_errors": int((p != Y[rep]).sum()),
                            "confusion": cm.tolist(),
                            "per_class_recall": (cm.diagonal() / cm.sum(1)).tolist()})
        print("  seed %4d acc %.4f errors %d"
              % (seed, acc, int((p != Y[rep]).sum())), flush=True)
        dump(os.path.join(KAN, "threeway_error_result.json"), res)
    means = {"acc_mean": float(np.mean([r["acc"] for r in res["runs"]])),
             "errors_mean": float(np.mean([r["n_errors"] for r in res["runs"]])),
             "acc_all_above": float(min(r["acc"] for r in res["runs"]))}
    res.update(means)
    dump(os.path.join(KAN, "threeway_error_result.json"), res)
    print(json.dumps(means, indent=2))


JOBS = {"nobase": (job_nobase, 20), "b256": (job_b256, 10), "sweep": (job_sweep, 10),
        "ablation": (job_ablation, 10), "l1": (job_l1, 10), "error": (job_error, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=list(JOBS))
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    fn, default_seeds = JOBS[args.job]
    if args.seeds is None:
        args.seeds = default_seeds
    print("job %s | %d seeds | three-way split" % (args.job, args.seeds), flush=True)
    gpu_free_check()
    fn(args)


if __name__ == "__main__":
    main()
