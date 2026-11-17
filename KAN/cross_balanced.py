# -*- coding: utf-8 -*-
"""Balanced accuracy and per-class recall for the cross-dataset table (P1-2).

The cross-dataset table of the paper reports accuracy only, on a target partition whose
classes are 1,233 TUM / 741 NORM (62/38). Accuracy is therefore inflated by the
majority class. This script retrains those five heads under the
identical protocol (batch 256, 15 epochs, no early stopping, 20 fixed seeds),
stores per-image predictions and reports balanced accuracy plus per-class recall,
mean +/- SD across seeds.

Outputs: cross_balanced_result.json, cross_balanced_preds.npz
Usage:   python cross_balanced.py
"""
import os
import sys
import json
import time

KAN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KAN)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, recall_score

import bspline_kan_extra as bkx
import bspline_kan_experiment as bke
from clean_lc25000_heads import build_kan_head, build_mlp_head, build_mlp_matched, n_params, set_seed, DEVICE

FEATS = os.path.join(KAN, "colon_gen_feats.npz")
OUT_JSON = os.path.join(KAN, "cross_balanced_result.json")
OUT_NPZ = os.path.join(KAN, "cross_balanced_preds.npz")
N_CLS = 2
BATCH = 256
SEEDS = [42, 123, 2024, 7, 99, 100, 200, 300, 400, 500,
         600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def predict(head, X, batch_size=512):
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(head(torch.from_numpy(X[i:i + batch_size]).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    d = np.load(FEATS)
    Xtr, Ytr = d["Xtr"].astype(np.float32), d["Ytr"]
    Xte, Yte = d["Xte"].astype(np.float32), d["Yte"]
    dim = Xtr.shape[1]
    print("train %s  test %s  classes %s" % (Xtr.shape, Xte.shape, np.bincount(Yte)), flush=True)

    # builders: the two KAN heads and their matched MLPs must be built from the
    # SAME module, so that the width search reproduces the widths reported in the paper
    cheb_ref = build_kan_head(dim, N_CLS)
    wide_ref, H_wide = build_mlp_matched(dim, n_cls=N_CLS, target=n_params(cheb_ref))
    bkan_ref = bke.build_bspline_kan_head(dim, n_cls=N_CLS)
    wide_bs_ref, H_bs = bke.build_mlp_matched(dim, bke.n_params(bkan_ref), n_cls=N_CLS)

    heads = {
        "narrow": (lambda: build_mlp_head(dim, N_CLS), n_params(build_mlp_head(dim, N_CLS))),
        "cheb": (lambda: build_kan_head(dim, N_CLS), n_params(cheb_ref)),
        "wide": (lambda: build_mlp_matched(dim, n_cls=N_CLS, target=n_params(cheb_ref))[0], n_params(wide_ref)),
        "bspline": (lambda: bke.build_bspline_kan_head(dim, n_cls=N_CLS), bke.n_params(bkan_ref)),
        "wide_bs": (lambda: bke.build_mlp_matched(dim, bke.n_params(bkan_ref), n_cls=N_CLS)[0], bke.n_params(wide_bs_ref)),
    }
    print("H_wide=%d (cheb %d vs %d) | H_bs=%d (bs %d vs %d)"
          % (H_wide, n_params(cheb_ref), n_params(wide_ref), H_bs,
             bke.n_params(bkan_ref), bke.n_params(wide_bs_ref)), flush=True)

    out = {"setting": "LC25000 colon -> CRC-VAL-HE-7K TUM/NORM",
           "class_counts": {"test": np.bincount(Yte).tolist()},
           "batch": BATCH, "epochs": 15, "early_stop": False, "seeds": SEEDS,
           "heads": {}}

    store = {}
    for name, (builder, params) in heads.items():
        accs, bacs, recs = [], [], []
        preds_all = []
        t0 = time.time()
        for seed in SEEDS:
            set_seed(seed)
            head = builder().to(DEVICE)
            bkx.train_model(head, Xtr, Ytr, Xte, Yte, N_CLS, batch_size=BATCH,
                            select="acc", early_stop=False)
            p = predict(head, Xte)
            accs.append(float((p == Yte).mean() * 100))
            bacs.append(float(balanced_accuracy_score(Yte, p) * 100))
            recs.append([float(v * 100) for v in recall_score(Yte, p, average=None, labels=[0, 1])])
            preds_all.append(p)
            print("  %-8s seed %4d acc %.3f bacc %.3f recall %s (%.0fs)"
                  % (name, seed, accs[-1], bacs[-1], np.round(recs[-1], 2), time.time() - t0), flush=True)

        a, b, r = np.array(accs), np.array(bacs), np.array(recs)
        out["heads"][name] = {
            "params": int(params),
            "acc_mean": float(a.mean()), "acc_sd": float(a.std(ddof=1)),
            "bacc_mean": float(b.mean()), "bacc_sd": float(b.std(ddof=1)),
            "recall_norm_mean": float(r[:, 0].mean()), "recall_norm_sd": float(r[:, 0].std(ddof=1)),
            "recall_tum_mean": float(r[:, 1].mean()), "recall_tum_sd": float(r[:, 1].std(ddof=1)),
            "acc": accs, "bacc": bacs, "recall": recs,
        }
        store[name] = np.array(preds_all, dtype=np.int16)
        print("  == %-8s acc %.3f+-%.3f | bacc %.3f+-%.3f | rec NORM %.2f TUM %.2f"
              % (name, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1),
                 r[:, 0].mean(), r[:, 1].mean()), flush=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        np.savez_compressed(OUT_NPZ, **store)

    print("written", OUT_JSON, OUT_NPZ, flush=True)


if __name__ == "__main__":
    main()
