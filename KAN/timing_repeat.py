# -*- coding: utf-8 -*-
"""Repeat inference-latency measurement (P1-1).

Five independent rounds, 200 timed passes each, 20 warm-up passes per round,
batch size 1, same GPU and same heads as the latency table of the paper. Writes
timing_repeat_result.json with the per-round values so that the reported latency
comes with a measured spread instead of "one round, no spread available".

Usage:  python timing_repeat.py
"""
import os
import sys
import json

KAN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KAN)

import torch
import torch.nn as nn
import numpy as np

from clean_lc25000_heads import build_kan_head, build_mlp_head, n_params, DEVICE
from bspline_kan_experiment import build_bspline_kan_head

D_IN, N_CLS, H_WIDE = 2048, 5, 528
ROUNDS, WARMUP, RUNS = 5, 20, 200
OUT = os.path.join(KAN, "timing_repeat_result.json")


def build_mlp_wide(d_in=D_IN, n_cls=N_CLS, H=H_WIDE):
    return nn.Sequential(
        nn.Linear(d_in, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, n_cls),
    )


def one_round(model, x):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(RUNS):
            model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / RUNS


def main():
    print("device %s" % DEVICE, flush=True)
    heads = {
        "wide": build_mlp_wide(),
        "narrow": build_mlp_head(D_IN, N_CLS),
        "cheb": build_kan_head(D_IN, N_CLS),
        "bspline": build_bspline_kan_head(D_IN, N_CLS),
    }
    x = torch.randn(1, D_IN, device=DEVICE)
    out = {"meta": {"d_in": D_IN, "n_cls": N_CLS, "batch": 1, "rounds": ROUNDS,
                    "warmup": WARMUP, "timed_passes": RUNS, "device": str(DEVICE),
                    "H_wide": H_WIDE}, "res": {}}

    for name, model in heads.items():
        model = model.to(DEVICE).eval()
        p = n_params(model)
        rounds = [float(one_round(model, x)) for _ in range(ROUNDS)]
        arr = np.array(rounds)
        out["res"][name] = {
            "params": int(p),
            "ms": rounds,
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        print("%-8s params=%9d  ms=%s  mean=%.4f sd=%.4f"
              % (name, p, ["%.3f" % v for v in rounds], arr.mean(), arr.std(ddof=1)),
              flush=True)

    base = out["res"]["wide"]["mean"]
    for name in out["res"]:
        out["res"][name]["rel"] = float(out["res"][name]["mean"] / base)
        print("%-8s relative latency = %.2fx" % (name, out["res"][name]["rel"]), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
