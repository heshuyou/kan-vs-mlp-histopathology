# -*- coding: utf-8 -*-
"""
Efficiency measurements for the KAN / narrow MLP / wide MLP heads:
  - parameter count (sum)
  - FLOPs (analytic: Linear=2*in*out; KAN=out*in*(2*(degree+1)-1) plus the Chebyshev basis cost)
  - single-image latency at batch=1 (CUDA events, averaged over several runs)

Usage: conda activate gpt2 && python efficiency_measure.py
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

from baselines import KANBlock, KANLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_IN = 2048
N_CLS = 5


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build_kan_head(d_in=D_IN, n_cls=N_CLS):
    return nn.Sequential(
        KANBlock(d_in, 128, 5, 0.2), KANBlock(128, 64, 5, 0.2),
        KANBlock(64, 32, 5, 0.2), KANLayer(32, n_cls, 5, 0.0),
    )


def build_mlp_narrow(d_in=D_IN, n_cls=N_CLS):
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_cls),
    )


def build_mlp_wide(d_in=D_IN, n_cls=N_CLS, H=528):
    return nn.Sequential(
        nn.Linear(d_in, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(H, H), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H, n_cls),
    )


def count_flops(m, d_in=D_IN):
    """Analytic FLOPs. Walks the modules and treats Linear and KANLayer separately."""
    total = 0
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            total += 2 * mod.in_features * mod.out_features
        elif isinstance(mod, KANLayer):
            in_f, out_f, deg = mod.in_features, mod.out_features, mod.degree
            # weighted sum: out*in sums over (degree+1) terms -> (degree+1) multiplies and degree adds
            total += out_f * in_f * (2 * (deg + 1) - 1)
            # Chebyshev basis: each input runs degree recurrences (2 multiplies and 1 add each)
            total += in_f * deg * 3
    return total


def measure_latency(m, d_in=D_IN, warmup=20, runs=200):
    m = m.to(DEVICE).eval()
    x = torch.randn(1, d_in, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            m(x)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end) / runs  # ms per forward
    return t


def main():
    heads = {
        "KAN（1.64M）": build_kan_head(),
        "narrow MLP (0.27M)": build_mlp_narrow(),
        "wide MLP (1.64M)": build_mlp_wide(),
    }
    print(f"device {DEVICE}, input dim {D_IN}, {N_CLS} output classes\n")
    print(f"{'head':<16}{'params':>12}{'FLOPs':>12}{'latency(ms)':>12}{'latency/wide MLP':>14}")
    base = None
    for name, m in heads.items():
        p = n_params(m)
        f = count_flops(m)
        t = measure_latency(m)
        if "wide MLP" in name:
            base = t
        print(f"{name:<16}{p:>12,}{f:>12,}{t:>12.4f}{'-' if base is None else '':>14}")
    # relative latency
    print("\nrelative inference latency (wide MLP = 1):")
    kan = measure_latency(build_kan_head())
    mlp_n = measure_latency(build_mlp_narrow())
    mlp_w = measure_latency(build_mlp_wide())
    print(f"  KAN      = {kan/mlp_w:.2f}x")
    print(f"  narrow MLP = {mlp_n/mlp_w:.2f}x")
    print("  wide MLP   = 1.00x")


if __name__ == "__main__":
    main()
