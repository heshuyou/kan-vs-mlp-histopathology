# -*- coding: utf-8 -*-
"""
Multi-seed experiment on frozen backbones after LC25000 cleaning (de-duplicated, leak-free):
  Cache the features of a frozen ResNet50 / EfficientNet-B0 and train only the head.
  Three heads are compared: KAN / MLP (same depth 128-64-32) / MLP-matched (three hidden layers of equal width, parameter count close to KAN).
  Fixed split (train_clean.txt / val_clean.txt); seeds affect only head initialisation and training randomness.
  Output: mean±std per head, paired t tests (KAN vs MLP / KAN vs MLP-matched), F test, parameter counts.

Usage: conda activate gpt2 && python clean_lc25000_heads.py --backbone resnet50 --seeds 20
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader

from baselines import KANBlock, KANLayer, PathologyDataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_LIST = os.path.join(BASE, "train_clean.txt")
VAL_LIST = os.path.join(BASE, "val_clean.txt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def build_backbone(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    else:
        raise ValueError(name)
    for p in m.parameters():
        p.requires_grad = False
    m.eval().to(DEVICE)
    return m, dim


def build_kan_head(d_in, n_cls=5):
    return nn.Sequential(
        KANBlock(d_in, 128, 5, 0.2), KANBlock(128, 64, 5, 0.2),
        KANBlock(64, 32, 5, 0.2), KANLayer(32, n_cls, 5, 0.0),
    )


def build_mlp_head(d_in, n_cls=5):
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_cls),
    )


def build_mlp_matched(d_in, n_cls=5, target=None):
    """MLP with three equal-width hidden layers of width H, sized to approach target (the KAN head)."""
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


def precompute(backbone, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(DEVICE)).cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labs, 0)


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
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_ds = PathologyDataset(TRAIN_LIST, BASE, transform)
    val_ds = PathologyDataset(VAL_LIST, BASE, transform)
    print(f"[{args.backbone}] train {len(train_ds)} | val {len(val_ds)}")

    backbone, dim = build_backbone(args.backbone)
    print(f"feature dim {dim}, precomputing frozen-backbone features ...", flush=True)
    cache_path = os.path.join(BASE, "KAN", f"lc25000_feats_{args.backbone}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
        print("loading cached features")
    else:
        Xtr, Ytr = precompute(backbone, train_ds)
        Xva, Yva = precompute(backbone, val_ds)
        np.savez(cache_path, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
        print(f"features cached: {cache_path}")
    print(f"train {Xtr.shape} | val {Xva.shape}")

    kan = build_kan_head(dim)
    mlp = build_mlp_head(dim)
    mlp_matched, H = build_mlp_matched(dim, target=n_params(kan))
    print(f"params: KAN {n_params(kan):,} | MLP {n_params(mlp):,} | MLP-matched(H={H}) {n_params(mlp_matched):,}")

    seeds = DEFAULT_SEEDS[:args.seeds]
    partial_path = os.path.join(BASE, "KAN", f"clean_lc25000_{args.backbone}_partial.json")
    results = {"kan": [], "mlp": [], "mlp_matched": []}
    start = 0
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            results = json.load(f)
        start = len(results["kan"])
        print(f"resuming from seed {start}/{len(seeds)}", flush=True)
    for i in range(start, len(seeds)):
        seed = seeds[i]
        set_seed(seed)
        a_kan = train_head(build_kan_head(dim), Xtr, Ytr, Xva, Yva, args.epochs)
        set_seed(seed)
        a_mlp = train_head(build_mlp_head(dim), Xtr, Ytr, Xva, Yva, args.epochs)
        set_seed(seed)
        a_mm = train_head(build_mlp_matched(dim, target=n_params(kan))[0], Xtr, Ytr, Xva, Yva, args.epochs)
        results["kan"].append(a_kan)
        results["mlp"].append(a_mlp)
        results["mlp_matched"].append(a_mm)
        print(f"  seed {seed:>4d}: KAN {a_kan:6.3f}% | MLP {a_mlp:6.3f}% | MLP-matched {a_mm:6.3f}%", flush=True)
        # Save incrementally so that an interrupted run loses nothing
        with open(os.path.join(BASE, "KAN", f"clean_lc25000_{args.backbone}_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)

    from scipy import stats
    out = {"backbone": args.backbone, "feature_dim": dim, "seeds": len(seeds),
           "params": {"kan": n_params(kan), "mlp": n_params(mlp), "mlp_matched": n_params(mlp_matched),
                      "mlp_matched_H": H}}
    print("\n================ summary ================")
    for k in ("kan", "mlp", "mlp_matched"):
        a = np.array(results[k])
        out[f"{k}_mean"] = a.mean()
        out[f"{k}_std"] = a.std(ddof=1)
        print(f"{k:12s}: {a.mean():.4f}% ± {a.std(ddof=1):.4f}%")

    for other in ("mlp", "mlp_matched"):
        kan_a = np.array(results["kan"])
        oth_a = np.array(results[other])
        d = kan_a - oth_a
        t, p = stats.ttest_rel(kan_a, oth_a)
        F = max(kan_a.var(ddof=1), oth_a.var(ddof=1)) / min(kan_a.var(ddof=1), oth_a.var(ddof=1))
        p_f = 2 * (1 - stats.f.cdf(F, len(kan_a) - 1, len(oth_a) - 1))
        out[f"kan_vs_{other}"] = {"diff_mean": d.mean(), "diff_std": d.std(ddof=1),
                                  "t": t, "p": p, "F": F, "p_f": p_f}
        print(f"KAN vs {other}: paired diff {d.mean():+.4f}%±{d.std(ddof=1):.4f}% | "
              f"t={t:.3f} p={p:.4f} | F={F:.2f} p={p_f:.4f}")

    out_path = os.path.join(BASE, "KAN", f"clean_lc25000_{args.backbone}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"results saved: {out_path}")


if __name__ == "__main__":
    main()
