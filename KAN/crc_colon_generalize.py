# -*- coding: utf-8 -*-
"""
Cross-dataset generalisation test (LC25000 colon -> CRC-VAL-HE-7K):
  train a binary head on the cleaned LC25000 colon images (colon_aca=0 / colon_n=1) and
  measure cross-dataset accuracy on TUM(->0) / NORM(->1) of CRC-VAL-HE-7K.
  Three heads: KAN / MLP / MLP-matched; several seeds. Features are cached to npy.

Usage: conda activate gpt2 && python crc_colon_generalize.py --seeds 20
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image

from baselines import KANBlock, KANLayer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_LIST = os.path.join(BASE, "train_clean.txt")
VAL_LIST = os.path.join(BASE, "val_clean.txt")
CRC = os.path.join(BASE, "CRC-VAL-HE-7K")
CACHE = os.path.join(BASE, "KAN", "colon_gen_feats.npz")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 2  # 0=adenocarcinoma (TUM), 1=benign (NORM)

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


def build_kan_head(d_in, n_cls=N_CLS):
    return nn.Sequential(
        KANBlock(d_in, 128, 5, 0.2), KANBlock(128, 64, 5, 0.2),
        KANBlock(64, 32, 5, 0.2), KANLayer(32, n_cls, 5, 0.0),
    )


def build_mlp_head(d_in, n_cls=N_CLS):
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_cls),
    )


def build_mlp_matched(d_in, n_cls=N_CLS, target=None):
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


def extract(paths, transform, batch_size=64):
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
    backbone.fc = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    feats = []
    with torch.no_grad():
        for s in range(0, len(paths), batch_size):
            batch = torch.stack([transform(Image.open(os.path.join(BASE, p)).convert("RGB"))
                                 for p in paths[s:s + batch_size]]).to(DEVICE)
            feats.append(backbone(batch).cpu().numpy())
    return np.concatenate(feats, 0)


def train_head(head, Xtr, Ytr, Xte, Yte, epochs=15, batch_size=256):
    head = head.to(DEVICE)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    te = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte))
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    te_dl = DataLoader(te, batch_size=batch_size)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            preds.append(head(xb).argmax(1).cpu().numpy())
            tgts.append(yb.numpy())
    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)
    return float((preds == tgts).mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    # training: every cleaned LC25000 colon image (labels 0=colon_aca, 1=colon_n)
    tr_paths, tr_labels = [], []
    for fp in (TRAIN_LIST, VAL_LIST):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p, l = line.rsplit(" ", 1)
                l = int(l)
                if l in (0, 1):
                    tr_paths.append(p)
                    tr_labels.append(l)
    tr_labels = np.array(tr_labels, dtype=np.int64)
    # test: CRC TUM(->0) / NORM(->1)
    te_paths = sorted(glob.glob(os.path.join(CRC, "TUM", "*"))) + sorted(glob.glob(os.path.join(CRC, "NORM", "*")))
    te_labels = np.array([0] * len(glob.glob(os.path.join(CRC, "TUM", "*"))) +
                         [1] * len(glob.glob(os.path.join(CRC, "NORM", "*"))), dtype=np.int64)
    te_paths = [os.path.relpath(p, BASE).replace("\\", "/") for p in te_paths]
    print(f"train (LC25000 colon) {len(tr_paths)} | test (CRC TUM/NORM) {len(te_paths)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr, Xte, Ytr, Yte = d["Xtr"], d["Xte"], d["Ytr"], d["Yte"]
        print("loading cached features")
    else:
        print("extracting LC25000 colon features ...")
        Xtr = extract(tr_paths, transform)
        print("extracting CRC TUM/NORM features ...")
        Xte = extract(te_paths, transform)
        Ytr, Yte = tr_labels, te_labels
        np.savez(CACHE, Xtr=Xtr, Xte=Xte, Ytr=Ytr, Yte=Yte)
        print(f"features cached: {CACHE}")
    print(f"train {Xtr.shape} | test {Xte.shape}")

    kan = build_kan_head(Xtr.shape[1])
    mlp = build_mlp_head(Xtr.shape[1])
    mlp_matched, H = build_mlp_matched(Xtr.shape[1], target=n_params(kan))
    print(f"params: KAN {n_params(kan):,} | MLP {n_params(mlp):,} | MLP-matched(H={H}) {n_params(mlp_matched):,}")

    seeds = DEFAULT_SEEDS[:args.seeds]
    res = {"kan": [], "mlp": [], "mlp_matched": []}
    for seed in seeds:
        for key, builder in (("kan", build_kan_head), ("mlp", build_mlp_head),
                             ("mlp_matched", lambda d: build_mlp_matched(d, target=n_params(kan))[0])):
            set_seed(seed)
            res[key].append(train_head(builder(Xtr.shape[1]), Xtr, Ytr, Xte, Yte, args.epochs))
        print(f"  seed {seed:>4d}: KAN {res['kan'][-1]:6.3f}% | MLP {res['mlp'][-1]:6.3f}% | "
              f"MM {res['mlp_matched'][-1]:6.3f}%", flush=True)
        with open(os.path.join(BASE, "KAN", "colon_gen_partial.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)

    from scipy import stats
    print("\n================ summary (cross-dataset accuracy) =================")
    for key in ("kan", "mlp", "mlp_matched"):
        a = np.array(res[key])
        print(f"{key:12s}: {a.mean():.3f}% ± {a.std(ddof=1):.3f}%")
    for other in ("mlp", "mlp_matched"):
        kan_a, oth_a = np.array(res["kan"]), np.array(res[other])
        d = kan_a - oth_a
        t, p = stats.ttest_rel(kan_a, oth_a)
        F = max(kan_a.var(ddof=1), oth_a.var(ddof=1)) / min(kan_a.var(ddof=1), oth_a.var(ddof=1))
        p_f = 2 * (1 - stats.f.cdf(F, len(kan_a) - 1, len(oth_a) - 1))
        print(f"KAN vs {other}: paired diff {d.mean():+.3f}% | t={t:.3f} p={p:.4f} | F={F:.2f} p={p_f:.4f}")

    out = {"seeds": len(seeds), "params": {"kan": n_params(kan), "mlp": n_params(mlp),
                                           "mlp_matched": n_params(mlp_matched), "H": H},
           "acc": {k: np.asarray(v).tolist() for k, v in res.items()}}
    with open(os.path.join(BASE, "KAN", "colon_gen_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("results saved to colon_gen_result.json")


if __name__ == "__main__":
    main()
