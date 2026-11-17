# -*- coding: utf-8 -*-
"""
Independent nine-class retraining on CRC-VAL-HE-7K (frozen ResNet50, head only):
  tests whether the finding that KAN brings no gain and is less stable also holds on the harder, unsaturated nine-class tissue task.
  Three heads: KAN / MLP (same depth 128-64-32) / MLP-matched (parameter count close to KAN).
  Fixed split train_crc.txt / val_crc.txt; seeds affect only head initialisation and training randomness.
  Metrics: accuracy + macro-F1; paired t test + F test. Features are cached to npy.

Usage: conda activate gpt2 && python crc_9class_heads.py --seeds 20
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
from PIL import Image

from baselines import KANBlock, KANLayer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_LIST = os.path.join(BASE, "train_crc.txt")
VAL_LIST = os.path.join(BASE, "val_crc.txt")
CACHE = os.path.join(BASE, "KAN", "crc9_resnet50_feats.npz")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 9
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

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


def load_list(fp):
    paths, labels = [], []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p, l = line.rsplit(" ", 1)
                paths.append(p)
                labels.append(int(l))
    return paths, np.array(labels, dtype=np.int64)


def extract_features(paths, transform, batch_size=64):
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


def train_head(head, Xtr, Ytr, Xva, Yva, epochs=15, batch_size=256):
    head = head.to(DEVICE)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va, batch_size=batch_size)
    opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1, best_state = -1.0, None
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
            for xb, yb in va_dl:
                xb = xb.to(DEVICE)
                preds.append(head(xb).argmax(1).cpu().numpy())
                tgts.append(yb.numpy())
        preds = np.concatenate(preds)
        tgts = np.concatenate(tgts)
        macro_f1 = _macro_f1(preds, tgts)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    # final accuracy + macro-F1 on the validation slice
    head.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in va_dl:
            xb = xb.to(DEVICE)
            preds.append(head(xb).argmax(1).cpu().numpy())
            tgts.append(yb.numpy())
    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)
    return float((preds == tgts).mean()) * 100, _macro_f1(preds, tgts)


def _macro_f1(preds, tgts):
    fs = []
    for c in range(N_CLS):
        tp = ((preds == c) & (tgts == c)).sum()
        fp = ((preds == c) & (tgts != c)).sum()
        fn = ((preds != c) & (tgts == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fs.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    tr_paths, tr_labels = load_list(TRAIN_LIST)
    va_paths, va_labels = load_list(VAL_LIST)
    print(f"CRC nine-class: train {len(tr_paths)} | val {len(va_paths)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        Xtr, Xva, Ytr, Yva = d["Xtr"], d["Xva"], d["Ytr"], d["Yva"]
        print("loading cached features")
    else:
        print("extracting frozen ResNet50 features (train) ...")
        Xtr = extract_features(tr_paths, transform)
        print("extracting frozen ResNet50 features (val) ...")
        Xva = extract_features(va_paths, transform)
        Ytr, Yva = tr_labels, va_labels
        np.savez(CACHE, Xtr=Xtr, Xva=Xva, Ytr=Ytr, Yva=Yva)
        print(f"features cached: {CACHE}")
    print(f"train {Xtr.shape} | val {Xva.shape}")

    kan = build_kan_head(Xtr.shape[1])
    mlp = build_mlp_head(Xtr.shape[1])
    mlp_matched, H = build_mlp_matched(Xtr.shape[1], target=n_params(kan))
    print(f"params: KAN {n_params(kan):,} | MLP {n_params(mlp):,} | MLP-matched(H={H}) {n_params(mlp_matched):,}")

    seeds = DEFAULT_SEEDS[:args.seeds]
    res = {"acc": {"kan": [], "mlp": [], "mlp_matched": []},
           "f1": {"kan": [], "mlp": [], "mlp_matched": []}}
    for seed in seeds:
        for key, builder in (("kan", build_kan_head), ("mlp", build_mlp_head),
                             ("mlp_matched", lambda d: build_mlp_matched(d, target=n_params(kan))[0])):
            set_seed(seed)
            acc, f1 = train_head(builder(Xtr.shape[1]), Xtr, Ytr, Xva, Yva, args.epochs)
            res["acc"][key].append(acc)
            res["f1"][key].append(f1)
        print(f"  seed {seed:>4d}: acc KAN {res['acc']['kan'][-1]:6.3f}% | MLP {res['acc']['mlp'][-1]:6.3f}% | "
              f"MM {res['acc']['mlp_matched'][-1]:6.3f}%  ||  macro-F1 KAN {res['f1']['kan'][-1]:.4f} | "
              f"MLP {res['f1']['mlp'][-1]:.4f} | MM {res['f1']['mlp_matched'][-1]:.4f}", flush=True)
        # Save incrementally so that an interrupted run loses nothing
        with open(os.path.join(BASE, "KAN", "crc9_partial.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)

    from scipy import stats
    print("\n================ summary ================")
    for key in ("kan", "mlp", "mlp_matched"):
        a = np.array(res["acc"][key])
        f = np.array(res["f1"][key])
        print(f"{key:12s}: acc {a.mean():.3f}%±{a.std(ddof=1):.3f}% | macro-F1 {f.mean():.4f}±{f.std(ddof=1):.4f}")
    for other in ("mlp", "mlp_matched"):
        kan_a, oth_a = np.array(res["acc"]["kan"]), np.array(res["acc"][other])
        d = kan_a - oth_a
        t, p = stats.ttest_rel(kan_a, oth_a)
        F = max(kan_a.var(ddof=1), oth_a.var(ddof=1)) / min(kan_a.var(ddof=1), oth_a.var(ddof=1))
        p_f = 2 * (1 - stats.f.cdf(F, len(kan_a) - 1, len(oth_a) - 1))
        print(f"KAN vs {other}: paired diff {d.mean():+.3f}% | t={t:.3f} p={p:.4f} | F={F:.2f} p={p_f:.4f}")

    out = {"seeds": len(seeds), "params": {"kan": n_params(kan), "mlp": n_params(mlp),
                                           "mlp_matched": n_params(mlp_matched), "H": H},
           "acc": {k: np.asarray(v).tolist() for k, v in res["acc"].items()},
           "f1": {k: np.asarray(v).tolist() for k, v in res["f1"].items()}}
    with open(os.path.join(BASE, "KAN", "crc9_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("results saved to crc9_result.json")


if __name__ == "__main__":
    main()
