"""
linear_probe_eval.py

Trains a single linear layer on frozen features and reports, per seed and as
mean +- sd over seeds: accuracy (%), macro-F1, balanced accuracy (%), per-class
recall, and the summed confusion matrix.

Changes relative to linear_probe.py, and why:

  1. train_one_seed() keeps the predictions of the best epoch, so macro-F1 and
     per-class metrics can be computed at all (the original returned only a
     scalar accuracy).
  2. --select lets the probe use the SAME model-selection criterion as the
     heads it is compared with. crc_9class_heads.py selects on validation
     macro-F1 (line 136 of that file); the original probe selected on
     validation accuracy. Pass --select f1 to match the CRC table.
  3. --seed-set lets the probe use the SAME seed list as the rest of the study.
     crc_9class_heads.py: DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 100, ...];
     the original probe used [42, 123, 2024, 7, 99, 2025, 314, ...].

Recommended invocation to make the CRC linear-probe row comparable with the
three heads of the in-domain table (same cached features, same seeds, same selection
criterion, same batch size):

    python linear_probe_eval.py --task crc9 --backbone resnet50 --seeds 20 \
        --select f1 --seed-set study --batch-size 256

Requires torch, numpy, scikit-learn; the .npz feature file must sit in the
current working directory.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)


class LinearProbe(nn.Module):
    """Single linear layer with input dropout, as specified in Sec. 2.2 of the paper."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


NPZ_FILES = {
    ("lc25000", "resnet50"):        "lc25000_feats_resnet50.npz",
    ("lc25000", "efficientnet_b0"): "lc25000_feats_efficientnet_b0.npz",
    ("crc9",    "resnet50"):        "crc9_resnet50_feats.npz",
}

_KEY_CANDIDATES = {
    "X_train": ["Xtr", "X_train", "train_feats", "feats_train", "x_train",
                "train_features"],
    "y_train": ["Ytr", "ytr", "y_train", "train_labels", "labels_train",
                "train_targets"],
    "X_val":   ["Xva", "X_val", "val_feats", "feats_val", "x_val",
                "val_features", "X_test", "test_feats"],
    "y_val":   ["Yva", "yva", "y_val", "val_labels", "labels_val",
                "val_targets", "y_test", "test_labels"],
}

# Seed lists. "study" is copied verbatim from crc_9class_heads.py (DEFAULT_SEEDS);
# "placeholder" is the list the original linear_probe.py carried.
SEED_SETS = {
    "study": [42, 123, 2024, 7, 99, 100, 200, 300, 400, 500,
              600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    "placeholder": [42, 123, 2024, 7, 99, 2025, 314, 271, 161, 500,
                    777, 888, 999, 101, 202, 303, 404, 505, 606, 707],
}


def _pick(npz, slot: str):
    for k in _KEY_CANDIDATES[slot]:
        if k in npz.files:
            return npz[k]
    raise KeyError(f"Could not find '{slot}' in npz. Available keys: {npz.files}")


def load_features(task: str, backbone: str = "resnet50"):
    fname = NPZ_FILES[(task, backbone)]
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Expected {fname} in the current directory. Available: {os.listdir('.')}")
    data = np.load(fname)
    print(f"[{task}/{backbone}] loaded {fname}, keys = {list(data.files)}")

    Xtr = torch.from_numpy(_pick(data, "X_train")).float()
    ytr = torch.from_numpy(_pick(data, "y_train")).long()
    Xva = torch.from_numpy(_pick(data, "X_val")).float()
    yva = torch.from_numpy(_pick(data, "y_val")).long()

    num_classes = int(ytr.max().item()) + 1
    print(f"  train={tuple(Xtr.shape)}, val={tuple(Xva.shape)}, "
          f"in_dim={Xtr.shape[1]}, classes={num_classes}")
    return Xtr, ytr, Xva, yva, Xtr.shape[1], num_classes


def train_one_seed(seed, Xtr, ytr, Xva, yva, in_dim, num_classes,
                   epochs=15, batch_size=512, lr=1e-3, weight_decay=1e-4,
                   label_smoothing=0.1, dropout=0.2, select="acc", device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LinearProbe(in_dim, num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                          shuffle=True, num_workers=0)
    Xva_dev, yva_dev = Xva.to(device), yva.to(device)
    yva_np = yva_dev.cpu().numpy()

    best = {"score": -1.0, "epoch": -1, "pred": None}
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva_dev).argmax(dim=1)
        pred_np = pred.cpu().numpy()
        score = ((pred_np == yva_np).mean() if select == "acc"
                 else f1_score(yva_np, pred_np, average="macro", zero_division=0))
        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch, "pred": pred_np}

    pred = best["pred"]
    return {
        "acc": float((pred == yva_np).mean() * 100),
        "epoch": best["epoch"],
        "macro_f1": float(f1_score(yva_np, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(yva_np, pred) * 100),
        "per_class_recall": recall_score(yva_np, pred, average=None,
                                         zero_division=0),
        "confusion": confusion_matrix(yva_np, pred),
    }


def _sd(x, ddof=1):
    return float(np.std(x, ddof=ddof))


def run(task, backbone="resnet50", n_seeds=20, epochs=15, batch_size=512,
        dropout=0.2, select="acc", seed_set="study", device=None, out=None):
    Xtr, ytr, Xva, yva, in_dim, num_classes = load_features(task, backbone)
    seeds = SEED_SETS[seed_set][:n_seeds]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seeds={len(seeds)} [{seed_set}]  batch={batch_size}  "
          f"epochs={epochs}  select={select}")

    rows = []
    for s in seeds:
        r = train_one_seed(s, Xtr, ytr, Xva, yva, in_dim, num_classes,
                           epochs=epochs, batch_size=batch_size, dropout=dropout,
                           select=select, device=device)
        rows.append(r)
        print(f"  seed={s:>5d}  acc={r['acc']:.3f}  macroF1={r['macro_f1']:.4f}  "
              f"balAcc={r['balanced_acc']:.3f}  best_ep={r['epoch']:>2d}")

    acc = np.array([r["acc"] for r in rows])
    f1 = np.array([r["macro_f1"] for r in rows])
    bacc = np.array([r["balanced_acc"] for r in rows])
    rec = np.stack([r["per_class_recall"] for r in rows], axis=0)
    cm = np.sum([r["confusion"] for r in rows], axis=0)

    summary = {
        "task": task, "backbone": backbone, "n_seeds": len(seeds),
        "seed_set": seed_set, "seeds": seeds,
        "batch_size": batch_size, "epochs": epochs, "dropout": dropout,
        "select": select,
        "acc_mean": float(acc.mean()), "acc_sd_ddof1": _sd(acc),
        "acc_sd_ddof0": _sd(acc, 0),
        "macro_f1_mean": float(f1.mean()), "macro_f1_sd": _sd(f1),
        "balanced_acc_mean": float(bacc.mean()), "balanced_acc_sd": _sd(bacc),
        "per_class_recall_mean": rec.mean(axis=0).tolist(),
        "per_class_recall_sd": rec.std(axis=0, ddof=1).tolist(),
        "acc_min": float(acc.min()), "acc_max": float(acc.max()),
        "confusion_pooled": cm.tolist(),
    }

    print(f"\n=== Linear probe | {task} | {backbone} | {len(seeds)} seeds "
          f"[{seed_set}] | select={select} | batch={batch_size} ===")
    print(f"  accuracy      {acc.mean():.3f} +- {_sd(acc):.3f}  (ddof=0: {_sd(acc, 0):.3f})")
    print(f"  macro-F1      {f1.mean():.4f} +- {_sd(f1):.4f}")
    print(f"  balanced acc  {bacc.mean():.3f} +- {_sd(bacc):.3f}")
    print("  per-class recall (mean, sd, n):")
    yva_np = yva.numpy()
    for c in range(num_classes):
        print(f"    class {c}: {rec[:, c].mean():.4f} +- {rec[:, c].std(ddof=1):.4f}"
              f"   n={(yva_np == c).sum()}")

    path = out or f"linearprobe_{task}_{backbone}_{seed_set}_{select}_n{len(seeds)}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(path.replace(".json", ".npz"), per_seed_acc=acc, per_seed_f1=f1,
             per_seed_bacc=bacc, per_class_recall=rec, pooled_confusion=cm)
    print(f"\n  wrote {path} (+ .npz with the per-seed arrays)")

    if task == "crc9" and seed_set == "placeholder" and select == "acc" \
            and batch_size == 512 and abs(acc.mean() - 96.100) > 0.15:
        print(f"\n  [warn] {acc.mean():.3f} differs from 96.100 by more than 0.15 pp.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["lc25000", "crc9"], default="crc9")
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"],
                    default="resnet50")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--select", choices=["acc", "f1"], default="acc",
                    help="criterion for the early-stopping epoch (crc_9class_heads.py uses f1)")
    ap.add_argument("--seed-set", choices=list(SEED_SETS), default="study",
                    help="which seed list to use; 'study' matches crc_9class_heads.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.task, args.backbone, args.seeds, args.epochs, args.batch_size,
        args.dropout, args.select, args.seed_set, args.device, args.out)
