# -*- coding: utf-8 -*-
"""
Error analysis after cleaning: train the KAN head on cached ResNet50 features (seed 42) and
produce the confusion matrix and the error pattern on val_clean, checking whether the claim that errors concentrate on lung squamous vs lung adenocarcinoma reproduces.

Usage: conda activate gpt2 && python error_analysis_clean.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from baselines import KANBlock, KANLayer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = os.path.join(BASE, "KAN", "lc25000_feats_resnet50.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["colon adenocarcinoma", "colon benign", "lung adenocarcinoma", "lung benign", "lung squamous"]


def build_kan_head(d_in, n_cls=5):
    return nn.Sequential(
        KANBlock(d_in, 128, 5, 0.2), KANBlock(128, 64, 5, 0.2),
        KANBlock(64, 32, 5, 0.2), KANLayer(32, n_cls, 5, 0.0),
    )


def train(head, Xtr, Ytr, Xva, Yva, epochs=15, batch_size=512, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    head.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in va_dl:
            xb = xb.to(DEVICE)
            preds.append(head(xb).argmax(1).cpu().numpy())
            tgts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def main():
    d = np.load(FEATS)
    Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
    Xtr, Xva = Xtr.astype(np.float32), Xva.astype(np.float32)
    print(f"train {Xtr.shape} | val {Xva.shape}")

    preds, tgts = train(build_kan_head(Xtr.shape[1]), Xtr, Ytr, Xva, Yva)
    n = len(tgts)
    acc = (preds == tgts).mean() * 100
    print(f"\nKAN head (seed 42) test accuracy: {acc:.2f}% | {int((preds != tgts).sum())}/{n} misclassified")

    # confusion matrix
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(tgts, preds):
        cm[t, p] += 1
    print("\nconfusion matrix (rows = true, columns = predicted):")
    header = "         " + "".join(f"{c:>8s}" for c in CLASS_NAMES)
    print(header)
    for i in range(5):
        print(f"{CLASS_NAMES[i]:>7s} " + "".join(f"{cm[i, j]:>8d}" for j in range(5)))

    # error pattern
    print("\nerror pattern (true -> predicted):")
    err = {}
    for t, p in zip(tgts, preds):
        if t != p:
            key = f"{CLASS_NAMES[t]}→{CLASS_NAMES[p]}"
            err[key] = err.get(key, 0) + 1
    for k, v in sorted(err.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} cases")

    # error count per class
    print("\nerrors per class:")
    for c in range(5):
        n_c = int((tgts == c).sum())
        e_c = int(((tgts == c) & (preds != c)).sum())
        print(f"  {CLASS_NAMES[c]}: {e_c}/{n_c} (error rate {e_c/n_c*100:.2f}%)")

    out = {"acc": acc, "confusion_matrix": cm.tolist(), "errors": err}
    with open(os.path.join(BASE, "KAN", "error_analysis_clean.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nresults saved to error_analysis_clean.json")


if __name__ == "__main__":
    main()
