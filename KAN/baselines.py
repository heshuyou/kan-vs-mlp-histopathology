# -*- coding: utf-8 -*-
"""
LC25000 baseline comparison scripts
Fair comparison against the KAN classifier:
  - mlp             : identical CNN feature extractor + MLP head (isolates the "KAN head vs MLP head" contribution)
  - resnet50        : ImageNet-pretrained ResNet50, frozen backbone, only the new head is trained (transfer-learning baseline)
  - efficientnet_b0 : ImageNet-pretrained EfficientNet-B0, frozen backbone, only the new head is trained

Kept consistent with KAN20260125.py: same data pipeline, same augmentation, same optimizer and loss, same image_size=256.
Depends only on torch / torchvision / numpy / PIL; metrics are computed in numpy (no sklearn, no matplotlib); results go to JSON + txt.

Usage example:
  python baselines.py --model mlp --epochs 30
  python baselines.py --model resnet50 --epochs 15
  python baselines.py --model efficientnet_b0 --epochs 15
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PathologyDataset(Dataset):
    """Pathology image dataset, identical to the KAN version."""

    def __init__(self, label_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, label = line.rsplit(" ", 1)
                self.samples.append((img_path, int(label)))
        print(f"loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.img_dir, img_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"image not found: {full_path}")
        image = Image.open(full_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------
# Lightweight CNN feature extractor identical to KAN20260125.py (used for the MLP baseline so the architecture matches)
# ----------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ----------------------------------------------------------------------
# KAN head (Chebyshev polynomial basis), identical to KAN20260125.py, kept for GPU reproduction
# ----------------------------------------------------------------------
class ChebyshevPolynomialLayer(nn.Module):
    def __init__(self, degree=5):
        super().__init__()
        self.degree = degree

    def compute_chebyshev_basis(self, x):
        x_scaled = 2.0 * torch.sigmoid(x) - 1.0
        chebyshev = [torch.ones_like(x_scaled), x_scaled]
        for n in range(1, self.degree):
            T_n = 2.0 * x_scaled * chebyshev[-1] - chebyshev[-2]
            chebyshev.append(T_n)
        return torch.stack(chebyshev, dim=1)

    def forward(self, x):
        return self.compute_chebyshev_basis(x)


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=5, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, degree + 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self.chebyshev = ChebyshevPolynomialLayer(degree)
        nn.init.kaiming_normal_(self.coefficients, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        batch_size = x.shape[0]
        cheb_basis = self.chebyshev(x).transpose(1, 2)  # [batch, in_features, degree+1]
        coeff_expanded = self.coefficients.unsqueeze(0).expand(batch_size, -1, -1, -1)
        weighted = torch.sum(coeff_expanded * cheb_basis.unsqueeze(1), dim=-1)
        output = torch.sum(weighted, dim=2) + self.bias
        return self.dropout(output)


class KANBlock(nn.Module):
    def __init__(self, in_features, out_features, degree=5, dropout=0.1, use_bn=True):
        super().__init__()
        self.kan_layer = KANLayer(in_features, out_features, degree, dropout)
        self.norm = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.norm(self.kan_layer(x)))


class KANClassifier(nn.Module):
    """CNN feature extractor + KAN head (256->128->64->32->5), reproducing KAN20260125.py."""

    def __init__(self, num_classes=NUM_CLASSES, feature_dim=256, kan_degree=5, dropout=0.2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.kan_classifier = nn.Sequential(
            KANBlock(feature_dim, 128, kan_degree, dropout),
            KANBlock(128, 64, kan_degree, dropout),
            KANBlock(64, 32, kan_degree, dropout),
            KANLayer(32, num_classes, kan_degree, dropout=0.0),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.kan_classifier(self.feature_extractor(x))


class MLPBaseline(nn.Module):
    """MLP head with the same layer sizes as the KAN head (256->128->64->32->5)."""

    def __init__(self, feature_dim=256, num_classes=NUM_CLASSES, dropout=0.2):
        super(MLPBaseline, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.head(self.feature_extractor(x))


def get_model(name):
    if name == "mlp":
        return MLPBaseline()
    if name == "kan":
        return KANClassifier()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model
    raise ValueError(f"unknown model: {name}")


def compute_metrics(preds, targets):
    """Compute accuracy and per-class P/R/F1 in numpy, without sklearn."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    acc = float((preds == targets).mean())
    per_class = []
    for c in range(NUM_CLASSES):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class.append({"class": CLASS_NAMES[c], "precision": round(p, 4),
                          "recall": round(r, 4), "f1": round(f1, 4), "support": int((targets == c).sum())})
    macro_p = float(np.mean([x["precision"] for x in per_class]))
    macro_r = float(np.mean([x["recall"] for x in per_class]))
    macro_f1 = float(np.mean([x["f1"] for x in per_class]))
    return acc, per_class, macro_p, macro_r, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "kan", "resnet50", "efficientnet_b0"])
    parser.add_argument("--data-dir", default="../")
    parser.add_argument("--train-file", default="train_list.txt")
    parser.add_argument("--test-file", default="val_list.txt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="baseline_results")
    args = parser.parse_args()

    # Default epoch budget: transfer learning on a frozen backbone converges fast, so fewer epochs suffice
    if args.epochs is None:
        args.epochs = 30 if args.model in ("mlp", "kan") else 15

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model: {args.model} | device: {device} | epochs: {args.epochs}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = PathologyDataset(os.path.join(args.data_dir, args.train_file), args.data_dir, train_transform)
    test_dataset = PathologyDataset(os.path.join(args.data_dir, args.test_file), args.data_dir, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = get_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {n_params:,} | trainable params: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.out, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    prev_best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tc, tot = 0.0, 0, 0
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tl += loss.item() * target.size(0)
            tc += (out.argmax(1) == target).sum().item()
            tot += target.size(0)
        train_loss = tl / tot
        train_acc = 100.0 * tc / tot

        model.eval()
        vl, vc, vtot = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                vl += criterion(out, target).item() * target.size(0)
                vc += (out.argmax(1) == target).sum().item()
                vtot += target.size(0)
        val_loss = vl / vtot
        val_acc = 100.0 * vc / vtot
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        # Resume support: save the best model and the progress every epoch so that an interrupted run loses nothing
        if val_acc > prev_best:
            torch.save(model.state_dict(), os.path.join(args.out, f"{args.model}_best.pth"))
            prev_best = val_acc
        with open(os.path.join(args.out, f"{args.model}_progress.json"), "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "epoch": epoch,
                       "best_val_acc": round(best_acc, 2), "last_val_acc": round(val_acc, 2)}, f)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        print(f"Epoch {epoch:2d}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.2f}% "
              f"| val_loss {val_loss:.4f} acc {val_acc:.2f}% | {time.time()-t0:.1f}s")

    # Final evaluation on the test partition
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            preds.extend(model(data).argmax(1).cpu().numpy())
            targets.extend(target.numpy())
    acc, per_class, macro_p, macro_r, macro_f1 = compute_metrics(preds, targets)

    result = {
        "model": args.model,
        "device": device,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "best_val_acc": round(best_acc, 2),
        "test_accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "history": history,
    }

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, f"{args.model}.json")
    out_txt = os.path.join(args.out, f"{args.model}.txt")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        f"model: {args.model}  device: {device}  epochs: {args.epochs}",
        f"trainable params: {n_trainable:,} / {n_params:,}",
        f"test accuracy: {acc*100:.2f}%  (best val: {best_acc:.2f}%)",
        f"macro P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}",
        "",
        "per-class metrics (precision / recall / f1 / support):",
    ]
    for c in per_class:
        lines.append(f"  {c['class']:<12s} {c['precision']:.4f}  {c['recall']:.4f}  {c['f1']:.4f}  {c['support']}")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nresults saved: {out_json} / {out_txt}")


if __name__ == "__main__":
    main()
