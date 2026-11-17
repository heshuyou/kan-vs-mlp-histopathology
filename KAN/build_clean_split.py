# -*- coding: utf-8 -*-
"""
Data cleaning. Reads the images only; it writes new split files and deletes nothing:
  1. LC25000 (jpg/): exact MD5 de-duplication (1,280 redundant images), keeping one copy per hash;
     stratified 80/20 split over the five class prefixes, and the hash step guarantees zero train/val content leakage.
  2. CRC-VAL-HE-7K: stratified 80/20 split over nine classes (the set has no duplicates to begin with).
Output: train_clean.txt / val_clean.txt / train_crc.txt / val_crc.txt (path + label)
Usage: conda activate gpt2 && python build_clean_split.py
"""
import os
import hashlib
import glob
from collections import defaultdict, Counter

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JPG = os.path.join(BASE, "jpg")
CRC = os.path.join(BASE, "CRC-VAL-HE-7K")

# LC25000 prefix -> label (0=colon adenocarcinoma, 1=colon benign, 2=lung adenocarcinoma, 3=lung benign, 4=lung squamous)
LC_PREFIX = [("colon_aca", 0), ("colonn", 1), ("lungaca", 2), ("lungn", 3), ("lungscc", 4)]
# CRC-VAL-HE-7K class -> label
CRC_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

SEED = 42
VAL_FRAC = 0.2


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def lc_label(path):
    name = os.path.basename(path).lower()
    for pref, lab in LC_PREFIX:
        if name.startswith(pref):
            return lab
    raise ValueError(f"unknown prefix: {name}")


def write_list(path, pairs):
    """pairs: [(relative path, label)], relative to BASE."""
    with open(path, "w", encoding="utf-8") as f:
        for rel, lab in pairs:
            f.write(f"{rel} {lab}\n")


def verify_no_leak(tr_pairs, va_pairs):
    tr_h = {md5(os.path.join(BASE, p)) for p, _ in tr_pairs}
    va_h = {md5(os.path.join(BASE, p)) for p, _ in va_pairs}
    return len(tr_h & va_h)


def main():
    # ---------- LC25000 ----------
    print("=== LC25000 de-duplication ===")
    all_p = sorted(glob.glob(os.path.join(JPG, "*.jpeg")))
    by_hash = defaultdict(list)
    for p in all_p:
        by_hash[md5(p)].append(p)
    n_raw = len(all_p)
    n_unique = len(by_hash)
    print(f"{n_raw} images, {n_unique} unique hashes, {n_raw - n_unique} redundant")

    # Keep one image per hash (the first in sorted order, so the result is deterministic)
    unique = [sorted(v)[0] for v in by_hash.values()]

    # Stratified 80/20
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for p in unique:
        by_class[lc_label(p)].append(p)
    tr_pairs, va_pairs = [], []
    for lab in sorted(by_class):
        ps = sorted(by_class[lab])
        rng.shuffle(ps)
        n_va = int(round(len(ps) * VAL_FRAC))
        for p in ps[n_va:]:
            tr_pairs.append((os.path.relpath(p, BASE).replace("\\", "/"), lab))
        for p in ps[:n_va]:
            va_pairs.append((os.path.relpath(p, BASE).replace("\\", "/"), lab))
        print(f"  class {lab}: {len(ps)} images -> train {len(ps)-n_va} / val {n_va}")

    leak = verify_no_leak(tr_pairs, va_pairs)
    assert leak == 0, f"leakage remains: {leak} images"
    tr_path = os.path.join(BASE, "train_clean.txt")
    va_path = os.path.join(BASE, "val_clean.txt")
    write_list(tr_path, tr_pairs)
    write_list(va_path, va_pairs)
    print(f"LC25000 after cleaning: train {len(tr_pairs)} / val {len(va_pairs)}, leakage {leak} (expected 0)")
    print(f"  wrote {os.path.basename(tr_path)} / {os.path.basename(va_path)}")

    # ---------- CRC-VAL-HE-7K ----------
    print("\n=== CRC-VAL-HE-7K stratified split ===")
    crc_tr, crc_va = [], []
    for lab, cls in enumerate(CRC_CLASSES):
        d = os.path.join(CRC, cls)
        ps = sorted(glob.glob(os.path.join(d, "*")))
        rng.shuffle(ps)
        n_va = int(round(len(ps) * VAL_FRAC))
        for p in ps[n_va:]:
            crc_tr.append((os.path.relpath(p, BASE).replace("\\", "/"), lab))
        for p in ps[:n_va]:
            crc_va.append((os.path.relpath(p, BASE).replace("\\", "/"), lab))
        print(f"  {cls}: {len(ps)} images -> train {len(ps)-n_va} / val {n_va}")
    tr_crc = os.path.join(BASE, "train_crc.txt")
    va_crc = os.path.join(BASE, "val_crc.txt")
    write_list(tr_crc, crc_tr)
    write_list(va_crc, crc_va)
    print(f"CRC split: train {len(crc_tr)} / val {len(crc_va)}")
    print(f"  wrote {os.path.basename(tr_crc)} / {os.path.basename(va_crc)}")


if __name__ == "__main__":
    main()
