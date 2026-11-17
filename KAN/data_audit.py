# -*- coding: utf-8 -*-
"""
Data audit. Read-only, no training: find the problems in LC25000 (jpg/) and CRC-VAL-HE-7K.
Checks:
  1. per-class image counts, file sizes, image-size distribution
  2. exact duplicates (identical MD5): within one dataset / across datasets / train-val leakage
  3. whether train_list.txt and val_list.txt share file names or hashes
Usage: conda activate gpt2 && python data_audit.py
"""
import os
import hashlib
import glob
from collections import Counter, defaultdict
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JPG = os.path.join(BASE, "jpg")
CRC = os.path.join(BASE, "CRC-VAL-HE-7K")
TRAIN_LIST = os.path.join(BASE, "train_list.txt")
VAL_LIST = os.path.join(BASE, "val_list.txt")

CRC_CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root, exts=('.jpeg', '.jpg', '.png', '.tif')):
    """Return {path: md5} for files whose name matches exts."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(p) and p.lower().endswith(exts):
            out[p] = None
    return out


def fill(paths):
    for p in paths:
        paths[p] = md5(p)
    return paths


def report_dups(name, paths):
    by_hash = defaultdict(list)
    for p, h in paths.items():
        by_hash[h].append(p)
    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    n_dup_imgs = sum(len(v) - 1 for v in dups.values())
    print(f"[{name}] {len(paths)} images, {len(by_hash)} unique hashes, {len(dups)} duplicate groups, {n_dup_imgs} redundant images")
    return by_hash


def main():
    print("=== 1. scan jpg/ (LC25000) ===")
    jpg_paths = fill(scan(JPG))
    jpg_hash = report_dups("LC25000", jpg_paths)

    # LC25000 class distribution (by file-name prefix)
    prefix_cnt = Counter()
    for p in jpg_paths:
        name = os.path.basename(p).lower()
        for pref in ('colon_aca', 'colonn', 'lungaca', 'lungn', 'lungscc'):
            if name.startswith(pref):
                prefix_cnt[pref] += 1
                break
        else:
            prefix_cnt['OTHER'] += 1
    print("   LC25000 prefix distribution:", dict(prefix_cnt))

    print("\n=== 2. scan CRC-VAL-HE-7K ===")
    crc_paths = fill(scan(CRC))
    crc_hash = report_dups("CRC-VAL-HE-7K", crc_paths)
    for c in CRC_CLASSES:
        n = sum(1 for p in crc_paths if os.path.sep + c + os.path.sep in p)
        print(f"   {c}: {n}")

    print("\n=== 3. cross-dataset overlap (LC25000 vs CRC-VAL-HE-7K, by MD5) ===")
    jpg_hset = set(jpg_hash.keys())
    crc_hset = set(crc_hash.keys())
    overlap = jpg_hset & crc_hset
    print(f"   exact overlapping hashes: {len(overlap)}")
    if overlap:
        # print the prefix distribution of the overlapping LC25000 images
        jp = defaultdict(list)
        for h in overlap:
            for p in jpg_hash[h]:
                jp[os.path.basename(p)[:20]].append(p)
        # count the classes of the overlapping LC25000 images
        oc = Counter()
        for h in overlap:
            for p in jpg_hash[h]:
                name = os.path.basename(p).lower()
                for pref in ('colon_aca', 'colonn', 'lungaca', 'lungn', 'lungscc'):
                    if name.startswith(pref):
                        oc[pref] += 1
                        break
        print("   class distribution of overlapping LC25000 images:", dict(oc))
        # class distribution of the overlapping CRC images
        cc = Counter()
        for h in overlap:
            for p in crc_hash[h]:
                for c in CRC_CLASSES:
                    if os.path.sep + c + os.path.sep in p:
                        cc[c] += 1
        print("   class distribution of overlapping CRC images:", dict(cc))

    print("\n=== 4. train_list vs val_list leakage ===")
    def load_list(fp):
        names = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line.split()[0])
        return names
    tr = load_list(TRAIN_LIST)
    va = load_list(VAL_LIST)
    tr_set, va_set = set(tr), set(va)
    print(f"   train {len(tr)} rows ({len(tr_set)} unique), val {len(va)} rows ({len(va_set)} unique)")
    print(f"   shared paths between train and val (same file name): {len(tr_set & va_set)}")
    # check by hash whether train and val share content
    tr_h = {md5(os.path.join(BASE, p)) for p in tr_set if os.path.exists(os.path.join(BASE, p))}
    va_h = {md5(os.path.join(BASE, p)) for p in va_set if os.path.exists(os.path.join(BASE, p))}
    print(f"   train hashes {len(tr_h)}, val hashes {len(va_h)}, hash intersection (content leakage): {len(tr_h & va_h)}")

    print("\n=== 5. image-size distribution (first 20 images per class) ===")
    dims = Counter()
    samples = []
    for p in list(jpg_paths)[:400]:
        samples.append(p)
    for p in list(crc_paths)[:200]:
        samples.append(p)
    for p in samples:
        try:
            with Image.open(p) as im:
                dims[im.size] += 1
        except Exception as e:
            print(f"   read failed {p}: {e}")
    print("   size distribution:", dict(dims))

    print("\n=== 6. abnormal file sizes (0 bytes or tiny) ===")
    bad = [p for p in list(jpg_paths) + list(crc_paths) if os.path.getsize(p) < 1024]
    print(f"   files below 1 KB: {len(bad)}")
    for p in bad[:20]:
        print("     ", p, os.path.getsize(p))


if __name__ == "__main__":
    main()
