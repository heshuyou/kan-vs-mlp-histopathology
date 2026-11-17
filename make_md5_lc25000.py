"""Rebuild md5_lc25000.csv, the per-image MD5 inventory of the cleaned LC25000 split.

The inventory backs the de-duplication described in README section 3: it lists every
image of the 23,720-image corpus with its MD5, class label and partition, so that a
reader can check which images are exact byte-level copies of one another.

Inputs, both at the repository root:

    train_clean.txt   18,976 rows (17,078 train partition + 1,898 selection slice)
    val_clean.txt      4,744 rows (report partition)

Image paths inside the list files are relative to the LC25000 root, for example
"jpg/lung_aca/lungaca1.jpeg", so the directory passed to --data-dir must contain jpg/.

Output: md5_lc25000.csv at the repository root, with columns

    path,md5,label,partition

Deterministic: the row order follows the two list files and the first occurrence of a
path wins, so re-running the script reproduces the shipped file byte for byte.
"""
import argparse
import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=ROOT,
                    help="directory that contains jpg/ (default: repository root)")
    ap.add_argument("--out", default=os.path.join(ROOT, "md5_lc25000.csv"),
                    help="output CSV (default: md5_lc25000.csv at the repository root)")
    args = ap.parse_args()

    rows = []
    for fname, part in (("train_clean.txt", "train"), ("val_clean.txt", "report")):
        with open(os.path.join(ROOT, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(None, 1)
                rows.append((rel, label, part))

    seen = {}
    out = []
    missing = 0
    for rel, label, part in rows:
        abspath = os.path.join(args.data_dir, rel)
        if not os.path.exists(abspath):
            missing += 1
            continue
        if rel in seen:
            continue
        seen[rel] = True
        out.append((rel, md5(abspath), label, part))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write("path,md5,label,partition\n")
        for rel, h, label, part in out:
            f.write("%s,%s,%s,%s\n" % (rel, h, label, part))

    n_train = sum(1 for r in out if r[3] == "train")
    n_report = sum(1 for r in out if r[3] == "report")
    n_hash = len({r[1] for r in out})
    print("rows %d (train %d, report %d), unique md5 %d, missing files %d"
          % (len(out), n_train, n_report, n_hash, missing))
    print("written:", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
