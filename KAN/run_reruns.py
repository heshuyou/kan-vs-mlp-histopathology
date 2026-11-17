# -*- coding: utf-8 -*-
"""Sequential driver for the re-run experiments.

Run it with the conda environment that has torch:
    python run_reruns.py

Every step is resumable: interrupted runs continue from the seeds already saved in
the output json.  Steps whose result file is already complete finish in seconds.
Use --only N to run a single step, --list to show the plan, --dry-run to print the
commands without executing them.
"""
import os, sys, subprocess, json, argparse

KAN = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    dict(name="LC25000 grouped split (ResNet50, 20 seeds, 5 heads)",
         out="lc25000_group2_resnet50_n20.json",
         copy_from="lc25000_group2_resnet50_partA.json",
         args=["rerun_protocol.py", "--task", "lc5", "--backbone", "resnet50",
               "--mode", "group3", "--groupset", "group2", "--seeds", "20",
               "--seed_start", "0", "--resume",
               "--heads", "probe,narrow,cheb,wide,bspline",
               "--out", "lc25000_group2_resnet50_n20.json"],
         minutes=60),
    dict(name="EfficientNet-B0 three-way split (20 seeds)",
         out="lc25000_threeway_efficientnet_b0_n20.json", copy_from=None,
         args=["rerun_protocol.py", "--task", "lc5", "--backbone", "efficientnet_b0",
               "--mode", "threeway", "--seeds", "20", "--seed_start", "0", "--resume",
               "--heads", "probe,narrow,cheb,wide,bspline",
               "--out", "lc25000_threeway_efficientnet_b0_n20.json"],
         minutes=50),
    dict(name="EfficientNet-B0 grouped split (20 seeds)",
         out="lc25000_group2_efficientnet_b0_n20.json", copy_from=None,
         args=["rerun_protocol.py", "--task", "lc5", "--backbone", "efficientnet_b0",
               "--mode", "group3", "--groupset", "group2", "--seeds", "20",
               "--seed_start", "0", "--resume",
               "--heads", "probe,narrow,cheb,wide,bspline",
               "--out", "lc25000_group2_efficientnet_b0_n20.json"],
         minutes=50),
    dict(name="Sample-size sweep 100/10/1% (ResNet50, 5 seeds)",
         out="lc25000_learning_curve_resnet50.json", copy_from=None,
         args=["learning_curve.py", "--backbone", "resnet50", "--seeds", "5",
               "--out", "lc25000_learning_curve_resnet50.json"],
         minutes=12),
    dict(name="LC25000 three-way split (ResNet50, 20 seeds) - already done, verify only",
         out="lc25000_threeway_resnet50_n20.json", copy_from=None,
         args=["rerun_protocol.py", "--task", "lc5", "--backbone", "resnet50",
               "--mode", "threeway", "--seeds", "20", "--seed_start", "0", "--resume",
               "--heads", "probe,narrow,cheb,wide,bspline",
               "--out", "lc25000_threeway_resnet50_n20.json"],
         minutes=0),
    dict(name="CRC nine-class three-way split (20 seeds) - already done, verify only",
         out="crc9_threeway_n20.json", copy_from=None,
         args=["rerun_protocol.py", "--task", "crc9", "--mode", "threeway",
               "--seeds", "20", "--seed_start", "0", "--resume",
               "--heads", "probe,narrow,cheb,wide", "--out", "crc9_threeway_n20.json"],
         minutes=0),
    dict(name="Aggregate every result json into a markdown table",
         out=None, copy_from=None,
         args=["summarize_reruns.py"], minutes=0),
]


def done_seeds(path, heads):
    if not os.path.exists(path):
        return 0
    try:
        d = json.load(open(path, encoding="utf-8"))
        return len(d["res"][heads[0]]["acc"])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="run only step N (1-based)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, s in enumerate(STEPS, 1):
            n = done_seeds(os.path.join(KAN, s["out"]), ["probe"]) if s["out"] else "-"
            print("%d. %-62s seeds_done=%s  ~%d min" % (i, s["name"], n, s["minutes"]))
        return

    for i, s in enumerate(STEPS, 1):
        if args.only and i != args.only:
            continue
        print("\n=== [%d/%d] %s  (about %d min) ===" % (i, len(STEPS), s["name"], s["minutes"]),
              flush=True)
        if s["copy_from"] and not os.path.exists(os.path.join(KAN, s["out"])):
            src = os.path.join(KAN, s["copy_from"])
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, os.path.join(KAN, s["out"]))
                print("  seeded from %s" % s["copy_from"], flush=True)
        cmd = [PY] + s["args"]
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            continue
        r = subprocess.run(cmd, cwd=KAN)
        if r.returncode != 0:
            print("  step failed with exit code %d; rerun the same script to resume."
                  % r.returncode, flush=True)
    print("\nFinished. Results are in the *_n20.json files.", flush=True)


if __name__ == "__main__":
    main()
