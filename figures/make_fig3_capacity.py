"""Generate Fig. 3: capacity curve on LC25000/ResNet50 features (three-way protocol).

One panel, one message: the KAN heads do not add capacity. The MLP capacity
curve (10 widths, 0.14M-4.2M, 10 seeds, +/-1 SD band) is the reference; the
Chebyshev heads at degree 3/5/8 and the B-spline head are placed on the same
axes, so a reader can see directly whether a head lies on, below or above the
curve at its own parameter budget.

Every number is read from the three-way result files (v30); the two annotation
values -- the MLP budget that matches the Chebyshev level, and the gap at the
shared 1.64M budget -- are computed by interpolation rather than typed in.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths are resolved from this file's own location, so the script runs from any
# working directory and on any platform. Set KAN_DATA to point at another copy
# of the result JSONs if they live outside this checkout.
OUT = os.path.dirname(os.path.abspath(__file__))
KAN = os.environ.get("KAN_DATA") or os.path.join(os.path.dirname(OUT), "KAN")
WIDTH_ORDER = [64, 96, 128, 192, 256, 384, 528, 640, 768, 1024]

C_MLP = "#2E7D6F"
C_CHEB = "#C1553B"
C_BS = "#5B4B8A"


def load(name):
    with open(os.path.join(KAN, name), encoding="utf-8") as f:
        return json.load(f)


sweep = load("threeway_sweep_result.json")
abl = load("threeway_ablation_result.json")
main20 = load("lc25000_threeway_resnet50_n20.json")

xs, ys, sds = [], [], []
for H in WIDTH_ORDER:
    s = sweep["summary"]["H%d" % H]
    xs.append(sweep["params"]["H%d" % H])
    ys.append(s["mean"])
    sds.append(s["sd"])
xs, ys, sds = map(np.asarray, (xs, ys, sds))

deg_pts = []
for k, lab in (("degree_3", "3"), ("degree_5", "5"), ("degree_8", "8")):
    s = abl["summary"][k]
    deg_pts.append((abl["params"][k], s["mean"], s["sd"], lab))
deg_pts.sort(key=lambda t: t[0])

bs = main20["summary"]["bspline"]
bs_p = main20["meta"]["params"]["bspline"]

# ---- annotation 1: which MLP budget reproduces the Chebyshev level? -------
kan_mean = abl["summary"]["degree_5"]["mean"]
eq_H = None
for i in range(len(xs) - 1):
    if ys[i] <= kan_mean <= ys[i + 1]:
        f = (kan_mean - ys[i]) / (ys[i + 1] - ys[i])
        eq_H = WIDTH_ORDER[i] + f * (WIDTH_ORDER[i + 1] - WIDTH_ORDER[i])
        break
eq_params = 2 * eq_H ** 2 + (2048 + 5 + 2) * eq_H + 5
eq_label = "%.2fM" % (eq_params / 1e6)

# ---- annotation 2: the gap at the shared 1.64M budget ---------------------
kan_p = abl["params"]["degree_5"]
y_curve = float(np.interp(np.log10(kan_p), np.log10(xs), ys))
gap = kan_mean - y_curve

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.9, 3.2))

ax.fill_between(xs, ys - sds, ys + sds, color=C_MLP, alpha=0.16, lw=0, zorder=1)
ax.plot(xs, ys, "-o", color=C_MLP, ms=5.0, lw=1.4, mec="black", mew=0.4,
        zorder=3, label="MLP head (10 widths)")
ax.errorbar(xs, ys, yerr=sds, color=C_MLP, lw=0.8, capsize=2.2, capthick=0.8,
            fmt="none", zorder=2)

dx, dy, ds = zip(*[(t[0], t[1], t[2]) for t in deg_pts])
ax.plot(dx, dy, "--s", color=C_CHEB, ms=5.0, lw=1.1, mec="black", mew=0.4,
        zorder=3, label="Chebyshev KAN (degree 3, 5, 8)")
ax.errorbar(dx, dy, yerr=ds, color=C_CHEB, lw=0.8, capsize=2.2, capthick=0.8,
            fmt="none", zorder=2)

ax.errorbar([bs_p], [bs["mean"]], yerr=[bs["sd"]], fmt="D", color=C_BS, ms=6.4,
            mec="black", mew=0.5, capsize=2.2, zorder=4, label="B-spline KAN")

# --- reading aid: the level the Chebyshev heads never leave ----------------
ax.axhline(kan_mean, color=C_CHEB, lw=0.7, ls=":", zorder=1)
ax.text(1.95e5, kan_mean + 0.055,
        "Chebyshev level\n$\\approx$ %s MLP" % eq_label, fontsize=7,
        color=C_CHEB, va="bottom", ha="left")

# the shared 1.64M budget: the Chebyshev point sits on the curve, so only the
# level line is drawn (no gap arrow -- the difference is 0.01 pp)

ax.set_xscale("log")
ax.set_xlim(1.0e5, 5.2e6)
ax.set_ylim(98.30, 99.90)
ax.set_xticks([2e5, 5e5, 1e6, 2e6, 4e6])
ax.set_xticklabels(["0.2M", "0.5M", "1M", "2M", "4M"])
ax.set_xlabel("Parameters (log scale)", fontsize=8.5)
ax.set_ylabel("Accuracy on the LC25000 report partition (%)", fontsize=8.5)
ax.grid(True, which="major", alpha=0.22, lw=0.4)
ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", handletextpad=0.5, borderpad=0.3, markerscale=0.8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_capacity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_capacity.png"), dpi=300, bbox_inches="tight")
print("saved; chebyshev %.3f, equivalent MLP %s, gap at 1.64M %+.3f pp"
      % (kan_mean, eq_label, gap))
