"""Generate Fig. 2: accuracy gain versus seed-to-seed spread (mean-stability plane).

Each KAN head is plotted at x = its accuracy minus that of the MLP matched to
its parameter budget (pp) and y = its standard deviation across seeds (pp, log
scale). The matched MLPs themselves are the grey markers on the x = 0 line;
the dotted connector runs from a KAN head to its own matched MLP, so the
vertical climb is the variance the head costs.

Data are read from the result files, not typed in, so the figure cannot drift
away from Tables 1-3.
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


def load(name):
    with open(os.path.join(KAN, name), encoding="utf-8") as f:
        return json.load(f)


def ms(arr):
    a = np.asarray(arr, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def cheb(p, key="kan", mkey="mlp_matched"):
    return p[key + "_mean"], p[key + "_std"], p[mkey + "_mean"], p[mkey + "_std"]


def ms_res(d, head):
    """Mean and SD across seeds, read straight from a re-run result json."""
    a = np.asarray(d["res"][head]["acc"], dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


lc_r50 = load("lc25000_threeway_resnet50_n20.json")
lc_b0 = load("lc25000_threeway_efficientnet_b0_n20.json")
crc_c = load("crc9_threeway_n20.json")
crc_b = load("crc9_threeway_bspline_n20.json")
crossg = load("colon_gen_result.json")
bs_cross = load("cross_bspline_20_result.json")

# label, Chebyshev (mean, sd, mlp_mean, mlp_sd), B-spline (mean, sd, mlp_mean, mlp_sd)
# The three in-domain settings come from the three-way re-runs (Tables 1-3 of the
# paper); the cross-domain row is unchanged because it was not re-run.
SETTINGS = [
    ("LC25000\nResNet50",
     (*ms_res(lc_r50, "cheb"), *ms_res(lc_r50, "wide")),
     (*ms_res(lc_r50, "bspline"), *ms_res(lc_r50, "wide_bs"))),
    ("LC25000\nEfficientNet-B0",
     (*ms_res(lc_b0, "cheb"), *ms_res(lc_b0, "wide")),
     (*ms_res(lc_b0, "bspline"), *ms_res(lc_b0, "wide_bs"))),
    ("CRC\n9-class",
     (*ms_res(crc_c, "cheb"), *ms_res(crc_c, "wide")),
     (*ms_res(crc_b, "bspline"), *ms_res(crc_b, "wide_bs"))),
    ("Cross-domain", (*ms(crossg["acc"]["kan"]), *ms(crossg["acc"]["mlp_matched"])),
     (*ms(bs_cross["bspline"]), *ms(bs_cross["mlp"]))),
]

C_CHEB = "#C1553B"
C_BS = "#5B4B8A"
C_MLP = "#9AA3AC"
C_LINK = "#C9CFD6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.1, 3.15))
ax.axvline(0, color="#666666", lw=0.8, ls="--", zorder=1)

for label, c, b in SETTINGS:
    for (mean, sd, mmean, msd), color, marker in (
            ((c[0], c[1], c[2], c[3]), C_CHEB, "o"),
            ((b[0], b[1], b[2], b[3]), C_BS, "D")):
        dx = mean - mmean
        ax.plot([0, dx], [msd, sd], color=C_LINK, lw=0.8, ls=":", zorder=2)
        ax.plot(dx, sd, marker=marker, ms=5.2, mfc=color, mec="black", mew=0.5, zorder=4)
        ax.plot(0, msd, marker="s", ms=3.6, mfc=C_MLP, mec="black", mew=0.4, zorder=3)

# hand-placed labels: only the two reading aids the figure needs
ax.annotate("cross-domain", xy=(-0.717, 3.03), xytext=(-0.97, 3.55),
            fontsize=7, color="#444444", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
ax.text(0.03, 1.15, "matched MLP\n($\\Delta = 0$)", fontsize=7, color="#555555",
        ha="left", va="center")

ax.set_yscale("log")
ax.set_ylim(0.010, 4.2)
ax.set_xlim(-1.02, 0.58)
ax.set_xlabel("Accuracy difference vs matched MLP (pp)", fontsize=9.2)
ax.set_ylabel("Standard deviation across seeds (pp, log)", fontsize=9.2)
ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3])
ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
ax.grid(True, which="major", alpha=0.25, lw=0.4)

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", color="none", mfc=C_CHEB, mec="black", ms=5.2, label="Chebyshev KAN"),
    Line2D([], [], marker="D", color="none", mfc=C_BS, mec="black", ms=5.2, label="B-spline KAN"),
    Line2D([], [], marker="s", color="none", mfc=C_MLP, mec="black", ms=3.6,
           label="matched MLP (its own)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True,
          framealpha=0.95, edgecolor="#CCCCCC", handletextpad=0.4, borderpad=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_stability.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_stability.png"), dpi=300, bbox_inches="tight")
print("ok")
