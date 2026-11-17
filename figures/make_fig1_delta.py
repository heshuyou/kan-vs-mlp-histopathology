"""Generate Fig. 1 (gain/loss of KAN heads against two MLP baselines).

Values are the margins recorded in the three-way result files listed in docs/RESULTS.md,
i.e. under the
three-way protocol (10% of the training pool held out for epoch selection,
report partition evaluated once, 20 seeds). Re-run this script whenever the
tables change.

Layout (v2, density reduced): horizontal grouped bars, one row per setting,
two series (vs the same-depth narrow MLP and vs the parameter-matched wide
MLP). Only the matched-MLP contrast is annotated with its value and with the
paired t-test outcome, because that contrast is the claim of the figure; bars
for which p >= 0.05 are drawn hollow and marked n.s., so that the figure
cannot be read as "the KAN wins in four of the eight pairs". The Chebyshev and
B-spline families are separated by a gap and labelled once, instead of by a
vertical rule plus two italic headers.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# (basis, setting, d_vs_narrow, d_vs_matched, p_vs_matched or None if untested)
GROUPS = [
    ("Chebyshev", "LC25000, ResNet50",        0.749, -0.094, 0.042),
    ("Chebyshev", "LC25000, EfficientNet-B0", 0.209, -0.085, 6.9e-4),
    ("Chebyshev", "CRC, 9-class",             1.319, -0.446, 3.0e-5),
    ("Chebyshev", "Cross-domain, colon",     -0.740, -0.720, 0.31),
    ("B-spline",  "LC25000, ResNet50",        1.179,  0.357, 5.6e-10),
    ("B-spline",  "LC25000, EfficientNet-B0", 0.285, -0.065, 2.0e-3),
    ("B-spline",  "CRC, 9-class",             1.922,  0.097, 0.055),
    ("B-spline",  "Cross-domain, colon",     -0.346, -0.279, 0.64),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

C_NARROW = "#c9cfd6"
C_MATCH = "#33536b"

# y positions: Chebyshev block on top, B-spline block below, one empty row between
ypos = [7.0, 6.0, 5.0, 4.0, 2.6, 1.6, 0.6, -0.4]
h = 0.34

fig, ax = plt.subplots(figsize=(6.2, 2.95))

narrow = [g[2] for g in GROUPS]
match = [g[3] for g in GROUPS]
pval = [g[4] for g in GROUPS]

ax.barh(np.array(ypos) + h / 2, narrow, h, label="vs narrow MLP (same depth)",
        color=C_NARROW, edgecolor="#5a6672", linewidth=0.6)
for i, v in enumerate(match):
    ns = pval[i] is not None and pval[i] >= 0.05
    ax.barh(ypos[i] - h / 2, v, h,
            color="white" if ns else C_MATCH,
            edgecolor="#1d3444", linewidth=0.8 if ns else 0.6,
            hatch="////" if ns else None,
            label="vs matched wide MLP" if i == 0 else None)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta$ accuracy (percentage points)", fontsize=7.5)
ax.set_yticks(ypos)
ax.set_yticklabels([g[0] + " \u2014 " + g[1] for g in GROUPS], fontsize=7)
ax.set_xlim(-1.45, 2.55)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax.set_ylim(-1.0, 7.6)
ax.tick_params(axis="y", length=0, pad=2)
ax.tick_params(axis="x", labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# annotate only the matched-MLP contrast (the claim of the figure)
for i, v in enumerate(match):
    ns = pval[i] is not None and pval[i] >= 0.05
    txt = f"{v:+.2f}" + (" n.s." if ns else "")
    if v >= 0:
        ax.text(v + 0.05, ypos[i] - h / 2, txt, ha="left", va="center", fontsize=6.6)
    else:
        ax.text(v - 0.05, ypos[i] - h / 2, txt, ha="right", va="center", fontsize=6.6)

# one separating rule between the two basis families, labelled once each
ax.axhline(3.3, color="#9aa3ab", linewidth=0.6, linestyle=(0, (4, 3)))
ax.text(2.5, 3.42, "Chebyshev", ha="right", va="bottom", fontsize=7, style="italic")
ax.text(2.5, 3.18, "B-spline", ha="right", va="top", fontsize=7, style="italic")

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_NARROW, edgecolor="#5a6672", linewidth=0.6),
           plt.Rectangle((0, 0), 1, 1, facecolor=C_MATCH, edgecolor="#1d3444"),
           plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#1d3444", hatch="////", linewidth=0.8)]
labels = ["vs narrow MLP", "vs matched MLP ($p<0.05$)", "vs matched MLP (n.s.)"]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, -0.19),
          ncol=3, frameon=False, fontsize=6.8, handlelength=1.1, columnspacing=1.1)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_delta.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_delta.png"), dpi=300, bbox_inches="tight")

with open(os.path.join(OUT, "fig1_delta_data.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["basis", "setting", "delta_vs_narrow_MLP_pp", "delta_vs_matched_MLP_pp", "p_vs_matched_MLP"])
    for g in GROUPS:
        wr.writerow([g[0], g[1].replace("\n", " "), f"{g[2]:.3f}", f"{g[3]:.3f}", g[4]])

print("ok")
