"""Generate Fig. 4: confusion matrix of the Chebyshev KAN head (three-way protocol).

Reads threeway_error_result.json (3 seeds on the report partition) and draws the
matrix of the seed named in SEED_FOR_FIGURE, in the same style as the earlier
version of this figure. The other seeds are summarised in the text, not drawn.
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
LABELS = ["Colon ACA", "Colon benign", "Lung ACA", "Lung benign", "Lung SCC"]
SEED_FOR_FIGURE = 42

with open(os.path.join(KAN, "threeway_error_result.json"), encoding="utf-8") as f:
    d = json.load(f)

run = next(r for r in d["runs"] if r["seed"] == SEED_FOR_FIGURE)
cm = np.asarray(run["confusion"], dtype=int)
n = int(cm.sum())

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.6, 4.1))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm[i, j]
        ax.text(j, i, "%d" % v, ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 * cm.max() else "black")

ax.set_xticks(range(len(LABELS)))
ax.set_yticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=42, ha="right")
ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_confusion.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_confusion.png"), dpi=220, bbox_inches="tight")

err = n - int(cm.trace())
off = cm.copy()
np.fill_diagonal(off, 0)
print("seed %d: n=%d, errors=%d, top confusions=%s"
      % (SEED_FOR_FIGURE, n, err,
         sorted([(int(off[i, j]), LABELS[i], LABELS[j])
                 for i in range(len(LABELS)) for j in range(len(LABELS))
                 if off[i, j] > 0], reverse=True)[:4]))
