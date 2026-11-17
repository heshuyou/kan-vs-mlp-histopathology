# Parameter-aligned audit of KAN and MLP classification heads in histopathology

Code, data splits, result files and figure sources for the paper *"Do KAN Heads
Outperform MLP Heads in Histopathology? A Parameter-Aligned Audit"*. The paper is
under review; citation metadata is in `CITATION.cff`.

The study keeps the backbone frozen, extracts one feature vector per patch, and asks a
single question: once the classification head is matched on parameter budget, does a
Kolmogorov–Arnold (KAN) head still beat an ordinary MLP head? Everything in this
repository exists to answer that question reproducibly. There is no new architecture and
no new training recipe.

---

## 1. What is here, and what is not

**Included**

| Path                                 | Content                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| `KAN/*.py`                           | every script that produced a number in the paper (22 files)                        |
| `KAN/*.json`                         | result record per table, figure or robustness check (23 files)                     |
| `KAN/*_preds.npz`                    | per-seed test predictions for the analyses that need them                          |
| `train_clean.txt`, `val_clean.txt`   | three-way split of the de-duplicated LC25000 corpus                                |
| `train_crc.txt`, `val_crc.txt`       | three-way split of CRC-VAL-HE-7K                                                   |
| `train_group2.txt`, `val_group2.txt` | cluster-aware (grouped) variant of the LC25000 split                               |
| `train_list.txt`, `val_list.txt`     | the official LC25000 split, kept only to document the 420-image train/test overlap |
| `figures/`                           | the four figures of the paper (PDF + PNG) and the scripts that draw them              |
| `docs/RESULTS.md`                    | table → result file → seed-count map                                               |
| `CHECKSUMS.sha256`                   | SHA256 of every file in this repository                                               |
| `md5_lc25000.csv`                    | per-image MD5 inventory of the de-duplicated corpus (23,720 rows)                 |
| `make_md5_lc25000.py`                | rebuilds the MD5 inventory from the split lists and the images                    |
| `legacy/`                            | archived outputs of the superseded two-zone protocol                              |

**Deliberately excluded**

* The image datasets. LC25000 and CRC-VAL-HE-7K are public but large and carry their own
  licences; download them from the original sources (Borkowski et al. 2019 for LC25000,
  Kather et al. 2019 for CRC-VAL-HE-7K) and place them as described in §2.
* Pre-extracted feature caches (`lc25000_feats_resnet50.npz` 185 MB,
  `lc25000_feats_efficientnet_b0.npz` 116 MB, `colon_gen_feats.npz` 90 MB,
  `crc9_resnet50_feats.npz` 56 MB, `lc25000_group_cache.npz` 378 MB). They are
  reproducible from the frozen backbones in minutes to a few hours and would blow past
  GitHub's file limits.
* Intermediate `*_partial.json` checkpoints and training logs. Only the JSON files
  shipped here are results.

---

## 2. Environment and data layout

Python 3.10+, PyTorch 2.x (CUDA optional but 10–20× faster), numpy, scipy,
scikit-learn, matplotlib, Pillow.

```bash
pip install -r requirements.txt
```

All scripts resolve paths relative to their own location: `KAN/` is the script
directory and its parent is the repository root. One exception: `KAN/baselines.py`
defaults its `--data-dir` to `../`, so launch it from inside `KAN/` or pass
`--data-dir` explicitly. Put the datasets at the root:

```
<repo>/
├── jpg/                    # LC25000: five class folders, .jpeg images
├── CRC-VAL-HE-7K/          # nine class folders
├── train_clean.txt  ...    # the split lists shipped here
├── KAN/                    # code + results (this release)
└── figures/
```

Every split list is plain text, one sample per line, `relative/path label`, e.g.

```
jpg/lung_aca/lungaca1.jpeg 0
```

---

## 3. Protocol (what "three-way" means)

* LC25000 is first de-duplicated by exact MD5: 1,280 redundant images are dropped, each
  hash keeping one copy (`build_clean_split.py`, `data_audit.py`).
* The de-duplicated corpus is split 80/20 into **train** and **holdout**; the holdout is
  halved into a **selection** slice and a **report** slice:
  17,078 / 1,898 / 4,744 patches. CRC-VAL-HE-7K: 5,169 / 575 / 1,436.
* The epoch is chosen on the selection slice; the report slice is evaluated **once**.
  Earlier work chose the epoch on the same images it reported, which is the leak this
  protocol removes.
* Heads: linear probe, narrow MLP (H=64/32), Chebyshev KAN, wide MLP matched to the KAN
  parameter budget, B-spline KAN, spline-only control. Same builder, same seeds, same
  optimiser: AdamW, lr 1e-3, weight decay 1e-4, 15 epochs, batch 512 (256 for the
  robustness job). Seeds: fixed list starting 42, 123, 2024, 7, 99, and so on; the same list is used
  across heads within a table.
* All in-domain numbers use 20 seeds; the ablations and sweeps use 10; the error
  analysis uses 3.

---

## 4. Reproducing the numbers

Run everything from `KAN/`:

```bash
cd KAN

# 0. split, audit and inventory (writes the split lists, the duplicate report and the MD5 table)
python build_clean_split.py
python data_audit.py
cd .. && python make_md5_lc25000.py && cd KAN

# 1. Tables 1–2: in-domain three-way comparison
python run_reruns.py                       # drives rerun_protocol.py; ~20 seeds per head

# 2. Table 4 and the ablations / sweep / error analysis (one job at a time)
python threeway_extras.py --job nobase     # Table 4b
python threeway_extras.py --job b256       # Table 4 robustness
python threeway_extras.py --job sweep      # Fig. 3
python threeway_extras.py --job ablation   # Table 5a/5b
python threeway_extras.py --job l1         # Table 5c
python threeway_extras.py --job error      # Fig. 4

# 3. Table 4a/4c: B-spline head at batch 512 on ResNet50 / EfficientNet-B0
python bspline_kan_experiment.py --batch 512
python bspline_kan_b0.py

# 4. Table 3: cross-dataset transfer and balanced accuracy
python crc_colon_generalize.py             # writes colon_gen_result.json
python cross_bspline_20.py                 # writes cross_bspline_20_result.json
python cross_balanced.py                   # writes cross_balanced_result.json (+ preds)

# 5. Table 6: latency, five independent rounds
python timing_repeat.py                    # writes timing_repeat_result.json

# 6. Figures (outputs land in ../figures; the scripts run from any directory)
cd ../figures
python make_fig1_delta.py
python make_fig2_stability.py
python make_fig3_capacity.py
python make_fig4_confusion.py
```

Each runner rewrites `<job>_partial.json` after every seed, so an interrupted run can
be inspected without losing completed work. Features are cached to `*.npz` on first
use; delete a cache to force recomputation.

---

## 5. Where each paper number lives

`docs/RESULTS.md` maps every table and figure to its file and seed count. Two details
worth knowing before reading the JSON:

* The MLP width matched to each KAN parameter budget is the top-level `H` field inside
  the result file (468, 528, 620, 652 or 708). `H` is a hidden width, `mlp_params` beside
  it is the parameter count that follows from it, and neither is a batch size; the batch
  size is `meta.batch` (512 for the LC25000 tables, 256 for the CRC-9 table). The one file
  name that carries `b512` refers to that batch.
* In `cross_balanced_result.json`, `heads.<name>.recall` holds per-class recalls. The
  key `recall_norm_mean` stores the recall of the **first** class (TUM histopathology,
  1,233 patches); the NORM class (741 patches) is the one that collapses.

---

## 6. Known caveats

* **Timing.** Latency was measured on one laptop RTX 3050 (4 GB), batch 1, five rounds
  of 20 warm-up + 200 timed passes. The B-spline head shows round-to-round drift
  (5.22–6.43 ms, SD 0.55) that the other three heads do not, and the ratios are
  session-relative: a second session on the same machine gave 8.6× for the Chebyshev head
  and 15.9× for the B-spline head against the same wide MLP, so treat the multiplier as
  an order-of-magnitude estimate rather than a hardware-independent constant.
* **Cross-domain experiment.** The label mapping between the two datasets is a
  semantic assumption (tumour ↔ tumour, normal ↔ normal), and under it the working point
  collapses toward the majority class: balanced accuracy lands near 60% while accuracy
  stays near 70%. Both are reported; neither ranks the heads.
* **Multiplicity.** The scripts emit raw p-values; the family-wise correction is applied
  in the paper, which reports the uncorrected values and the corrected outcome for the
  family of eight matched-MLP comparisons.
* **Legacy scripts.** `efficiency_measure.py` and `error_analysis_clean.py` are retained
  for provenance; the numbers reported come from `timing_repeat.py` and
  `threeway_extras.py --job error`.

---

## 7. Licence and citation

Code: MIT (see `LICENSE`). Split lists and result files: CC BY 4.0. The image datasets
keep their own licences and are not redistributed here.

`CITATION.cff` is provided; GitHub will render a "Cite this repository" button from it.

---

## 8. Notes on provenance

Comments and documentation in this repository were drafted with AI assistance and then
reviewed and edited by the authors. Every number in the result files was produced by the
scripts in `KAN/` as shipped here; `docs/RESULTS.md` maps each table and figure to the
file it came from.
