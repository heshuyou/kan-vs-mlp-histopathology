# RESULTS: mapping of paper numbers to files and seeds

Paper: *Do KAN Heads Outperform MLP Heads in Histopathology? A Parameter-Aligned
Audit* 

Every computed number in the paper comes from one of the scripts in `KAN/`. Table 7 is
a literature summary and has no result file of its own. The `--job`
runs of `threeway_extras.py` write `<job>_result.json` plus a `*_partial.json` that is rewritten after every seed, so an interrupted run can be inspected without losing work;
only the JSON files shipped in this repository are results; the partial checkpoints are not.

**Protocol (all in-domain numbers).** Stratified 80/20 re-split of the de-duplicated
LC25000 corpus, with the holdout halved into a selection and a report slice:
17,078 / 1,898 / 4,744 patches. CRC-VAL-HE-7K: 5,169 / 575 / 1,436. The epoch is chosen
on the selection slice and the report slice is evaluated once. Seeds are fixed lists,
identical across heads within a table.

| Paper item                       | Content                                                                      | File                                                                                                                     | Seeds |
| -------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----- |
| Table 1                          | LC25000 five-class, probe / narrow / Chebyshev / wide (H=528), ResNet50      | `lc25000_threeway_resnet50_n20.json`                                                                                     | 20    |
| Table 1                          | same, EfficientNet-B0 (wide H=468)                                           | `lc25000_threeway_efficientnet_b0_n20.json`                                                                              | 20    |
| Table 2                          | CRC nine-class, probe / narrow / Chebyshev / wide (H=528)                    | `crc9_threeway_n20.json`                                                                                                 | 20    |
| Table 2                          | CRC nine-class, B-spline / wide (H=708)                                      | `crc9_threeway_bspline_n20.json`                                                                                         | 20    |
| Table 3                          | cross-dataset, Chebyshev block and narrow MLP (accuracy)                     | `colon_gen_result.json`                                                                                                  | 20    |
| Table 3                          | cross-dataset, B-spline block, incl. F test (accuracy)                       | `cross_bspline_20_result.json`                                                                                           | 20    |
| Table 3                          | cross-dataset, balanced accuracy and per-class recall for all five heads     | `cross_balanced_result.json` (+ `cross_balanced_preds.npz`)                                                              | 20    |
| Table 4 (a)                      | ResNet50, B-spline vs wide MLP H=708 at batch 512                            | `bspline_kanresnet50_b512_result.json`                                                                                   | 20    |
| Table 4 (b)                      | spline-only control (no linear base weight) vs MLP H=652                     | `threeway_nobase_result.json`                                                                                            | 20    |
| Table 4 (c)                      | EfficientNet-B0, B-spline vs wide MLP H=620                                  | `lc25000_threeway_efficientnet_b0_n20.json`                                                                              | 20    |
| Table 4 robustness               | B-spline at batch 256                                                        | `threeway_b256_result.json`                                                                                              | 10    |
| Table 5 (a)(b)                   | spline order / depth ablations                                               | `threeway_ablation_result.json`                                                                                          | 10    |
| Table 5 (c)                      | L1 strength                                                                  | `threeway_l1_result.json`                                                                                                | 10    |
| Table 6                          | parameter count, analytic FLOPs, single-image latency (five rounds)          | `timing_repeat_result.json`; FLOPs are computed analytically in `threeway_extras.py`                                     | —     |
| Table 7                          | literature accuracy on LC25000                                               | no file: figures taken from the cited papers                                                                             | —     |
| Fig. 1                           | delta-to-matched-MLP bar chart                                               | `../figures/fig1_delta_data.csv` (written by `make_fig1_delta.py`)                                                       | —     |
| Fig. 2                           | seed-to-seed stability                                                       | computed in `make_fig2_stability.py` from the per-seed records inside the three-way result files                         | 20    |
| Fig. 3                           | parameter sweep, ten MLP widths plus the KAN points                          | `threeway_sweep_result.json`                                                                                             | 10    |
| Fig. 4                           | confusion matrix and error structure                                         | `threeway_error_result.json`                                                                                             | 3     |
| §5.5 grouped variant             | cluster-aware split (1,537 clusters)                                         | `lc25000_group2_resnet50_n20.json`, `lc25000_group2_efficientnet_b0_n20.json`, `lc25000_group2_stats.json`               | 20    |
| §5.5 earlier (two-zone) protocol | epoch selected on the report partition, superseded by the three-way protocol | `legacy/clean_lc25000_resnet50.json`; the two-zone numbers quoted in §5.5 come from this run | 20    |
| §4.1 probe saturation            | linear-probe ceilings, both backbones                                        | `linearprobe_lc25000_resnet50_study_acc_n20.json`, `linearprobe_lc25000_efficientnet_b0_study_acc_n20.json`              | 20    |
| §4.2 selection/leak check        | probe and head behaviour when the epoch sees the report slice                | `selection_split_result.json`                                                                                            | 10    |

## Notes

- `H468`/`H528`/`H620`/`H652`/`H708` denote the **MLP width matched to the corresponding
  KAN parameter budget**. They are the top-level `H` field inside the result file, not part
  of the file name, and `mlp_params` beside them is the resulting parameter count. The
  batch size is `meta.batch`.
- Parameter counts live in `meta.params` (older files) or in the top-level
  `bspline_params` / `mlp_params` fields (B-spline files).
- `cross_balanced_result.json` stores per-class recalls under `heads.<name>.recall`.
  The prefixed key `recall_norm_mean` holds the recall of the **first** class
  (TUM histopathology, 1,233 patches); the collapsing class is NORM (741 patches).
- Head builders and the shared training routine are defined once, in
  `clean_lc25000_heads.py`, `bspline_kan_experiment.py` and `bspline_kan_extra.py`;
  `rerun_protocol.py` holds `train_head`, `stratified_indices` and `gpu_free_check`,
  which `threeway_extras.py` imports unchanged. This is why heads within a table share
  seeds, optimiser and epoch budget by construction.
- **Two-zone files.** `legacy/` holds the three outputs of the earlier two-zone protocol,
  in which the report partition itself served for epoch selection:
  `bspline_kan_b0_result.json`, `mlp_at_nobase_budget_result.json` and
  `clean_lc25000_resnet50.json`. No table or figure in the paper is drawn from the first
  two; the third supplies the two-zone numbers quoted in section 5.5. Every number in
  Tables 1-5 comes from the three-way files listed above. The generators,
  `bspline_kan_b0.py` and `mlp_at_nobase_budget.py`, stay in `KAN/` and write there when
  re-run.
- **MD5 inventory.** `md5_lc25000.csv` at the repository root holds one row per image of the
  cleaned corpus (`path,md5,label,partition`, 23,720 rows, 23,720 distinct hashes) and is
  written by `make_md5_lc25000.py` at the repository root, which rebuilds it byte for byte
  from the split lists and the image files.
- The de-duplicated split, the MD5 inventory and the seed lists are produced by
  `build_clean_split.py`; `data_audit.py` reproduces the duplicate audit, including the
  420 images that the official split shares between train and test.
