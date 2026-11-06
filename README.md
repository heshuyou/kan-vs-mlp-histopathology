# kan-vs-mlp-histopathology

# Code and Data Splits for the KAN Classification-Head Study

This repository contains the source code and the cleaned data splits used in the paper
*"Attributing the Apparent Performance Gain of Kolmogorov-Arnold Classification Heads
in Histopathological Image Classification."*

## Contents

```
code_repo/
├── README.md              # this file
├── requirements.txt       # Python dependencies
├── code/                  # all experiment and figure-generation scripts
│   ├── baselines.py             # shared KAN (Chebyshev) layer/block definitions
│   ├── build_clean_split.py     # MD5 deduplication + leakage-free stratified split
│   ├── data_audit.py            # dataset audit (duplicates / leakage detection)
│   ├── clean_lc25000_heads.py   # main LC25000 experiment (KAN vs MLP heads)
│   ├── crc_9class_heads.py      # nine-class CRC-VAL-HE-7K experiment
│   ├── crc_colon_generalize.py  # cross-dataset generalization (colon -> CRC)
│   ├── param_sweep.py           # MLP parameter sweep
│   ├── kan_ablation_clean.py    # degree / depth ablation
│   ├── bspline_kan_experiment.py# B-spline KAN (LC25000)
│   ├── bspline_kan_extra.py     # B-spline KAN (CRC + cross-domain)
│   ├── kan_regularization.py    # L1-regularization ablation
│   ├── efficiency_measure.py    # FLOPs and inference latency
│   ├── error_analysis_clean.py  # confusion matrix / error analysis
│   ├── verify_params.py         # per-layer parameter-count check
│   ├── compute_macro_f1.py      # macro-F1 for the 708-wide MLP
│   ├── gen_figures5.py          # generates the paper figures (PDF + PNG)
│   └── gen_graphical_abstract.py# graphical abstract
└── (data splits, at this root level)
    ├── train_clean.txt / val_clean.txt   # cleaned LC25000 split (18,976 / 4,744)
    ├── train_crc.txt  / val_crc.txt      # CRC-VAL-HE-7K split (5,744 / 1,436)
    └── train_list.txt / val_list.txt     # original LC25000 split (for reference)
```

## Dependencies

Python 3.10, PyTorch (with CUDA recommended), torchvision, numpy, scipy, matplotlib,
Pillow. See `requirements.txt`.

## Data

The two datasets are publicly available and are **not** included in this repository:

- **LC25000** (lung & colon histopathology, 768x768, 5 classes, 25,000 images):
  https://github.com/tampapath/lung_colon_image_set (or
  https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).
  Place the `*.jpeg` images in a `jpg/` folder at the repository root.
- **CRC-VAL-HE-7K** (colorectal tissue, 224x224, 9 classes, 7,180 images):
  https://zenodo.org/record/1214456. Place the 9 class subfolders (ADI, BACK, DEB, LYM,
  MUC, MUS, NORM, STR, TUM) in a `CRC-VAL-HE-7K/` folder at the repository root.

The scripts assume the following layout (matching the original experiment setup):

```
<repo root>/
├── code/                 # this repository's scripts
├── jpg/                  # LC25000 images (download)
├── CRC-VAL-HE-7K/        # CRC-VAL-HE-7K images (download)
├── train_clean.txt ...   # the provided data splits
```

## How to reproduce

All scripts are run from the `code/` directory with the project's Python environment.

```bash
cd code

# 1. (Optional) audit the raw data
python data_audit.py

# 2. Build the cleaned, leakage-free splits
python build_clean_split.py

# 3. Main experiments (features are cached to .npz on first run)
python clean_lc25000_heads.py --backbone resnet50 --seeds 20
python clean_lc25000_heads.py --backbone efficientnet_b0 --seeds 20
python crc_9class_heads.py --seeds 20
python crc_colon_generalize.py --seeds 20

# 4. Ablations, sweep, B-spline, regularization
python param_sweep.py --seeds 10
python kan_ablation_clean.py --seeds 10
python bspline_kan_experiment.py --seeds 10
python bspline_kan_extra.py --seeds 20
python kan_regularization.py --seeds 10

# 5. Efficiency and error analysis
python efficiency_measure.py
python error_analysis_clean.py
python verify_params.py
python compute_macro_f1.py

# 6. Figures
python gen_figures5.py
python gen_graphical_abstract.py
```

## Notes

- The frozen-backbone features are cached to `*.npz` files in the `code/` directory, so
  head-only experiments (steps 3-6) are fast once the cache exists.
- **Hardware**: the experiments were run on an NVIDIA RTX 3050 Laptop GPU (4 GB). Set the
  machine to "never sleep" before long runs; most scripts save intermediate results so
  that interrupted runs can be resumed by simply re-running the same command.
- The Chebyshev degree-8 KAN head uses a reduced batch size (128) to fit the 4 GB GPU
  (see `kan_ablation_clean.py`).
