# Quantile IV with Multivariable Outcomes

This project implements a Quantile Instrumental Variables (Quantile IV) method to estimate causal effects when dealing with multivariable outcomes. It builds upon a deep learning framework using PyTorch Lightning and includes simulation, training, and evaluation pipelines.

## 🔧 Environment Setup

To get started, create and activate a dedicated conda environment:

```bash
conda create -n mlmr310 python=3.10
conda activate mlmr310
```

## 🧱 Installation

Navigate to the project root directory and install the package in editable mode:

```bash
cd ~/ml-mr/ml-mr-main
pip install -e .
```

Install required Python dependencies:

```bash
pip install numpy pandas torch scikit-learn pytorch-lightning statsmodels dill jupyter matplotlib seaborn wandb
```

---

## 📁 Repository Components

### `simulate_mv_data_for_quantile_iv.py`

This script generates simulated data for one exposure variable (`T1`) and multiple outcomes (`Y1`, `Y2`, `Y3`). It uses SNPs as instruments and includes a confounding variable `U`. The generated files include:
- `X.csv` (instruments)
- `T.csv` (exposure)
- `Y.csv` (outcomes)
- `Z.csv` (confounders)
- `merged_mv.csv` (all variables together)

### `quantile_iv.py`

This is the core implementation of the Quantile IV estimator. It includes:
- Exposure model (predicting quantiles of the exposure)
- Outcome model (predicting expected outcomes using quantiles)
- Full training pipeline for both models
- CLI support via `argparse`
- Support for multivariable outcomes through `--outcome-dim`
- Functions to save estimated causal effects and plots

The main function `fit_quantile_iv()` performs model training and saves results in a specified output directory.

### `data.py`

Defines dataset classes used for training and inference:
- `IVDatasetWithGenotypes` loads the input data (X, T, Y, covariates)
- Handles argument parsing for file paths and variable names

This file is crucial for preprocessing and batching data for the quantile IV training pipeline.

### `MREstimator_ate.py`

This script loads a trained Quantile IV model and:
- Computes Average Treatment Effects (ATE) using the `.ate()` method
- Compares estimated ATE with the true slopes used in simulation
- Optionally plots the estimated `do(Y)` curves side-by-side for each outcome using data saved in `causal_estimates.csv`

---

## 🚀 Example Workflow

1. **Generate simulated data:**
```bash
python simulate_mv_data_for_quantile_iv.py
```

2. **Train Quantile IV estimator:**
```bash
python -m ml_mr.estimation.quantile_iv \
  --data-dir simulated_data_mv \
  --exposure T1 \
  --outcomes Y1 Y2 Y3 \
  --instruments snp_0 snp_1 snp_2 snp_3 snp_4 snp_5 snp_6 snp_7 snp_8 snp_9 \
  --covariables U \
  --n-quantiles 5 \
  --output-dir quantile_iv_output_mv \
  --outcome-dim 3 \
  --exposure-max-epochs 1000 \
  --outcome-max-epochs 1000
```

3. **Run evaluation and plotting:**
```bash
python MREstimator_ate.py
```

---

## 📈 Outputs

- `quantile_iv_output_mv/causal_estimates.csv`: Estimated `do(Y)` values for a range of `X`
- `quantile_iv_output_mv/*.png`: Plots of estimated causal curves
- `meta.json`: Metadata and training configuration
