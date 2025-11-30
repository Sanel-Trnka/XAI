# Project Setup & Usage Guide

This guide walks you through preparing the environment, training and evaluating the models, and monitoring runs with TensorBoard.

## Prerequisites

- Python 3.10 or newer (project tested on Python 3.12)
- `pip` and a virtual environment tool such as `venv`

## 1. Environment Setup

Run the following commands from the project root to create an isolated environment and install dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Data Layout

- Training data: `dataset/data.csv`
- Held-out test set: `dataset/test.csv`

Both CSV files ship with the repository. If you replace them, keep the same column structure (the target column must remain `diagnosis`).

## 3. Configuration Files

Training and evaluation always load a config file (default `configs/wdbc.py`). Configs are plain Python modules with top-level assignments. Key fields:

| Field | Description |
| --- | --- |
| `dataset_csv`, `testset_csv` | Relative paths to the training and test CSV files |
| `target_column` | Column containing the labels (`diagnosis`) |
| `ignore_columns` | Columns removed before modeling (e.g., `id`) |
| `training_split` | Train/validation ratio used when creating loaders |
| `hidden_layers`, `activation` | MLP topology definition |
| `optimizer`, `learning_rate`, `momentum` | Optimizer choice and hyperparameters |
| `batch_size`, `epochs` | Training loop parameters |
| `max_depth`, `min_samples_leaf` | Decision-tree sweep bounds for the interpretable baseline |

Two presets are provided:

- `configs/wdbc.py`: full 30-feature baseline (batch size 32, 24 epochs).
- `configs/wdbc_important_features.py`: reduced six-feature variant (batch size 16, 60 epochs, tighter tree regularization).

Create additional configs by copying one of these files and editing the fields.

## 4. Training

Basic usage:

```bash
python train.py --config=configs/wdbc.py [--save-model]
```

Arguments:

- `--config`: Path to a config file. Defaults to `configs/wdbc.py` if omitted. Always run commands from the repo root so relative paths resolve correctly.
- `--save-model`: If provided, the script saves two checkpoints under the current run directory (`results/<timestamp>/`):
	- `last_model/`: final-epoch MLP, optimizer, scaler, and the final decision tree.
	- `best_model/`: best validation-accuracy MLP and the best-performing decision tree encountered during the depth sweep.

Outputs produced during training:

- Console metrics for each epoch (loss, accuracy) for both training and validation splits.
- Decision-tree validation accuracy per tested depth and a plotted tree figure logged to TensorBoard.
- TensorBoard summaries written to `results/<timestamp>/` (see Section 6).

## 5. Evaluation

Evaluation loads a saved run (usually from `results/<timestamp>/best_model` or `last_model`). Supply the directory containing `model.pth`, `scaler.pkl`, and `clf.pkl`.

```bash
python eval.py --config=configs/wdbc.py --model "results/2025-11-10 16:50:47.582907/best_model"
```

Notes:

- The `--config` argument must match the config that produced the checkpoint (or another config with identical architecture and preprocessing choices).
- The `--model` argument is required. Wrap the path in quotes if it contains spaces (timestamps include spaces by default).
- The evaluator prints confusion matrices and accuracies for both the MLP and decision tree, then generates LIME/SHAP explanations and logs them to a fresh TensorBoard run directory under `results/`.

## 6. TensorBoard Monitoring

All training and evaluation runs emit TensorBoard summaries beneath `results/`. Launch TensorBoard from the project root and point it at the entire folder to browse any run:

```bash
tensorboard --logdir results --port 6006
```

Open http://localhost:6006 in a browser to inspect loss/accuracy curves, decision-tree figures, LIME explanations, and SHAP plots. Use the dropdown in TensorBoard to switch between timestamps.

With the environment configured and commands above, you can reproduce results, explore new hyperparameters, and visualize model explanations end-to-end.
