"""End-to-end training script for the WDBC experiments.

It handles two phases:
1. Train a black-box MLP classifier used later for LIME/SHAP analyses.
2. Fit a white-box decision tree to support human-interpretable inspection.

Both models, plus their preprocessing artifacts, can optionally be saved.
"""

from utils.common import get_args, get_config, load_data, load_train_valid_dataset, get_optimizer, save_model, generate_run_dir_path, get_x_labels
from models.common import Config, ExecMode
from argparse import Namespace
from pandas import DataFrame
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn import Sequential
import torch.nn as nn
from model.MLP import load_model
import utils.mlp_training as MLPTraining
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
import utils.dt_training as DTTraining
from sklearn import tree
import matplotlib.pyplot as plt
import os
import copy

pandas.set_option("future.no_silent_downcasting", True)

run_dir_path = generate_run_dir_path()

writer = SummaryWriter(log_dir=run_dir_path)

args: Namespace = get_args(ExecMode.TRAIN)
cfg: Config = get_config(args.config)

print("Loaded Configuration: {}".format(cfg))

base_df: DataFrame = load_data(cfg.dataset_csv)
print(base_df.head())

# Fit a single scaler on the training split so we can reuse it for validation/testing
scaler = StandardScaler()

train_ds, valid_ds, _, _ = load_train_valid_dataset(base_df, cfg, scaler)

train_loader: DataLoader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
valid_loader: DataLoader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=True)

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

print("Lenth training data: {}, length validation data: {}".format(train_N, valid_N))

# Training of an MLP for Black Box analysis

INPUT_SIZE = train_ds[0][0].shape.numel()

model: Sequential = load_model(INPUT_SIZE, cfg)

loss_function: torch.nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

optimizer: Adam | SGD = get_optimizer(model, cfg)

writer.add_text(tag='mlp', text_string='This is a simple MLP with these configs: {}'.format(cfg))

# Standard supervised training loop with TensorBoard tracking per epoch
for epoch in range(cfg.epochs):
    print('Epoch: {}'.format(epoch))
    train_loss, train_accuracy = MLPTraining.train(model, train_loader, train_N, optimizer, loss_function)
    valid_loss, valid_accuracy = MLPTraining.validate(model, valid_loader, valid_N, loss_function)
    writer.add_scalar("MLP/Loss/train", train_loss, epoch)
    writer.add_scalar("MLP/Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("MLP/Loss/valid", valid_loss, epoch)
    writer.add_scalar("MLP/Accuracy/valid", valid_accuracy, epoch)

    # Track best model by validation accuracy (store state, but DO NOT save yet).
    # We'll save the best MLP together with the best decision tree (clf) later.
    if epoch == 0:
        best_valid = valid_accuracy
        best_epoch = 0
        # store initial state as best until a better one is found
        # helper to move tensors to cpu for safe storage
        def _cpuify(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            if isinstance(obj, dict):
                return {k: _cpuify(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cpuify(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_cpuify(v) for v in obj)
            return obj

        best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        best_optimizer_state = _cpuify(optimizer.state_dict())
        best_optimizer_type = "Adam" if isinstance(optimizer, Adam) else "SGD"
        best_scaler = copy.deepcopy(scaler)
    else:
        if valid_accuracy > best_valid:
            best_valid = valid_accuracy
            best_epoch = epoch
            print(f"New best validation accuracy {best_valid} at epoch {best_epoch}")
            # capture the best model/optimizer/scaler states in CPU memory for later saving
            best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            # cpuify optimizer state (nested structure)
            def _cpuify(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu()
                if isinstance(obj, dict):
                    return {k: _cpuify(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_cpuify(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_cpuify(v) for v in obj)
                return obj

            best_optimizer_state = _cpuify(optimizer.state_dict())
            best_optimizer_type = "Adam" if isinstance(optimizer, Adam) else "SGD"
            best_scaler = copy.deepcopy(scaler)

writer.flush()

# Training a Decision Tree for White Box Analysis

# Reloading datasets 
_, _, train_df, valid_df = load_train_valid_dataset(base_df, cfg)

# Track (max_depth_config, realized_depth, accuracy) so we can log the best tree
best_accuracy = (0,0,0)
fig = plt.figure(figsize=(24, 16))

labels = get_x_labels(train_df, cfg)

for max_depth in range(1, cfg.max_depth+1):
    clf = DTTraining.train(train_df, cfg, max_depth)
    accuracy = DTTraining.eval(clf, valid_df, cfg)
    print("Accuray: {}".format(accuracy))
    if accuracy > best_accuracy[2]:
        depth = clf.get_depth()
        best_accuracy = (max_depth, depth, accuracy)
        tree.plot_tree(clf, filled=True, feature_names=labels)
    writer.add_scalar("DecisionTree/valid/accuracy", accuracy, max_depth)

print("Best accuracy of {} found with allowed maximum depth of {} and actual depth of {}".format(best_accuracy[2], best_accuracy[0], best_accuracy[1]))

plt.tight_layout()
writer.add_figure("DecisionTree/tree-plot", fig)
plt.close(fig)

writer.flush()
writer.close()

if args.save_model:
    # Save the final (last) model and associated artifacts into a dedicated folder
    last_dir = os.path.join(run_dir_path, "last_model")
    os.makedirs(last_dir, exist_ok=True)
    # clf (decision tree) should be defined after DT training above; pass it so it's saved too
    save_model(model, optimizer, scaler, clf, last_dir)
    # Also save the best MLP model (by validation accuracy) together with the best decision tree found.
    try:
        best_dir = os.path.join(run_dir_path, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        # Reconstruct model and optimizer objects, load stored best states and save using save_model()
        # Build a fresh model with the same input size
        best_model = load_model(INPUT_SIZE, cfg)
        best_model.load_state_dict(best_model_state_dict)
        # Create optimizer for this model and load stored optimizer state if available
        best_optimizer = get_optimizer(best_model, cfg)
        try:
            best_optimizer.load_state_dict(best_optimizer_state)
        except Exception:
            # If optimizer load fails, ignore — saving model weights and scaler is still useful
            pass
        # Use stored scaler and the best clf found during DT training
        save_model(best_model, best_optimizer, best_scaler, clf, best_dir)
    except NameError:
        # If for some reason best states were never set, skip best save
        print("Best model state not available; skipping saving best_model folder.")