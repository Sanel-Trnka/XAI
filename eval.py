"""Evaluation & explainability entry point for trained WDBC models.

Loads a saved run (MLP + decision tree + scaler), reports accuracy/CM,
and produces LIME/SHAP summaries that get written to TensorBoard.
"""

from utils.common import get_args, load_model, get_run_dir_path, get_config, load_data, load_test_dataset, load_train_valid_dataset, get_x_labels, generate_run_dir_path
from utils.lime import LimeExplainer
from utils.shap import ShapExplainer
from argparse import Namespace
from models.common import ExecMode, Config
from pandas import DataFrame
import pandas
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
import torch

pandas.set_option("future.no_silent_downcasting", True)

run_dir_path: str = generate_run_dir_path()

args: Namespace = get_args(ExecMode.EVAL)
cfg: Config = get_config(args.config)

writer = SummaryWriter(log_dir=run_dir_path)

# Restore artifacts from disk (model weights, optimizer, scaler, decision tree)
model, optimizer, scaler, clf = load_model(get_run_dir_path(args.model), cfg)

# Load train and test data (train split is reused to fit explainers)
base_df: DataFrame = load_data(cfg.dataset_csv)
_, _, train_df, _ = load_train_valid_dataset(base_df, cfg, scaler)
test_df: DataFrame = load_data(cfg.testset_csv)
x_test, y_test = load_test_dataset(test_df, cfg, scaler)

labels = get_x_labels(test_df, cfg)

print("Labels: {}".format(labels))

print(test_df.head())

print("Test_df on row 0: {}".format(test_df.iloc[0]))
print("Test_df label on row 0: {}".format(y_test[0]))

# Confusion matrix for MLP (neural network)
with torch.no_grad():
    # model expects tensor inputs; convert DataFrame to torch.Tensor
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    logits = model(x_test_tensor).view(-1)
    probs = torch.sigmoid(logits)
    y_pred_mlp = (probs >= 0.5).long().numpy()

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("MLP Confusion Matrix (rows=true, cols=pred):")
print(cm_mlp)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")

# Confusion matrix for DecisionTree classifier (clf)
x_test_no_scaler, _ = load_test_dataset(test_df, cfg, None)
y_pred_clf = clf.predict(x_test_no_scaler.values)
cm_clf = confusion_matrix(y_test, y_pred_clf)
print("Decision Tree Confusion Matrix (rows=true, cols=pred):")
print(cm_clf)
clf_accuracy = accuracy_score(y_test, y_pred_clf)
print(f"Decision Tree Test Accuracy: {clf_accuracy:.4f}")

# LIME explanations on every test sample and SHAP on train samples
explainer_lime: LimeExplainer = LimeExplainer(model, writer, train_df, labels, cfg)
explainer_shap: ShapExplainer = ShapExplainer(model, writer, train_df, labels, cfg)
for idx in range(len(x_test.values)):
    # Explain each test instance individually so we can inspect cases later
    explainer_lime.explain_instance(x_test.iloc[idx], idx)
explainer_lime.summarize_explanations()
explainer_shap.summarize_explanations()

writer.flush()
writer.close()