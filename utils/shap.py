"""Convenience wrapper around SHAP for consistent logging & preprocessing."""

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from pandas import DataFrame, Series
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter

from models.common import Config
from utils.common import data_prep


class ShapExplainer:
    """Computes SHAP values for the MLP and emits TensorBoard-ready figures."""

    def __init__(self, model: Sequential, writer: SummaryWriter, x_df: DataFrame, labels: list[str], cfg: Config):
        """Prepare background data and SHAP explainer from the provided dataset."""
        x, y = data_prep(x_df, cfg)
        print("Training data shape: {}".format(x.values.shape))
        
        self.model = model
        self.writer = writer
        self.cfg = cfg  # store config for consistent preprocessing
        self.x = x
        self.y = y
        self.class_names = ["Benign", "Malignant"]
        self.random_state = getattr(cfg, "random_seed", 42)
        self.log_dir = Path(writer.log_dir) if getattr(writer, "log_dir", None) else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.explanations: list[dict[str, Any]] = []
        
        # Determine feature names robustly
        if labels and len(labels) == x.shape[1]:
            self.feature_names = labels
        else:
            if labels and len(labels) != x.shape[1]:
                print(f"Warning: Provided labels length {len(labels)} != feature dimension {x.shape[1]}. Using data_prep-derived column names.")
            self.feature_names = list(x.columns)
        
        # Background data for KernelExplainer
        background_size = min(100, len(x))
        background_data = shap.sample(x.values, background_size)
        
        self.explainer = shap.KernelExplainer(
            model=self._predict_proba,
            data=background_data,
            link="identity"
        )
        
    def explain(self, sample_size: int | None = None, nsamples: int | str = "auto") -> dict[str, Any]:
        """Compute SHAP values for a (possibly sampled) subset of the training data."""
        if sample_size is not None and sample_size <= 0:
            raise ValueError("sample_size must be positive when provided")

        if sample_size is not None and sample_size < len(self.x):
            sampled_x = self.x.sample(n=sample_size, random_state=self.random_state)
        else:
            sampled_x = self.x

        sampled_y = self.y.loc[sampled_x.index]

        print(
            "Generating SHAP values for {} samples with {} features".format(
                len(sampled_x), sampled_x.shape[1]
            )
        )

        shap_values = self.explainer.shap_values(sampled_x.values, nsamples=nsamples)
        explanation = {
            "features": sampled_x,
            "targets": sampled_y,
            "shap_values": shap_values,
            "nsamples": nsamples,
        }
        self.explanations.append(explanation)
        return explanation

    def summarize_explanations(self, sample_size: int | None = None, nsamples: int | str = "auto") -> None:
        """Create SHAP summary plots for benign, malignant and combined cohorts."""
        needs_new_explanation = sample_size is not None or nsamples != "auto" or not self.explanations
        explanation = (
            self.explain(sample_size=sample_size, nsamples=nsamples)
            if needs_new_explanation
            else self.explanations[-1]
        )

        shap_values_seq = self._ensure_sequence(explanation["shap_values"])
        features = explanation["features"]
        targets = explanation["targets"]

        benign_mask = targets.values == 0
        malignant_mask = targets.values == 1

        plots: list[tuple[str, np.ndarray, DataFrame]] = []

        # Benign summary
        benign_values = (
            shap_values_seq[0][benign_mask]
            if len(shap_values_seq) > 0 else np.empty((0, features.shape[1]))
        )
        plots.append(("Benign", benign_values, features[benign_mask]))

        # Malignant summary
        malignant_values = (
            shap_values_seq[1][malignant_mask]
            if len(shap_values_seq) > 1 else np.empty((0, features.shape[1]))
        )
        plots.append(("Malignant", malignant_values, features[malignant_mask]))

        # Combined summary uses class-conditional SHAP values
        combined_values = self._combine_by_true_class(shap_values_seq, targets.values)
        plots.append(("Both", combined_values, features))

        for class_name, values, feats in plots:
            fig = self._build_summary_figure(values, feats, class_name)
            tag = f"SHAP/summary_{class_name.lower()}"
            step = len(self.explanations)
            if self.writer:
                self.writer.add_figure(tag, fig, global_step=step)
            self._save_figure(fig, f"shap_summary_{class_name.lower()}.png")
            plt.close(fig)

    def _predict_proba(self, x):
        """Model wrapper that returns probabilities compatible with SHAP."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.model(x_tensor)

            # Handle logits for binary classification (single output neuron)
            logits = outputs.squeeze()

            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            probs_pos = torch.sigmoid(logits).cpu().numpy()
            probs_pos = np.atleast_1d(probs_pos).astype(float)

            probs_neg = 1.0 - probs_pos
            probs = np.vstack([probs_neg, probs_pos]).T
        return probs

    def _ensure_sequence(self, values: Any) -> Sequence[np.ndarray]:
        """Normalize different SHAP output shapes into a list of ndarrays."""
        if isinstance(values, (list, tuple)):
            return [np.asarray(v) for v in values]

        arr = np.asarray(values)
        if arr.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            return [arr[:, :, idx] for idx in range(arr.shape[2])]
        if arr.ndim == 2:
            return [arr]
        raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")

    def _combine_by_true_class(self, shap_values: Sequence[np.ndarray], targets: np.ndarray) -> np.ndarray:
        """Pick class-specific SHAP values based on the ground-truth labels."""
        if not shap_values:
            return np.empty((0, len(self.feature_names)))

        combined = np.zeros_like(shap_values[0])
        for class_idx in range(min(len(shap_values), len(self.class_names))):
            mask = targets == class_idx
            if mask.any():
                combined[mask] = shap_values[class_idx][mask]
        return combined

    def _build_summary_figure(self, shap_values: np.ndarray, features: DataFrame, class_name: str):
        """Create a SHAP summary plot (or placeholder) for a given cohort."""
        if shap_values.size == 0 or len(features) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"No samples available for {class_name}",
                    ha="center", va="center", fontsize=12)
            ax.axis("off")
            return fig

        max_display = min(20, shap_values.shape[1])
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=self.feature_names,
            show=False,
            max_display=max_display,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP Summary - {class_name} Samples", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Persist matplotlib figures to disk next to the TensorBoard logdir."""
        if self.log_dir is None:
            return
        output_path = self.log_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=200)