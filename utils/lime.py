"""Wrapper around lime_tabular explainer with TensorBoard logging helpers."""

from lime.lime_tabular import LimeTabularExplainer
from pandas import DataFrame, Series 
from torch.nn import Sequential
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.common import data_prep
from models.common import Config
import matplotlib.pyplot as plt


class LimeExplainer:
    """Convenience facade for running LIME explanations and aggregations."""

    explanations: list = []

    def __init__(self, model: Sequential, writer: SummaryWriter, x_df: DataFrame, feature_names: list[str], cfg: Config):
        """Instantiate the tabular explainer with the preprocessed training data."""
        x, y = data_prep(x_df, cfg)
        print("Training data shape: {}".format(x.values.shape))
        
        # Initialize LIME explainer with training data
        self.explainer = LimeTabularExplainer(
            training_data=x.values,
            feature_names=feature_names,
            class_names=['Benign', 'Malignant'],
            mode='classification',
            discretize_continuous=True
        )
        
        self.model = model
        self.writer = writer
        self.feature_names = feature_names

    def explain_instance(self, row: Series, id: int, num_features=10):
        """Generate a LIME explanation for a single row and log the plots."""
        x_np = row.values.reshape(1, -1)
        probs = self._predict_proba(x_np)
        predicted_class = int(np.argmax(probs[0]))
        print("Instance {}: Probability: {} | Predicted Class: {}".format(
            id, probs[0], 'Malignant' if predicted_class == 1 else 'Benign'))
        
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            data_row=row.values,
            predict_fn=self._predict_proba,
            num_features=num_features,
            top_labels=2
        )

        self.explanations.append({
            'explanation': explanation,
            'predicted_class': predicted_class,
            'probs': probs[0],
            'num_features': num_features
        })
        
        # Create matplotlib figure for the explanation
        fig = self._create_explanation_figure(explanation, predicted_class, probs[0], num_features)
        
        # Log to TensorBoard
        self.writer.add_figure(
            'LIME/instance_{}'.format(id),
            fig,
            global_step=id
        )
        
        # Close the figure to free memory
        plt.close(fig)
        
        print("LIME explanation for instance {} logged to TensorBoard".format(id))
        
        return explanation
    

    def summarize_explanations(self):
        """Aggregate stored explanations into summary plots for quick inspection."""
        if not self.explanations:
            print("No explanations to summarize")
            return
        
        print("Summarizing {} explanations...".format(len(self.explanations)))
        
        # Collect feature weights for each feature across all explanations
        feature_weights = {feature: [] for feature in self.feature_names}
        feature_frequency = {feature: 0 for feature in self.feature_names}
        
        for exp_data in self.explanations:
            explanation = exp_data['explanation']
            predicted_class = exp_data['predicted_class']
            
            # Get explanation for the predicted class
            exp_list = explanation.as_list(label=predicted_class)
            
            # Process each feature in the explanation
            for feature_desc, weight in exp_list:
                # Extract the actual feature name from the description
                # LIME returns feature descriptions like "feature_name <= 0.5"
                feature_name = self._extract_feature_name(feature_desc)
                
                if feature_name in feature_weights:
                    feature_weights[feature_name].append(weight)
                    feature_frequency[feature_name] += 1
        
        # Create summary visualizations
        self._create_summary_figures(feature_weights, feature_frequency)
    
    def _extract_feature_name(self, feature_desc):
        """Extract the base feature name from LIME's feature description."""
        # LIME descriptions are like "feature_name <= 0.5" or "0.3 < feature_name <= 0.5"
        # We need to extract just the feature name
        for feature in self.feature_names:
            if feature in feature_desc:
                return feature
        # If no exact match, return the description as-is
        return feature_desc.split()[0] if ' ' in feature_desc else feature_desc
    
    def _create_summary_figures(self, feature_weights, feature_frequency):
        """Create and log summary figures to TensorBoard."""
        
        # Figure 1: Boxplot of feature importance distribution
        fig1 = self._create_boxplot_figure(feature_weights)
        self.writer.add_figure('LIME_Summary/feature_importance_distribution', fig1)
        plt.close(fig1)
        
        # Figure 2: Average feature importance
        fig2 = self._create_average_importance_figure(feature_weights)
        self.writer.add_figure('LIME_Summary/average_feature_importance', fig2)
        plt.close(fig2)
        
        # Figure 3: Feature frequency in explanations
        fig3 = self._create_frequency_figure(feature_frequency)
        self.writer.add_figure('LIME_Summary/feature_frequency', fig3)
        plt.close(fig3)
    
    def _create_boxplot_figure(self, feature_weights):
        """Create boxplot showing distribution of feature importances."""
        # Filter out features that never appeared
        active_features = {k: v for k, v in feature_weights.items() if len(v) > 0}
        
        if not active_features:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No feature weights collected', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by median absolute importance
        sorted_features = sorted(active_features.items(), 
                                key=lambda x: np.median(np.abs(x[1])), 
                                reverse=True)
        
        # Take top 20 features for readability
        top_n = min(20, len(sorted_features))
        sorted_features = sorted_features[:top_n]
        
        feature_names = [f[0] for f in sorted_features]
        weights_data = [f[1] for f in sorted_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))
        
        # Create horizontal boxplot
        bp = ax.boxplot(weights_data, vert=False, patch_artist=True,
                       labels=feature_names,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        ax.set_xlabel('Feature Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Feature Importances Across All Explanations\n(Top {} Features)'.format(top_n), 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_average_importance_figure(self, feature_weights):
        """Create bar chart of average feature importance."""
        # Calculate mean absolute importance for each feature
        avg_importance = {}
        for feature, weights in feature_weights.items():
            if len(weights) > 0:
                avg_importance[feature] = np.mean(np.abs(weights))
        
        if not avg_importance:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No feature weights collected', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 features
        top_n = min(20, len(sorted_features))
        sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, importances, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Average Absolute Importance', fontsize=12, fontweight='bold')
        ax.set_title('Average Feature Importance Across All Explanations\n(Top {} Features)'.format(top_n), 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(importances):
            ax.text(v + 0.01 * max(importances), i, '{:.3f}'.format(v), 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def _create_frequency_figure(self, feature_frequency):
        """Create bar chart showing how often each feature appears in explanations."""
        # Filter out features with zero frequency
        active_freq = {k: v for k, v in feature_frequency.items() if v > 0}
        
        if not active_freq:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No features appeared in explanations', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by frequency
        sorted_features = sorted(active_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 features
        top_n = min(20, len(sorted_features))
        sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        frequencies = [f[1] for f in sorted_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
        
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(features)))
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, frequencies, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Frequency (Number of Explanations)', fontsize=12, fontweight='bold')
        ax.set_title('Feature Frequency in Top Explanations\n(Top {} Features)'.format(top_n), 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(frequencies):
            ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
        
        plt.tight_layout()
        return fig


    def _create_explanation_figure(self, explanation, predicted_class, probs, num_features):
        """Build matplotlib figures combining feature weights and prediction probs."""
        # Get the explanation for the predicted class
        exp_list = explanation.as_list(label=predicted_class)
        
        # Sort by absolute weight (most important features first)
        exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
        exp_list = exp_list[:num_features]
        
        # Extract feature names and weights
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Feature importance bar chart
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, weights, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=9)
        ax1.set_xlabel('Feature Weight', fontsize=11)
        ax1.set_title('LIME Feature Importance\n(Predicted: {})'.format(
            'Malignant' if predicted_class == 1 else 'Benign'), fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right plot: Prediction probabilities
        class_names = ['Benign', 'Malignant']
        bar_colors = ['skyblue', 'coral']
        
        ax2.bar(class_names, probs, color=bar_colors, alpha=0.7)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add probability values on top of bars
        for i, (prob, class_name) in enumerate(zip(probs, class_names)):
            ax2.text(i, prob + 0.02, '{:.3f}'.format(prob), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        return fig

    def _predict_proba(self, x):
        """Helper passed to LIME that returns class probabilities for samples."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.model(x_tensor)

            # Handle outputs that may be shape (N, 1) or (N,)
            logits = outputs.squeeze()

            # Ensure logits is a torch tensor of shape (N,)
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            probs_pos = torch.sigmoid(logits).cpu().numpy()

            # Ensure 1D array
            probs_pos = np.atleast_1d(probs_pos).astype(float)

            # Build two-column probability array: [P(class0), P(class1)]
            probs_neg = 1.0 - probs_pos
            probs = np.vstack([probs_neg, probs_pos]).T
            return probs
    

