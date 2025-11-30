"""Utility helpers for fitting and evaluating DecisionTree baselines."""

from sklearn import tree
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter
from models.common import Config
from utils.common import data_prep
from sklearn.tree import DecisionTreeClassifier

def train(train_df: DataFrame, cfg: Config, max_depth: int):
    """Fit a decision tree on the prepared dataframe for a given depth budget."""
    x_df, y_df = data_prep(train_df, cfg)

    clf: DecisionTreeClassifier = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=cfg.min_samples_leaf)
    clf = clf.fit(x_df.values, y_df.values)
    return clf


def eval(clf: DecisionTreeClassifier, valid_df: DataFrame, cfg: Config):
    """Evaluate a trained classifier and return accuracy as a float."""
    x_df, y_df = data_prep(valid_df, cfg)
    accuracy = 0

    pred = clf.predict(x_df.values)

    for i in range(len(pred)):
        accuracy += pred[i] == y_df.to_list()[i]

    accuracy /= len(pred)
    return accuracy
    