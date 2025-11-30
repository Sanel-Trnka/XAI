"""Utility helpers shared by training/evaluation/analysis scripts."""

from models.common import Config, ExecMode
import sys
import inspect
import os
import json
from typing import Any, Mapping
import argparse
from pandas import DataFrame, Series
import pandas as pd
from data.WBDCDataset import WBDCDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, NotFittedError
from torch.optim import Adam, SGD
from torch.nn import Sequential
import torch
import datetime
import joblib
from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn

def get_config(path: str) -> Config:
    """Load a configuration file from disk (py/json/yaml/toml) into `Config`."""
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore

    try:
        import tomllib  # Python 3.11+
    except Exception:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            tomllib = None  # type: ignore

    def _resolve(p: str) -> str:
        candidates = [p]
        main_dir = get_base_path()
        if main_dir:
            candidates.append(os.path.join(main_dir, p))
            candidates.append(os.path.join(main_dir, os.path.basename(p)))
        candidates.append(os.path.join(os.getcwd(), p))
        for c in candidates:
            if os.path.isfile(c):
                return c
        return p  # fall back; will raise if not found

    def _load(fp: str) -> Mapping[str, Any]:
        ext = os.path.splitext(fp)[1].lower()
        with open(fp, "rb") as f:
            # Explicit loaders by extension
            if ext in (".json",):
                return json.load(f)
            if ext in (".yml", ".yaml"):
                if not yaml:
                    raise RuntimeError("PyYAML not installed")
                return yaml.safe_load(f) or {}
            if ext in (".toml",):
                if not tomllib:
                    raise RuntimeError("tomllib/tomli not available")
                return tomllib.load(f)
            if ext in (".py",):
                # Execute a simple Python config file of top-level assignments
                src = f.read().decode("utf-8")
                ns: dict[str, Any] = {}
                exec(compile(src, fp, "exec"), {}, ns)
                return {k: v for k, v in ns.items() if not k.startswith("_") and not callable(v)}
            # Try JSON -> YAML -> TOML
            data = None
            f.seek(0)
            try:
                data = json.load(f)
                return data
            except Exception:
                pass
            f.seek(0)
            if yaml:
                try:
                    data = yaml.safe_load(f) or {}
                    return data
                except Exception:
                    pass
            f.seek(0)
            if tomllib:
                try:
                    data = tomllib.load(f)
                    return data
                except Exception:
                    pass
            # Finally, as a last resort, try interpreting as a Python config
            f.seek(0)
            try:
                src = f.read().decode("utf-8")
                ns: dict[str, Any] = {}
                exec(compile(src, fp, "exec"), {}, ns)
                return {k: v for k, v in ns.items() if not k.startswith("_") and not callable(v)}
            except Exception:
                pass
        raise ValueError(f"Unsupported or unreadable config: {fp}")

    def _build(cfg_dict: Mapping[str, Any]) -> Config:
        if hasattr(Config, "model_validate"):  # pydantic v2
            return Config.model_validate(cfg_dict)  # type: ignore[attr-defined]
        if hasattr(Config, "parse_obj"):  # pydantic v1
            return Config.parse_obj(cfg_dict)  # type: ignore[attr-defined]
        return Config(**cfg_dict)  # dataclass/attrs/vanilla

    cfg_path = _resolve(path)
    cfg_dict = _load(cfg_path)
    return _build(cfg_dict)

def get_base_path():
    """Gets the absolute dir path of the script that was executed."""
    
    # Check if '__main__' module has a '__file__' attribute
    # This check is necessary for environments like interactive interpreters
    if not hasattr(sys.modules['__main__'], '__file__'):
        # Return None or raise an exception, e.g., when run in REPL
        return None
        
    # Get the file path from the __main__ module
    main_script_file = inspect.getfile(sys.modules['__main__'])
    
    # Return the absolute path
    return os.path.dirname(os.path.abspath(main_script_file))

def get_args(mode: ExecMode) -> argparse.Namespace:
    """Parse CLI arguments, toggling required flags based on execution mode."""
    parser = argparse.ArgumentParser(description="{} script".format("Training" if mode == ExecMode.TRAIN else "Evaluation"))
    
    if mode == ExecMode.TRAIN:
        parser.add_argument(
            "--save-model",
            action="store_true",
            default=False,
            help="Save model weights and scaler"
        )

    elif mode == ExecMode.EVAL:
        parser.add_argument(
            "--model",
            type=str,
            help="Path to the model directory"
        )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/wdbc.py",
        help="Path to the config file (default: configs/wdbc.py)"
    )

    args = parser.parse_args()

    # Validate that required arguments exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    return args


def load_data(csv_path: str) -> DataFrame:
    """Read a CSV relative to the project root into a pandas DataFrame."""
    df = pd.read_csv(os.path.join(get_base_path(), csv_path))
    return df


def load_train_valid_dataset(base_df: DataFrame, cfg: Config, scaler: StandardScaler | None = None) -> tuple[WBDCDataset, WBDCDataset, DataFrame, DataFrame]:
    """Split the base dataframe, optionally scale features, and wrap them as datasets."""
    x: DataFrame = base_df.copy()
    y: Series = x.pop(cfg.target_column)
    x_train, x_valid, y_train, y_test = train_test_split(
        x,
        y,
        train_size=cfg.training_split,
        random_state=124,
        shuffle=True,
    )

    ignore_cols_train = [x_train.pop(ignore_column) for ignore_column in cfg.ignore_columns]
    ignore_cols_valid = [x_valid.pop(ignore_column) for ignore_column in cfg.ignore_columns]

    if scaler is None:
        # Do not scale if no scaler provided
        scaled_x_train = x_train.copy()
        scaled_x_valid = x_valid.copy()
    else:
        try:
            check_is_fitted(scaler)
            scaled_x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
        except NotFittedError:
            scaled_x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)

        scaled_x_valid = pd.DataFrame(scaler.transform(x_valid), columns=x_valid.columns, index=x_valid.index)

    # re-add ignored columns
    for idx, ignore_column in enumerate(cfg.ignore_columns):
        scaled_x_train[ignore_column] = ignore_cols_train[idx].values
        scaled_x_valid[ignore_column] = ignore_cols_valid[idx].values

    # re-add target column
    scaled_x_train[cfg.target_column] = y_train.values
    scaled_x_valid[cfg.target_column] = y_test.values

    train_ds = WBDCDataset(scaled_x_train, cfg)
    valid_ds = WBDCDataset(scaled_x_valid, cfg)

    print("X_train: {}".format(scaled_x_train))
    print("X_test: {}".format(scaled_x_valid))

    return train_ds, valid_ds, scaled_x_train, scaled_x_valid

def load_test_dataset(test_df: DataFrame, cfg: Config, scaler: StandardScaler|None = None) -> tuple[DataFrame, Series]:
    """Apply the shared preprocessing pipeline to the hold-out test dataframe."""
    x_test, y_test = data_prep(test_df, cfg)
    if scaler is not None:
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    return x_test, y_test


def get_optimizer(model: Sequential, cfg: Config):
    """Instantiate the optimizer defined in the config for the supplied model."""
    optimizer = None
    if cfg.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    return optimizer


def generate_run_dir_path() -> str:
    """Create a timestamped run directory under results/ for logging and artifacts."""
    return os.path.join(get_base_path(), "results/{}".format(datetime.datetime.now()))

def get_run_dir_path(model_path: str) -> str:
    """Resolve a possibly relative model folder path into an absolute path."""
    return os.path.join(get_base_path(), model_path)


def save_model(model: Sequential, optimizer: Adam | SGD, scaler: StandardScaler, clf: DecisionTreeClassifier, dir: str):
    """Persist weights/optimizer along with the scaler and decision tree via joblib."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer_type": "Adam" if isinstance(optimizer, Adam) else "SGD",
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(dir, "model.pth"))
    joblib.dump(scaler, os.path.join(dir, "scaler.pkl"))
    joblib.dump(clf, os.path.join(dir, "clf.pkl"))


def load_model(path: str, cfg: Config) -> tuple[Sequential, Adam | SGD, StandardScaler, DecisionTreeClassifier]:
    """Restore the MLP/optimizer/scaler/decision-tree tuple from disk."""
    model_fp = os.path.join(path, "model.pth")
    scaler_fp = os.path.join(path, "scaler.pkl")
    clf_fp = os.path.join(path, "clf.pkl")

    if not os.path.isfile(model_fp):
        raise FileNotFoundError(f"Model checkpoint not found: {model_fp}")
    if not os.path.isfile(scaler_fp):
        raise FileNotFoundError(f"Scaler file not found: {scaler_fp}")
    if not os.path.isfile(clf_fp):
        raise FileNotFoundError(f"Decision Tree file not found: {clf_fp}")

    checkpoint = torch.load(model_fp, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError("Invalid checkpoint format: expected keys 'model' and 'optimizer'")

    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid model state_dict format")

    # Helper to extract numeric index from parameter key (handles simple "0.weight" or more complex prefixes)
    def _extract_index_from_key(key: str) -> int:
        parts = key.split('.')
        for p in parts:
            if p.isdigit():
                return int(p)
        # fallback: try first part as int
        try:
            return int(parts[0])
        except Exception as e:
            raise ValueError(f"Cannot extract module index from state_dict key: {key}") from e

    # Collect numeric indices present in the state_dict
    indices = set()
    weight_map = {}
    for k, v in state_dict.items():
        try:
            idx = _extract_index_from_key(k)
        except ValueError:
            continue
        indices.add(idx)
        # track weight tensors specifically for linear inference
        if k.endswith(".weight"):
            weight_map[idx] = v

    if not indices:
        raise ValueError("No module indices found in state_dict; cannot reconstruct model")

    max_idx = max(indices)

    # Reconstruct sequential modules: if index has a weight -> Linear inferred from weight shape, otherwise default to ReLU
    modules = []
    for i in range(0, max_idx + 1):
        if i in weight_map:
            wt = weight_map[i]
            if not hasattr(wt, "shape") or len(wt.shape) != 2:
                # unexpected param shape; try to fall back to direct load with empty sequential
                raise ValueError(f"Unsupported weight tensor shape for inferred Linear at index {i}: {getattr(wt, 'shape', None)}")
            out_features, in_features = wt.shape[0], wt.shape[1]
            modules.append(nn.Linear(in_features, out_features))
        else:
            # No parameters at this index -> likely an activation / non-param layer. Default to ReLU.
            modules.append(nn.ReLU())

    # Build Sequential and load state_dict (keys like "0.weight" will match)
    model = nn.Sequential(*modules)
    # Load with strict=True so mismatches surface as errors
    model.load_state_dict(state_dict)
    model.eval()

    optimizer = None
    if checkpoint.get("optimizer_type") == "Adam":
        optimizer = Adam(model.parameters(), cfg.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif checkpoint.get("optimizer_type") == "SGD":
        optimizer = SGD(model.parameters(), cfg.learning_rate, cfg.momentum)
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        # If optimizer_type missing or unknown, try to infer; default to Adam
        optimizer = Adam(model.parameters(), cfg.learning_rate)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            # ignore if cannot load optimizer state
            pass

    scaler = joblib.load(scaler_fp)
    clf = joblib.load(clf_fp) 

    return model, optimizer, scaler, clf
    

def data_prep(df: DataFrame, cfg: Config) -> tuple[DataFrame, Series]:
    """Drop ignored columns, encode labels, and return feature/target frames."""
    x_df = df.copy()
    for column in cfg.ignore_columns:
        if column in df.columns:
            x_df.pop(column) 
    y_df = x_df.pop(cfg.target_column)
    # Replace string labels and ensure integer dtype
    y_df = y_df.replace({'M': 1, 'B': 0})
    y_numeric = pd.to_numeric(y_df, errors='coerce')
    if y_numeric.isnull().any():
        raise ValueError(f"Non-numeric targets after replacement: {y_df[y_numeric.isnull()].unique()}")
    y_df = y_numeric.astype(int)
    return x_df, y_df


def get_x_labels(df: DataFrame, cfg: Config):
    """Return ordered feature names excluding ignored columns and the target."""
    ignore = set(cfg.ignore_columns or [])
    ignore.add(cfg.target_column)
    labels = [col for col in df.columns if col not in ignore]
    return labels