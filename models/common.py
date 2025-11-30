"""Shared configuration schema and mode enums for training/evaluation scripts."""

from typing import Literal
from pydantic import BaseModel, Field
from enum import Enum

class Config(BaseModel):
    """Strictly validates all tunable knobs for both training and evaluation."""
    dataset_csv: str                                        # Dataset file path
    testset_csv: str                                        # Testset file path
    target_column: str                                      # Ground truth column in Dataset
    learning_rate: float                                    #·Adjust learning rate
    momentum: float                                         # Momentum for SGD Optimizer
    training_split: float = Field(..., gt=0.0, le=1.0)      # Train/Validate set ratio
    optimizer: Literal['Adam', 'SGD']                       # Either use Adam or SGD optimizer
    hidden_layers: list[int]                                # Adjust these fully connected layers
    activation: Literal['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU'] # Either use ReLU, Sigmoid or tanh
    ignore_columns: list[str]                               # Columns in Dataset to ignore i.e. not relevant for training 
    batch_size: int                                         # Batch size of training
    epochs: int = Field(..., gt=0)                          # Epochs of training
    max_depth: int = Field(..., gt=0)                       # Max depth of Decision Tree
    min_samples_leaf: int = Field(..., gt=0)                # Min Samples per leaf of Decision Tree


class ExecMode(Enum):
    """Lightweight flag to distinguish training vs evaluation CLI flows."""
    TRAIN = 1
    EVAL = 2