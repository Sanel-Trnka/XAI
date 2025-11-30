"""Simple configurable feed-forward network builder used across scripts."""

import torch.nn as nn
from torch.nn import Sequential
from models.common import Config

def load_model(input_size: int, cfg: Config) -> Sequential:
    """Create a sequential MLP whose hidden layers mirror `cfg.hidden_layers`."""
    modules = [
        nn.Linear(input_size, cfg.hidden_layers[0])
    ]

    _add_activation_function(modules, cfg)

    for idx_layer in range(1, len(cfg.hidden_layers)):
        modules.append(nn.Linear(cfg.hidden_layers[idx_layer-1], cfg.hidden_layers[idx_layer]))
        _add_activation_function(modules, cfg)

    modules.append(nn.Linear(cfg.hidden_layers[-1], 1))

    model = nn.Sequential(*modules)

    return model


def _add_activation_function(modules: list, cfg: Config):
    """Append the activation defined in the config to the provided module list."""
    if cfg.activation == 'ReLU':
        modules.append(nn.ReLU())
    if cfg.activation == 'LeakyReLU':
        modules.append(nn.LeakyReLU())
    if cfg.activation == 'Sigmoid':
        modules.append(nn.Sigmoid())
    if cfg.activation == 'Tanh':
        modules.append(nn.Tanh())