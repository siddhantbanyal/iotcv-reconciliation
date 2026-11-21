"""Test support: make `src/` importable and provide a tiny config loader.

Works under both `pytest tests/` and `python -m unittest discover tests`.
"""

import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def load_default_cfg() -> dict:
    cfg_path = os.path.join(_ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def small_cfg(seed: int = 0, ticks: int = 200) -> dict:
    """A faster, smaller world for unit tests."""
    cfg = load_default_cfg()
    cfg["seed"] = seed
    cfg["world"]["n_bins"] = 8
    cfg["world"]["n_zones"] = 4
    cfg["world"]["n_skus"] = 5
    cfg["world"]["ticks"] = ticks
    return cfg


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)
