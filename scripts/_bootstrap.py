"""Shared bootstrap for CLI scripts: make src/ importable and load/merge configs."""

import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    """Load a config; if it isn't default.yaml, merge it on top of default.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    default_path = os.path.join(ROOT, "configs", "default.yaml")
    if os.path.abspath(path) != os.path.abspath(default_path):
        with open(default_path) as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    return cfg
