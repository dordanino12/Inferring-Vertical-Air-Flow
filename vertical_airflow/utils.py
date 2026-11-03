"""Shared helpers: configs, model construction, checkpoints, k-fold layout."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import yaml

from .models import PretrainedTemporalUNet, PretrainedTemporalUNetMitB1

BACKBONES = {"mit_b1": PretrainedTemporalUNetMitB1, "resnet18": PretrainedTemporalUNet}


def load_config(path: str) -> dict:
    """Load a YAML config; ``base: other.yaml`` entries are merged recursively."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        cfg = _merge(parent, cfg)
    return cfg


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(backbone: str, in_channels: int, use_conv_lstm: bool = True, freeze_encoder: bool = True):
    """Build a model exactly as configured for training (refiner disabled)."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}', expected one of {list(BACKBONES)}")
    return BACKBONES[backbone](
        out_channels=1,
        lstm_layers=1 if use_conv_lstm else 0,
        freeze_encoder=freeze_encoder,
        in_channels=in_channels,
        use_conv_lstm=use_conv_lstm,
        use_refiner=False,
    )


def load_model_from_checkpoint(path: str, device="cpu"):
    """Rebuild a model from a released checkpoint (architecture is read from the file)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    state = ckpt["model_state"]
    use_conv_lstm = cfg.get("use_conv_lstm", any("lstm" in k for k in state))
    model = build_model(cfg["type"], cfg["in_channels"], use_conv_lstm)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), ckpt


def fold_dirs(kfold_root: str):
    """Sorted fold_* sub-directories of a k-fold dataset root."""
    return sorted(d for d in os.listdir(kfold_root)
                  if d.startswith("fold_") and os.path.isdir(os.path.join(kfold_root, d)))


def find_fold_checkpoint(models_dir: str, fold_name: str):
    for name in sorted(os.listdir(models_dir)):
        if name.endswith(f"_{fold_name}_best_bin_loss.pt") or name == f"{fold_name}.pt":
            return os.path.join(models_dir, name)
    return None
