"""K-fold ensemble loading and inference, shared by the figure scripts."""
from __future__ import annotations

import os

import numpy as np
import torch

from .dataset import NPZSequenceDataset
from .utils import find_fold_checkpoint, fold_dirs, load_config, load_model_from_checkpoint


class Ensemble:
    """The K fold models of one experiment, each with its training-set normalisation.

    ``folds`` restricts the ensemble to some fold names (e.g. ["fold_01_val_r0-1_c0-2"]).
    """

    def __init__(self, config_path, checkpoints, data_root="data", device=None, folds=None, verbose=True):
        self.cfg = load_config(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = os.path.join(data_root, self.cfg["data"]["kfold_dir"])
        self.one_sat = bool(self.cfg["data"].get("use_one_satellite", False))
        self.target = self.cfg["data"].get("target", "layer")
        self.models, self.norms = [], []
        for fold in fold_dirs(self.root):
            if folds and fold not in folds:
                continue
            ck = find_fold_checkpoint(checkpoints, fold)
            if ck is None:
                continue
            model, _ = load_model_from_checkpoint(ck, self.device)
            self.models.append(model)
            self.norms.append(NPZSequenceDataset(os.path.join(self.root, fold, "train_w.npz"), self.one_sat, verbose=False))
            if verbose:
                print(f"[ensemble] {self.cfg['name']}: loaded {ck}")
        if not self.models:
            raise SystemExit(f"no checkpoints for {self.cfg['name']} in {checkpoints}")

    def test_set(self, npz=None):
        path = npz or os.path.join(self.root, "test_w.npz")
        return NPZSequenceDataset(path, self.one_sat, verbose=False).use_normalization_of(self.norms[0])

    @torch.no_grad()
    def predict(self, x):
        """x: normalised input [B, T, C, H, W] -> (mean, std) over the folds, in m/s, numpy [B, T, 1, H, W]."""
        x = x.to(self.device)
        preds = np.stack([ds.denormalize(m(x)[0].cpu().numpy()) for m, ds in zip(self.models, self.norms)])
        return preds.mean(axis=0), preds.std(axis=0)
