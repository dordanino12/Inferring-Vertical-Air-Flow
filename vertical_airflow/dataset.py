"""
Sequence dataset.

Each ``.npz`` file holds
    X : float32 [N, T=12, 2, 128, 128]  rendered radiance of satellites A and B
    Y : float32 [N, T=12, 1, 128, 128]  ground-truth vertical velocity V_z [m/s]

``__getitem__`` returns (x, y, mask):
    x    : X / norm_const                       (norm_const = max(X) of this file, >= 1)
    y    : clip(Y / scale, -1, 1)               (symmetric scaling, max|Y| -> 0.95)
    mask : 1 where satellite-A radiance > CLOUD_THRESHOLD (cloud mask M)

Validation / test sets must use the *training* set's ``norm_const`` and
``scale`` - use ``dataset.use_normalization_of(train_dataset)``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CLOUD_THRESHOLD = 1.1   # radiance threshold for the cloud mask (raw, un-normalised units)
TARGET_NORM = 0.95      # max |V_z| of the training set is mapped to this value


class NPZSequenceDataset(Dataset):
    def __init__(self, npz_path: str, use_one_satellite: bool = False, verbose: bool = True):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
        self.use_one_satellite = bool(use_one_satellite)
        if self.use_one_satellite:            # monocular: keep satellite A only
            self.X = self.X[:, :, 0:1]
        self.N, self.T, self.C, self.H, self.W = self.X.shape

        # (numpy scalar types kept as in the training code for bit-exact results)
        self.norm_const = max(np.max(self.X), 1.0)
        max_abs = max(np.max(self.Y), np.abs(np.min(self.Y)))
        self.scale = max_abs / TARGET_NORM
        if self.scale == 0:
            self.scale = 1.0
        if verbose:
            mode = "mono" if self.use_one_satellite else "stereo"
            print(f"[data] {npz_path}: N={self.N}, T={self.T}, C={self.C} ({mode}), "
                  f"V_z range [{self.Y.min():.2f}, {self.Y.max():.2f}] m/s")

    def use_normalization_of(self, other: "NPZSequenceDataset"):
        self.norm_const, self.scale = other.norm_const, other.scale
        return self

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        mask = (x[:, 0:1] > CLOUD_THRESHOLD).float()
        x = x / self.norm_const
        y = torch.from_numpy(np.clip(self.Y[idx] / self.scale, -1.0, 1.0).astype(np.float32))
        return x, y, mask

    def denormalize(self, y_norm):
        """Normalised velocity -> m/s (numpy array or torch tensor, device preserved)."""
        if isinstance(y_norm, torch.Tensor):
            device = y_norm.device
            y_raw = y_norm.detach().cpu().numpy() * self.scale
            return torch.from_numpy(y_raw.astype(np.float32)).to(device)
        return y_norm * self.scale
