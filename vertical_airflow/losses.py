"""
Training losses (Sec. 3.6 and 4.4 of the paper).

* ``bin_loss``   - inverse-frequency velocity-bin weighted L1 (Eq. 6).
* ``power_loss`` - magnitude-weighted L1, weight = 1 + 4|V_z|^8 on normalised V_z
                   ("L_power", Eq. 7). Also used as a warm-up before the bin loss.
* ``l1_loss``    - plain masked L1 (loss ablation).

Mask M (``use_mask``):
  True          - envelope: only cloudy pixels count (M=1 cloud, M=0 elsewhere);
                  the bin histogram is also computed on cloudy pixels only.
  "slice_mask"  - horizontal layers: the cloud mask of the middle frame (t=5),
                  dilated by 5x5, is applied to all frames; cloudy pixels get
                  weight 1, all other pixels ``unmasked_weight`` (0.9).
  False         - all pixels, equal weight.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BIN_WIDTH = 0.1          # m/s, width of a velocity bin
MAX_BIN_WEIGHT = 100.0   # cap on inverse-frequency weights
MIDDLE_FRAME = 5         # frame whose cloud mask defines the layer mask
MASK_DILATION = 5        # dilation of the layer mask (pixels)


def layer_mask(mask: torch.Tensor) -> torch.Tensor:
    """Dilated cloud mask of the middle frame, broadcast to every frame. mask: [B, T, 1, H, W]."""
    m = F.max_pool2d(mask[:, MIDDLE_FRAME], kernel_size=MASK_DILATION, stride=1, padding=MASK_DILATION // 2)
    return m.unsqueeze(1).expand_as(mask)


def spatial_weights(mask, use_mask, unmasked_weight):
    """Per-pixel weight M, or None for no masking."""
    if mask is None or use_mask is False:
        return None
    if use_mask == "slice_mask":
        lm = layer_mask(mask)
        return torch.where(lm > 0.5, torch.ones_like(lm), torch.full_like(lm, unmasked_weight))
    return mask


def bin_loss(y_pred, y, mask, use_mask, dataset, bin_min, bin_max, unmasked_weight=0.9):
    """Inverse-frequency velocity-bin weighted L1 loss (Eq. 6).

    y_pred, y: normalised velocities [B, T, 1, H, W]; bins are defined in m/s
    on the de-normalised ground truth, over [bin_min, bin_max] of the training set.
    """
    abs_diff = (y_pred - y).abs()
    bin_min, bin_max = float(bin_min), float(bin_max)
    num_bins = int(math.ceil((bin_max - bin_min) / BIN_WIDTH))

    # pixels that define the velocity histogram: cloudy (envelope) / mask region (layers)
    hist_sel = None
    if mask is not None and use_mask == "slice_mask":
        hist_sel = layer_mask(mask) > 0.5
    elif mask is not None and use_mask is True:
        hist_sel = mask > 0.5
    if hist_sel is not None and hist_sel.sum() == 0:
        hist_sel = None

    y_ms = dataset.denormalize(y).clamp(bin_min, bin_max - 1e-6)
    y_flat = y_ms[hist_sel] if hist_sel is not None else y_ms.flatten()

    # inverse-frequency bin weights, normalised to mean 1 over non-empty bins
    bin_counts = torch.zeros(num_bins, device=y.device)
    for i in range(num_bins):
        lo = bin_min + i * BIN_WIDTH
        bin_counts[i] = ((y_flat >= lo) & (y_flat < lo + BIN_WIDTH)).sum().float()
    bin_weights = torch.zeros(num_bins, device=y.device)
    non_empty = bin_counts > 0
    bin_weights[non_empty] = y_flat.numel() / (bin_counts[non_empty] + 1e-8)
    if non_empty.any():
        bin_weights[non_empty] = bin_weights[non_empty] / (bin_weights[non_empty].mean() + 1e-8)
    bin_weights = torch.clamp(bin_weights, max=MAX_BIN_WEIGHT)

    pixel_w = torch.zeros_like(y_ms)
    for i in range(num_bins):
        lo = bin_min + i * BIN_WIDTH
        pixel_w[(y_ms >= lo) & (y_ms < lo + BIN_WIDTH)] = bin_weights[i]

    m = spatial_weights(mask, use_mask, unmasked_weight)
    w = pixel_w if m is None else pixel_w * m
    denom = w.sum()
    if m is not None and denom < 1e-8:
        return torch.zeros((), device=y.device)
    return (abs_diff * w).sum() / (denom + 1e-8)


def power_loss(y_pred, y, mask, use_mask, unmasked_weight=0.9):
    """Magnitude-weighted L1 (Eq. 7): weight = 1 + 4 |y|^8 on normalised velocities."""
    abs_diff = (y_pred - y).abs()
    weight = 1.0 + 4.0 * (y.abs() ** 8)
    m = spatial_weights(mask, use_mask, unmasked_weight)
    if m is None:
        return (abs_diff * weight).mean()
    return (abs_diff * m * weight).sum() / ((m * weight).sum() + 1e-8)


def l1_loss(y_pred, y, mask, use_mask, unmasked_weight=0.9):
    """Plain L1 with the spatial mask M (loss ablation)."""
    abs_diff = (y_pred - y).abs()
    m = spatial_weights(mask, use_mask, unmasked_weight)
    if m is None:
        return abs_diff.mean()
    return (abs_diff * m).sum() / (m.sum() + 1e-8)


class LossSchedule:
    """Selects the loss for a given epoch.

    schedule="power_to_bins" (default): power loss for ``power_epochs`` epochs,
    then a linear blend into the bin loss over ``interp_epochs`` epochs, then the
    bin loss. schedule="bins" / "power" / "l1": a single loss throughout.
    """

    def __init__(self, cfg: dict, dataset, bin_min: float, bin_max: float):
        self.schedule = cfg.get("schedule", "power_to_bins")
        self.power_epochs = int(cfg.get("power_epochs", 10))
        self.interp_epochs = int(cfg.get("interp_epochs", 5))
        self.use_mask = cfg["use_mask"]
        self.unmasked_weight = float(cfg.get("unmasked_weight", 0.9))
        self.dataset, self.bin_min, self.bin_max = dataset, bin_min, bin_max

    def mode(self, epoch: int):
        """Return (mode, interpolation factor) for a 1-based epoch."""
        if self.schedule != "power_to_bins":
            return self.schedule, 1.0
        if epoch <= self.power_epochs:
            return "power", 0.0
        if epoch <= self.power_epochs + self.interp_epochs:
            return "interp", (epoch - self.power_epochs) / self.interp_epochs
        return "bins", 1.0

    def is_final_loss(self, epoch: int) -> bool:
        """Checkpoints are only selected once the final loss is active."""
        return self.schedule != "power_to_bins" or self.mode(epoch)[0] == "bins"

    def __call__(self, y_pred, y, mask, epoch: int):
        mode, a = self.mode(epoch)
        if mode == "power":
            return power_loss(y_pred, y, mask, self.use_mask, self.unmasked_weight)
        if mode == "l1":
            return l1_loss(y_pred, y, mask, self.use_mask, self.unmasked_weight)
        lb = bin_loss(y_pred, y, mask, self.use_mask, self.dataset, self.bin_min, self.bin_max, self.unmasked_weight)
        if mode == "bins":
            return lb
        lp = power_loss(y_pred, y, mask, self.use_mask, self.unmasked_weight)
        return (1 - a) * lp + a * lb
