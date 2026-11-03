"""
Stage 4 - assemble sequences and spatial K-fold splits (Sec. 4.1).

Inputs: the renders (render_dataset.py) and one ground-truth map set
(velocity_maps.py). Consecutive groups of T=12 rendered time steps form one
sequence (one orbital overpass, 220 s). For each of the 49 spatial samples of
a group, the stereo images and the V_z map (of view 0) are resized from
256x256 to 128x128 (area interpolation), and camera noise is added to both
views (dark current, read noise, 10-bit quantisation).

Spatial split of the 7x7 sample grid (no spatial overlap between splits):
    test       : rows 5-6, cols 4-6 (fixed), surrounded by a fixed ignored halo
    validation : one of 4 blocks per fold; its 8-connected halo is dropped
    train      : the remaining cells
Output:
    <output>/test_w.npz
    <output>/fold_0K_val_r<a>-<b>_c<c>-<d>/{train_w.npz, val_w.npz}
    each with X: float32 [N, 12, 2, 128, 128] and Y: float32 [N, 12, 1, 128, 128]

    python data_generation/build_kfold_dataset.py --renders /path/to/renders --maps /path/to/maps_1000m --output data/layer_1000m
"""
import argparse
import gc
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

SEQ_LEN, GRID, NUM_SAMPLES, OUT_RES = 12, 7, 49, 128
VAL_BLOCKS = [((0, 1), (0, 2)), ((0, 1), (4, 6)), ((2, 3), (0, 6)), ((5, 6), (0, 2))]
TEST_CELLS = {(5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)}
IGNORE_CELLS = {(4, c) for c in range(3, 7)} | {(5, 3), (6, 3)}

# --- camera noise model ---
CONVERSION_FACTOR = 178.6304426659069   # electrons per radiance unit
EXPOSURE_TIME_US = 205
DARK_CURRENT_RATE = 4.72e-6
READ_NOISE_VAR = 5.29                   # [e^2]
FULL_WELL_CAPACITY = 10600              # [e]
BIT_DEPTH_LEVELS = 1024                 # 10 bit


def sensor_noise(img, rng):
    e = img * CONVERSION_FACTOR
    dark = DARK_CURRENT_RATE * EXPOSURE_TIME_US
    e = e + rng.normal(dark, dark ** 0.5, size=e.shape)
    e = e + rng.normal(0.0, READ_NOISE_VAR ** 0.5, size=e.shape)
    e = np.clip(e, 0, FULL_WELL_CAPACITY)
    e = np.round(e * (BIT_DEPTH_LEVELS / FULL_WELL_CAPACITY)) * (FULL_WELL_CAPACITY / BIT_DEPTH_LEVELS)
    return (e / CONVERSION_FACTOR).astype(np.float32)


def block_cells(block):
    (r0, r1), (c0, c1) = block
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def halo(cells):
    out = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < GRID and 0 <= cc < GRID and (rr, cc) not in cells:
                    out.add((rr, cc))
    return out


FOLD_CELLS = [block_cells(b) for b in VAL_BLOCKS]
FOLD_HALOS = [halo(c) for c in FOLD_CELLS]


def split_of(cell, fold):
    if cell in TEST_CELLS:
        return "test"
    if cell in IGNORE_CELLS or cell in FOLD_HALOS[fold]:
        return None
    return "val" if cell in FOLD_CELLS[fold] else "train"


def find(folder, sample, view, cache={}):
    if folder not in cache:
        cache[folder] = sorted(f for f in os.listdir(folder) if f.endswith(".pkl")) if os.path.isdir(folder) else []
    key = f"sample_{sample:03d}_"
    for f in cache[folder]:
        if key in f and f"_view_{view}" in f:
            return os.path.join(folder, f)
    return None


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resize(a):
    return cv2.resize(np.nan_to_num(np.asarray(a), nan=0.0), (OUT_RES, OUT_RES), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--component", default="w", choices=["u", "v", "w"])
    ap.add_argument("--no-noise", action="store_true", help="do not add camera noise")
    ap.add_argument("--seed", type=int, default=0, help="seed of the camera noise (the original data was not seeded)")
    ap.add_argument("--max-sequences", type=int, default=None, help="only build the first N time groups (for testing)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    folders = sorted((f for f in os.listdir(args.renders) if f.isdigit()), key=int)
    splits = {"test": [[], []]}
    folds = [{"train": [[], []], "val": [[], []]} for _ in VAL_BLOCKS]

    starts = list(range(0, len(folders) - SEQ_LEN + 1, SEQ_LEN))[:args.max_sequences]
    for start in tqdm(starts, desc="sequences"):
        group = folders[start:start + SEQ_LEN]
        for s in range(NUM_SAMPLES):
            cell = (s // GRID, s % GRID)
            targets = [split_of(cell, k) for k in range(len(VAL_BLOCKS))]
            if all(t is None for t in targets):
                continue
            xs, ys = [], []
            for folder in group:
                p0 = find(os.path.join(args.renders, folder), s, 0)
                p1 = find(os.path.join(args.renders, folder), s, 1)
                pm = find(os.path.join(args.maps, folder), s, 0)
                if not (p0 and p1 and pm):
                    break
                img0, img1 = resize(load(p0)["render"]), resize(load(p1)["render"])
                if not args.no_noise:
                    img0, img1 = sensor_noise(img0, rng), sensor_noise(img1, rng)
                xs.append(np.stack([img0, img1]).astype(np.float32))
                vmap = np.asarray(load(pm)[f"{args.component}_map"])
                vmap = vmap if vmap.ndim == 3 else vmap[None]
                ys.append(np.stack([resize(h) for h in vmap]).astype(np.float32))
            if len(xs) < SEQ_LEN:
                continue
            X, Y = np.stack(xs), np.stack(ys)
            if targets[0] == "test":
                splits["test"][0].append(X); splits["test"][1].append(Y)
                continue
            for k, t in enumerate(targets):
                if t:
                    folds[k][t][0].append(X); folds[k][t][1].append(Y)

    def save(lists, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        X, Y = np.stack(lists[0]), np.stack(lists[1])
        np.savez_compressed(path, X=X, Y=Y)
        print(f"{path}: X {X.shape}, Y {Y.shape}")
        del X, Y
        gc.collect()

    tag = args.component
    save(splits["test"], os.path.join(args.output, f"test_{tag}.npz"))
    for k, (block, f) in enumerate(zip(VAL_BLOCKS, folds), start=1):
        (r0, r1), (c0, c1) = block
        d = os.path.join(args.output, f"fold_{k:02d}_val_r{r0}-{r1}_c{c0}-{c1}")
        save(f["train"], os.path.join(d, f"train_{tag}.npz"))
        save(f["val"], os.path.join(d, f"val_{tag}.npz"))


if __name__ == "__main__":
    main()
