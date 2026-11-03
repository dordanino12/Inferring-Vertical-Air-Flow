"""
Stage 1 - LES netCDF -> volumetric samples.

Each LES time step (512 x 512 x 200 voxels, 20 m) is cut into 128 x 128 x 200
patches with stride 64 (a 7 x 7 grid, 49 overlapping samples). For every patch
the extinction coefficient is computed from the microphysics (calc_beta.py)
and saved together with the velocity components:

    <output>/<time id>/sample_NNN.pkl  ->  {'U','V','W','beta_ext' : float32 [Z=200, Y=128, X=128], 'metadata'}

NNN = row * 7 + col of the patch grid.

    python data_generation/les_to_volumes.py --input /path/to/netCDF --output /path/to/volumes
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_beta import process_cloud_vars  # noqa: E402

PATCH, STRIDE = 128, 64


def time_id(path):
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def patches_from_nc(nc_path, out_dir, save_lwc_reff=False):
    os.makedirs(out_dir, exist_ok=True)
    nc = Dataset(nc_path, "r")
    nx, ny = nc.variables["x"].shape[0], nc.variables["y"].shape[0]
    p = nc.variables["p"][:]
    count = 0
    for i in range((ny - PATCH) // STRIDE + 1):          # Y
        for j in range((nx - PATCH) // STRIDE + 1):      # X
            ys, xs = slice(i * STRIDE, i * STRIDE + PATCH), slice(j * STRIDE, j * STRIDE + PATCH)
            qn = nc.variables["QN"][0, :, ys, xs]
            ncd = nc.variables["NC"][0, :, ys, xs]
            tabs = nc.variables["TABS"][0, :, ys, xs]
            lwc, reff, beta = process_cloud_vars(qn, ncd, tabs, p)
            data = {
                "metadata": {"source_file": os.path.basename(nc_path), "id": count, "grid_idx": (i, j),
                             "coords_x": (xs.start, xs.stop), "coords_y": (ys.start, ys.stop)},
                "U": np.ma.filled(nc.variables["U"][0, :, ys, xs], 0.0).astype(np.float32),
                "V": np.ma.filled(nc.variables["V"][0, :, ys, xs], 0.0).astype(np.float32),
                "W": np.ma.filled(nc.variables["W"][0, :, ys, xs], 0.0).astype(np.float32),
                "beta_ext": np.ma.filled(beta, 0.0).astype(np.float32),
            }
            if save_lwc_reff:
                data["LWC"] = np.ma.filled(lwc, 0.0).astype(np.float32)
                data["Reff"] = np.ma.filled(reff, 0.0).astype(np.float32)
            with open(os.path.join(out_dir, f"sample_{count:03d}.pkl"), "wb") as f:
                pickle.dump(data, f)
            count += 1
    nc.close()
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with *_<10-digit time id>.nc LES files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=None, help="first time id (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last time id (inclusive)")
    ap.add_argument("--save-lwc-reff", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.nc")), key=time_id)
    files = [f for f in files if (args.start is None or time_id(f) >= args.start) and (args.end is None or time_id(f) <= args.end)]
    print(f"{len(files)} netCDF files")
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        n = patches_from_nc(f, os.path.join(args.output, name), args.save_lwc_reff)
        print(f"{name}: {n} samples")


if __name__ == "__main__":
    main()
