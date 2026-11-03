"""
Step 1 - AirMSPI (ER-2) multi-angle images -> model input sequence (Sec. 5).

Reads the AirMSPI L1B2 HDF files of one sortie (sorted by time = view angle),
crops a CROP x CROP window around a fixed geographic centre from the 355 nm
intensity, resizes it to 128 x 128 and normalises the whole sequence jointly
to [0, 1]. The single camera is duplicated into both input channels (the
monocular model uses only channel 0).

    python real_data/seac4rs/prepare_airmspi.py --hdf-dir /path/to/AirMSPI/20130812 --output airmspi_sequence.npz
Output: X float32 [1, T, 2, 128, 128], Y zeros (no ground truth).
"""
import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image


def read_band(path, band):
    with h5py.File(path, "r") as f:
        img = f[f"HDFEOS/GRIDS/{band}/Data Fields/I"][()]
        lat = f["HDFEOS/GRIDS/Ancillary/Data Fields/Latitude"][()]
        lon = f["HDFEOS/GRIDS/Ancillary/Data Fields/Longitude"][()]
    while img.ndim > 2:
        img = img[0]
    return img, lat, lon


def crop_around(img, lat, lon, center, size, fill):
    img = np.where(img > -900, img, fill)                       # HDF fill value -999
    d2 = (lat - center[0]) ** 2 + (lon - center[1]) ** 2
    d2[~((lat > -90) & (lon > -180))] = np.inf
    y, x = np.unravel_index(np.argmin(d2), d2.shape)
    h = size // 2
    y0, y1, x0, x1 = y - h, y + h, x - h, x + h
    H, W = img.shape
    crop = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)))
    return np.pad(crop, pad, mode="constant", constant_values=fill) if any(sum(p) for p in pad) else crop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lat", type=float, default=33.6119)
    ap.add_argument("--lon", type=float, default=-85.6623)
    ap.add_argument("--crop", type=int, default=1024, help="crop size in AirMSPI pixels before resizing")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--band", default="355nm_band")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="divide by headroom * max cloud radiance (1.0 = sequence maximum maps to 1, as in the paper)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.hdf_dir, "*.hdf")))
    if not files:
        raise SystemExit(f"no .hdf files in {args.hdf_dir}")
    # global minimum valid radiance over all frames (background level)
    gmin = np.inf
    for f in files:
        img = read_band(f, args.band)[0]
        if np.any(img > -900):
            gmin = min(gmin, float(np.min(img[img > -900])))
    gmin = 0.0 if gmin == np.inf else gmin

    frames = []
    for f in files:
        img, lat, lon = read_band(f, args.band)
        crop = np.maximum(crop_around(img, lat, lon, (args.lat, args.lon), args.crop, gmin), gmin)
        frames.append(np.array(Image.fromarray(crop, mode="F").resize((args.size, args.size), Image.Resampling.BILINEAR)))
        print(f"{os.path.basename(f)}")

    x = np.repeat(np.stack(frames)[:, None], 2, axis=1)[None].astype(np.float32)   # [1, T, 2, H, W]
    x = np.maximum(x - gmin, 0.0)
    cloud = x[x > 0.05]
    norm = float(np.max(cloud)) if cloud.size else (float(np.max(x[x > 0])) if np.any(x > 0) else 1.0)
    norm = (norm or 1.0) * args.headroom
    x = x / norm
    np.savez(args.output, X=x, Y=np.zeros((1, x.shape[1], 1, args.size, args.size), np.float32))
    print(f"saved {args.output}: X {x.shape}")


if __name__ == "__main__":
    main()
