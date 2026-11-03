"""
Stage 2 - render every volume sample from the two satellites.

LES time folders (sorted) are assigned cyclically to the T=12 acquisition
times of the overpass CSV (t = 0, 20, ..., 220 s): folder k is imaged at time
index k mod 12, so 12 consecutive LES time steps form one orbital overpass
(satellite motion, sun direction and cloud evolution all advance together).

    <output>/<time id>/sample_NNN_time_<t>_view_<k>.pkl -> {'render': float64 [256, 256], 'timestamp': t, 'satellite_idx': k}

    python data_generation/render_dataset.py --input /path/to/volumes --output /path/to/renders
    # split across GPUs / machines with --start/--end (time ids, inclusive);
    # the time index is always computed from the position in the FULL folder list.
"""
import argparse
import os
import pickle
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mitsuba_renderer import MitsubaRenderer  # noqa: E402

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "overpass_2satellites.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="output of les_to_volumes.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--overpass-csv", default=DEFAULT_CSV)
    ap.add_argument("--spp", type=int, default=8192)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--g", type=float, default=0.85, help="Henyey-Greenstein asymmetry parameter")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--ocean-reflectance", type=float, default=0.03, help="albedo of the ocean surface (negative = none)")
    ap.add_argument("--start", default=None, help="first time-id folder to render (inclusive)")
    ap.add_argument("--end", default=None, help="last time-id folder to render (inclusive)")
    ap.add_argument("--samples", type=int, nargs="*", default=None, help="render only these sample indices")
    args = ap.parse_args()

    df = pd.read_csv(args.overpass_csv)
    times = sorted(df["utc time"].unique())
    rows_at = {t: df.index[df["utc time"] == t].tolist() for t in times}

    folders = sorted(d.name for d in os.scandir(args.input) if d.is_dir() and d.name.isdigit())
    for k, folder in enumerate(tqdm(folders, desc="time steps")):
        if (args.start and int(folder) < int(args.start)) or (args.end and int(folder) > int(args.end)):
            continue
        t = times[k % len(times)]
        in_dir, out_dir = os.path.join(args.input, folder), os.path.join(args.output, folder)
        os.makedirs(out_dir, exist_ok=True)
        renderer = MitsubaRenderer(args.overpass_csv, rows_at[t], spp=args.spp, g_value=args.g,
                                   image_res=args.resolution, vol_path=os.path.join(out_dir, "_tmp_cloud.vol"),
                                   seed=args.seed,
                                   ocean_reflectance=None if args.ocean_reflectance < 0 else args.ocean_reflectance)
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".pkl"))
        if args.samples is not None:
            files = [f for f in files if int(f.split("_")[1].split(".")[0]) in args.samples]
        for f in files:
            images = renderer.render(os.path.join(in_dir, f))
            base = os.path.splitext(f)[0]
            for sat, img in enumerate(images):
                with open(os.path.join(out_dir, f"{base}_time_{t}_view_{sat}.pkl"), "wb") as fo:
                    pickle.dump({"render": img, "timestamp": t, "satellite_idx": sat}, fo)
        if os.path.exists(renderer.vol_path):
            os.remove(renderer.vol_path)


if __name__ == "__main__":
    main()
