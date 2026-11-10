"""
Export the released LES fields: extinction coefficient + vertical velocity per time step.

The raw LES files (11 float32 fields, 2.3 GB per time step, ~2 TB in total) are reduced to
the two fields the data pipeline needs:

    beta_ext  [1/m]  extinction coefficient from the microphysics (calc_beta.py), lossless
    W         [m/s]  vertical velocity, rounded to 1e-3 m/s (|error| <= 5e-4 m/s)

Both are stored as compressed NetCDF4 (zlib level 4 + shuffle), about 49 MB per time
step (~42 GB for the 848 steps). The layout matches the raw LES ([time=1, z, y, x], x/y/z
in metres) and the time id stays the last "_" token of the file name, so
les_to_volumes.py reads these files with the same command as the raw LES:

    <output>/BOMEX_512x512x200_20m_beta_w_<10-digit time id>.nc
    <output>/SHA256SUMS                      (verify with: sha256sum -c SHA256SUMS)

    python data_generation/export_beta_w.py --input /path/to/LES_netCDF --output les_beta_w --workers 6

Finished files are skipped, so an interrupted export can be re-run with the same command.
"""
import argparse
import glob
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_beta import process_cloud_vars  # noqa: E402

PREFIX = "BOMEX_512x512x200_20m_beta_w"
W_DECIMALS = 3          # W is rounded to 1e-3 m/s
CHUNKS = (1, 50, 128, 128)


def time_id(path):
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def out_name(path):
    return f"{PREFIX}_{time_id(path):010d}.nc"


def export_one(src, dst):
    tmp = dst + ".part"
    with Dataset(src, "r") as nc:
        field = lambda k: np.ma.filled(nc.variables[k][0], 0.0).astype(np.float32)  # noqa: E731
        p = np.asarray(nc.variables["p"][:])
        _, _, beta = process_cloud_vars(field("QN"), field("NC"), field("TABS"), p)
        beta = np.ma.filled(beta, 0.0).astype(np.float32)
        w = field("W")
        with Dataset(tmp, "w", format="NETCDF4") as out:
            out.title = "BOMEX LES: extinction coefficient and vertical velocity"
            out.source_file = os.path.basename(src)
            out.history = ("beta_ext from QN, NC, TABS, p with calc_beta.py (Eqs. 1-5, Q_ext = 2); "
                           f"W rounded to 1e-{W_DECIMALS} m/s; created {time.strftime('%Y-%m-%d')}")
            for d in ("time", "z", "y", "x"):
                out.createDimension(d, len(nc.dimensions[d]))
            for d in ("time", "z", "y", "x"):
                v = out.createVariable(d, "f4", (d,))
                v[:] = nc.variables[d][:]
                v.units = getattr(nc.variables[d], "units", "s" if d == "time" else "m")
            b = out.createVariable("beta_ext", "f4", ("time", "z", "y", "x"), zlib=True, complevel=4,
                                   shuffle=True, chunksizes=CHUNKS)
            b.units, b.long_name = "1/m", "extinction coefficient"
            b[0] = beta
            v = out.createVariable("W", "f4", ("time", "z", "y", "x"), zlib=True, complevel=4,
                                   shuffle=True, chunksizes=CHUNKS, least_significant_digit=W_DECIMALS)
            v.units, v.long_name = "m/s", "vertical velocity"
            v[0] = w
    os.replace(tmp, dst)
    return os.path.getsize(dst)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with the raw LES *_<10-digit time id>.nc files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=4, help="parallel processes (~3 GB RAM each)")
    ap.add_argument("--no-checksums", action="store_true", help="do not write SHA256SUMS")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input, "*.nc")), key=time_id)
    todo = [f for f in files if not os.path.exists(os.path.join(args.output, out_name(f)))]
    print(f"{len(files)} LES files, {len(files) - len(todo)} already exported, {len(todo)} to do", flush=True)

    t0, total = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(export_one, f, os.path.join(args.output, out_name(f))): f for f in todo}
        for k, job in enumerate(as_completed(jobs), 1):
            total += job.result()
            rate = (time.time() - t0) / k
            print(f"[{k}/{len(todo)}] {out_name(jobs[job])}  {total / 1e9:.1f} GB so far, "
                  f"~{(len(todo) - k) * rate / 60:.0f} min left", flush=True)

    if not args.no_checksums:
        outs = sorted(glob.glob(os.path.join(args.output, f"{PREFIX}_*.nc")))
        with open(os.path.join(args.output, "SHA256SUMS"), "w") as f:
            for o in outs:
                f.write(f"{sha256(o)}  {os.path.basename(o)}\n")
        size = sum(os.path.getsize(o) for o in outs)
        print(f"{len(outs)} files, {size / 1e9:.1f} GB, checksums in {args.output}/SHA256SUMS")


if __name__ == "__main__":
    main()
