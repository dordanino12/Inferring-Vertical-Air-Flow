"""
Step 2 - DC-8 in-situ vertical wind (MMS, 1 Hz) for the overlap window.

Reads the ICARTT (.ict) file of the NASA DC-8 Meteorological Measurement
System, applies the file's scale factors, and keeps the samples in the time
window of the AirMSPI overpass (default: 2013-08-12 18:02:03 - 18:24:00 UTC).

    python real_data/seac4rs/prepare_dc8.py --ict SEAC4RS-MMS-1HZ_DC8_20130812_R0.ict --output dc8_window.csv
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

FILL_VALUES = [-999, -9999, -99999, -999999, -9999999, -99999999]
SCALE = {"W": 1e-3, "G_LAT": 1e-5, "G_LONG": 1e-5, "G_ALT": 0.1}


def read_ict(path):
    with open(path) as f:
        lines = f.readlines()
    n_header = int(lines[0].split(",")[0])
    y, m, d = (int(v) for v in lines[6].split(",")[:3])
    df = pd.read_csv(path, skiprows=n_header - 1, sep=",")
    df.columns = [c.strip() for c in df.columns]
    df = df.replace(FILL_VALUES, np.nan)
    for col, s in SCALE.items():
        if col in df.columns:
            df[col] = df[col] * s
    df["Time (UTC)"] = datetime(y, m, d) + pd.to_timedelta(df["TIME_UTC"], unit="s")
    return df.rename(columns={"G_LAT": "Latitude", "G_LONG": "Longitude", "G_ALT": "Altitude (m)",
                              "W": "Vertical Velocity (m/s)"})[
        ["Time (UTC)", "Latitude", "Longitude", "Altitude (m)", "Vertical Velocity (m/s)"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ict", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", default="2013-08-12 18:02:03")
    ap.add_argument("--end", default="2013-08-12 18:24:00")
    args = ap.parse_args()
    df = read_ict(args.ict)
    win = df[(df["Time (UTC)"] >= pd.to_datetime(args.start)) & (df["Time (UTC)"] <= pd.to_datetime(args.end))]
    win.to_csv(args.output, index=False)
    w = win["Vertical Velocity (m/s)"].dropna()
    print(f"{len(win)} samples ({w.size} valid W) in [{args.start}, {args.end}], W range [{w.min():.2f}, {w.max():.2f}] m/s -> {args.output}")


if __name__ == "__main__":
    main()
