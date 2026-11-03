"""
Stage 3 - ground-truth vertical-velocity maps (Sec. 3.2, Fig. 3).

* envelope : for every pixel of satellite view k, march the camera ray through
             the volume (20 m steps) and take the velocity of the first voxel
             with beta_ext > 0 (the visible cloud envelope). Clear-sky pixels = NaN.
* layer    : V_z on a horizontal plane at altitude Z, seen from nadir (a camera
             at 600 km above the domain centre; ray/plane intersection on a
             750 m reference plane, then sampled at height Z). Includes clear air.

    <output>/<time id>/sample_NNN_time_<t>_view_<k>_first_hit.pkl     (envelope)
    <output>/<time id>/sample_NNN_time_<t>_view_<k>_slice_<Z>m.pkl     (layer)
        -> {'u_map','v_map','w_map': float64 [256, 256]}

    python data_generation/velocity_maps.py --mode envelope --input /path/to/volumes --output /path/to/maps_envelope
    python data_generation/velocity_maps.py --mode layer --height 1000 --input /path/to/volumes --output /path/to/maps_1000m
"""
import argparse
import ast
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "overpass_2satellites.csv")
LOOK_AT = np.array([0.0, 0.0, 1500.0])           # [m]
NADIR_CAMERA = np.array([0.0, 0.0, 600e3])       # [m]
REFERENCE_PLANE_Z = 750.0                        # [m]


class CloudRayCaster:
    """Ray casting through a [Z, Y, X] volume with ``voxel_size`` m voxels centred at (0, 0)."""

    def __init__(self, pkl_path, voxel_size=20.0):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.vol_beta, self.vol_w = data["beta_ext"], data["W"]
        self.vol_u, self.vol_v = data.get("U", self.vol_w), data.get("V", self.vol_w)
        self.nz, self.ny, self.nx = self.vol_beta.shape
        self.voxel_size = voxel_size
        self.min_bound = np.array([-self.nx * voxel_size / 2, -self.ny * voxel_size / 2, 0.0])
        self.max_bound = np.array([self.nx * voxel_size / 2, self.ny * voxel_size / 2, self.nz * voxel_size])

    @staticmethod
    def get_rays(cam_pos, look_at, resolution, fov=0.25):
        H, W = resolution
        cam_dir = (look_at - cam_pos) / np.linalg.norm(look_at - cam_pos)
        cam_right = np.cross(cam_dir, np.array([-1, 0, 0]))
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_dir)
        cam_up /= np.linalg.norm(cam_up)
        scale = np.tan(np.deg2rad(fov * 0.5))
        i, j = np.meshgrid(np.arange(W), np.arange(H))
        x = -((2 * (i + 0.5) / W - 1) * (W / H) * scale)
        y = (1 - 2 * (j + 0.5) / H) * scale
        d = x[..., None] * cam_right + y[..., None] * cam_up + cam_dir
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(cam_pos, d.shape), d

    def first_hit(self, cam_pos, look_at, resolution=(256, 256), step_size=20.0):
        """Velocity at the first cloudy voxel along each ray (cloud envelope)."""
        H, W = resolution
        ro, rd = self.get_rays(np.asarray(cam_pos), np.asarray(look_at), resolution)
        ro, rd = ro.reshape(-1, 3), rd.reshape(-1, 3)
        maps = [np.full(ro.shape[0], np.nan) for _ in range(3)]

        inv_d = 1.0 / (rd + 1e-6)                                   # ray / box intersection
        t0, t1 = (self.min_bound - ro) * inv_d, (self.max_bound - ro) * inv_d
        tmin = np.maximum(np.minimum(t0, t1).max(axis=1), 0)
        tmax = np.minimum(np.maximum(t0, t1).min(axis=1), 1e11)
        valid = np.where(tmax > tmin)[0]
        if len(valid) == 0:
            return [m.reshape(H, W) for m in maps]

        t_start, t_end = tmin[valid], tmax[valid]
        steps = np.arange(int(np.max(t_end - t_start) / step_size) + 2) * step_size
        t_vals = t_start[:, None] + steps[None, :]
        pts = ro[valid][:, None, :] + rd[valid][:, None, :] * t_vals[:, :, None]
        g = ((pts - self.min_bound) / self.voxel_size).astype(int)
        gx = np.clip(g[:, :, 0], 0, self.nx - 1)
        gy = np.clip(g[:, :, 1], 0, self.ny - 1)
        gz = np.clip(g[:, :, 2], 0, self.nz - 1)
        beta = self.vol_beta[gz, gy, gx]
        beta[~(t_vals <= t_end[:, None])] = 0

        hit = beta > 0
        idx = np.argmax(hit, axis=1)
        has_hit = np.any(hit, axis=1)
        r = np.arange(len(valid))
        hz, hy, hx = gz[r, idx], gy[r, idx], gx[r, idx]
        for m, vol in zip(maps, (self.vol_u, self.vol_v, self.vol_w)):
            out = np.full(len(valid), np.nan)
            out[has_hit] = vol[hz, hy, hx][has_hit]
            m[valid] = out
        return [m.reshape(H, W) for m in maps]

    def z_slice(self, cam_pos, look_at, target_z, resolution=(256, 256), reference_plane_z=REFERENCE_PLANE_Z):
        """Velocity on the horizontal plane Z=target_z (XY from the ray / reference-plane intersection)."""
        H, W = resolution
        ro, rd = self.get_rays(np.asarray(cam_pos), np.asarray(look_at), resolution)
        ro, rd = ro.reshape(-1, 3), rd.reshape(-1, 3).copy()
        dz = rd[:, 2]
        dz[np.abs(dz) < 1e-6] = 1e-6
        t_hit = (reference_plane_z - ro[:, 2]) / dz
        pts = ro + t_hit[:, None] * rd
        pts[:, 2] = target_z
        valid = (t_hit > 0) & np.all(pts >= self.min_bound, axis=1) & np.all(pts <= self.max_bound, axis=1)
        maps = [np.full(ro.shape[0], np.nan) for _ in range(3)]
        idx = np.where(valid)[0]
        if len(idx):
            g = ((pts[idx] - self.min_bound) / self.voxel_size).astype(int)
            gx = np.clip(g[:, 0], 0, self.nx - 1)
            gy = np.clip(g[:, 1], 0, self.ny - 1)
            gz = np.clip(g[:, 2], 0, self.nz - 1)
            for m, vol in zip(maps, (self.vol_u, self.vol_v, self.vol_w)):
                m[idx] = vol[gz, gy, gx]
        return [m.reshape(H, W) for m in maps]


def load_cameras(csv_path):
    """{utc time: [satellite position (m) per view]} in the ray-caster's frame."""
    df = pd.read_csv(csv_path)
    times = sorted(df["utc time"].unique())
    cams = {}
    for t in times:
        cams[t] = []
        for _, row in df[df["utc time"] == t].iterrows():
            e, n, u = ast.literal_eval(row["sat ENU coordinates [km]"])
            cams[t].append(np.array([-n, e, u]) * 1000.0)
    return times, cams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["envelope", "layer"], required=True)
    ap.add_argument("--height", type=float, help="layer altitude Z [m] (layer mode)")
    ap.add_argument("--input", required=True, help="output of les_to_volumes.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--overpass-csv", default=DEFAULT_CSV)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    if args.mode == "layer" and args.height is None:
        ap.error("--height is required in layer mode")

    times, cams = load_cameras(args.overpass_csv)
    res = (args.resolution, args.resolution)
    folders = sorted(d.name for d in os.scandir(args.input) if d.is_dir() and d.name.isdigit())
    for k, folder in enumerate(tqdm(folders, desc="time steps")):
        if (args.start and int(folder) < int(args.start)) or (args.end and int(folder) > int(args.end)):
            continue
        t = times[k % len(times)]
        out_dir = os.path.join(args.output, folder)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(f for f in os.listdir(os.path.join(args.input, folder)) if f.endswith(".pkl")):
            caster = CloudRayCaster(os.path.join(args.input, folder, f))
            base = os.path.splitext(f)[0]
            for view, cam in enumerate(cams[t]):
                if args.mode == "envelope":
                    u, v, w = caster.first_hit(cam, LOOK_AT, res)
                    suffix = "first_hit"
                else:
                    u, v, w = caster.z_slice(NADIR_CAMERA, LOOK_AT, args.height, res)
                    suffix = f"slice_{int(args.height)}m"
                with open(os.path.join(out_dir, f"{base}_time_{t}_view_{view}_{suffix}.pkl"), "wb") as fo:
                    pickle.dump({"u_map": u, "v_map": v, "w_map": w}, fo, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
