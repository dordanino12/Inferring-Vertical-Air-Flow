"""
Physics-based rendering of a cloud volume from orbit with Mitsuba 3 (Sec. 3.4).

Volumetric path tracing (``volpath``, unlimited depth) through a heterogeneous
medium whose extinction is ``beta_ext``; single-scattering albedo 1,
Henyey-Greenstein phase function (g = 0.85 in the paper), sun as a directional
emitter, a dark diffuse surface below the cloud emulating the ocean (albedo
0.03), and perspective cameras placed at the satellite positions of an overpass
CSV. This is the scene used to render the paper's dataset.
"""
import os
import pickle
import struct

import mitsuba as mi
import numpy as np
import pandas as pd

mi.set_variant("cuda_ad_rgb" if "cuda_ad_rgb" in mi.variants() else "llvm_ad_rgb")
try:  # prefer CUDA when a GPU is available
    import torch
    if not torch.cuda.is_available():
        mi.set_variant("llvm_ad_rgb")
except ImportError:
    pass

SUN_IRRADIANCE = 131.4


class MitsubaRenderer:
    def __init__(self, overpass_csv, overpass_indices, spp, g_value=0.85, cloud_width=128, image_res=256,
                 fov=0.25, voxel_res=0.02, scene_scale=1e3, cloud_zrange=(0.0, 4.0), vol_path=None, seed=41,
                 ocean_reflectance=0.03):
        """
        overpass_indices : CSV rows (one per satellite) of a single acquisition time
        cloud_width      : horizontal size of the volume [voxels]; voxel_res [km]
        fov              : camera field of view [deg]; image_res: image size [pixels]
        ocean_reflectance: albedo of the diffuse ocean surface below the cloud (None = no surface)
        """
        self.overpass_csv = overpass_csv
        self.overpass_indices = list(overpass_indices)
        self.satellites = len(self.overpass_indices)
        self.spp = spp
        self.g_value = g_value
        self.voxel_res = voxel_res
        self.scene_scale = scene_scale
        self.W = cloud_width * voxel_res               # horizontal extent of the volume [km]
        self.cloud_zcenter = sum(cloud_zrange) / 2
        self.fov = fov
        self.film_dim = image_res
        self.vol_path = vol_path
        self.seed = seed
        self.ocean_reflectance = ocean_reflectance
        self.sensors, self.scene = [], None
        self._read_overpass_csv()
        self._create_sensors()

    def _read_overpass_csv(self):
        df = pd.read_csv(self.overpass_csv)
        coords = [df.loc[i, "sat ENU coordinates [km]"] for i in self.overpass_indices]
        self.sat_Wx = [float(c[1:-1].split(",")[0]) for c in coords]
        self.sat_Wy = [float(c[1:-1].split(",")[1]) for c in coords]
        self.sat_H = [float(c[1:-1].split(",")[2]) for c in coords]
        first = self.overpass_indices[0]
        self.sun_azimuth = float(df.loc[first, "sun azimuth [deg]"])
        self.sun_zenith = float(df.loc[first, "sun zenith [deg]"])

    def _create_sensors(self):
        cloud_target_z = self.cloud_zcenter * 2.5      # camera aim point (scene units)
        for i in range(self.satellites):
            to_world = mi.scalar_rgb.Transform4f.look_at(
                target=[0, 0, cloud_target_z],
                origin=[self.sat_Wy[i], self.sat_Wx[i], self.sat_H[i]],
                up=[1, 0, 0])
            self.sensors.append(mi.load_dict({
                "type": "perspective", "fov": self.fov, "to_world": to_world,
                "film": {"type": "hdrfilm", "width": self.film_dim, "height": self.film_dim, "filter": {"type": "tent"}},
            }))

    def write_vol_file(self, sample_path, param_type="beta_ext"):
        """Write the sample's extinction field ([Z, Y, X] in the pickle) as a Mitsuba .vol grid."""
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        data = np.transpose(sample[param_type], (2, 1, 0))
        if data.ndim == 3:
            data = np.expand_dims(data, axis=3)
        if data.dtype != np.float32:     # (as in the original code: flip only applied to non-float32 input)
            data = np.fliplr(data.astype(np.float32))
        os.makedirs(os.path.dirname(os.path.abspath(self.vol_path)), exist_ok=True)
        with open(self.vol_path, "wb") as f:
            f.write(b"VOL\x03")
            f.write(struct.pack("<i", 1))                 # float32 encoding
            f.write(struct.pack("<i", data.shape[2]))
            f.write(struct.pack("<i", data.shape[0]))
            f.write(struct.pack("<i", data.shape[1]))
            f.write(struct.pack("<i", data.shape[3]))     # channels
            f.write(struct.pack("<6f", *np.array([0, 0, 0, data.shape[2], data.shape[0], data.shape[1]], dtype=np.float32)))
            data.tofile(f)

    def scene_dict(self):
        az, ze = np.deg2rad(self.sun_azimuth), np.deg2rad(self.sun_zenith)
        sun_dir = [-np.sin(ze) * np.sin(az), -np.sin(ze) * np.cos(az), np.cos(ze)]
        T = mi.scalar_rgb.Transform4f
        scene = {
            "type": "scene",
            "integrator": {"type": "volpath", "max_depth": -1, "rr_depth": 1000},
            "object": {   # transparent box that holds the cloud medium
                "type": "cube",
                "bsdf": {"type": "null"},
                "to_world": T.scale(self.W / 2 * 1e3 / self.scene_scale).translate([0, 0, 2 * self.cloud_zcenter]),
                "interior": {
                    "type": "heterogeneous",
                    "albedo": 1.0,
                    "phase": {"type": "hg", "g": self.g_value},
                    "sigma_t": {
                        "type": "gridvolume",
                        "filename": self.vol_path,
                        "to_world": T.rotate([0, 1, 0], -90).scale(self.W * 1e3 / self.scene_scale)
                                     .translate([-0.5 + self.cloud_zcenter, -0.5, -0.5]),
                    },
                    "scale": self.scene_scale,
                },
            },
            "emitter": {"type": "directional", "direction": sun_dir,
                        "irradiance": {"type": "rgb", "value": SUN_IRRADIANCE}},
        }
        if self.ocean_reflectance is not None:   # dark diffuse surface emulating the ocean
            half = (self.W / 2 + 4) * 1e3 / self.scene_scale
            scene["ocean"] = {"type": "cube", "to_world": T.scale(half).translate([0, 0, -half]),
                              "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": self.ocean_reflectance}}}
        return scene

    def render(self, sample_path):
        """Render one volume sample from every satellite. Returns float [n_sat, H, W] radiance."""
        self.write_vol_file(sample_path)
        self.scene = mi.load_dict(self.scene_dict())
        images = [np.array(mi.render(self.scene, sensor=s, spp=self.spp, seed=self.seed))[:, :, 0] for s in self.sensors]
        return np.stack(images).astype(np.float64)
