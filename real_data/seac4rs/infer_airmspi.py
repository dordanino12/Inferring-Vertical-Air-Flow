"""
Step 3 - infer V_z on the AirMSPI sequence with the monocular 900 m ensemble.

The sequence is processed causally: for every frame t the model sees frames
1..t and the prediction for frame t is kept (as an online system would). The
K fold models are averaged. Background (clear-sky) pixels are excluded and the
inferred velocities of all frames are pooled for the distribution comparison.

    python real_data/seac4rs/infer_airmspi.py --npz real_data/seac4rs/data/airmspi.npz \
        --checkpoints checkpoints/layer_900m_mono --kfold-dir data/layer_900m --output-dir results/seac4rs
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from vertical_airflow.dataset import NPZSequenceDataset  # noqa: E402
from vertical_airflow.utils import find_fold_checkpoint, fold_dirs, load_model_from_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="AirMSPI input sequence, e.g. real_data/seac4rs/data/airmspi.npz")
    ap.add_argument("--checkpoints", required=True, help="fold checkpoints of the layer_900m_mono model")
    ap.add_argument("--kfold-dir", required=True, help="matching training data (for the de-normalisation scale)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--save-maps", action="store_true", help="also save a PDF of the inferred map per frame")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    models, norms = [], []
    for fold in fold_dirs(args.kfold_dir):
        ck = find_fold_checkpoint(args.checkpoints, fold)
        if ck is None:
            continue
        m, c = load_model_from_checkpoint(ck, device)
        models.append(m)
        norms.append(NPZSequenceDataset(os.path.join(args.kfold_dir, fold, "train_w.npz"),
                                        c["config"]["use_one_satellite"], verbose=False))
        print(f"loaded {ck}")
    one_sat = norms[0].use_one_satellite

    X = np.load(args.npz)["X"].astype(np.float32)          # [1, T, 2, H, W], already in [0, 1]
    T = X.shape[1]
    x_all = torch.from_numpy(X).to(device)
    if one_sat:
        x_all = x_all[:, :, :1]
    background_level = float(X[0, :, 0].min())

    maps, pooled = [], []
    with torch.no_grad():
        for t in range(1, T + 1):
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                preds = [ds.denormalize(m(x_all[:, :t])[0].squeeze(0).cpu().numpy()) for m, ds in zip(models, norms)]
            pred = np.mean(np.stack(preds), axis=0)[t - 1, 0]
            cloud = X[0, t - 1, 0] > background_level + 1e-4
            maps.append(np.where(cloud, pred, np.nan))
            pooled.append(pred[cloud])
            print(f"frame {t}/{T}: {cloud.sum()} cloudy pixels")

    v = np.concatenate(pooled)
    np.savez(os.path.join(args.output_dir, "airmspi_inferred_vz.npz"), maps=np.stack(maps), pooled=v)
    np.savetxt(os.path.join(args.output_dir, "model_vz_pooled.csv"), v, header="Model Vertical Velocity (m/s)", comments="")
    print(f"{v.size} inferred values, range [{v.min():.2f}, {v.max():.2f}] m/s -> {args.output_dir}")

    if args.save_maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for t, mp in enumerate(maps, start=1):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mp, cmap="jet", vmin=-5.55, vmax=5.55)
            ax.axis("off"); fig.colorbar(im, ax=ax, label="V_z [m/s]")
            fig.savefig(os.path.join(args.output_dir, f"vz_frame_{t:02d}.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
