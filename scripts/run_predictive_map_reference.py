#!/usr/bin/env python3
"""Reproduce released predictive-coding measurements without executing notebooks.

Requires a clean checkout of jgornet/predictive-coding-recovers-maps and its
published checkpoint/NumPy files. Saved-array results are descriptive: those
arrays are not documented as held-out trajectories by the authors.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mc_binding.predictive_map_metrics import (
    circular_positions, distance_summary, pair_distances, ridge_position_decode,
)


REFERENCE_COMMIT = "fed80b72925a4d3f3e9ec4b70f71ecaedec75c13"
FILES = {
    "environment": ("predictive_coding.ckpt", "predictive-coder-environment-images.npy",
                    "predictive-coder-environment-positions.npy"),
    "path": ("predictive_coding.ckpt", "predictive-coder-path-images.npy",
             "predictive-coder-path-positions.npy"),
    "circle": ("predictive-coder-circle.ckpt", "predictive-coder-circle-images.npy",
               "predictive-coder-circle-positions.npy"),
    "autoencoder": ("autoencoder.ckpt", "autoencoder-environment-images.npy",
                    "autoencoder-environment-positions.npy"),
    "circle_autoencoder": ("autoencoder-circle.ckpt", "autoencoder-circle-images.npy",
                           "autoencoder-circle-positions.npy"),
}

# The official EnvironmentDataset applies ToTensor() and then these channel
# statistics. Keep the published .npy untouched; the optional correction below
# is a separately labeled diagnostic for its apparent extra factor of 1/255.
ENV_MEAN = np.asarray([121.6697, 149.3242, 154.9510], dtype=np.float32) / 255
ENV_STD = np.asarray([40.7521, 47.7267, 103.2739], dtype=np.float32) / 255


def environment_rgb(normalized):
    values = np.asarray(normalized, dtype=np.float32)
    channel_shape = (1,) * (values.ndim - 3) + (3, 1, 1)
    return values * ENV_STD.reshape(channel_shape) + ENV_MEAN.reshape(channel_shape)


def restore_released_environment_scale(normalized):
    """Undo the apparent extra /255, then reapply official normalization."""
    rgb = environment_rgb(normalized) * 255
    channel_shape = (1,) * (rgb.ndim - 3) + (3, 1, 1)
    return (rgb - ENV_MEAN.reshape(channel_shape)) / ENV_STD.reshape(channel_shape)


def environment_scale_audit(images):
    indices = sorted({0, len(images) // 4, len(images) // 2, 3 * len(images) // 4, len(images) - 1})
    samples = np.stack([np.asarray(images[i, images.shape[1] // 2]) for i in indices])
    rgb = environment_rgb(samples)
    return {"sample_indices": indices, "rgb_min": float(rgb.min()),
            "rgb_max": float(rgb.max()), "rgb_mean": float(rgb.mean()),
            "rgb_max_near_one_over_255": bool(rgb.max() <= 1.1 / 255)}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(repo, weights, case):
    repo, weights = Path(repo).resolve(), Path(weights).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError(f"Expected official reference commit {REFERENCE_COMMIT}; got {commit}")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official checkout has local changes")
    if not (repo / "predictive_coding/models/encoder_decoder.py").is_file():
        raise ValueError("Official model implementation is missing")
    paths = [weights / name for name in FILES[case]]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Download released files first: {missing}")
    return paths, {"repo_commit": commit, "official_files": {
        path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths}}


def endpoint_positions(positions, count):
    positions = np.asarray(positions)
    if positions.ndim == 3:
        positions = positions[:, -1]
    if positions.ndim != 2 or len(positions) != count or positions.shape[1] < 2:
        raise ValueError("Official position array does not align with saved sequences")
    return np.asarray(positions[:, :2], dtype=float)


def validate_saved_arrays(images, positions, autoencoder):
    """Validate released array layout without assuming the model's max length."""
    expected = (images.ndim == 4 and images.shape[1:] == (3, 64, 64)) if autoencoder else (
        images.ndim == 5 and 2 <= images.shape[1] <= 20 and images.shape[2:] == (3, 64, 64))
    if not expected:
        kind = "[N,3,64,64]" if autoencoder else "[N,L,3,64,64] with 2<=L<=20"
        raise ValueError(f"Expected released {kind} images, got {images.shape}")
    if len(images) < 6:
        raise ValueError("Too few released images")
    endpoint_positions(positions, len(images))
    return {"image_sequences": int(len(images)),
            "sequence_length": None if autoencoder else int(images.shape[1]),
            "position_columns": int(positions.shape[-1])}


def run(args):
    paths, provenance = preflight(args.repo, args.weights, args.case)
    images = np.load(paths[1], mmap_mode="r", allow_pickle=False)
    positions_raw = np.load(paths[2], mmap_mode="r", allow_pickle=False)
    autoencoder = "autoencoder" in args.case
    layout = validate_saved_arrays(images, positions_raw, autoencoder)
    if args.restore_released_scale and args.case != "environment":
        raise ValueError("Scale restoration is audited only for the environment case")
    scale_audit = environment_scale_audit(images) if args.case == "environment" else None
    if args.restore_released_scale and not scale_audit["rgb_max_near_one_over_255"]:
        raise ValueError("Saved images do not exhibit the suspected extra /255 scale")
    if args.check_only:
        print(json.dumps({"state": "preflight_complete", "case": args.case,
                          **provenance, "saved_array_layout": layout,
                          "released_image_scale_audit": scale_audit}, indent=2))
        return
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from predictive_coding.models.encoder_decoder import Autoencoder, PredictiveCoder

    positions = endpoint_positions(positions_raw, len(images))
    device = torch.device(args.device)
    model = Autoencoder(3, 3, [2, 2, 2, 2]) if autoencoder else PredictiveCoder(3, 3, [2, 2, 2, 2], seq_len=20)
    state = torch.load(paths[0], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    latent_chunks, total_error, persistence_error, total_pixels = [], 0.0, 0.0, 0
    example = None
    for start in range(0, len(images), args.batch_size):
        saved_batch = np.array(images[start:start + args.batch_size], copy=True)
        if args.restore_released_scale:
            saved_batch = restore_released_environment_scale(saved_batch)
        batch = torch.from_numpy(saved_batch).float().to(device)
        with torch.inference_mode():
            latent = model.get_latents(batch)
            if not autoencoder:
                latent = latent[:, -1]
            latent = latent.cpu().numpy()
            predicted = model(batch) if autoencoder else model(batch)[:, :-1]
            target = batch if autoencoder else batch[:, 1:]
            if example is None:
                candidate_targets = target.reshape(-1, *target.shape[-3:])
                candidate_predictions = predicted.reshape(-1, *predicted.shape[-3:])
                frame_index = int(candidate_targets.var(dim=(1, 2, 3)).argmax())
                pframe = candidate_predictions[frame_index]
                tframe = candidate_targets[frame_index]
                example = (tframe.detach().cpu().numpy(), pframe.detach().cpu().numpy())
            total_error += float(torch.sum((predicted - target) ** 2).item())
            if not autoencoder:
                persistence_error += float(torch.sum((batch[:, :-1] - target) ** 2).item())
            total_pixels += target.numel()
        latent_chunks.append(latent)
    latents = np.concatenate(latent_chunks)
    is_circle = "circle" in args.case
    geometry = circular_positions(positions) if is_circle else positions
    latent_distance, physical_distance = pair_distances(
        latents, positions, seed=args.seed,
        circular_period=30.0 if is_circle else None)
    distance = distance_summary(latent_distance, physical_distance)
    # This deterministic split is an audit of the saved arrays, not a claim of
    # independently routed test trajectories. The published decoder is in-sample.
    cutoff = int(.8 * len(latents))
    train_idx = np.unique(np.linspace(0, cutoff - 1, min(cutoff, 1000), dtype=int))
    test_idx = np.unique(np.linspace(cutoff, len(latents) - 1,
                                     min(len(latents) - cutoff, 1000), dtype=int))
    decoder, predictions = ridge_position_decode(latents[train_idx], geometry[train_idx],
                                                  latents[test_idx], geometry[test_idx])
    paper_decoder = None
    if args.paper_decoder:
        # Deliberately reproduce the notebook's in-sample readout. This is a
        # reference-fidelity measurement, never a held-out spatial claim.
        from predictive_coding.analysis import Latents, PositionDecoder
        released = Latents(latents.astype(np.float32), positions.astype(np.float32))
        fitted = PositionDecoder(in_dim=128, device=str(device))
        fitted.train(released)
        errors = fitted.error(released)
        paper_decoder = {"mean_error": float(np.mean(errors)),
                         "median_error": float(np.median(errors)),
                         "train_and_evaluate_same_saved_latents": True,
                         "iterations": 8000}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, frame, title in zip(axes, example, ("Actual next image" if not autoencoder else "Actual image",
                                                "Predicted next image" if not autoencoder else "Reconstruction")):
        rgb = (frame + 1) / 2 if is_circle else environment_rgb(frame)
        ax.imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "image-example.png", dpi=180)
    plt.close(fig)
    plt.figure(figsize=(6, 5))
    plt.scatter(physical_distance, latent_distance, s=2, alpha=.16, rasterized=True)
    plt.xlabel("physical distance (lattice units; circular wrap for circle)")
    plt.ylabel("latent Euclidean distance")
    plt.tight_layout()
    plt.savefig(output / "latent-vs-physical.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.scatter(geometry[test_idx, 0], geometry[test_idx, 1], s=8, label="actual")
    plt.scatter(predictions[:, 0], predictions[:, 1], s=8, label="decoded")
    plt.legend()
    plt.xlabel("world coordinate 1")
    plt.ylabel("world coordinate 2")
    plt.tight_layout()
    plt.savefig(output / "position-decoding.png", dpi=180)
    plt.close()
    report = {"state": "complete", "case": args.case, **provenance,
              **layout,
              "input_scale": "restored_from_released_x255" if args.restore_released_scale else "released_unchanged",
              "released_image_scale_audit": scale_audit,
              "next_image_mse": None if autoencoder else total_error / total_pixels,
              "copy_previous_frame_mse": None if autoencoder else persistence_error / total_pixels,
              "reconstruction_mse": total_error / total_pixels if autoencoder else None,
              "pixel_metric_scope": "Saved arrays; restored scale is a diagnostic transform, not an author-validated replacement; evaluation split not documented as independent",
              "latent_distance": distance,
              "position_decoder": decoder,
              "paper_position_decoder": paper_decoder,
              "position_decoder_scope": "Ridge, up to 1000 evenly sampled rows from first 80% versus up to 1000 from last 20% of saved sequences; possible temporal/route leakage",
              "circular_geometry": is_circle,
              "not_yet_reproduced": ([] if args.paper_decoder else ["published convolutional position decoder"]) +
                                     ["independent circular-corridor route test"]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--case", choices=sorted(FILES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--paper-decoder", action="store_true",
                        help="Also run the official 8000-iteration in-sample PositionDecoder")
    parser.add_argument("--restore-released-scale", action="store_true",
                        help="Environment case only: undo apparent extra /255 without changing the released file")
    a = parser.parse_args()
    if not a.check_only and (a.output is None or a.batch_size < 1):
        parser.error("--output and positive --batch-size required for analysis")
    run(a)
