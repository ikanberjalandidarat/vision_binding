#!/usr/bin/env python3
"""Exploratory spatial-distance test on verified original validation paths.

This is not the paper's fitted distance model or a route-invariance result.
It compares frozen checkpoint activations with raw RGB on identical local pairs.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_predictive_map_paths import read_state
from run_predictive_map_original_pngs import load_episode
from run_predictive_map_reference import ENV_MEAN, ENV_STD, REFERENCE_COMMIT, file_hash


def local_pairs(latents, rgb, positions):
    from scipy.stats import spearmanr

    latent, pixel, physical, lags = [], [], [], []
    for lag in range(5, min(91, len(latents))):
        position_distance = np.linalg.norm(positions[lag:] - positions[:-lag], axis=1)
        keep = position_distance > .5
        if not keep.any():
            continue
        latent.append(np.linalg.norm(latents[lag:] - latents[:-lag], axis=1)[keep])
        pixel.append(np.linalg.norm(rgb[lag:] - rgb[:-lag], axis=1)[keep])
        physical.append(position_distance[keep])
        lags.append(np.full(int(keep.sum()), lag, dtype=int))
    if not physical:
        raise ValueError("No usable local pairs in verified path")
    latent = np.concatenate(latent)
    pixel = np.concatenate(pixel)
    physical = np.concatenate(physical)
    lags = np.concatenate(lags)
    result = {"pairs": int(len(physical)),
              "latent_vs_physical_spearman": float(spearmanr(latent, physical).statistic),
              "rgb_vs_physical_spearman": float(spearmanr(pixel, physical).statistic),
              "by_temporal_separation": {}}
    for label, lower, upper in (("5to15", 5, 15), ("16to30", 16, 30),
                                ("31to60", 31, 60), ("61to90", 61, 90)):
        mask = (lags >= lower) & (lags <= upper)
        if mask.sum() >= 20 and np.ptp(physical[mask]) > 0:
            result["by_temporal_separation"][label] = {
                "pairs": int(mask.sum()),
                "latent_spearman": float(spearmanr(latent[mask], physical[mask]).statistic),
                "rgb_spearman": float(spearmanr(pixel[mask], physical[mask]).statistic)}
    return result, (physical, latent, pixel)


def run(args):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repo = args.repo.resolve()
    manifest = json.loads(args.manifest.read_text())
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT or manifest["repo_commit"] != commit:
        raise ValueError("Manifest and official checkout must match the pinned commit")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official checkout has local changes")
    if manifest["state"] != "complete" or len(manifest["selected_paths"]) < 4:
        raise ValueError("A completed manifest with four or more continuous paths is required")
    sys.path.insert(0, str(repo))
    from predictive_coding.models.encoder_decoder import PredictiveCoder

    model = PredictiveCoder(3, 3, [2, 2, 2, 2], seq_len=20)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(args.device)
    report_paths, cached = [], []
    plot_samples = []
    for path_index, path in enumerate(manifest["selected_paths"]):
        states, images = [], []
        for episode, expected_hash in zip(path["episodes"], path["state_sha256"]):
            state, digest = read_state(repo, episode)
            if digest != expected_hash:
                raise ValueError(f"State changed since manifest: {episode}")
            png_paths = [f"{episode}/{i}.png" for i in range(10)]
            frames, _ = load_episode(repo, png_paths)
            states.append(state)
            images.append(frames)
        states = np.concatenate(states)
        images = np.concatenate(images)
        # The official model's layer normalization spans its input clip. Take
        # only the final latent of a sliding ten-frame window, so no frame
        # after the labeled endpoint is present anywhere in the forward pass.
        windows = np.stack([images[end - 9:end + 1] for end in range(9, len(images))])
        features = []
        for start in range(0, len(windows), 8):
            batch = torch.from_numpy(windows[start:start + 8]).float().to(args.device)
            with torch.inference_mode():
                latent = model.get_latents(batch)
            if tuple(latent.shape[1:]) != (10, 128, 8, 8):
                raise ValueError(f"Unexpected checkpoint latent shape {tuple(latent.shape)}")
            features.append(latent[:, -1].cpu().numpy().reshape(len(batch), -1))
        features = np.concatenate(features)
        endpoint_states = states[9:]
        rgb = np.clip(images[9:] * ENV_STD[None, :, None, None] + ENV_MEAN[None, :, None, None], 0, 1)
        rgbs = rgb.reshape(len(rgb), -1)
        metrics, pairs = local_pairs(features, rgbs, endpoint_states[:, :2])
        report_paths.append({"path_index": path_index,
                             "start_git_episode_index": path["start_git_episode_index"],
                             "source_frames": int(len(states)), "feature_endpoints": int(len(features)),
                             "max_boundary_step": path["max_boundary_step"],
                             **metrics})
        cached.append((features, endpoint_states, path_index))
        rng = np.random.default_rng(731 + path_index)
        ids = rng.choice(len(pairs[0]), size=min(300, len(pairs[0])), replace=False)
        plot_samples.append(tuple(values[ids] for values in pairs))
        print(f"Analyzed {path_index + 1}/{len(manifest['selected_paths'])} verified paths", flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "latents-and-states.npz",
                        latents=np.concatenate([item[0] for item in cached]),
                        states=np.concatenate([item[1] for item in cached]),
                        path_ids=np.concatenate([np.full(len(item[0]), item[2], dtype=np.int16) for item in cached]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    physical = np.concatenate([sample[0] for sample in plot_samples])
    for axis, index, title in zip(axes, (1, 2), ("Checkpoint latent", "Raw RGB pixels")):
        axis.scatter(physical, np.concatenate([sample[index] for sample in plot_samples]), s=3, alpha=.2)
        axis.set_xlabel("Physical distance (provisional coordinate units)")
        axis.set_ylabel("Feature Euclidean distance")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output / "local-distances.png", dpi=180)
    plt.close(fig)
    latent_scores = np.array([item["latent_vs_physical_spearman"] for item in report_paths])
    rgb_scores = np.array([item["rgb_vs_physical_spearman"] for item in report_paths])
    report = {"state": "complete", "protocol": "original_validation_contiguous_windows_v1",
              "scope": "Descriptive within-path local pairs; route identities and coordinate semantics not independently verified",
              "repo_commit": commit, "checkpoint_sha256": file_hash(args.checkpoint),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "paths": len(report_paths),
              "source_frames": int(sum(item["source_frames"] for item in report_paths)),
              "feature_endpoints": int(sum(item["feature_endpoints"] for item in report_paths)),
              "feature_context": "Final latent of a sliding 10-frame window; no frames after endpoint supplied",
              "median_path_spearman_latent": float(np.median(latent_scores)),
              "median_path_spearman_rgb": float(np.median(rgb_scores)),
              "paths_latent_exceeds_rgb": int(np.sum(latent_scores > rgb_scores)),
              "per_path": report_paths,
              "not_a_paper_reproduction": "No published log-distance fit, route-heldout probe, place-field test, or circular corridor"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
