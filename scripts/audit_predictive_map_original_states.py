#!/usr/bin/env python3
"""Read-only coverage audit of the authors' original validation episode states."""

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

from run_predictive_map_reference import REFERENCE_COMMIT


def git_output(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main(args):
    repo = args.repo.resolve()
    commit = git_output(repo, "rev-parse", "HEAD").decode().strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError(f"Expected pinned reference commit {REFERENCE_COMMIT}; got {commit}")
    if git_output(repo, "status", "--porcelain").strip():
        raise ValueError("Official checkout has local changes")
    names = git_output(repo, "ls-tree", "-r", "--name-only", "HEAD", "datasets/val-dataset").decode().splitlines()
    episodes = sorted(str(Path(name).parent) for name in names if name.endswith("/state.npz"))
    if len(episodes) < args.episodes:
        raise ValueError(f"Only {len(episodes)} episodes available")
    indices = np.unique(np.concatenate((np.arange(min(8, len(episodes))),
                                        np.linspace(0, len(episodes) - 1, args.episodes, dtype=int))))
    states, selected = [], []
    for index in indices:
        episode = episodes[int(index)]
        blob = git_output(repo, "show", f"HEAD:{episode}/state.npz")
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            state = np.asarray(data["arr_0"], dtype=np.float32)
        if state.shape != (10, 3) or not np.isfinite(state).all():
            raise ValueError(f"Unexpected state in {episode}: {state.shape}")
        states.append(state)
        selected.append({"episode": episode, "git_episode_index": int(index),
                         "state_sha256": hashlib.sha256(blob).hexdigest(),
                         "first": state[0].tolist(), "last": state[-1].tolist()})
        if len(states) % 32 == 0:
            print(f"Checked {len(states)}/{len(indices)} state files", flush=True)
    array = np.stack(states)
    coords = array[:, :, :2]
    all_coords = coords.reshape(-1, 2)
    steps = np.linalg.norm(np.diff(coords, axis=1), axis=2)
    adjacent = np.flatnonzero(np.diff(indices) == 1)
    boundary_steps = np.linalg.norm(coords[adjacent + 1, 0] - coords[adjacent, -1], axis=1)
    extent = np.ptp(all_coords, axis=0)
    occupancy = {}
    for bins in (8, 16):
        edges = [np.linspace(all_coords[:, axis].min(), all_coords[:, axis].max() + 1e-6, bins + 1)
                 for axis in range(2)]
        occupancy[str(bins)] = {"occupied_bins": int(np.count_nonzero(np.histogram2d(
            all_coords[:, 0], all_coords[:, 1], bins=edges)[0])), "total_bins": bins * bins}
    # Across-episode revisit counts are descriptive: route identities and angle
    # units have not yet been verified from the recording code.
    endpoints = array[:, -1, :]
    delta = endpoints[:, None, :2] - endpoints[None, :, :2]
    distance = np.linalg.norm(delta, axis=2)
    upper = np.triu(np.ones(distance.shape, dtype=bool), 1)
    near = upper & (distance <= 2)
    report = {"state": "complete", "scope": "Dispersed original validation state.npz files; column meanings provisional",
              "repo_commit": commit, "total_episodes_in_git": len(episodes),
              "sampled_episodes": len(selected), "rows": int(array.shape[0] * array.shape[1]),
              "coordinate_1_range": [float(all_coords[:, 0].min()), float(all_coords[:, 0].max())],
              "coordinate_2_range": [float(all_coords[:, 1].min()), float(all_coords[:, 1].max())],
              "coordinate_extent": extent.tolist(),
              "within_episode_step_distance": {"median": float(np.median(steps)),
                                               "p95": float(np.percentile(steps, 95)),
                                               "max": float(steps.max())},
              "adjacent_episode_boundaries": {"count": int(len(boundary_steps)),
                                              "median_step": float(np.median(boundary_steps)) if len(boundary_steps) else None,
                                              "max_step": float(boundary_steps.max()) if len(boundary_steps) else None},
              "occupancy": occupancy,
              "endpoint_pairs_within_2_coordinate_units": int(near.sum()),
              "column_3_range": [float(array[:, :, 2].min()), float(array[:, :, 2].max())],
              "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")
    main(args)
