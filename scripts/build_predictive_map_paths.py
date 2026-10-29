#!/usr/bin/env python3
"""Select long, position-continuous paths from the authors' validation episodes.

This is a read-only manifest builder. The first two state columns are treated
as provisional planar coordinates; no route identity is inferred.
"""

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


def read_state(repo, episode):
    blob = git_output(repo, "show", f"HEAD:{episode}/state.npz")
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        state = np.asarray(archive["arr_0"], dtype=np.float32)
    if state.shape != (10, 3) or not np.isfinite(state).all():
        raise ValueError(f"Invalid state for {episode}: {state.shape}")
    return state, hashlib.sha256(blob).hexdigest()


def build(repo, windows, episodes_per_window, max_boundary_step):
    revision = git_output(repo, "rev-parse", "HEAD").decode().strip()
    if revision != REFERENCE_COMMIT or git_output(repo, "status", "--porcelain").strip():
        raise ValueError("Official checkout must be clean and at the pinned reference commit")
    names = git_output(repo, "ls-tree", "-r", "--name-only", "HEAD", "datasets/val-dataset").decode().splitlines()
    episodes = sorted(str(Path(name).parent) for name in names if name.endswith("/state.npz"))
    if len(episodes) < windows * episodes_per_window:
        raise ValueError("Insufficient validation episodes")
    # Spread candidate windows across the recording and avoid overlapping them.
    starts = np.unique(np.linspace(0, len(episodes) - episodes_per_window,
                                   windows, dtype=int))
    selected = []
    rejected = []
    for start in starts:
        group = episodes[int(start):int(start) + episodes_per_window]
        records = [read_state(repo, episode) for episode in group]
        states = np.stack([item[0] for item in records])
        boundaries = np.linalg.norm(states[1:, 0, :2] - states[:-1, -1, :2], axis=1)
        within = np.linalg.norm(np.diff(states[:, :, :2], axis=1), axis=2)
        extent = np.ptp(states[:, :, :2].reshape(-1, 2), axis=0)
        entry = {"start_git_episode_index": int(start), "episodes": group,
                 "state_sha256": [item[1] for item in records],
                 "max_boundary_step": float(boundaries.max()),
                 "max_within_episode_step": float(within.max()),
                 "median_boundary_step": float(np.median(boundaries)),
                 "path_extent": extent.tolist()}
        if np.any(boundaries > max_boundary_step) or np.any(within > max_boundary_step) or np.linalg.norm(extent) < 3:
            rejected.append(entry)
        else:
            selected.append(entry)
        print(f"Checked {len(selected) + len(rejected)}/{len(starts)} candidate paths", flush=True)
    if len(selected) < 4:
        raise RuntimeError(f"Only {len(selected)} continuous paths; no spatial run should proceed")
    return {"state": "complete", "protocol": "original_validation_contiguous_windows_v1",
            "repo_commit": revision, "episode_length": 10,
            "episodes_per_path": episodes_per_window,
            "max_boundary_step_allowed": max_boundary_step,
            "selected_paths": selected, "rejected_paths": rejected,
            "route_identity_known": False,
            "coordinate_columns": "first two state values, provisional pending recording-code check"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", type=int, default=24)
    p.add_argument("--episodes-per-window", type=int, default=10)
    p.add_argument("--max-boundary-step", type=float, default=1.5)
    a = p.parse_args()
    if a.windows < 4 or a.episodes_per_window < 2 or a.max_boundary_step <= 0:
        p.error("Invalid path selection parameters")
    report = build(a.repo.resolve(), a.windows, a.episodes_per_window, a.max_boundary_step)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(a.output)
