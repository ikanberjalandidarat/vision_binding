"""CPU-only, trajectory-heldout probes of frozen video features."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import atomic_json


def ridge_predict(x, y, target, alpha):
    """All centering/scaling is fitted on the training partition only."""
    mean, scale = x.mean(0), x.std(0).clip(1e-6)
    x, target = (x - mean) / scale, (target - mean) / scale
    center = y.mean(0)
    if len(x) < x.shape[1]:
        weights = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y - center)
    else:
        weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - center))
    return target @ weights + center


def occupancy_map(position, values, edges, min_count=3):
    """Signed channel means, not biological firing rates; NaN = unsupported."""
    counts = np.histogram2d(*position.T, bins=edges)[0]
    totals = np.histogram2d(*position.T, bins=edges, weights=values)[0]
    means = np.full_like(totals, np.nan)
    np.divide(totals, counts, out=means, where=counts >= min_count)
    return means, counts


def circular_labels(y, episodes, seed):
    """Preserve within-trajectory smoothness; this is a diagnostic, not a p-value."""
    out = y.copy()
    rng = np.random.default_rng(seed)
    for eid in sorted(set(episodes)):
        ix = np.flatnonzero(episodes == eid)
        if len(ix) < 4:
            raise ValueError("Too few clips for circular-shift control")
        offset = int(rng.integers(max(1, len(ix)//4), max(2, 3*len(ix)//4)))
        out[ix] = np.roll(y[ix], offset, axis=0)
    return out


def analyze(root):
    root = Path(root)
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Extraction must finish before analysis")
    rows = json.loads((root / "clips.json").read_text())
    split = np.array([r["split"] for r in rows])
    episodes = np.array([r["episode"] for r in rows])
    for eid in set(episodes):
        if len(set(split[episodes == eid])) != 1:
            raise ValueError("Episode leakage across partitions")
    for geometry in {r["geometry"] for r in rows}:
        parts = {("train" if r["split"] == "heldout_route" else r["split"])
                 for r in rows if r["geometry"] == geometry}
        if len(parts) != 1:
            raise ValueError("Geometry leakage across partitions")
    tr, va = split == "train", split == "validation"
    if min(tr.sum(), va.sum()) < 4:
        raise ValueError("Insufficient train/validation clips")
    y = np.array([r["position"] for r in rows], dtype=float)
    heading = np.deg2rad([r["heading"] for r in rows])
    elapsed = np.array([r["ticks"][-1] for r in rows], dtype=float) / 20.
    nuisance = np.column_stack([np.sin(heading), np.cos(heading), elapsed, elapsed**2])
    with np.load(root / rows[0]["cache"]) as cache:
        keys = sorted(k for k in cache.files if not k.startswith("tokens_"))
    features = {k: [] for k in keys}
    for row in rows:
        with np.load(root / row["cache"]) as cache:
            for key in keys:
                features[key].append(cache[key])
    features = {k: np.asarray(v, dtype=float) for k, v in features.items()}
    features["heading_and_elapsed_time"] = nuisance
    summary = []
    predictions = {}
    shuffled = circular_labels(y[tr], episodes[tr], 731)
    for name, x in features.items():
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite activation")
        for control in ("true_labels", "circular_shift_training_labels"):
            train_y = y[tr] if control == "true_labels" else shuffled
            alphas = (.1, 1., 10., 100., 1000.)
            errors = [np.linalg.norm(ridge_predict(x[tr], train_y, x[va], a)-y[va], axis=1).mean()
                      for a in alphas]
            alpha = alphas[int(np.argmin(errors))]
            pred = ridge_predict(x[tr], train_y, x, alpha)
            predictions[name + "/" + control] = pred
            for part in ("validation", "heldout_route", "test"):
                mask = split == part
                for eid in sorted(set(episodes[mask])):
                    selected = mask & (episodes == eid)
                    error = np.linalg.norm(pred[selected] - y[selected], axis=1)
                    baseline = np.linalg.norm(y[selected] - y[tr].mean(0), axis=1)
                    summary.append(dict(representation=name, control=control, split=part,
                        episode=eid, clips=int(selected.sum()), alpha=alpha,
                        mean_error_blocks=float(error.mean()), median_error_blocks=float(np.median(error)),
                        training_mean_baseline_blocks=float(baseline.mean())))
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    atomic_json(out / "probe-results.json", summary)
    np.savez_compressed(out / "predictions.npz", **predictions)
    # Native channel choices made exclusively using train variance. Identical
    # bins for train and heldout routes; unsupported bins remain missing.
    maps, selected_units = {}, {}
    edges = (np.arange(-8., 9., 1.), np.arange(0., 13., 1.))
    for name, x in features.items():
        if not name.startswith("video_"):
            continue
        units = np.argsort(x[tr].var(0))[-6:].tolist()
        selected_units[name] = units
        for geometry in sorted(set(r["geometry"] for r in rows if r["split"] == "train")):
            gm = np.array([r["geometry"] == geometry for r in rows])
            for part in ("train", "heldout_route"):
                mask = gm & (split == part)
                for unit in units:
                    field, counts = occupancy_map(y[mask], x[mask, unit], edges)
                    prefix = f"{name}/{geometry}/{part}/unit{unit}"
                    maps[prefix] = field
                    maps[prefix + "/occupancy"] = counts
    np.savez_compressed(out / "signed-channel-maps.npz", **maps)
    atomic_json(out / "review.json", dict(
        subject="Frozen pretrained video encoder; no added memory module",
        results=summary, selected_native_channels=selected_units,
        primary="Episode-level heldout-route error in blocks; report across episodes, not IID frames",
        controls="Repeated final frame; heading/time; train-mean position; circular-shift labels",
        limitations=["20 trajectories validate pipeline, not cell emergence.",
            "Loop paths do not densely cover the arena. Gridness is deliberately not scored.",
            "Signed activation maps are not nonnegative firing rates.",
            "Route-heldout decoding alone does not remove heading/appearance confounds.",
            "One circular-shift control is not a significance test.",
            "No causal-use or learned-navigation claim."]))
    print(out / "review.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction")
    analyze(parser.parse_args().extraction)
