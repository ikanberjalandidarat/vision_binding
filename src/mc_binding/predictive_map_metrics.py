"""Shared, label-only spatial metrics for reference and frozen-video features.

Inputs are already extracted activations. Minecraft poses never enter a model.
Synthetic examples in tests are NON-MINECRAFT fixtures.
"""

import numpy as np


def flatten_features(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim < 2 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Need at least three finite feature rows")
    return values.reshape(len(values), -1)


def circular_positions(positions, circumference=30.0):
    """Embed the released circular corridor's wrapped second coordinate."""
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("Need at least two position coordinates")
    angle = 2 * np.pi * p[:, 1] / circumference
    radius = circumference / (2 * np.pi)
    return np.column_stack((p[:, 0], radius * np.cos(angle), radius * np.sin(angle)))


def pair_distances(features, positions, max_pairs=20000, seed=731, circular_period=None):
    """Independent non-identical pairs; deterministic cap avoids quadratic memory."""
    x = flatten_features(features)
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or len(p) != len(x) or not np.isfinite(p).all():
        raise ValueError("Features and positions must align and be finite")
    n = len(x)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(seed)
        pairs = set()
        while len(pairs) < max_pairs:
            a = rng.integers(0, n, size=min(max_pairs, 2 * (max_pairs - len(pairs))))
            b = rng.integers(0, n, size=len(a))
            pairs.update((int(min(u, v)), int(max(u, v))) for u, v in zip(a, b) if u != v)
        i, j = np.asarray(sorted(pairs), dtype=np.int64).T
    delta = np.abs(p[i] - p[j])
    if circular_period is not None:
        if circular_period <= 0 or p.shape[1] < 2:
            raise ValueError("Circular corridor requires positive period and two coordinates")
        wrapped = np.mod(delta[:, 1], circular_period)
        delta[:, 1] = np.minimum(wrapped, circular_period - wrapped)
    return np.linalg.norm(x[i] - x[j], axis=1), np.linalg.norm(delta, axis=1)


def _ranks(values):
    """Average ranks for ties, including repeated physical positions."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1, len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def distance_summary(latent, physical):
    latent, physical = np.asarray(latent), np.asarray(physical)
    if len(latent) != len(physical) or len(latent) < 3:
        raise ValueError("Need at least three aligned distance pairs")
    a, b = _ranks(latent), _ranks(physical)
    score = None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    return {"pairs": len(a), "spearman_latent_vs_physical": score,
            "mean_latent_distance": float(np.mean(latent)),
            "mean_physical_distance": float(np.mean(physical))}


def ridge_position_decode(train_features, train_positions, test_features, test_positions, alpha=100.0):
    """Fit only on train rows; report Euclidean error on separate test rows."""
    train, test = flatten_features(train_features), flatten_features(test_features)
    y, truth = np.asarray(train_positions, float), np.asarray(test_positions, float)
    if train.shape[1] != test.shape[1] or y.ndim != 2 or truth.ndim != 2 or y.shape[1] != truth.shape[1] or len(y) != len(train) or len(truth) != len(test):
        raise ValueError("Incompatible feature/position matrices")
    mean_x, mean_y = train.mean(0), y.mean(0)
    centered = train - mean_x
    # Dual form scales with observations, not the 8192-channel reference latent.
    kernel = centered @ centered.T
    weights = centered.T @ np.linalg.solve(kernel + alpha * np.eye(len(train)), y - mean_y)
    pred = (test - mean_x) @ weights + mean_y
    baseline = np.linalg.norm(truth - mean_y, axis=1)
    errors = np.linalg.norm(pred - truth, axis=1)
    return {"train_rows": len(train), "test_rows": len(test),
            "mean_error": float(errors.mean()), "median_error": float(np.median(errors)),
            "train_mean_baseline_error": float(baseline.mean())}, pred
