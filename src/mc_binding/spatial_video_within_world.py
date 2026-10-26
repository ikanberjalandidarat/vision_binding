"""Within-world route retrieval from frozen Minecraft video features.

Only cached encoder activations are inputs to retrieval. Simulator poses and
headings are used afterward to score place identity and spatial relationships.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .io import atomic_json
from .spatial_video_dataset import load_diverse_v2


LAYERS = (3, 11, 23)
FEATURES = tuple(f"{condition}_layer{layer}"
                 for condition in ("video", "repeat_last_frame") for layer in LAYERS)


def heading_difference(a, b):
    """Pairwise smallest yaw difference in degrees."""
    return np.abs((np.asarray(a)[:, None] - np.asarray(b)[None, :] + 180) % 360 - 180)


def compare_world(keys, queries, key_position, query_position,
                  key_heading, query_heading):
    """Score route-1 queries against route-0 keys, without fitting a readout."""
    keys, queries = np.asarray(keys, float), np.asarray(queries, float)
    key_position, query_position = np.asarray(key_position, float), np.asarray(query_position, float)
    if (keys.ndim != 2 or queries.ndim != 2 or keys.shape[1] != queries.shape[1]
            or key_position.shape != (len(keys), 3)
            or query_position.shape != (len(queries), 3)
            or len(keys) < 2 or len(queries) < 1):
        raise ValueError("Expected route-0 and route-1 feature matrices with XYZ positions")
    if not np.isfinite(keys).all() or not np.isfinite(queries).all():
        raise ValueError("Nonfinite cached feature")
    distance = np.linalg.norm(query_position[:, None, :] - key_position[None, :, :], axis=-1)
    heading = heading_difference(query_heading, key_heading)
    oracle = distance.min(axis=1)
    eligible = oracle <= 2.0
    cross_view = ((distance <= 2.0) & (heading >= 90.0)).any(axis=1)

    # Centering uses only unlabeled route-0 vectors from this world. Cosine
    # retrieval has no pose labels, decoder fit, or validation-set tuning.
    center = keys.mean(axis=0)
    key_vector, query_vector = keys - center, queries - center
    key_vector /= np.maximum(np.linalg.norm(key_vector, axis=1, keepdims=True), 1e-12)
    query_vector /= np.maximum(np.linalg.norm(query_vector, axis=1, keepdims=True), 1e-12)
    similarity = query_vector @ key_vector.T
    chosen = np.argmax(similarity, axis=1)
    predicted = distance[np.arange(len(queries)), chosen]

    near = distance <= 2.0
    far = distance >= 6.0
    near_cross_view = near & (heading >= 90.0)
    near_same_view = near & (heading <= 45.0)
    far_same_view = far & (heading <= 45.0)

    def mean_or_none(values):
        return float(np.mean(values)) if len(values) else None

    def hit_rate(mask):
        return mean_or_none(predicted[mask] <= 2.0) if mask.sum() >= 3 else None

    return dict(
        key_clips=len(keys), query_clips=len(queries),
        eligible_queries=int(eligible.sum()), cross_view_queries=int(cross_view.sum()),
        near_pairs=int(near.sum()), far_pairs=int(far.sum()),
        near_cross_view_pairs=int(near_cross_view.sum()),
        near_same_view_pairs=int(near_same_view.sum()),
        far_same_view_pairs=int(far_same_view.sum()),
        mean_oracle_distance=mean_or_none(oracle),
        mean_retrieved_distance=mean_or_none(predicted),
        mean_excess_over_oracle=mean_or_none(predicted - oracle),
        mean_random_key_distance=mean_or_none(distance.mean(axis=1)),
        hit_at_2_blocks=hit_rate(eligible),
        random_key_hit_at_2_blocks=mean_or_none(near[eligible].mean(axis=1)) if eligible.sum() >= 3 else None,
        cross_view_hit_at_2_blocks=hit_rate(cross_view),
        near_far_cosine_margin=(float(similarity[near].mean() - similarity[far].mean())
                                if near.sum() >= 3 and far.sum() >= 3 else None),
        near_cross_view_cosine=(float(similarity[near_cross_view].mean())
                                if near_cross_view.sum() >= 3 else None),
        near_same_view_cosine=(float(similarity[near_same_view].mean())
                               if near_same_view.sum() >= 3 else None),
        cross_view_minus_same_view_cosine=(
            float(similarity[near_cross_view].mean() - similarity[near_same_view].mean())
            if near_cross_view.sum() >= 3 and near_same_view.sum() >= 3 else None),
        cross_view_near_vs_same_view_far_margin=(
            float(similarity[near_cross_view].mean() - similarity[far_same_view].mean())
            if near_cross_view.sum() >= 3 and far_same_view.sum() >= 3 else None),
    )


def analyze(extraction, dataset, output):
    extraction, dataset, output = Path(extraction), Path(dataset), Path(output)
    if json.loads((extraction / "status.json").read_text()) != {"state": "complete", "clips": 188}:
        raise ValueError("Require the complete 188-clip diverse-v2 extraction")
    manifest = json.loads((extraction / "manifest.json").read_text())
    entries, provenance = load_diverse_v2(dataset)
    if (manifest.get("protocol") != "diverse_environments_v2"
            or manifest.get("dataset_sha256") != provenance["dataset_sha256"]
            or manifest.get("split_protocol") != provenance["split_protocol"]
            or not manifest.get("raw_png_hashes_verified")):
        raise ValueError("Extraction and dataset provenance differ")
    rows = json.loads((extraction / "clips.json").read_text())
    if len(rows) != 188:
        raise ValueError("Missing extraction clip rows")

    poses = {}
    for entry in entries:
        for frame in entry["frames"]:
            pose = frame["pose"]
            poses[(entry["episode"], frame["elapsed_ticks"])] = [pose["x"], pose["y"], pose["z"]]
    by_world = {f"env-{i:02d}": {0: [], 1: []} for i in range(8)}
    for row in rows:
        world, route = row["geometry"], row["route"]
        if world not in by_world or route not in (0, 1):
            raise ValueError("Unexpected world or route in extraction")
        tick = row["ticks"][-1]
        position = poses[(row["episode"], tick)]
        if not np.allclose([position[0], position[2]], row["position"]):
            raise ValueError("Clip endpoint and source pose differ")
        cache = (extraction / row["cache"]).resolve()
        if extraction.resolve() not in cache.parents:
            raise ValueError("Cache path escapes extraction")
        with np.load(cache) as data:
            if not set(FEATURES).issubset(data.files):
                raise ValueError(f"Missing cached feature in {cache}")
            features = {name: np.asarray(data[name], dtype=float) for name in FEATURES}
        by_world[world][route].append(dict(position=position, heading=row["heading"], features=features))

    results = []
    for world, routes in by_world.items():
        key, query = routes[0], routes[1]
        if not key or not query:
            raise ValueError(f"Missing route features for {world}")
        kp, qp = [np.array([r["position"] for r in route]) for route in (key, query)]
        kh, qh = [np.array([r["heading"] for r in route]) for route in (key, query)]
        for name in FEATURES:
            k = np.stack([r["features"][name] for r in key])
            q = np.stack([r["features"][name] for r in query])
            result = compare_world(k, q, kp, qp, kh, qh)
            results.append(dict(world=world, representation=name, **result))
    report = dict(
        state="complete", question="Within-world same-place recognition and near/far spatial relationships",
        extraction=str(extraction), dataset_sha256=provenance["dataset_sha256"],
        route_protocol="Route 0 keys; route 1 queries; no pose used for feature retrieval",
        feature_metric="Cosine similarity after centering on unlabeled route-0 vectors",
        position_metric="3D Euclidean distance from source Minecraft XYZ; <=2 blocks = nearby",
        minimum_pairs_or_queries_to_report_conditional_metric=3,
        interpretation=("Exploratory; sparse route overlap, especially village; pairwise comparisons are correlated. "
                        "Route-0 reference vectors are an external retrieval set, not an internally maintained map."),
        results=results)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    print(output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.extraction, args.dataset, args.output)
