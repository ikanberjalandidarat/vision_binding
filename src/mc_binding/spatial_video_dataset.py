"""Versioned adapter and real-frame gate for the eight-world Minecraft dataset."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .io import atomic_json, digest, file_hash


# One world per category. Environment-heldout scores are exploratory, not
# within-category generalization. Route 1 in training worlds is also held out.
SPLITS = {0: "train", 1: "train", 2: "train", 3: "train",
          4: "validation", 5: "validation", 6: "test", 7: "test"}
SPLIT_PROTOCOL = "diverse_v2_env_0to3_train_4to5_validation_6to7_test_route1_heldout"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_diverse_v2(root, verify_images=False):
    """Return model-ready entries and a fingerprint without changing the data."""
    root = Path(root).resolve()
    entries, sources, provenance, image_count = [], set(), [], 0
    for index, split in SPLITS.items():
        folder = root / f"env-{index:02d}"
        status_path = folder / "status.json"
        spec_path = folder / "specification.json"
        manifest_path = folder / "manifest.json"
        sequences_path = folder / "sequences.json"
        status = json.loads(status_path.read_text())
        spec = json.loads(spec_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        routes = json.loads(sequences_path.read_text())
        if status.get("state") != "complete" or spec["id"] != folder.name:
            raise ValueError(f"{folder}: incomplete or wrong environment ID")
        if manifest.get("protocol") != "diverse_environments_v2" or manifest["spec_sha256"] != digest(spec):
            raise ValueError(f"{folder}: protocol or specification hash mismatch")
        if (manifest["seed"], manifest["rgb"], manifest["tick_ms"],
                manifest["capture_every_ticks"], manifest["action_chunk_ticks"]) != (731, [448, 280], 50, 8, 2):
            raise ValueError(f"{folder}: incompatible capture settings")
        if manifest["routes"] != spec["routes"] or len(routes) != 2:
            raise ValueError(f"{folder}: routes differ from the recorded specification")
        sources.add(manifest["source_hash"])
        provenance.append(dict(environment=folder.name, status_sha256=_sha(status_path),
                               specification_sha256=_sha(spec_path), manifest_sha256=_sha(manifest_path),
                               sequences_sha256=_sha(sequences_path)))
        for route, original in enumerate(routes):
            if (original.get("route") != route or not original.get("is_minecraft")
                    or not original.get("route_complete")
                    or original["waypoints_reached"] != original["waypoints_total"]
                    or original["waypoints_total"] != len(spec["routes"][route])):
                raise ValueError(f"{folder}: route {route} incomplete or mislabeled")
            frames = []
            for frame_number, frame in enumerate(original["frames"]):
                relative = Path(frame["frame"])
                if (relative.is_absolute() or ".." in relative.parts or
                        relative.parts[:1] != (f"route-{route}",) or
                        frame["elapsed_ticks"] != frame_number * 8):
                    raise ValueError(f"{folder}: route {route} invalid frame path or clock")
                image_path = folder / relative
                if verify_images:
                    if file_hash(image_path) != frame["sha256"]:
                        raise ValueError(f"{image_path}: PNG byte hash mismatch")
                    with Image.open(image_path) as image:
                        if image.format != "PNG" or image.mode != "RGB" or image.size != (448, 280):
                            raise ValueError(f"{image_path}: unexpected image format, mode or size")
                if frame["elapsed_ticks"] >= 72:
                    frames.append({**frame, "frame": str(Path(folder.name) / relative)})
                image_count += 1
            entries.append(dict(episode=original["episode"], geometry=folder.name,
                                route=route, split=split,
                                heldout_route=(split == "train" and route == 1),
                                is_minecraft=True, route_complete=True, frames=frames))
    return entries, dict(
        protocol="diverse_environments_v2", split_protocol=SPLIT_PROTOCOL,
        split_by_environment={f"env-{i:02d}": SPLITS[i] for i in range(8)},
        source_hashes=sorted(sources), image_count=image_count,
        model_frames_from_tick=72,
        dataset_sha256=digest(provenance), raw_png_hashes_verified=verify_images)


def verify(root, output):
    from .spatial_video import clips
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"state": "running"})
    try:
        entries, provenance = load_diverse_v2(root, verify_images=True)
        windows = list(clips(entries))
        counts = {}
        for window in windows:
            counts[window["split"]] = counts.get(window["split"], 0) + 1
        if any(counts.get(part, 0) < 4 for part in ("train", "heldout_route", "validation", "test")):
            raise ValueError(f"Insufficient nonoverlapping 16-frame clips: {counts}")
        report = {**provenance, "state": "complete", "episodes": len(entries),
                  "clips_by_split": counts,
                  "scope": "All source PNG bytes matched recorded SHA256, RGB mode and dimensions; model not run"}
        atomic_json(output / "report.json", report)
        atomic_json(output / "status.json", {"state": "complete"})
        print(output / "report.json")
    except BaseException as exc:
        atomic_json(output / "status.json", {"state": "error", "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.dataset, args.output)
