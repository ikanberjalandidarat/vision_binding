"""Audit diverse-dataset-v2 recording metadata without downloading raw RGB."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def cell(pose):
    return (math.floor(pose["x"]), math.floor(pose["z"]))


def heading_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def route_metrics(entry):
    frames = entry["frames"]
    ticks = [f["elapsed_ticks"] for f in frames]
    poses = [f["pose"] for f in frames]
    cells = {}
    for frame in frames:
        if frame["elapsed_ticks"] >= 72:  # exclude the scripted initial heading scan
            cells.setdefault(cell(frame["pose"]), []).append(frame["pose"]["yaw"])
    return {
        "frames": len(frames),
        "last_recorded_tick": ticks[-1],
        "end_tick": entry["end_tick"],
        "capture_intervals": sorted(set(b - a for a, b in zip(ticks, ticks[1:]))),
        "unique_image_hashes": len({f["sha256"] for f in frames}),
        "unique_visited_cells_after_scan": len(cells),
        "height_range": [min(p["y"] for p in poses), max(p["y"] for p in poses)],
        "waypoints_reached": entry["waypoints_reached"],
        "waypoints_total": entry["waypoints_total"],
    }, cells


def audit(root):
    results = []
    failures = []
    source_hashes = set()
    standard = (731, (448, 280), 50, 8, 2)
    for index in range(8):
        folder = root / f"env-{index:02d}"
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        spec = json.loads((folder / "specification.json").read_text())
        check = json.loads((folder / "build-check.json").read_text())
        entries = json.loads((folder / "sequences.json").read_text())
        source_hashes.add(manifest["source_hash"])
        errors = []
        if status.get("state") != "complete":
            errors.append("status is not complete")
        if spec["id"] != folder.name or manifest["spec_sha256"] != sha256_json(spec):
            errors.append("specification ID or digest mismatch")
        settings = (
            manifest["seed"], tuple(manifest["rgb"]), manifest["tick_ms"],
            manifest["capture_every_ticks"], manifest["action_chunk_ticks"],
        )
        if settings != standard or manifest["routes"] != spec["routes"]:
            errors.append("nonstandard manifest or route specification")
        if len(entries) != 2 or [e["route"] for e in entries] != [0, 1]:
            errors.append("expected two ordered routes")
        expected = check["expected"]
        actual = check["actual"]
        if math.dist(expected, [actual[k] for k in ("x", "y", "z")]) > 0.1:
            errors.append("build start pose mismatch")
        route_rows, route_cells = [], []
        for route, entry in enumerate(entries):
            metrics, cells = route_metrics(entry)
            route_rows.append(metrics)
            route_cells.append(cells)
            frames = entry["frames"]
            if not entry["is_minecraft"] or not entry["route_complete"]:
                errors.append(f"route {route}: incomplete or not labeled Minecraft")
            if entry["waypoints_reached"] != entry["waypoints_total"] or entry["waypoints_total"] != len(spec["routes"][route]):
                errors.append(f"route {route}: waypoint count mismatch")
            if metrics["capture_intervals"] != [8] or frames[0]["elapsed_ticks"] != 0:
                errors.append(f"route {route}: irregular capture clock")
            if not 0 <= entry["end_tick"] - metrics["last_recorded_tick"] < 8:
                errors.append(f"route {route}: end tick is inconsistent")
            if metrics["unique_image_hashes"] != len(frames):
                errors.append(f"route {route}: repeated exact image hashes")
            for k, frame in enumerate(frames):
                pose = frame["pose"]
                if any(not math.isfinite(pose[field]) for field in ("x", "y", "z", "yaw", "pitch")):
                    errors.append(f"route {route}: nonfinite pose at frame {k}")
                    break
                if not (-15 < pose["x"] < 15 and -15 < pose["z"] < 15 and 199.5 < pose["y"] < 211):
                    errors.append(f"route {route}: out-of-bounds pose at frame {k}")
                    break
                if len(frame["actions_since_previous"]) != (0 if k == 0 else 4):
                    errors.append(f"route {route}: action-clock mismatch at frame {k}")
                    break
                if len(frame["sha256"]) != 64 or Path(frame["frame"]).is_absolute():
                    errors.append(f"route {route}: malformed frame reference at frame {k}")
                    break
        first, second = (set(c) for c in route_cells)
        shared = first & second
        union = first | second
        heading_differences = [
            min(heading_distance(a, b) for a in route_cells[0][spot] for b in route_cells[1][spot])
            for spot in shared
        ]
        row = {
            "id": folder.name,
            "name": spec["name"],
            "source_hash": manifest["source_hash"],
            "recording_checks_passed": not errors,
            "errors": errors,
            "routes": route_rows,
            "shared_visited_cells": len(shared),
            "route_cell_jaccard": len(shared) / len(union),
            "shared_cells_with_90deg_view_change_fraction": sum(d >= 90 for d in heading_differences) / len(shared),
            "cross_route_identical_image_hashes": len(
                {f["sha256"] for f in entries[0]["frames"]}
                & {f["sha256"] for f in entries[1]["frames"]}
            ),
        }
        results.append(row)
        failures.extend(f"{folder.name}: {message}" for message in errors)
    return {
        "scope": "metadata only; raw PNG bytes and rendered visual quality are not verified here",
        "environment_count": len(results),
        "route_count": sum(len(row["routes"]) for row in results),
        "recording_metadata_checks_passed": not failures,
        "failures": failures,
        "source_hashes": sorted(source_hashes),
        "environments": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    if not report["recording_metadata_checks_passed"]:
        raise SystemExit(1)
