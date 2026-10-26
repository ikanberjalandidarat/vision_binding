"""NON-MINECRAFT synthetic fixtures for the diverse-v2 adapter and hash gate."""

import json
from pathlib import Path

import pytest
from PIL import Image

from mc_binding.io import digest, file_hash
from mc_binding.spatial_video_dataset import load_diverse_v2


def fixture_dataset(root):
    for index in range(8):
        folder = root / f"env-{index:02d}"
        folder.mkdir()
        spec = dict(id=folder.name, routes=[[[0, 200, 0]], [[0, 200, 0]]])
        (folder / "status.json").write_text('{"state":"complete"}')
        (folder / "specification.json").write_text(json.dumps(spec))
        manifest = dict(protocol="diverse_environments_v2", spec_sha256=digest(spec),
                        seed=731, rgb=[448, 280], tick_ms=50, capture_every_ticks=8,
                        action_chunk_ticks=2, routes=spec["routes"],
                        source_hash="NON-MINECRAFT-synthetic-fixture")
        (folder / "manifest.json").write_text(json.dumps(manifest))
        routes = []
        for route in range(2):
            frames = []
            image_dir = folder / f"route-{route}"
            image_dir.mkdir()
            for step in range(10):
                path = image_dir / f"frame-{step * 8:05d}.png"
                Image.new("RGB", (448, 280), (index, route, step)).save(path)
                frames.append(dict(frame=str(path.relative_to(folder)), elapsed_ticks=step * 8,
                                   sha256=file_hash(path), pose=dict(x=0, y=200, z=0, yaw=0)))
            routes.append(dict(episode=f"NON-MINECRAFT-{index}-{route}",
                               is_minecraft=True, route=route, route_complete=True,
                               waypoints_reached=1, waypoints_total=1, frames=frames))
        (folder / "sequences.json").write_text(json.dumps(routes))
    return root


def test_adapter_splits_paths_and_hash_gate(tmp_path):
    """Synthetic RGB and labels exercise production checks; no Minecraft was run."""
    root = fixture_dataset(tmp_path)
    entries, provenance = load_diverse_v2(root, verify_images=True)
    assert len(entries) == 16 and provenance["image_count"] == 160
    assert provenance["raw_png_hashes_verified"]
    assert entries[0]["split"] == "train" and not entries[0]["heldout_route"]
    assert entries[1]["heldout_route"] and entries[1]["frames"][0]["elapsed_ticks"] == 72
    assert entries[1]["frames"][0]["frame"].startswith("env-00/route-1/")
    assert [entries[i * 2]["split"] for i in range(8)] == ["train"] * 4 + ["validation"] * 2 + ["test"] * 2

    target = root / "env-07" / "route-1" / "frame-00072.png"
    Image.new("RGB", (448, 280), "black").save(target)
    with pytest.raises(ValueError, match="PNG byte hash mismatch"):
        load_diverse_v2(root, verify_images=True)
