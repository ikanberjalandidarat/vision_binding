"""Frozen-video spatial pilot. No pose input, recurrent state or trained encoder.

Commands are deliberately separate: collect (real simulator), extract (GPU),
probe (CPU). A completed fixture test is not a Minecraft/model validation.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .io import atomic_json, source_hash, environment


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def partition(entry):
    if entry["split"] not in {"train", "validation", "test"}:
        raise ValueError("Unknown geometry split")
    return "heldout_route" if entry["split"] == "train" and entry["route"] >= 2 else entry["split"]


def clips(entries, length=16, hop=16, tick_step=8):
    """Nonoverlapping windows on an exact simulation clock, endpoint labels.

    Accept any starting clock phase. Extra heading-probe frames and duplicate
    timestamps are not resampled. Missing ticks cause rejection, not duplication.
    """
    if length < 2 or length % 2 or hop < length or tick_step < 1:
        raise ValueError("Need even clip length, hop >= length, positive tick step")
    seen = set()
    geometry_splits = {}
    for entry in entries:
        eid = entry["episode"]
        if eid in seen:
            raise ValueError("Duplicate episode")
        seen.add(eid)
        previous = geometry_splits.setdefault(entry["geometry"], entry["split"])
        if previous != entry["split"]:
            raise ValueError("Geometry leaks across train/validation/test")
        if not entry.get("is_minecraft") or not entry.get("route_complete"):
            raise ValueError("Only complete real Minecraft trajectories are eligible")
        frames = {}
        for frame in entry["frames"]:
            tick = frame["elapsed_ticks"]
            frames.setdefault(tick, frame)
        next_start = 0
        for start in sorted(frames):
            if start < next_start:
                continue
            ticks = list(range(start, start + length * tick_step, tick_step))
            if not all(t in frames for t in ticks):
                continue
            next_start = start + hop * tick_step
            selected = [frames[t] for t in ticks]
            pose = selected[-1]["pose"]
            yield dict(episode=eid, geometry=entry["geometry"], route=entry["route"],
                       split=partition(entry), ticks=ticks, frames=selected,
                       position=[pose["x"], pose["z"]], heading=pose["yaw"])


def collect_command(a):
    # Reuse only deterministic collection, never its Qwen encoder or GRU fit.
    from .hippocampal_study import collect
    from types import SimpleNamespace
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(dataset=a.dataset, render_python=a.render_python,
                               scenes=5, steps=a.steps, stride=4, cognitive_routes=True)
    atomic_json(out / "manifest.json", dict(experiment="frozen_video_spatial",
                settings=vars(settings), subject="RGB-only pretrained V-JEPA 2",
                source_hash=source_hash(), dataset_sha256=sha256(Path(a.dataset) / "manifest.json"),
                collection="Scripted loops with privileged pose for collection/scoring only"))
    atomic_json(out / "status.json", {"state": "running"})
    try:
        entries = collect(settings, out)
        if len(entries) != 20 or not all(e["route_complete"] for e in entries):
            raise ValueError("20 complete trajectories required; inspect route completion")
        atomic_json(out / "status.json", {"state": "complete", "trajectories": 20})
    except BaseException as exc:
        atomic_json(out / "status.json", {"state": "error", "error": repr(exc)})
        raise


def capture_blocks(encoder, layers, video):
    """Raw post-block residual channels, BEFORE final encoder LayerNorm."""
    import torch
    values, handles = {}, []
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(encoder.blocks) for i in layers):
        raise ValueError("Invalid block indices")
    try:
        for index in layers:
            def hook(module, args, output, index=index):
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise ValueError("Expected block output [batch,tokens,channels]")
                values[index] = output.detach().float().cpu()
            handles.append(encoder.blocks[index].register_forward_hook(hook))
        with torch.inference_mode():
            encoder(video)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != set(layers):
        raise ValueError("Not all block hooks executed")
    return values


def extract_command(a):
    import torch
    from PIL import Image
    root, repo = Path(a.dataset).resolve(), Path(a.repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean official V-JEPA checkout for provenance")
    entries = json.loads((root / "sequences.json").read_text())
    windows = list(clips(entries, a.frames, a.frames))
    if not windows:
        raise ValueError("No eligible video clips")
    if a.smoke:
        windows = windows[:1]
    layers = [int(x) for x in a.layers.split(",")]
    # Conservative bound: float32 pooled + optional full tokens, two conditions.
    estimated = len(windows) * len(layers) * 1024 * 4 * (2 + (a.frames // 2 * 256 if a.tokens else 0))
    if estimated > a.max_cache_gb * 1e9:
        raise ValueError(f"Cache estimate {estimated / 1e9:.2f} GB exceeds limit")
    if a.dry_run:
        print(json.dumps(dict(clips=len(windows), estimated_bytes=estimated, repo_commit=commit)))
        return
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "status.json", {"state": "running"})
    try:
        # Explicit local checkpoint avoids changing upstream download URLs or
        # silently falling back to random initialization.
        encoder, predictor = torch.hub.load(str(repo), "vjepa2_vit_large",
                                           source="local", pretrained=False)
        del predictor
        weights = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module.").removeprefix("backbone."): v
                 for k, v in weights["target_encoder"].items()}
        incompatible = encoder.load_state_dict(state, strict=False)
        if incompatible.missing_keys or set(incompatible.unexpected_keys) - {"pos_embed"}:
            raise ValueError(f"Checkpoint mismatch: {incompatible}")
        del weights, state
        encoder.eval().requires_grad_(False).to(a.device)
        transform = torch.hub.load(str(repo), "vjepa2_preprocessor",
                                   source="local", crop_size=256)
        atomic_json(out / "manifest.json", dict(
            model="V-JEPA 2 ViT-L/16", frozen=True, repo_commit=commit,
            checkpoint_sha256=sha256(a.checkpoint), sequences_sha256=sha256(root / "sequences.json"),
            clip_protocol="exact_spacing_any_phase_v2",
            layers_zero_based=layers, frames=a.frames, tick_step=8, label="last frame X,Z",
            extraction="raw block output; mean of spatial tokens in final temporal tubelet",
            conditions=["video", "repeat_last_frame"], tokens=a.tokens,
            expected_cache_bytes=estimated, parameters=sum(p.numel() for p in encoder.parameters()),
            torch_version=torch.__version__, pose_is_encoder_input=False))
        atomic_json(out / "environment.json", dict(environment=environment(),
                    source_hash=source_hash(), settings=vars(a)))
        rows = []
        for i, window in enumerate(windows):
            images = []
            for frame in window["frames"]:
                path = (root / frame["frame"]).resolve()
                if root not in path.parents or sha256(path) != frame["sha256"]:
                    raise ValueError("Frame path or hash mismatch")
                with Image.open(path) as image:
                    pixels = np.array(image.convert("RGB"))
                # Fixed HUD mask; no pose-dependent crop and no coordinates.
                pixels[int(.88 * len(pixels)):] = 127
                images.append(pixels)
            payload = {}
            for condition, buffer in [("video", images), ("repeat_last_frame", [images[-1]] * a.frames)]:
                video = transform(buffer)[0].unsqueeze(0).to(a.device)
                if tuple(video.shape) != (1, 3, a.frames, 256, 256):
                    raise ValueError(f"Unexpected preprocessing shape {video.shape}")
                values = capture_blocks(encoder, layers, video)
                for layer, value in values.items():
                    if tuple(value.shape) != (1, a.frames // 2 * 256, 1024):
                        raise ValueError(f"Unexpected ViT-L token shape {value.shape}")
                    grid = value.reshape(a.frames // 2, 16, 16, 1024)
                    payload[f"{condition}_layer{layer}"] = grid[-1].mean((0, 1)).numpy()
                    if a.tokens and condition == "video":
                        payload[f"tokens_layer{layer}"] = grid.numpy()
            np.savez_compressed(out / f"clip-{i:05d}.npz", **payload)
            rows.append({k: v for k, v in window.items() if k != "frames"})
            rows[-1]["cache"] = f"clip-{i:05d}.npz"
            atomic_json(out / "clips.json", rows)
            print(f"Extracted {i+1}/{len(windows)}", flush=True)
        atomic_json(out / "status.json", {"state": "smoke_complete" if a.smoke else "complete",
                                         "clips": len(rows)})
    except BaseException as exc:
        atomic_json(out / "status.json", {"state": "error", "error": repr(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--dataset", required=True)
    c.add_argument("--render-python", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--steps", type=int, default=3000)
    e = sub.add_parser("extract")
    for name in ("dataset", "repo", "checkpoint", "output"):
        e.add_argument("--" + name, required=True)
    e.add_argument("--frames", type=int, default=16)
    e.add_argument("--layers", default="3,11,23")
    e.add_argument("--device", default="cuda")
    e.add_argument("--tokens", action="store_true")
    e.add_argument("--max-cache-gb", type=float, default=2)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--smoke", action="store_true", help="One real clip; cannot be used for probe claims")
    a = p.parse_args()
    (collect_command if a.command == "collect" else extract_command)(a)


if __name__ == "__main__":
    main()
