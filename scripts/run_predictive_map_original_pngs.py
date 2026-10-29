#!/usr/bin/env python3
"""Audit the released predictive checkpoint on the authors' original validation PNGs.

Reads PNG blobs through Git so Oscar's sparse checkout need not materialize the
82,630-image validation set. No notebook or official source file is modified.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_predictive_map_reference import ENV_MEAN, ENV_STD, REFERENCE_COMMIT, file_hash


def git_output(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def episode_paths(repo, limit):
    paths = git_output(repo, "ls-tree", "-r", "--name-only", "HEAD", "datasets/val-dataset").decode().splitlines()
    episodes = {}
    for path in paths:
        p = Path(path)
        if p.suffix == ".png" and p.stem.isdigit():
            episodes.setdefault(str(p.parent), {})[int(p.stem)] = path
    complete = [(episode, [frames[i] for i in range(10)])
                for episode, frames in sorted(episodes.items()) if all(i in frames for i in range(10))]
    if len(complete) < limit:
        raise ValueError(f"Only {len(complete)} complete validation episodes available")
    return complete[:limit]


def load_episode(repo, paths):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ENV_MEAN.tolist(), ENV_STD.tolist()),
    ])
    frames, hashes = [], []
    for path in paths:
        blob = git_output(repo, "show", f"HEAD:{path}")
        hashes.append(hashlib.sha256(blob).hexdigest())
        with Image.open(io.BytesIO(blob)) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError(f"Unexpected original PNG mode/size: {path}, {image.mode}, {image.size}")
            frames.append(transform(image).numpy())
    return np.stack(frames), hashes


def run(args):
    repo = args.repo.resolve()
    if git_output(repo, "rev-parse", "HEAD").decode().strip() != REFERENCE_COMMIT:
        raise ValueError("Official checkout is not at the pinned reference commit")
    if git_output(repo, "status", "--porcelain").strip():
        raise ValueError("Official checkout has local changes")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = episode_paths(repo, args.episodes)
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(repo))
    from predictive_coding.models.encoder_decoder import PredictiveCoder

    model = PredictiveCoder(3, 3, [2, 2, 2, 2], seq_len=20)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(args.device)
    records = []
    model_sse = copy_sse = rgb_model_sse = rgb_copy_sse = 0.0
    pixel_count = 0
    example = None
    for episode, paths in selected:
        frames, hashes = load_episode(repo, paths)
        batch = torch.from_numpy(frames[None]).float().to(args.device)
        with torch.inference_mode():
            prediction = model(batch)[:, :-1]
            target = batch[:, 1:]
            previous = batch[:, :-1]
            model_error = float(((prediction - target) ** 2).sum().item())
            copy_error = float(((previous - target) ** 2).sum().item())
            if example is None:
                example = tuple(v[0, 4].detach().cpu().numpy() for v in (previous, prediction, target))
            count = target.numel()
            rgb_scale = torch.as_tensor(ENV_STD, device=args.device).reshape(1, 1, 3, 1, 1)
            rgb_model_sse += float((((prediction - target) * rgb_scale) ** 2).sum().item())
            rgb_copy_sse += float((((previous - target) * rgb_scale) ** 2).sum().item())
        model_sse += model_error
        copy_sse += copy_error
        pixel_count += count
        records.append({"episode": episode, "png_sha256": hashes,
                        "next_image_mse_normalized": model_error / count,
                        "copy_previous_mse_normalized": copy_error / count})
        print(f"Checked {len(records)}/{len(selected)} original episodes", flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, frame, title in zip(axes, example, ("Previous", "Predicted next", "Actual next")):
        rgb = np.clip(frame * ENV_STD[:, None, None] + ENV_MEAN[:, None, None], 0, 1)
        axis.imshow(rgb.transpose(1, 2, 0))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "image-example.png", dpi=180)
    plt.close(fig)
    report = {"state": "complete", "scope": "Original Git-tracked validation PNGs; first 10 frames per selected episode",
              "repo_commit": REFERENCE_COMMIT, "checkpoint_sha256": file_hash(checkpoint),
              "episodes": len(selected), "predicted_frame_pairs": 9 * len(selected),
              "next_image_mse_normalized": model_sse / pixel_count,
              "copy_previous_mse_normalized": copy_sse / pixel_count,
              "next_image_mse_rgb_0to1": rgb_model_sse / pixel_count,
              "copy_previous_mse_rgb_0to1": rgb_copy_sse / pixel_count,
              "per_episode": records}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.episodes < 1:
        p.error("--episodes must be positive")
    run(a)
