#!/usr/bin/env python3
"""Test whether the official checkpoint's earlier outputs depend on later frames.

Only a copied input tensor is perturbed. The official model, dataset, and
notebooks are never edited. This is a model-forward diagnostic, not training.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_predictive_map_original_pngs import load_episode
from run_predictive_map_reference import REFERENCE_COMMIT, file_hash


def difference(a, b):
    delta = (a - b).abs()
    return {"max_absolute": float(delta.max().item()),
            "mean_absolute": float(delta.mean().item())}


def run(args):
    import torch

    repo = args.repo.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError("Expected pinned official reference commit")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official checkout has local changes")
    names = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only",
                                     "HEAD", "datasets/val-dataset"], text=True).splitlines()
    episodes = sorted({str(Path(name).parent) for name in names if name.endswith("/state.npz")})
    if len(episodes) < 8:
        raise ValueError("Too few original validation episodes")
    original, _ = load_episode(repo, [f"{episodes[0]}/{i}.png" for i in range(10)])
    alternate, _ = load_episode(repo, [f"{episodes[7]}/{i}.png" for i in range(10)])
    original = torch.from_numpy(original[None]).float().to(args.device)
    alternate = torch.from_numpy(alternate[None]).float().to(args.device)
    sys.path.insert(0, str(repo))
    from predictive_coding.models.encoder_decoder import PredictiveCoder

    model = PredictiveCoder(3, 3, [2, 2, 2, 2], seq_len=20)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(args.device)
    attention_heads = {name: module.num_heads for name, module in model.named_modules()
                       if hasattr(module, "num_heads")}
    checks = {}
    with torch.inference_mode():
        base_prediction = model(original)[:, 4].clone()
        base_latent = model.get_latents(original)[:, 4].clone()
        repeated_prediction = model(original)[:, 4]
        repeated_latent = model.get_latents(original)[:, 4]
        checks["identical_input_repeat"] = {
            "prediction": difference(base_prediction, repeated_prediction),
            "latent": difference(base_latent, repeated_latent)}
        for changed_frame in (5, 9):
            altered = original.clone()
            altered[:, changed_frame] = alternate[:, changed_frame]
            changed_prediction = model(altered)[:, 4]
            changed_latent = model.get_latents(altered)[:, 4]
            checks[f"replace_future_frame_{changed_frame}"] = {
                "input_change_mean_absolute": float((original[:, changed_frame] - alternate[:, changed_frame]).abs().mean().item()),
                "prediction_at_frame_4": difference(base_prediction, changed_prediction),
                "latent_at_frame_4": difference(base_latent, changed_latent)}
    report = {"state": "complete", "scope": "Same-length ten-frame original validation clip; change only a future frame",
              "repo_commit": commit, "checkpoint_sha256": file_hash(args.checkpoint),
              "question": "Does output at frame 4 change when only frame 5 (its target) or frame 9 changes?",
              "source_observation": "Official encoder_decoder.py applies F.layer_norm(f, f.shape[1:]) over the temporal dimension",
              "attention_heads_in_loaded_model": attention_heads,
              "episodes": [episodes[0], episodes[7]], "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
