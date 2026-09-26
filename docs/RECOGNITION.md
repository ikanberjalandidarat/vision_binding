# From V3 to recognition calibration

V3 established visible Minecraft structures and correct feet-position/yaw telemetry. It did not establish clean model inputs or recognition accuracy. This workflow adds a small fixed-pose pilot, not an independent-world benchmark. No OpenAI API or API key is used: inference uses Qwen on Oscar.

## 1. Capture on Oscar

Commit/push the new code yourself, then `git pull --ff-only origin main` on Oscar. Use a compute allocation. With the existing environments:

```bash
cd ~/projects/zef-binding
export MC_BINDING_SCRATCH=/oscar/scratch/$USER/zef-binding
export MC_RENDER_ENV="$MC_BINDING_SCRATCH/envs/render"
export MC_BINDING_ENV="$MC_BINDING_SCRATCH/envs/model"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export JAVA_HOME="$MC_RENDER_ENV"
export PATH="$MC_RENDER_ENV/bin:$PATH"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"

xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding capture-pairs --n 1 --output "$MC_BINDING_SCRATCH/runs/recognition-captures-01"
```

This reuses the working V3 arena/chunk-loading recipe. It creates six images: recipient pair, its two isolated objects, disjoint donor pair, its two isolated objects. There are no neural-network calls during capture. Output directories must be new. If interrupted, keep the diagnostics and retry into a new directory; capture resume is not implemented. The manifest records errors, actual camera pose, commands, raw and cleaned image hashes, cleanup settings and source hash.

After the first configuration works, `--n 4` captures all four combinations of independently swapping colors and types between the two fixed positions, for both recipient and donor. These are calibration configurations in one arena, not four independent worlds. Donor descriptions are disjoint, but changing templates also changes silhouettes; geometry matching is not claimed.

## 2. Review images locally

```bash
rsync -av zfawnia@ssh.ccv.brown.edu:/oscar/scratch/zfawnia/zef-binding/runs/ /Users/lianfei/Documents/RES/Minecraft-Binding/runs/oscar/
```

Inspect `recognition-captures-01/contact_sheet.png`, the clean `frames/`, and the corresponding `raw/` captures. Confirm the shapes, colors and screen order. Candidate boxes come from fixed color thresholds within the retained viewport, not calibrated projection or validated segmentation. Isolated renders preserve each object's paired-scene position and scale. These boxes/labels are never sent to Qwen. If thresholds fail (especially green/yellow), preserve the raw image and fix annotation before inference.

Cleanup is deliberately explicit: crop raw 448x280 frames to `(0,40,448,195)`; cover the raw crosshair rectangle `(218,134,229,146)` with RGB `(128,128,128)`. The output is 448x155. The small grey square is a preprocessing artifact, not reconstructed scenery. Exactly the same cleanup applies to every frame. Raw frames remain intact. Candidate objects touching the crop or overlapping the cover cause an error. This is suitable only for this fixed pose and layout; camera-motion experiments need a separate cleanup validation. There is no claim of native HUD removal.

## 3. Pin the checkpoint, then recognize

Use the **model** environment, not the render environment (MineStudio installed a different Transformers version there). On an active CUDA compute allocation, create a run-local config with the resolved checkpoint revision:

```bash
"$MC_BINDING_ENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi
config = json.loads(Path('configs/qwen_pilot.json').read_text())
config['revision'] = HfApi().model_info(config['model_id']).sha
out = Path(os.environ['MC_BINDING_SCRATCH']) / 'runs' / 'qwen-recognition.json'
with out.open('x') as stream:
    json.dump(config, stream, indent=2)
print('Pinned revision:', config['revision'])
print('Config:', out)
PY
```

The public model does not require an OpenAI key. The first inference downloads checkpoint weights to `HF_HOME`; allow time and scratch space. Reuse the config file on restart instead of resolving a new revision. Once you have reviewed the captures:

```bash
"$MC_BINDING_ENV/bin/python" -m mc_binding recognize --dataset "$MC_BINDING_SCRATCH/runs/recognition-captures-01" --config "$MC_BINDING_SCRATCH/runs/qwen-recognition.json" --output "$MC_BINDING_SCRATCH/runs/recognition-baseline-01" --reviewed-captures
```

One configuration yields 24 responses: each visible object is queried for color, type, and color+type, both paired and isolated. Four configurations yield 96 responses. Prompts contain no candidate vocabulary, answer hints, IDs, coordinates or expected labels. Strict and predeclared-synonym scoring run side by side; aliases default empty. Raw responses are preserved so `block`, `column`, or full-sentence answers can be diagnosed without changing the strict results. Do not add aliases based on which response would score correctly.

Completed image-level probe groups are saved atomically. Restart the identical command to resume; changed data/config/code/environment is rejected. Use a new output directory if any of those change. Read `results.jsonl`, `summary.json`, `status.json` and `manifest.json`, or download them with the same rsync command. Summary rates are descriptive; no independent-trial confidence intervals are claimed for these repeated configurations.

## Acceptance gate

Before self/donor patches: review cleanup and labels, confirm both paired and isolated recognition (including the donor vocabulary), inspect invalid/raw answers and type confusions, and freeze the scoring rules. If the model calls the tower a block or the pillar a tower, diagnose stimulus naming and/or redesign shapes on calibration data before declaring binding errors. This capture manifest is intentionally separate from the existing patch dataset contract. Converting it into patch-ready data requires the ROI/preprocessing and self-patch checks; do not pass it to `mc-binding patch` yet.

Local validation covered cleanup on V3, color-based ordering, no input-array mutation, crosshair-overlap rejection, four-way counterbalancing, checkpoint/resume, hash checks and raw-RGB-only model calls using a test double. Real MineStudio pair capture and Qwen recognition are still to be tested on Oscar.
