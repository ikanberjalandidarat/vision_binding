# Replication and behavior experiments

This implements new capture specifications, V replication, offline destination choices,
and single-trial movement replay. Real rendering, model inference and controller validation
must run on Oscar. Motion/occlusion rescue is the next stage, not an implemented result.
No training or speed improvement is claimed.

## 1. Render one configuration first

After you commit/push the changes yourself and pull them on Oscar:

```bash
cd ~/projects/zef-binding
mkdir -p logs
sbatch scripts/slurm/replication_capture.sbatch 1
```

The job prints six captures into
`/oscar/scratch/$USER/zef-binding/runs/vision-captures-replication-1-JOBID`.
Check `manifest.json` state is `captured_needs_visual_review`, then copy that folder
using rsync from your Mac. Review `contact_sheet.png` and raw frames for clipping,
wrong colors, text, ceiling contamination, and donor/recipient alignment. The first
configuration contains yellow stairs and a blue pillar; the donor swaps their colors.
Do not mark uninspected captures as reviewed.

```bash
# Mac; replace JOBID with the capture job number
rsync -av zfawnia@ssh.ccv.brown.edu:/oscar/scratch/zfawnia/zef-binding/runs/vision-captures-replication-1-JOBID/ runs/oscar/vision-captures-replication-1-JOBID/
```

After the one-configuration render is visually valid, run the full set:

```bash
sbatch scripts/slurm/replication_capture.sbatch 24
```

This produces 144 images: 24 configurations × recipient/donor × pair/two isolates.
The fixed factorial design is:

- Palettes: yellow/blue, red/yellow, red/blue.
- Shape pairs: stairs/pillar, arch/tower.
- Two object placements (different depth/spacing).
- Both color assignments for every combination.

It includes familiar comparisons as well as new combinations. It uses the same arena
and camera, not 24 independent worlds. Reverse assignments and paired sides are
repeated measurements. Yellow is the new color; do not describe this as broad palette
coverage. Color masks and silhouette alignment remain gates. Never relax a failed
alignment threshold merely to obtain a model result.

## 2. Run V replication

After reviewing the full capture set, replace the placeholder in this command:

```bash
sbatch scripts/slurm/replication_model.sbatch vision-pilot \
  vision-captures-replication-24-CAPTURE_JOBID --reviewed-captures
```

The output is `vision-pilot-replication-MODEL_JOBID`. The configuration tests V30,
V31, and simultaneous V30+31, with the existing self/random/other-object/background
controls. Expected complete row count for 24 configurations is **912**: 192 clean
recognition records plus 720 patch records. A failed clean recognition gate stops the
run and saves diagnostics; this failure is a result to inspect, not a reason to silently
exclude a scene. First-answer-token scores include red, blue and yellow spelling
variants, with token IDs recorded. They are not normalized two-choice probabilities.

The HTML report includes actual images and token overlays. Compare results by palette,
shape pair and placement, not only pooled success. Keep these predefined sites fixed
when inspecting this replication; any newly chosen site needs another held-out check.

## 3. Run destination choice

After reviewing replication results:

```bash
sbatch scripts/slurm/replication_model.sbatch destination \
  vision-captures-replication-24-CAPTURE_JOBID --reviewed-captures
```

Output: `destination-replication-MODEL_JOBID`. Expected complete count: **672** rows,
24 × (4 clean choices + 3 patch sets × 4 conditions × 2 requested colors).

The prompt asks which side to walk toward for a specified color. Both object regions
are patched together, avoiding an ambiguous two-red-object internal counterfactual.
Conditions are self, both_objects, norm-matched random and token-count-matched
background. Clean recipient and donor choices must both be correct before patches
are interpreted. Invalid answers are retained, never silently converted to a side.
Self-patches must reproduce clean text exactly. `report.html` shows images and answers.

These are offline destination choices, not observed movement and not shape/color
conjunction tasks. The same vision masks and grid-alignment checks are required.

## 4. Validate actual approach behavior

Acquire an interactive CPU/render allocation using your usual Oscar allocation command
and restore the render environment variables. Model and renderer remain separate:
this stage reads a completed decision run and does not load Qwen.

```bash
export MC_BINDING_SCRATCH="/oscar/scratch/$USER/zef-binding"
export MC_RENDER_ENV="$MC_BINDING_SCRATCH/envs/render"
export JAVA_HOME="$MC_RENDER_ENV"
export PATH="$MC_RENDER_ENV/bin:$PATH"
export MINESTUDIO_DIR="$MC_BINDING_SCRATCH/minestudio"
export HF_HOME="$MC_BINDING_SCRATCH/huggingface"
export XDG_CACHE_HOME="$MC_BINDING_SCRATCH/cache"

# Replace directory names with actual completed runs.
xvfb-run -a "$MC_RENDER_ENV/bin/python" -m mc_binding approach \
  --dataset "$MC_BINDING_SCRATCH/runs/vision-captures-replication-24-CAPTURE_JOBID" \
  --decisions "$MC_BINDING_SCRATCH/runs/destination-replication-MODEL_JOBID" \
  --trial 'f0000:baseline:clean_recipient:yellow' \
  --output "$MC_BINDING_SCRATCH/runs/approach-clean-01"
```

First validate clean approaches to both colors on the same layout. Then replay matched
`f0000:layer30:both_objects:yellow`, `f0000:layer30:self:yellow`, and clean donor trials
such as `f0000:baseline:clean_color_swap:yellow`, each into a new output directory.
Use actual trial keys from `results.jsonl`; do not assume an intervention changes a choice.

The simulator rebuilds the decision's input scene, verifies camera pose, and compares
its initial cleaned frame to the saved input before moving. A mismatch stops movement
and saves a frame for inspection. Once movement begins, only camera and forward
controls are used; no teleport-to-goal shortcut. Telemetry steers toward a waypoint two
blocks in front of the chosen object's bounding box. Arrival tolerance is 0.8 blocks,
with a bounded step budget. This uses privileged world coordinates, not visual navigation.

`approach.json` distinguishes:
- reaching the selected waypoint (controller performance),
- reaching the goal requested in the actual rendered scene (task success).

A donor-induced wrong choice can navigate successfully while failing the actual task.
Frames and per-step actions/poses are saved; initial render failures, unexpected action
schemas and elevation changes fail explicitly. Camera action sign and forward movement
still need a real simulator smoke check. A run finishing does not imply arrival success.

## 5. Motion/occlusion rescue protocol — gated follow-up

Do not reuse fixed-camera token indices once the camera moves. That would patch the
wrong image regions. Implement and validate frame-specific object correspondence first.
For occlusion, color-threshold masks cannot identify hidden object regions; use reviewed
amodal geometry/projection or matched-pose unobstructed captures with explicit provenance.

Prespecify paired episodes with identical scene, instruction and camera trajectory:
clean view, occluded view, self control, wrong-object donor control, and matched-pose
clean V donor rescue. Save frame IDs, poses, intervention timestamps and occluder state.
Measure failure rate across all episodes, choice recovery and destination arrival, reporting
both all-episode results and rescue conditional on a baseline failure. A privileged clean
counterfactual donor tests mechanism; it is not a deployable autonomous-agent solution.
Only later test whether a genuinely available past observation supplies a useful memory.
