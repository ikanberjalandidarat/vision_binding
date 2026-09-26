# Minecraft visual binding

Experiment backbone for the supplied [workup](docs/minecraft_binding_agent_workup%20%281%29.md) and [original notebook](notebooks/thoughtful_emoji.ipynb). **No commits are made by the assistant.**

Implemented: deterministic two-object scene specifications, explicitly labeled fixture images, recorded-frame validation and contact sheets, Qwen2-VL baseline/decoder-patching runner, strict and configurable synonym scoring, atomic family checkpoints/resume, paired analysis, and Oscar job templates.

**Current gate:** Minecraft rendering and Qwen GPU inference have not been validated. The local smoke uses flat diagnostic drawings, not Minecraft. MineStudio has a real render/reset/camera smoke entry point, but is not yet a validated dataset backend. Stage C tracking and Stage D learning experiments remain behind the workup's gates.

## Local quick start

Use a separate Python 3.10 environment on Oscar. The CPU backbone also supports Python 3.9.

```bash
python -m pip install -e '.[test]'
mc-binding doctor --output runs/environment.json
mc-binding smoke --backend fixture --output data/smoke
mc-binding generate --backend fixture --n 20 --output data/debug20
mc-binding validate data/debug20
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The hook tests require PyTorch (`pip install torch`, or install the `model` extra). `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated plugins from a shared base environment. Images, model weights, environments and outputs are ignored by Git. Original notebook/workup are preserved.

`data/debug20/contact_sheet.png` contains annotation overlays; clean model images contain no labels. Every image has a hash. Dataset generation refuses to overwrite an existing directory. Use **different seeds** for calibration and test; split labels alone do not produce independent geometry. Only freeze a 400-family test manifest after the 20-family debugging gates pass.

## Qwen commands (CUDA allocation)

```bash
python -m pip install -e '.[model,test]'
```

Set `revision` in `configs/qwen_pilot.json` to the checkpoint's immutable Hugging Face commit hash. This is intentionally unset rather than silently downloading a moving `main` revision. The initial configuration explicitly uses 4-bit loading, matching the supplied notebook. Layers 21–27 are a replication candidate band, not validated Minecraft heads. `query_heads` and `kv_heads` are separate; `null` means the whole projection.

```bash
mc-binding baseline --dataset data/debug20 --config configs/qwen_pilot.json --output runs/fixture-baseline --allow-fixture
mc-binding patch --dataset data/debug20 --config configs/qwen_pilot.json --output runs/fixture-patch --allow-fixture
mc-binding analyze runs/fixture-patch
```

These example runs are **fixture diagnostics only**. Validated Minecraft datasets will use the same interface without `--allow-fixture`. For color-only/type-only probes, copy the config and change `task` to `color`/`type`; use a different output directory.

A patch family contains 24 rows: clean, three exact self-patches, and five interventions across four donor contexts. Self-patches must exactly preserve generated text before donor runs proceed. Q edits the final prompt token. V edits bounding-box-overlapping visual tokens at the original target's screen side in both contexts. ROI mapping uses the actual processor image grid and full-image resize. Unequal token counts/grids are logged as skipped; no interpolation is invented. Changed templates can therefore make V/Q+V unavailable while Q remains evaluable.

Runs save immutable settings, model/processor metadata, library versions, image hashes, prompts, object tables, raw responses, both parsers' scores, head-channel indices and token positions. Hooks apply once during prefill; cached autoregressive decoding continues afterward. Q/K/V hooks precede RoPE; output hooks act on `o_proj` input. The CLI matrix exercises Q/V, while K and output hooks are available for later diagnostics. Run one writer per output directory. Completed family files are atomic; an interrupted family is rerun in full. Resume refuses configuration/data/environment changes.

Analysis reports baseline eligibility, skipped and invalid counts, all-scene and clean-correct transfer rates, Wilson intervals and paired family bootstrap intervals. Q versus random Q-only is primary; other contrasts are secondary. Synonym aliases start empty and must be frozen from stimulus validation. No response-dependent aliases or pink-to-magenta rule is applied.

See [Oscar setup](docs/OSCAR.md) and [implementation status](docs/IMPLEMENTATION.md) for the remaining gates and provenance.

## Next after the successful V3 smoke

See [recognition calibration](docs/RECOGNITION.md) for the new `capture-pairs` and `recognize` commands. They generate raw/clean recipient and disjoint-donor frames plus isolated-object controls, then score color/type/pair prompts on saved RGB. This is a separate fixed-pose calibration dataset; it is not yet accepted by the patch runner. Native HUD suppression is not assumed: a documented fixed crop and small crosshair cover are applied consistently, with raw captures retained.

For the first color-only decoder intervention, see [Q color pilot](docs/Q_COLOR_PILOT.md). `q-pilot` uses reviewed recorded pairs, checks clean and exact self-Q baselines, and compares donor Q with norm-matched random Q using the validated BF16 model settings. It does not require ROI-token mapping.
