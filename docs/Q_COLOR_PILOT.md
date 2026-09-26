# Color-only Q-transfer diagnostic

The unquantized BF16/fast-processor recognition run answered all eight color-only probes correctly. Shape names were unreliable, including on isolated frames. This pilot therefore scores colors only and makes no color–type binding claim.

The command `q-pilot` accepts the existing `recognition_pilot_v1` capture manifest. It does not consume ROI token maps or edit V. The candidate box problem on the isolated green structure cannot affect these Q edits. Screen-side object ordering in the paired images must still be visually reviewed. The saved RGB is the only visual input; annotations are used offline for scoring.

## Run on Oscar

After committing/pushing the changes yourself and pulling on Oscar, run in an active GPU allocation using the **model** environment and the previously pinned BF16 config:

```bash
"$MC_BINDING_ENV/bin/python" -m mc_binding q-pilot \
  --dataset "$MC_BINDING_SCRATCH/runs/recognition-captures-6713013-v2" \
  --config "$MC_BINDING_SCRATCH/runs/qwen-recognition-bf16.json" \
  --output "$MC_BINDING_SCRATCH/runs/q-color-pilot-01" \
  --reviewed-captures
```

The runner requires `load_in_4bit: false`, `dtype: bfloat16`, and `use_fast: true`. It forces `task: color` and generation-score checks in its resolved manifest. Numerical failure stops the run. Generation remains greedy with caching; Q interventions fire once at prefill's final prompt token, on projection outputs before RoPE. Candidate layers 21–27 come from the earlier synthetic work; this is not a validated Minecraft localization claim.

For each captured configuration:

1. Repeat all eight clean color probes across pair and isolated images. Require every one to be strictly correct. Stop on failure, retaining the raw diagnostic rows; do not silently exclude it.
2. Capture decoder Q projections for both requested sides in both paired contexts. Check exact self-Q replacement for both recipient sides. Both generated strings must match clean before any donor interventions run.
3. Cross both recipient target sides with four donor contexts (same/opposite requested side × same/disjoint colors). Evaluate donor Q and a norm-matched random Q control for each context.

This yields 8 clean + 2 self + 16 intervention = **26 rows per configuration**. Whole configurations checkpoint atomically; incomplete configurations rerun from the beginning. The latest incomplete results are kept separately in `diagnostics/`. Resume with identical settings, source, environment and capture hashes; use a new output directory after changes.

Random controls modify exactly the selected Q channels and match the donor perturbation norm separately in each layer in float32 before casting to the model dtype. BF16 rounding can slightly alter the realized perturbation. Seeds, channels, positions, shapes and pre-cast norms are logged. One random draw per context is a plumbing control, not a complete random-control distribution.

## Reading results

Download `q-color-pilot-01/` with the usual rsync command. Inspect `status.json`, `results.jsonl`, `summary.json`, `manifest.json` and `diagnostics/`. The summary focuses on opposite-instruction, disjoint-color trials:

- Recipient's other color: consistent with selection transfer.
- Donor-selected object's color: donor-color transfer.
- Original recipient color: retained original answer.
- Other colors / invalid text: reported separately via raw rows and scoring flags.

Flags can overlap in same-content or same-side controls. The summary gives the paired Q-minus-random-Q difference without confidence intervals. Both target sides share the same images and the four counterbalanced layouts share one arena; these are not independent-world samples. A positive result here does not establish abstract object addresses, identity tracking, localization, or color–type binding.

The existing one-configuration dataset is sufficient for a first mechanical diagnostic. Before interpreting transfer behavior, capture/review all four counterbalanced configurations with `capture-pairs --n 4` into a fresh directory and run this command on that dataset. The same baseline and self-patch gates apply. Scaling to independent worlds and held-out scenes remains later work.

Local tests cover the matrix, clean/self gates, resume, Q-only channel selection, seeded norm controls, and non-finite generation handling with test doubles. Actual Qwen self-patch equality and donor effects require Oscar execution. No effect is assumed.
