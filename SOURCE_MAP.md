# Source map and deliberate changes

The release was extracted from `Rubric_RL_source_20260930`. `SOURCE_MANIFEST.json` records original paths, SHA-256 hashes, and release hashes for the extracted files. Original source files remain untouched.

| Release location | Snapshot location |
| --- | --- |
| `verl/` | `verl/` (patched distributed trainer and supporting runtime) |
| `rubric_generation/` | Four initializer/prepared-dataset modules from `rubric_generation/` |
| `prompts/` | `examples/visual_rl/format_prompt/rubric_prompt/` plus legacy evolution templates |
| `rewards/` | `examples/visual_rl/reward_function/rubric_llm_judge.py` and `rubric_jev_judge.py` |
| `tests/runtime/` | Original evolution, judge, teacher, and rollout-padding tests |

## Paper alignment

`rubric_generator_rewards.py` now credits each semantically valid criterion that passes the empirical minimum-support check by one, then divides by the proposed count. It no longer rewards proximity to a 50/50 split with `4p(1-p)`. Invalid, duplicate, and excess slots remain in the denominator; unavailable audits exclude their entire proposal group rather than create synthetic rewards. The highest-reward parseable proposal is selected and all available proposals train the generator.

`rubric_weight_allocation.py` divides the full signed root weight among the parent and accepted children. The old equal-family branch retained only 80% of that weight. The paper configuration explicitly enables the corrected branch.

The configuration sets eight proposals, exact dead/saturated thresholds (0/1), minimum split support two, six horizontal candidates, four vertical parents, constant teacher mixing 0.5, and effective generator weights 0.125 each. Offline initial rubrics come from the frozen base model; online initialization is disabled. Configuration values are aligned to the current manuscript, so this release is not an unchanged replay of every historical experiment.

## Training and resume integration

`train.py` and `liverubric/launch.py` call the original offline initializer and `verl.trainer.opsd_main`. They resolve local paths, validate data and settings, store the run configuration, and check initialization identity. The prepared-dataset validator accepts arbitrary complete dataset sizes and the configured initialization tensor-parallel size, replacing the snapshot's hard-coded 4,500 rows and eight GPUs.

The trainer's rubric state writes now use the existing extra-state checkpoint hook. A completion manifest is written after model/optimizer/teacher/dataloader/rubric state; the latest tracker is updated atomically afterwards. Checkpoint pruning follows publication. File logging appends when resuming. The logger configuration's tuple annotation accepts multiple backends under current OmegaConf validation. Retained runtime tests are adjusted for relocated assets and the new checkpoint hooks; new checks cover the resolved configuration and minimum-support gate.

## Retained behavior

The original proposal parsing, semantic checks, numerical-grounding gate, batching, API failure handling, signed reward aggregation, role-specific normalization, rank padding, joint optimizer step, teacher updates, and FSDP checkpoint code are retained. All extracted prompt files are byte-identical to the snapshot. The public repository omits the five-question HealthBench demonstration JSON; its hash is retained in the manifest and the original file remains available locally.

The runtime retains a 20-root cap for accepted horizontal additions and a 10-step retry cooldown. Vertical parents with pending or active children are skipped; the runtime does not recursively evolve arbitrary descendants. The four-parent budget applies per question; the configuration removes the historical global 128-target cap. These details matter when interpreting long training runs.

Supporting framework modules and disabled legacy configuration branches remain because shared imports depend on them. Baseline projects, launch scripts for private servers, datasets, old result files, and model checkpoints are omitted. The previous lightweight reference implementation was archived locally before this extraction.
