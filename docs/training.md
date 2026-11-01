# Training details

Training code for **Rubrics That Keep Teaching: Joint Policy Learning and Rubric Evolution for Open-Ended Reinforcement Learning**.

This release uses the original Ray, FSDP, and vLLM training pipeline. One policy generates task responses and rubric evolution proposals, then receives a joint update from the solver, horizontal generator, and vertical generator objectives. A separate frozen teacher conditions on the evolved rubric and supplies token-level credit. The fixed KL reference remains independent of that teacher.

## Where to start

| File | Responsibility |
| --- | --- |
| `train.py`, `liverubric/launch.py` | Prepare initial rubrics, validate inputs, launch training, resume |
| `configs/liverubric_4gpu.yaml` | Four-GPU HealthBench training configuration |
| `verl/trainer/opsd_trainer.py` | Actual training loop and same-rollout reward update |
| `verl/trainer/rubric_generator_training.py` | Sample, audit, evaluate, select, and commit proposals |
| `verl/trainer/rubric_generator_rewards.py` | Proposal reward and within-group advantages |
| `verl/trainer/rubric_evolution.py` | Criterion states, empirical checks, persistent rubric state |
| `verl/workers/actor/rubric_joint_update.py` | One optimizer step over all three roles |
| `rewards/rubric_llm_judge.py` | Criterion scoring, batching, retries, and signed reward aggregation |

`rubric_generation/` contains the original offline initializer. The local extraction preserves 23 prompt assets byte-for-byte, including demonstrations and full semantic verification prompts. The public release includes 22 assets; the HealthBench demonstration JSON is prepared locally as described in [the prompt guide](../prompts/README.md). The criterion-judge prompt also remains in the judge source. `verl/` includes the patched runtime needed by the real distributed update; installing an upstream `verl` package does not supply these project changes.

## Environment

Use a separate **Linux/CUDA** environment, Python 3.10 or newer, and four GPUs for the supplied configuration. The snapshot pins vLLM 0.11.0 and Transformers 4.57.3; other dependencies retain the snapshot's unpinned requirements. A frozen installation lockfile and a four-GPU smoke run are still needed before claiming end-to-end reproducibility of this extracted release.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'flash-attn>=2.4.3' --no-build-isolation
python -m pip install -e '.[test]'
```

Run from this checkout so the prompt and reward assets remain available. Keep the base model as a local Hugging Face directory with `.safetensors`, model configuration, and tokenizer files. Configure the judge endpoint/model in the YAML. The paper configuration names `deepseek-v4-flash-0731`; access to this model must be available on your endpoint. Supply its credential through `DEEPSEEK_API_KEY`, not in source or configuration files.

## Data and training

Input is JSONL with `prompt` (a string or chat messages) and `answer` (may be empty for rubric-scored open-ended tasks):

```json
{"prompt": [{"role": "user", "content": "Explain the requested topic."}], "answer": ""}
```

Training uses at least 64 questions with the default batch size. The initializer reads only each target question and up to five fixed question/rubric demonstrations, excludes a target's own demonstration, and strips reference answers from the demonstration messages. It generates and validates 7–20 signed criteria per question. Failed initialization stops training and leaves a reusable per-question cache. Source/model/prompt fingerprints prevent reuse with changed inputs. The included demonstrations are the original HealthBench prompt assets; adapt them deliberately for other training domains.

Inspect the resolved settings and commands without loading a model or calling a judge:

```bash
python train.py --model /models/Qwen3-4B-Instruct-2507 \
  --train-file /datasets/train.jsonl --output runs/healthbench --dry-run
```

Start initialization and training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh \
  --model /models/Qwen3-4B-Instruct-2507 \
  --train-file /datasets/train.jsonl --output runs/healthbench
```

Initialization runs in its own process and releases GPU memory before RL starts. Online initialization stays disabled. Add `--val-file /datasets/val.jsonl` for periodic validation; validation rows must include their evaluation `rubric`. Omit it to train without periodic validation. `--init-dir` reuses an existing initialization cache after identity checks. `--config` selects an edited YAML for a new run.

The supplied configuration uses batch 64, two epochs, eight solver rollouts, and eight proposals per applicable evolution operation. Up to six horizontal candidates and four vertical parents are considered; each vertical parent receives up to two children. Proposal reward is `V + D - P`, with `D` the fraction passing semantic checks and minimum support on both sides of the rollout split. Eligible splits receive equal credit. Parent/child weights sum to the original signed parent weight.

Effective solver : horizontal : vertical loss weights are **1 : 0.125 : 0.125** (`0.25 × 0.5` for each generator role). Teacher mixing is **0.5**, with reweight clipping **0.2**, no decay, and a refresh every **10** optimizer steps. See `SOURCE_MAP.md` for differences from the archived training snapshot and retained evolution limits.

## Save and resume

```bash
python train.py --output runs/healthbench --resume latest
# Or select a completed checkpoint:
python train.py --output runs/healthbench \
  --resume runs/healthbench/checkpoints/global_step_100
```

Resume uses the saved configuration and validates unchanged data, model initialization identity, code, and prompts. Keep the same GPU world size and local model/data locations. Checkpoints retain actor parameters, optimizer, scheduler, worker RNG, stateful dataloader, the independent teacher, and evolved rubric state. Initial rubrics and their cache remain in `initialization/`; the evolved state contains their root criteria. The completion manifest is published only after these states are written, followed by an atomic update of the latest-checkpoint tracker. Missing state fails explicitly. Training logs are appended on resume.

Save every 10 steps by default; `save_limit: -1` keeps all checkpoints. These are distributed training checkpoints, not standalone Hugging Face model exports. Existing checkpoints from other code versions require an explicit migration; this launcher does not silently load them.

```text
runs/healthbench/
  launch.json                 # Inputs, hashes, code identity
  resolved_config.json        # Immutable run configuration
  initialization/             # Per-question cache, manifest, prepared train.jsonl
  rubric_logs/                # Raw proposals, audit decisions, evolution diagnostics
  checkpoints/
    experiment_log.jsonl
    checkpoint_tracker.json
    global_step_10/
      actor/                  # Actor, optimizer, scheduler/RNG, independent teacher
      dataloader.pt
      rubric_evolution_state.json
      liverubric_checkpoint.json
```

## Checks

```bash
python -m unittest discover -s tests -p 'test_release.py' -v
# In the complete training environment, also run the retained runtime tests:
python -m unittest discover -s tests/runtime -v
```

The release tests cover real proposal rewards, family weights, original prompt hashes, initialization prompt construction, launch/resume plans, checkpoint completeness, and a CPU teacher save/restore. Runtime tests retain the original evolution, judge, teacher, and rollout-padding checks. They do not replace a CUDA distributed training run. Full training data, model weights, private API keys, and experiment outputs are not bundled. See `THIRD_PARTY_NOTICES.md` for source attribution.
