# Prompt assets

All published `.jinja` and system-prompt `.txt` files preserve the original training text. The active paths are:

| Stage | Asset |
| --- | --- |
| Initial rubric | `question_only_initial_rubric_system.txt` + local demonstration JSON |
| Solver response | `rubric_raw.jinja` |
| Horizontal proposal | `rubric_gap_proposal.jinja` |
| Vertical proposal | `rubric_evolution_batch_proposal.jinja` |
| Semantic audit | `rubric_gap_meta_verify.jinja`, `rubric_child_meta_verify.jinja` |
| Self-teacher | `rubric_guided_rlsd_teacher.jinja` |

The criterion-judge prompt is embedded in `rewards/rubric_llm_judge.py`. Legacy templates are retained for the original runtime's alternate branches.

## Local HealthBench demonstrations

The training prompt uses five fixed question/rubric demonstrations. Their text is excluded from GitHub because [HealthBench asks users not to publish benchmark examples online](https://openai.com/index/healthbench/). Obtain the original demonstration file from your local training assets and copy it into this ignored path:

```bash
cp /path/to/local/healthbench_example_level_fewshot.json prompts/
```

The expected SHA-256 is recorded under `prompts/healthbench_example_level_fewshot.json` in [`SOURCE_MANIFEST.json`](../SOURCE_MANIFEST.json). Check the original file from the repository root:

```bash
python -c 'import hashlib,json,pathlib; p=pathlib.Path("prompts/healthbench_example_level_fewshot.json"); m=json.load(open("SOURCE_MANIFEST.json")); e=next(x for x in m["files"] if x["release"]==str(p)); assert hashlib.sha256(p.read_bytes()).hexdigest()==e["source_sha256"]; print("Original demonstrations verified")'
```

For independent experiments, you can prepare your own five demonstrations using this schema and the official dataset. Each assistant message is a JSON array of `criterion`/`points` objects; preserve the delimiters in the user message:

```json
[
  {
    "messages": [
      {"role": "user", "content": "### Target Conversation\nuser: <question>\n\nGenerate the example-level grading rubric as a JSON array."},
      {"role": "assistant", "content": "[{\"criterion\": \"<requirement>\", \"points\": 5}]"}
    ]
  }
]
```

A different demonstration set defines a different experiment and will not match the original checksum. Initialization records the rendered prompt hashes and prevents cache reuse after they change. The builder uses at most five demonstrations, strips reference answers, and excludes any demonstration matching the target question.
