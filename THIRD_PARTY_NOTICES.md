# Third-party notices

The extracted training runtime derives from **verl** and preserves source headers bearing `Copyright 2024 Bytedance Ltd. and/or its affiliates`, along with other notices in individual files. Those files use the Apache License, Version 2.0; a copy is included in `third_party/verl-LICENSE`. The runtime contains local project modifications and should not be treated as an unmodified upstream release.

The HealthBench demonstration JSON is prepared locally and excluded from this public release. Its source is the original training project's prompt assets. The question-only builder strips demonstration reference answers and excludes a matching target demonstration before model input. See the [official dataset card](https://huggingface.co/datasets/openai/healthbench) and the [HealthBench release's request to avoid publishing benchmark examples](https://openai.com/index/healthbench/).

`SOURCE_MAP.md` describes the paper-alignment and integration changes. These notices preserve existing attribution; a project-wide license for newly authored code has not yet been selected.
