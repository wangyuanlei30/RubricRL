"""Frozen-base, resumable question-only initialization before any RL update.

Run as a separate process so vLLM and its GPU allocations exit before training.
No expert fallback: every training row must have a validated generated rubric.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path

from rubric_generation.rubric_initializer import (
    QUESTION_ONLY, InitialRubricPromptBuilder, InitialRubricRequest,
    _conversation_to_text, canonical_question_text, parse_initial_rubric,
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def question_messages(builder, question: str) -> list[dict]:
    messages = list(builder.build(InitialRubricRequest(question=question, mode=QUESTION_ONLY)).messages)
    # A demonstration may itself be one of the 4500 training questions. Exclude
    # its own gold rubric instead of showing the target's answer in-context.
    kept = [messages[0]]
    for index in range(1, len(messages) - 1, 2):
        user = messages[index]['content']
        example_q = user.split('### Target Conversation\n', 1)[1].rsplit(
            '\n\nGenerate the example-level grading rubric', 1)[0]
        if canonical_question_text(example_q) != canonical_question_text(question):
            kept.extend(messages[index:index + 2])
    return kept + [messages[-1]]


def validate_completion(text: str, question: str) -> list[dict]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not 7 <= len(raw) <= 20:
        raise ValueError('Expected a JSON list with 7 to 20 criteria')
    for item in raw:
        if not isinstance(item, dict) or set(item) != {'criterion', 'points'}:
            raise ValueError('Expected criterion/points fields')
        if not isinstance(item['criterion'], str) or not item['criterion'].strip():
            raise ValueError('Empty criterion')
        if type(item['points']) is not int or not 0 < abs(item['points']) <= 10:
            raise ValueError('Expected nonzero integer points in [-10, 10]')
    roots = parse_initial_rubric(text, question=question, mode=QUESTION_ONLY)
    if len(roots) != len(raw) or not any(item['weight'] > 0 for item in roots):
        raise ValueError('Duplicate criteria or no positive criteria')
    return roots


def prepare(args) -> tuple[list[dict], list[dict], dict]:
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    builder = InitialRubricPromptBuilder(str(args.prompt_dir), few_shot_examples=5)
    records = []
    for row in rows:
        question = _conversation_to_text(row['prompt'])
        if not question:
            raise ValueError('Empty training question')
        messages = question_messages(builder, question)
        records.append({'question': question, 'messages': messages, 'key': digest(messages)})
    weights = sorted(args.model.glob('*.safetensors'))
    if not weights:
        raise ValueError('No local base-model safetensors weights')
    manifest = {
        'version': 1, 'source_sha256': file_hash(args.source), 'model_path': str(args.model.resolve()),
        'model_files': [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights],
        'model_config_sha256': file_hash(args.model / 'config.json'),
        'tokenizer_config_sha256': file_hash(args.model / 'tokenizer_config.json'),
        'initializer_code_sha256': file_hash(Path(__file__).with_name('rubric_initializer.py')),
        'offline_code_sha256': file_hash(Path(__file__)),
        'prompt_hashes': [r['key'] for r in records], 'count': len(rows),
        'temperature': 0.7, 'top_p': 1.0, 'seed': 1, 'max_tokens': 4096,
        'max_model_len': 36864, 'tensor_parallel_size': args.tp,
        'reference_answers': False, 'target_rubrics_in_prompt': False,
        'few_shot_max': 5, 'self_demonstration_excluded': True,
    }
    return rows, records, manifest


def run(args) -> None:
    rows, records, manifest = prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Initialization identity changed; use a new output directory')
        atomic_json(manifest_path, manifest)
        cache = args.output / 'cache'
        cache.mkdir(exist_ok=True)
        unique = {r['key']: r for r in records}
        completed = {}
        for key, record in unique.items():
            path = cache / f'{key}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['key'] != key:
                    raise ValueError('Cache identity mismatch')
                completed[key] = validate_completion(saved['completion'], record['question'])
        pending = [r for key, r in unique.items() if key not in completed]
        if pending:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False)
            for record in pending:
                record['tokens'] = tokenizer.apply_chat_template(record['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                if len(record['tokens']) + 4096 > 36864:
                    raise ValueError('Initialization prompt exceeds context; refusing truncation')
            llm = LLM(model=str(args.model), tensor_parallel_size=args.tp, dtype='bfloat16',
                max_model_len=36864, gpu_memory_utilization=0.65, max_num_seqs=args.batch_size,
                max_num_batched_tokens=32768, enable_chunked_prefill=True, enforce_eager=True,
                seed=1, trust_remote_code=False)
            for attempt in range(3):
                remaining = []
                params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=4096, seed=1 + attempt)
                for start in range(0, len(pending), args.batch_size):
                    chunk = pending[start:start + args.batch_size]
                    outputs = llm.generate([{'prompt_token_ids': r['tokens']} for r in chunk], params, use_tqdm=False)
                    if len(outputs) != len(chunk):
                        raise RuntimeError('Initialization output count mismatch')
                    for record, output in zip(chunk, outputs, strict=True):
                        answer = output.outputs[0]
                        try:
                            if answer.finish_reason == 'length':
                                raise ValueError('Truncated initialization')
                            roots = validate_completion(answer.text, record['question'])
                        except (ValueError, TypeError) as exc:
                            with (args.output / 'failures.jsonl').open('a') as log:
                                log.write(json.dumps({'key': record['key'], 'attempt': attempt + 1,
                                    'error': str(exc), 'completion': answer.text}, ensure_ascii=False) + '\n')
                            remaining.append(record)
                            continue
                        atomic_json(cache / f"{record['key']}.json", {'key': record['key'],
                            'completion': answer.text, 'attempt': attempt + 1,
                            'prompt_tokens': len(record['tokens']), 'output_tokens': len(answer.token_ids)})
                        completed[record['key']] = roots
                    print(f'Initialized {len(completed)}/{len(unique)} unique questions', flush=True)
                pending = remaining
                if not pending:
                    break
            if pending:
                raise RuntimeError(f'{len(pending)} questions failed initialization; RL will not start')
        dataset = args.output / 'train.jsonl'
        temp = dataset.with_suffix('.jsonl.tmp')
        with temp.open('w') as stream:
            for row, record in zip(rows, records, strict=True):
                result = deepcopy(row)
                roots = completed[record['key']]
                result.update(rubric=roots, rubric_list=[r['description'] for r in roots],
                    rubric_count=len(roots), rubric_origin='offline_question_only_base',
                    generated_rubric_metadata={'prompt_key': record['key'], 'mode': QUESTION_ONLY})
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        temp.replace(dataset)
        atomic_json(args.output / 'ready.json', {'count': len(rows), 'dataset_sha256': file_hash(dataset),
            'manifest_sha256': file_hash(manifest_path), 'unique_questions': len(unique)})
        print(f'Offline initialization complete: {dataset}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0, help='Testing only; full launcher never sets this')
    args = parser.parse_args()
    if args.tp < 1 or args.batch_size < 1 or args.limit < 0:
        parser.error('Invalid size')
    run(args)


if __name__ == '__main__':
    main()
