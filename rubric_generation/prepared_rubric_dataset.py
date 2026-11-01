"""Validate a published offline rubric dataset without regenerating its cache."""
import json
from pathlib import Path
from types import SimpleNamespace

from .offline_question_rubrics import file_hash, prepare


def validate_prepared_dataset(directory: Path, *, source: Path, model: Path,
                              prompt_dir: Path, tensor_parallel_size: int = 4) -> Path:
    directory = directory.resolve()
    ready = json.loads((directory / 'ready.json').read_text())
    dataset = directory / 'train.jsonl'
    manifest_path = directory / 'manifest.json'
    for key, path in [('dataset_sha256', dataset), ('manifest_sha256', manifest_path)]:
        if ready[key] != file_hash(path):
            raise ValueError(f'Published initialization checksum mismatch: {path}')
    if 'recovery_report_sha256' in ready:
        if ready['recovery_report_sha256'] != file_hash(directory / 'recovery_report.json'):
            raise ValueError('Initialization recovery record changed')
    original, records, expected = prepare(SimpleNamespace(source=source, model=model,
        prompt_dir=prompt_dir, limit=0, tp=tensor_parallel_size))
    if json.loads(manifest_path.read_text()) != json.loads(json.dumps(expected)):
        raise ValueError('Prepared initialization differs from the requested source/model/prompts')
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not rows or ready['count'] != len(rows) or len(rows) != len(original):
        raise ValueError('Expected one prepared row for every source question')
    for row, source_row, record in zip(rows, original, records, strict=True):
        if (row['prompt'] != source_row['prompt'] or row.get('rubric_origin') != 'offline_question_only_base'
                or row['generated_rubric_metadata']['prompt_key'] != record['key']
                or not row['rubric'] or row['rubric_count'] != len(row['rubric'])
                or not any(item['weight'] > 0 for item in row['rubric'])):
            raise ValueError('Prepared row has invalid provenance or rubric')
    return dataset
