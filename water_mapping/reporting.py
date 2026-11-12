"""Write a small, named evidence bundle rather than many loose JSON files."""
from pathlib import Path
import csv
from .data import write_json


def save_study(output, payload):
    """Save study.json and, when present, a directly readable evaluation.csv.

    The notebook/CLI retain frozen_config.json as the explicit selection boundary.
    All other machine-readable evidence is grouped under named study.json keys.
    Write the CSV first, then atomically replace the study status/evidence file.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = payload.get('result', {}).get('evaluation', [])
    if rows:
        target = output / 'evaluation.csv'
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
    temporary = output / 'study.json.tmp'
    write_json(temporary, payload)
    temporary.replace(output / 'study.json')
