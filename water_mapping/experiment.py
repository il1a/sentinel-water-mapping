"""Calibration and evaluation stages shared by the notebook and command line.

Only validation labels select parameters. Evaluation consumes a saved, frozen
configuration. All loops retain one chip rather than an entire split in memory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import os
import platform
import sys
import time

import numpy as np

from .core import aggregate, classify, confusion, smooth_power, tiled_mean
from .data import load_chip, load_splits, read_json, recorded_selection, sha256, write_json

WIDTHS = (1, 3, 5)
THRESHOLDS_DB = (-25.0, -22.0, -19.0, -17.0, -15.0)


def environment_record() -> dict:
    packages = {}
    for name in ['numpy', 'scipy', 'tifffile', 'imagecodecs', 'matplotlib', 'pandas',
                 'psutil', 'numba', 'numba-cuda', 'nbformat', 'nbclient', 'ipykernel']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(),
                packages=packages, cpu_logical_count=os.cpu_count(),
                thread_environment={name: os.environ.get(name) for name in
                                    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})


def protocol(root: Path) -> dict:
    """Prespecified extension, recorded before inspecting new test outcomes."""
    root = Path(root)
    return {
        'title': 'Expanded Sen1Floods11 hand-label evaluation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split_counts': {s: len(rows) for s, rows in load_splits(root).items()},
        'split_sha256': {s: sha256(root / 'manifests/splits' / f'flood_{s}_data.csv')
                         for s in ['train', 'valid', 'test', 'bolivia']},
        'widths': list(WIDTHS), 'thresholds_db': list(THRESHOLDS_DB),
        'selection': 'Maximum mean defined chip water IoU on all 89 validation chips; ties retain candidate order.',
        'no_smoothing_selection': 'Select width 1 independently by the same validation objective.',
        'raw_threshold_policy': 'Strict VH dB < threshold for width 1; smoothed linear power < 10**(threshold_db/10) for widths 3 and 5.',
        'dtype': 'float64 linear power and smoothing output',
        'mask': 'Both SAR bands finite for smoothing; score only those centers with label >= 0.',
        'edge': 'SciPy half-sample symmetric reflect; integer-rounded valid-neighbor counts.',
        'prior_inspected_test_images': [r[0] for r in recorded_selection(root)['test']],
        'evaluation': 'Freeze validation winners before test scoring. Report all 90 test chips and complementary 80 separately; Bolivia 15 separately. Training scores are descriptive only.',
        'limits': ['The 10 previously inspected test chips are not pristine holdout.',
                   'Standard splits share events; pixels and chips are spatially correlated.',
                   'Observed surface water, not independently verified newly flooded land.',
                   'Tiling retains whole input/output arrays, not out-of-core execution.',
                   'No area estimates from geographic-grid pixel counts.'],
    }


def calibrate(root: Path, rows: list, *, progress=False) -> tuple[list, list]:
    """Score a fixed 15-candidate grid, smoothing once per width and chip."""
    scores = {(width, threshold): [] for width in WIDTHS for threshold in THRESHOLDS_DB}
    details = []
    for index, row in enumerate(rows, 1):
        chip = load_chip(root, row)
        for width in WIDTHS:
            mean_power = smooth_power(chip.power, chip.valid, width)
            for threshold in THRESHOLDS_DB:
                prediction = classify(chip.vh_db, mean_power, chip.valid, width, threshold)
                result = confusion(prediction, chip.labels, chip.valid)
                scores[width, threshold].append(result)
                details.append(dict(image=chip.name, size=width, threshold_db=threshold, **result))
        if progress and (index % 20 == 0 or index == len(rows)):
            print(f'Validation: {index}/{len(rows)} chips', flush=True)
    candidates = []
    for (width, threshold), rows_for_candidate in scores.items():
        summary = aggregate(rows_for_candidate)
        if summary['macro_iou'] is None:
            raise ValueError('No defined validation IoU: cannot choose a detector')
        candidates.append(dict(size=width, threshold_db=threshold,
                               macro_iou=summary['macro_iou'], aggregate=summary))
    return candidates, details


def freeze(root: Path, output: Path, candidates: list) -> dict:
    """Persist validation winners before the caller begins test evaluation."""
    selected = max(candidates, key=lambda row: row['macro_iou'])
    baseline = max((row for row in candidates if row['size'] == 1), key=lambda row: row['macro_iou'])
    config = dict(selected=selected, no_smoothing_baseline=baseline,
                  frozen_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_before_this_run_test_scoring=True,
                  prior_test_inspection_disclosed=True,
                  validation_chips=selected['aggregate']['chips'],
                  raw_threshold_policy='strict_db_v1',
                  source_sha256={name: sha256(Path(root) / name) for name in
                                 ['water_mapping/core.py', 'water_mapping/data.py', 'water_mapping/experiment.py']})
    write_json(Path(output) / 'frozen_config.json', config)
    return config


def evaluation_summaries(evaluation: list, prior_test: set) -> dict:
    groups = {
        'train': lambda r: r['split'] == 'train',
        'valid': lambda r: r['split'] == 'valid',
        'test_all_90': lambda r: r['split'] == 'test',
        'test_previously_inspected_10': lambda r: r['split'] == 'test' and r['image'] in prior_test,
        'test_new_80': lambda r: r['split'] == 'test' and r['image'] not in prior_test,
        'bolivia_separate_15': lambda r: r['split'] == 'bolivia',
    }
    return {group: {method: aggregate([r for r in evaluation if predicate(r) and r['method'] == method])
                    for method in ['selected', 'no_smoothing']}
            for group, predicate in groups.items()}


def evaluate(root: Path, selected: dict, frozen: dict, *, progress=False) -> dict:
    """Stream declared splits with fixed settings; also verify tiled equivalence."""
    evaluation, checks, equivalence = [], [], []
    for split, rows in selected.items():
        for index, row in enumerate(rows, 1):
            chip = load_chip(root, row)
            checks.append(dict(split=split, **chip.check))
            for method, key in [('selected', 'selected'), ('no_smoothing', 'no_smoothing_baseline')]:
                config = frozen[key]
                mean_power = smooth_power(chip.power, chip.valid, config['size'])
                prediction = classify(chip.vh_db, mean_power, chip.valid,
                                      config['size'], config['threshold_db'])
                evaluation.append(dict(split=split, image=chip.name,
                                       event=chip.name.split('_')[0], method=method,
                                       prediction_sha256=hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest(),
                                       **confusion(prediction, chip.labels, chip.valid)))
                if method == 'selected':
                    tiles = tiled_mean(chip.power, chip.valid, config['size'])
                    same_nan = bool(np.array_equal(np.isnan(mean_power), np.isnan(tiles)))
                    close = bool(np.allclose(mean_power, tiles, atol=1e-12, rtol=1e-10, equal_nan=True))
                    tiled_prediction = classify(chip.vh_db, tiles, chip.valid,
                                                config['size'], config['threshold_db'])
                    disagreement = int(np.count_nonzero(prediction != tiled_prediction))
                    finite = np.isfinite(mean_power) & np.isfinite(tiles)
                    max_error = float(np.max(np.abs(mean_power[finite] - tiles[finite]))) if finite.any() else None
                    if not same_nan or not close or disagreement:
                        raise AssertionError(f'Whole/tiled detector mismatch: {chip.name}')
                    equivalence.append(dict(image=chip.name, nan_masks_equal=same_nan,
                                            numeric_allclose=close, max_abs_power_error=max_error,
                                            prediction_disagreements=disagreement))
            if progress and (index % 50 == 0 or index == len(rows)):
                print(f'Evaluate {split}: {index}/{len(rows)} chips', flush=True)
    prior_test = {r[0] for r in recorded_selection(root)['test']}
    event_rows = []
    for split in selected:
        for event in sorted({r['event'] for r in evaluation if r['split'] == split}):
            for method in ['selected', 'no_smoothing']:
                records = [r for r in evaluation if (r['split'], r['event'], r['method']) == (split, event, method)]
                event_rows.append(dict(split=split, event=event, method=method, **aggregate(records)))
    return dict(evaluation=evaluation, input_checks=checks, cpu_equivalence=equivalence,
                accuracy_summary=evaluation_summaries(evaluation, prior_test), event_summary=event_rows)
