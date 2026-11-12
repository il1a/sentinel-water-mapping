"""Harmonized evaluation of released Otsu maps, without calibration or training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import re

import numpy as np
import tifffile

from .core import aggregate, classify, confusion, smooth_power
from .data import (BASE_URL, acquire_pairs, load_chip, load_splits, read_json,
                   recorded_selection, sha256, verified_download, contained_data_path)
from .experiment import environment_record
from .reporting import save_study

METHODS = ('selected', 'no_smoothing', 'released_otsu')
COUNT_KEYS = ('tp', 'fp', 'fn', 'tn', 'scored_pixels')
REFERENCE = 'reference'


class CoverageError(ValueError):
    def __init__(self, image, check):
        self.check = check
        super().__init__(f"{image}: released prediction coverage incomplete: {check['uncovered_scored_pixels']} scored pixels; decision needed")


def otsu_name(image):
    if not re.fullmatch(r'[A-Za-z-]+_[0-9]+_S1Hand\.tif', image):
        raise ValueError('Invalid image identity')
    return image.replace('_S1Hand.tif', '_S1OtsuLabelHand.tif')


def prediction_valid(values, nodata):
    """Binary prediction coverage; even a binary-valued nodata tag is excluded."""
    valid = np.isfinite(values) & np.isin(values, (0, 1))
    if nodata is not None:
        valid &= ~np.isnan(values) if np.isnan(nodata) else values != nodata
    return valid


def sensitivity_nonwater(counts, missing_water, missing_nonwater):
    """Pooled sensitivity only: hypothetical missing pixels assigned nonwater."""
    result = {k: counts[k] for k in ('tp', 'fp', 'fn', 'tn')}
    if missing_water < 0 or missing_nonwater < 0:
        raise ValueError('Missing pixel counts must be nonnegative')
    result['fn'] += missing_water
    result['tn'] += missing_nonwater
    return {k: v for k, v in summarize([result]).items()
            if k not in ('macro_iou', 'chips', 'defined_iou_chips', 'undefined_iou_chips')}


def validate_prediction(values, common_mask, prediction_geo, reference_geo, nodata, *, name, image):
    """Reject unknown classes, grids, and missing predictions on the common mask.

    Negative one is permitted only as a declared missing value outside the common
    mask. Explicit nodata is never silently converted into the nonwater class.
    """
    if name != otsu_name(image):
        raise ValueError('Released prediction identity mismatch')
    if values.ndim != 2 or values.shape != common_mask.shape:
        raise ValueError('Released prediction shape mismatch')
    for key in ('GeographicTypeGeoKey', 'GTModelTypeGeoKey', 'GTRasterTypeGeoKey'):
        if key not in reference_geo or prediction_geo.get(key) != reference_geo[key]:
            raise ValueError(f'Released prediction grid mismatch: {key}')
    for key in ('ModelPixelScale', 'ModelTiepoint'):
        if key not in prediction_geo or key not in reference_geo:
            raise ValueError(f'Missing grid metadata: {key}')
        a, b = np.asarray(prediction_geo[key]), np.asarray(reference_geo[key])
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-12):
            raise ValueError(f'Released prediction grid mismatch: {key}')
    if 'ModelTransformation' in prediction_geo or 'ModelTransformation' in reference_geo:
        raise ValueError('Unsupported transformed grid')
    missing = ~np.isfinite(values)
    if nodata is not None:
        missing |= (np.isnan(values) if np.isnan(nodata) else values == nodata)
    if np.any(~missing & ~np.isin(values, (0, 1))):
        raise ValueError('Unexpected released prediction class')
    uncovered = int(np.count_nonzero(common_mask & missing))
    check = dict(shape=list(values.shape), dtype=str(values.dtype),
                classes=[float(x) for x in np.unique(values) if np.isfinite(x)],
                nodata='NaN' if nodata is not None and np.isnan(nodata) else nodata,
                missing_pixels=int(missing.sum()), uncovered_scored_pixels=uncovered,
                scored_pixels=int(common_mask.sum()), grid_matches=True)
    if uncovered:
        raise CoverageError(image, check)
    return check


def acquire_otsu(root, selected):
    pins_path = root / 'manifests/released_otsu_manifest.json'
    pins = {x['name']: x for x in read_json(pins_path)} if pins_path.exists() else {}
    def fetch(job):
        split, image = job
        name = otsu_name(image)
        relative = Path('data/S1OtsuLabelHand') / name
        path = contained_data_path(root, relative)
        url = BASE_URL + 'S1OtsuLabelHand/' + name
        verified_download(path, url, pins.get(name))
        entry = dict(split=split, image=image, name=name, relative_path=relative.as_posix(),
                     url=url, bytes=path.stat().st_size, sha256=sha256(path))
        if name in pins and any(entry[k] != pins[name][k] for k in ('bytes', 'sha256', 'url')):
            raise ValueError(f'Released map provenance mismatch: {name}')
        return entry
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(fetch, [(s, row[0]) for s, rows in selected.items() for row in rows]))


def assert_reference_match(actual, expected):
    for key in COUNT_KEYS:
        if actual[key] != int(expected[key]):
            raise ValueError(f'Frozen reference count mismatch: {key}')
    if actual['prediction_sha256'] != expected['prediction_sha256']:
        raise ValueError('Frozen reference fingerprint mismatch')


def summarize(rows):
    result = aggregate(rows)
    n = result['scored_pixels']
    result['overall_accuracy'] = (result['tp'] + result['tn']) / n if n else None
    result['undefined_iou_chips'] = result['chips'] - result['defined_iou_chips']
    return result


def group_results(rows, prior):
    predicates = {
        'test_all_90': lambda r: r['split'] == 'test',
        'test_previously_inspected_10': lambda r: r['split'] == 'test' and r['image'] in prior,
        'test_new_80': lambda r: r['split'] == 'test' and r['image'] not in prior,
        'bolivia_separate_15': lambda r: r['split'] == 'bolivia',
    }
    summaries, events = {}, []
    for group, predicate in predicates.items():
        subset = [r for r in rows if predicate(r)]
        summaries[group] = {m: summarize([r for r in subset if r['method'] == m]) for m in METHODS}
        for event in sorted({r['event'] for r in subset}):
            for method in METHODS:
                events.append(dict(group=group, event=event, method=method,
                                   **summarize([r for r in subset if r['event'] == event and r['method'] == method])))
    return summaries, events


def paired_defined(rows, prior):
    """Secondary macro sensitivity on exactly the same defined chips per method."""
    cohorts = {
        'test_all_90': lambda r: r['split'] == 'test',
        'test_previously_inspected_10': lambda r: r['split'] == 'test' and r['image'] in prior,
        'test_new_80': lambda r: r['split'] == 'test' and r['image'] not in prior,
        'bolivia_separate_15': lambda r: r['split'] == 'bolivia',
    }
    result = {}
    for cohort, predicate in cohorts.items():
        subset = [r for r in rows if predicate(r)]
        included, excluded = [], []
        for name in sorted({r['image'] for r in subset}):
            chip_rows = {r['method']: r for r in subset if r['image'] == name}
            if set(chip_rows) != set(METHODS):
                raise ValueError('Paired sensitivity requires all methods')
            undefined = [m for m in METHODS if chip_rows[m]['iou'] is None]
            if undefined:
                excluded.append(dict(image=name, undefined_methods=undefined,
                                     reason='Zero TP+FP+FN; no reference or predicted water'))
            else:
                included.append(name)
        result[cohort] = dict(chips=len(included), included_images=included, excluded=excluded,
            macro_iou={m: summarize([r for r in subset if r['image'] in included and r['method'] == m])['macro_iou']
                       for m in METHODS})
    return result


def run_comparison(root, output, *, progress=False, coverage_policy="strict"):
    """Verify frozen reference and all map coverage before scoring any Otsu maps.

    A nonempty output or any historical evidence destination is rejected. Failed
    runs retain explicit status=failed. Intersection is an explicit opt-in; its
    coverage exclusions and full-mask frozen reference scores remain in evidence.
    """
    if coverage_policy not in ("strict", "intersection"):
        raise ValueError("coverage_policy must be strict or intersection")
    root, output = Path(root).resolve(), Path(output).resolve()
    for protected in ('reference',):
        if output.is_relative_to(root / protected):
            raise ValueError('Historical evidence is immutable')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a fresh, empty output directory')
    evidence = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    environment=environment_record(), source_sha256={p: sha256(root / p) for p in
                    ('water_mapping/core.py', 'water_mapping/data.py', 'water_mapping/otsu.py', 'scripts/compare_otsu.py', 'manifests/released_otsu_manifest.json')},
                    protocol=dict(coverage_policy=coverage_policy, mask='both SAR bands finite & label >= 0 & released Otsu valid; identical for all methods' if coverage_policy == 'intersection' else 'both SAR bands finite & label >= 0; identical for all methods',
                    scope='90 official test and 15 Bolivia chips; no training, tuning, or deep learning',
                    methods='Frozen validation-selected global VH thresholds versus released event-adaptive VH Otsu maps',
                    metrics='Shared confusion/aggregate core; undefined ratios null; mean defined chip IoU excludes undefined chips',
                    limitations=['Present-day v1.1 comparison, not exact paper reproduction.',
                    'Paper epsilon-based empty-chip IoU convention differs; we do not change our metric.',
                    'Previously inspected test and Bolivia results cannot become pristine holdouts.',
                    'Reading released maps is not Otsu inference; no runtime comparison.']))
    save_study(output, evidence)
    try:
        splits = load_splits(root)
        selected = {s: splits[s] for s in ('test', 'bolivia')}
        prior = {r[0] for r in recorded_selection(root)['test']}
        if len(prior) != 10 or not prior.issubset({r[0] for r in selected['test']}):
            raise ValueError('Prior test membership mismatch')
        evidence['protocol']['prior_inspected_test_images'] = sorted(prior)
        evidence['protocol']['split_sha256'] = {s: sha256(root / f'manifests/splits/flood_{s}_data.csv') for s in splits}
        reference_dir = root / REFERENCE
        frozen = read_json(reference_dir / 'frozen_config.json')
        for source, expected_hash in frozen['release_source_sha256'].items():
            if sha256(root / source) != expected_hash:
                raise ValueError(f'Frozen reference numerical source changed: {source}')
        for key, width in (('selected', 3), ('no_smoothing_baseline', 1)):
            if frozen[key]['size'] != width or frozen[key]['threshold_db'] != -22 or frozen['raw_threshold_policy'] != 'strict_db_v1':
                raise ValueError('Unexpected frozen reference configuration')
        with (reference_dir / 'evaluation.csv').open(encoding='utf-8', newline='') as stream:
            archive = list(csv.DictReader(stream))
        archive = [r for r in archive if r['split'] in selected]
        expected = {(r['split'], r['image'], r['method']): r for r in archive}
        expected_keys = {(s, row[0], m) for s, rs in selected.items() for row in rs for m in METHODS[:2]}
        if len(archive) != 210 or set(expected) != expected_keys:
            raise ValueError('Frozen reference rows missing, duplicated, or unexpected')
        evidence['frozen_reference'] = dict(directory=REFERENCE, configuration=frozen,
            files_sha256={p: sha256(reference_dir / p) for p in ('frozen_config.json', 'evaluation.csv')})
        evidence['input_manifest'] = acquire_pairs(root, selected)
        evidence['released_otsu_manifest'] = acquire_otsu(root, selected)
        checks, rows = [], []
        # First pass validates all predictions and reproduces all frozen results.
        for split, pairs in selected.items():
            for pair in pairs:
                chip = load_chip(root, pair)
                path = root / 'data/S1OtsuLabelHand' / otsu_name(chip.name)
                with tifffile.TiffFile(path) as pred_file, tifffile.TiffFile(root / 'data/S1Hand' / chip.name) as image_file:
                    tag = pred_file.pages[0].tags.get('GDAL_NODATA')
                    nodata = float(tag.value.rstrip('\x00')) if tag else None
                    values = pred_file.asarray()
                    try:
                        check = validate_prediction(values, chip.valid & (chip.labels >= 0),
                            pred_file.geotiff_metadata, image_file.geotiff_metadata, nodata, name=path.name, image=chip.name)
                    except CoverageError as exc:
                        check = exc.check
                missing = ~np.isfinite(values)
                if nodata is not None:
                    missing |= np.isnan(values) if np.isnan(nodata) else values == nodata
                excluded = missing & chip.valid & (chip.labels >= 0)
                check.update(excluded_water_pixels=int(np.count_nonzero(excluded & (chip.labels == 1))),
                             excluded_nonwater_pixels=int(np.count_nonzero(excluded & (chip.labels == 0))),
                             intersection_scored_pixels=check['scored_pixels'] - check['uncovered_scored_pixels'])
                checks.append(dict(split=split, image=chip.name, **check))
                for method, key in (('selected', 'selected'), ('no_smoothing', 'no_smoothing_baseline')):
                    config = frozen[key]
                    prediction = classify(chip.vh_db, smooth_power(chip.power, chip.valid, config['size']),
                                          chip.valid, config['size'], config['threshold_db'])
                    record = _row(split, chip, method, prediction)
                    assert_reference_match(record, expected[split, chip.name, method])
                    rows.append(record)
        evidence['input_checks'] = checks
        evidence['frozen_reference']['verification'] = dict(rows=210, counts_exact=True, prediction_fingerprints_exact=True)
        gaps = [c for c in checks if c['uncovered_scored_pixels']]
        evidence['coverage_summary'] = dict(chips=len(checks), affected_chips=len(gaps),
            uncovered_scored_pixels=sum(c['uncovered_scored_pixels'] for c in checks),
            scored_pixels=sum(c['scored_pixels'] for c in checks))
        evidence['coverage_summary']['by_split'] = {}
        for split in selected:
            group = [c for c in checks if c['split'] == split]
            total = sum(c['scored_pixels'] for c in group)
            lost = sum(c['uncovered_scored_pixels'] for c in group)
            evidence['coverage_summary']['by_split'][split] = dict(chips=len(group),
                affected_chips=sum(c['uncovered_scored_pixels'] > 0 for c in group),
                original_scored_pixels=total, intersection_scored_pixels=total-lost,
                excluded_pixels=lost, excluded_fraction=lost/total if total else None,
                excluded_water_pixels=sum(c['excluded_water_pixels'] for c in group),
                excluded_nonwater_pixels=sum(c['excluded_nonwater_pixels'] for c in group))
        if gaps and coverage_policy == "strict":
            raise ValueError(f"Released predictions missing on {len(gaps)} chips / {evidence['coverage_summary']['uncovered_scored_pixels']} common-mask pixels; decision needed. See input_checks.")
        if progress:
            print('All 105 released grids validated; all 210 full-mask frozen counts and fingerprints match.', flush=True)
        full_rows = rows
        rows = []
        for split, pairs in selected.items():
            for pair in pairs:
                chip = load_chip(root, pair)
                values = tifffile.imread(root / 'data/S1OtsuLabelHand' / otsu_name(chip.name))
                # Validation already established classes and explicit nodata. Binary
                # class membership is the prediction-coverage mask, never a label mask.
                nodata = next(c['nodata'] for c in checks if c['image'] == chip.name)
                if nodata == 'NaN':
                    nodata = float('nan')
                common_valid = chip.valid & prediction_valid(values, nodata)
                for method, key in (('selected', 'selected'), ('no_smoothing', 'no_smoothing_baseline')):
                    config = frozen[key]
                    prediction = classify(chip.vh_db, smooth_power(chip.power, chip.valid, config['size']),
                                          chip.valid, config['size'], config['threshold_db'])
                    rows.append(_row(split, chip, method, prediction, valid=common_valid))
                rows.append(_row(split, chip, 'released_otsu', values == 1, valid=common_valid))
                if len({r['scored_pixels'] for r in rows[-3:]}) != 1:
                    raise AssertionError('Method denominators differ')
        full_summaries, _ = group_results(full_rows, prior)
        evidence['frozen_reference']['full_mask_accuracy_summary'] = full_summaries
        evidence['frozen_reference']['full_mask_evaluation'] = full_rows
        rows.sort(key=lambda r: (r['split'], r['image'], METHODS.index(r['method'])))
        summaries, events = group_results(rows, prior)
        evidence['result'] = dict(evaluation=rows, accuracy_summary=summaries, event_summary=events)
        missing_water = sum(c['excluded_water_pixels'] for c in checks if c['split'] == 'bolivia')
        missing_nonwater = sum(c['excluded_nonwater_pixels'] for c in checks if c['split'] == 'bolivia')
        bolivia_otsu = summaries['bolivia_separate_15']['released_otsu']
        evidence['sensitivity'] = dict(
            paired_defined_chip=paired_defined(rows, prior),
            description=f'Hypothetical missing-pixel treatment, not recovered Otsu predictions: assign missing released pixels to nonwater ({missing_water} additional FN and {missing_nonwater} TN).',
            bolivia_original_mask={m: full_summaries['bolivia_separate_15'][m] for m in METHODS[:2]},
            bolivia_common_mask=summaries['bolivia_separate_15'],
            otsu_missing_as_nonwater_pooled=sensitivity_nonwater(bolivia_otsu, missing_water, missing_nonwater))
        evidence['status'] = 'complete'
        evidence['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save_study(output, evidence)
        return evidence
    except Exception as exc:
        evidence.pop('result', None)
        # A CSV written before a failed final JSON save is not a completed bundle.
        csv_path = output / 'evaluation.csv'
        if csv_path.exists():
            csv_path.replace(output / 'evaluation.incomplete.csv')
        evidence.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save_study(output, evidence)
        raise


def _row(split, chip, method, prediction, *, valid=None):
    counts = confusion(prediction, chip.labels, chip.valid if valid is None else valid)
    n = counts['scored_pixels']
    return dict(split=split, image=chip.name, event=chip.name.split('_')[0], method=method,
                prediction_sha256=hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest(),
                **counts, overall_accuracy=(counts['tp'] + counts['tn']) / n if n else None)
