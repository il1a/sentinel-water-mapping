"""Generate the teaching notebook deterministically; do not execute experiments."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def add(kind, source):
    source = source.strip() + '\n'
    cell = nbf.v4.new_markdown_cell(source) if kind == 'md' else nbf.v4.new_code_cell(source)
    cell['id'] = hashlib.sha256((kind + source).encode()).hexdigest()[:12]
    cells.append(cell)

add('md', '''# Surface-water mapping with Sen1Floods11
## A reproducible threshold experiment and computational comparison

**Question:** How well does a simple Sentinel-1 VH backscatter threshold identify labeled surface water, and how do whole-array CPU, halo-tiled CPU, and optional GPU smoothing compare on the same machine?

The detector is deliberately simple. No learning model is trained. Accuracy and computational performance are separate questions: validation selects the detector; released event-adaptive Otsu maps provide a same-pixel threshold-method comparison; numerical checks and timing assess smoothing implementations. The labels describe water observed during flood events, not independently verified newly flooded land.

Read from top to bottom, then use **Restart Kernel and Run All**. The notebook calls the same importable functions as the command-line workflow. It does not maintain a second numerical implementation.''')
add('md', '''## 1. Execution contract

Use the project's documented Python environment. The first complete run may download approximately **0.7 GB** of public image/label files, plus the released Otsu classification maps for 105 evaluation chips. Later runs verify cached bytes against available hashes. A fresh kernel is important because thread settings must be set before importing scientific libraries.

Each run creates a fresh UTC timestamped subdirectory under ignored `runs/notebook/`. This prevents a skipped benchmark or a failed run from being confused with older output. The small verification files under `reference/` are read-only inputs, not output folders. CPU evaluation is sufficient; CUDA is optional and unavailable GPU support is reported rather than treated as an accuracy failure. Set `RUN_BENCHMARK = False` to skip timing, without skipping accuracy evaluation.''')
add('code', '''import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from pathlib import Path
from datetime import datetime, timezone
import sys
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'water_mapping/core.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Start Jupyter in the repository or its notebooks directory.')
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/notebook' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT.mkdir(parents=True, exist_ok=False)
RUN_BENCHMARK = True
print('Repository:', ROOT.name)
print('New outputs:', OUT.relative_to(ROOT))''')
add('code', '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from water_mapping.core import smooth_power, classify
from water_mapping.data import load_splits, recorded_selection, acquire_pairs, load_chip, read_json
from water_mapping.experiment import (protocol, calibrate, freeze,
                                      evaluate, environment_record)
from water_mapping.reporting import save_study
from water_mapping.otsu import run_comparison
from water_mapping.visualization import (example_figure, calibration_figure,
                                         summary_table, event_figure, timing_figure, memory_table)

environment = environment_record()
print('Python:', sys.version.split()[0], '| NumPy:', np.__version__)
print('Thread limits:', environment['thread_environment'])''')
add('md', '''## 2. Dataset, split roles, and provenance

Each paired chip contains a two-band Sentinel-1 SAR image and a hand-labeled water mask on a 512 x 512 grid. VH denotes cross-polarized radar backscatter. Water can have low backscatter, motivating the threshold, but low backscatter is not unique to water.

The official manifests contain **252 training, 89 validation, 90 test, and 15 separate Bolivia chips: 446 unique pairs**. Training chips are used for illustration and timing; their accuracy scores are descriptive. All 89 validation chips select settings. The 90-chip test is scored only after freezing those settings. Bolivia is reported separately rather than silently added to the test set.

Chip identity is disjoint across these manifests, but events occur across standard splits. This is not a geographically independent event-generalization test. Ten test chips were inspected in the earlier study. The full test and its complementary 80-chip extension are therefore reported separately from those ten; rerunning cannot make previously inspected data an untouched holdout.''')
add('code', '''splits = load_splits(ROOT)
pilot = recorded_selection(ROOT)
roles = {'train': 'Illustration/timing; descriptive accuracy',
         'valid': 'Select settings', 'test': 'Frozen evaluation',
         'bolivia': 'Separate external-region description'}
display(pd.DataFrame([{'split': k, 'chips': len(v), 'role': roles[k]}
                      for k, v in splits.items()]).set_index('split'))
run_protocol = protocol(ROOT)
save_study(OUT, {'status': 'declared', 'protocol': run_protocol, 'environment': environment})
print('Earlier inspected test chips:', len(run_protocol['prior_inspected_test_images']))''')
add('code', '''# Downloads missing files only; verifies cached files and existing pinned hashes.
input_manifest = acquire_pairs(ROOT, splits)
print(f"Verified {len(input_manifest)} TIFFs, {sum(x['bytes'] for x in input_manifest) / 1e9:.3f} GB.")''')
add('md', '''### Validity and scoring masks

The loader checks dimensions, data types, label values, geographic reference, and matching image/label grids. Labels are **1 = water, 0 = non-water, -1 = ignored**. A SAR pixel is valid only when both bands are finite.

Smoothing uses the SAR validity mask, never the reference labels. Scoring then intersects valid SAR centers with labels 0 or 1. Thus ignored labels do not enter the confusion counts, but valid SAR observations at those locations can contribute to neighboring means. These distinct masks avoid using ground truth to construct the detector.''')
add('md', '''## 3. Detector: average power, then apply a strict threshold

Decibels are logarithmic. Convert VH from dB to linear power using `10 ** (VH / 10)` before averaging a neighborhood. A masked mean divides by the number of valid neighbors; empty neighborhoods remain undefined. Image edges use half-sample symmetric reflection, repeating the edge sample. Only valid center pixels can be classified as water.

For widths 3 and 5, classify water when smoothed power is strictly below `10 ** (threshold_dB / 10)`. Width 1 is the unsmoothed baseline and compares raw VH directly with the dB threshold. This explicit strict-dB convention avoids a portability issue when power conversion rounds an exactly-on-threshold raw value to either side.''')
add('code', '''# A teaching calculation, not an alternative detector implementation.
example_db = np.array([-30., -10.])
mean_db = example_db.mean()
db_of_mean_power = 10 * np.log10(np.mean(10 ** (example_db / 10)))
print(f'Mean of dB values: {mean_db:.2f} dB')
print(f'dB of mean linear power: {db_of_mean_power:.2f} dB')
print('These are different operations; the detector uses the second.')''')
add('md', '''### A fixed training example

The first recorded training chip is selected by identity, not by visual quality or test performance. The illustration uses a prespecified 5 x 5 window and -22 dB threshold to explain the computation. It is not evidence that these settings will win the expanded validation search. Gray background shows VH; blue indicates water in the two masks. The reference and prediction panels share the scoring mask: white indicates an ignored reference label or invalid SAR. In the VH panel, white indicates invalid SAR only. The full prediction remains available from the shared detector on all valid SAR centers, including locations without reference labels. Display scaling affects only the grayscale picture, not classification.''')
add('code', '''example = load_chip(ROOT, pilot['train'][0])
example_mean = smooth_power(example.power, example.valid, 5)
example_prediction = classify(example.vh_db, example_mean, example.valid, 5, -22)
fig = example_figure(example, example_prediction)
fig.savefig(OUT / 'training_example.png', dpi=150, bbox_inches='tight')
plt.show()
print('Scored pixels:', example.check['scored_pixels'], '| ignored labels:', example.check['label_ignore_pixels'])''')
add('md', '''## 4. Select on all 89 validation chips, then freeze

The fixed search contains **15 candidates**: widths {1, 3, 5} crossed with thresholds {-25, -22, -19, -17, -15} dB. No test score enters this search.

Water intersection over union (**IoU**) is TP / (TP + FP + FN), where TP, FP, and FN are true-positive, false-positive, and false-negative water pixels. **Macro IoU** averages defined chip IoUs with equal chip weight. A chip with no true or predicted water has an undefined IoU and is omitted from that mean, with its exclusion count reported. **Pooled IoU** first sums the confusion counts across chips; large water regions therefore contribute more. Both range from 0 to 1, with larger values indicating better agreement.

Selection maximizes validation macro IoU; ties retain the fixed candidate order. The unsmoothed comparator chooses its own best threshold within width 1 under the same objective. This avoids handicapping the baseline with a threshold chosen for a different filter.''')
add('code', '''candidates, validation_details = calibrate(ROOT, splits['valid'], progress=False)
fig = calibration_figure(candidates)
fig.savefig(OUT / 'validation_grid.png', dpi=150, bbox_inches='tight')
plt.show()''')
add('code', '''# Persist before calling evaluate: this is the boundary between selection and scoring.
frozen = freeze(ROOT, OUT, candidates)
display(pd.DataFrame([{'method': label, 'width': frozen[key]['size'],
                       'threshold_dB': frozen[key]['threshold_db'],
                       'validation_macro_IoU': frozen[key]['macro_iou'],
                       'defined_validation_IoUs': frozen[key]['aggregate']['defined_iou_chips']}
                      for label, key in [('Selected', 'selected'),
                                         ('Unsmoothed', 'no_smoothing_baseline')]]).set_index('method').round(4))
print('Saved', (OUT / 'frozen_config.json').relative_to(ROOT), 'before test scoring.')''')
add('md', '''## 5. Apply frozen settings to all 446 chips

This pass loads one image/label pair at a time and retains compact counts and checks, not a full split of image arrays. Both selected and independently tuned unsmoothed detectors are evaluated. Whole-array and halo-tiled CPU predictions are also checked for every chip.

The full 90-chip test contains both the 10 previously inspected chips and the 80-chip extension. Their scores are related summaries, not independent experiments. Training and validation scores are descriptive and must not be presented as held-out accuracy. Bolivia's 15 chips remain separate.''')
add('code', '''# Read the persisted settings rather than selecting again from evaluation results.
frozen = read_json(OUT / 'frozen_config.json')
result = evaluate(ROOT, splits, frozen, progress=False)
print(f"Evaluated {len(result['input_checks'])} unique chips with two frozen detectors.")''')
add('md', '''### Released Otsu maps: a same-pixel comparison

Otsu chooses a VH threshold adaptively for an event; our two detectors use global thresholds fixed by validation. We evaluate the authors' released classification maps, without generating Otsu thresholds or retuning either detector.

The shared comparison function loads the saved reference **3 x 3, -22 dB** and **raw VH, -22 dB** settings. It verifies every recomputed fixed-detector count and prediction fingerprint against the original full run. Before scoring, all 105 released maps must match the SAR grids and supported classes/nodata. The comparison uses the original scoring mask intersected with released-Otsu validity, identically for all three methods. Missing predictions are not treated as non-water.

The ordinary 90-chip test has complete coverage, so its mask is unchanged. In Bolivia, the common mask excludes **64 reference-water pixels**, or **0.00227%** of the original 2,824,434 scored pixels, leaving **2,824,370**. These exclusions are small but not random with respect to reference class. Original-mask fixed-detector results remain preserved, and a conservative hypothetical sensitivity check below describes the effect of assigning those missing Otsu predictions to non-water.

This is a **harmonized evaluation of released maps** on current v1.1 data, not an exact reproduction of the paper's scores. The paper's epsilon convention gives IoU 1 to an empty reference/prediction pair; our macro IoU excludes undefined chip ratios. Defined-chip counts therefore remain visible. Reading predictions does not measure Otsu generation time, so no Otsu runtime comparison is made.''')
add('code', '''OTSU_OUT = OUT / 'released_otsu'
otsu = run_comparison(ROOT, OTSU_OUT, progress=False, coverage_policy='intersection')
verification = otsu['frozen_reference']['verification']
assert verification['counts_exact'] and verification['prediction_fingerprints_exact']
for group, methods in otsu['frozen_reference']['full_mask_accuracy_summary'].items():
    for method in ('selected', 'no_smoothing'):
        for key, value in result['accuracy_summary'][group][method].items():
            assert methods[method][key] == value, (group, method, key)
coverage = otsu['coverage_summary']['by_split']
assert coverage['test']['excluded_pixels'] == 0
assert coverage['bolivia']['excluded_pixels'] == coverage['bolivia']['excluded_water_pixels'] == 64
assert coverage['bolivia']['intersection_scored_pixels'] == 2824370
# The ordinary test mask is unchanged; all three overlapping test cohorts must match.
for group, methods in otsu['result']['accuracy_summary'].items():
    if group == 'bolivia_separate_15':
        continue
    for method in ('selected', 'no_smoothing'):
        for key, value in result['accuracy_summary'][group][method].items():
            # Sorted chip order can change the final bit of a floating-point mean.
            if key == 'macro_iou' and value is not None:
                assert np.isclose(methods[method][key], value, rtol=0, atol=1e-15)
            else:
                assert methods[method][key] == value, (group, method, key)
comparison = summary_table(otsu['result']['accuracy_summary'],
                           ['test_all_90', 'test_previously_inspected_10',
                            'test_new_80', 'bolivia_separate_15'])
comparison = comparison.drop(columns=['chips', 'F1']).rename(index={
    'test_all_90': 'Full test (90)',
    'test_previously_inspected_10': 'Previously inspected (10)',
    'test_new_80': 'Additional test (80)',
    'bolivia_separate_15': 'Bolivia separately (15)',
    'selected': 'Fixed 3x3 / -22 dB',
    'no_smoothing': 'Fixed raw VH / -22 dB',
    'released_otsu': 'Released Otsu'})
print('Common-mask comparison: test unchanged; Bolivia excludes 64 missing reference-water pixels.')
display(comparison)
print(f"All {len(otsu['input_checks'])} released maps passed grid/class checks; "
      f"{verification['rows']} frozen-detector count and fingerprint checks matched.")
paired = otsu['sensitivity']['paired_defined_chip']['test_all_90']
display(Markdown(
    f"**Paired-chip sensitivity:** on the same {paired['chips']} test chips with defined IoU "
    "for every method, macro IoU is "
    f"{paired['macro_iou']['selected']:.4f} (3x3), "
    f"{paired['macro_iou']['no_smoothing']:.4f} (raw VH), and "
    f"{paired['macro_iou']['released_otsu']:.4f} (Otsu). "
    "This secondary check uses a narrower chip set; it does not replace the primary metric."))
original = otsu['sensitivity']['bolivia_original_mask']
common_otsu = otsu['sensitivity']['bolivia_common_mask']['released_otsu']
scenario = otsu['sensitivity']['otsu_missing_as_nonwater_pooled']
display(Markdown(
    "**Bolivia mask sensitivity.** Original-mask pooled/macro IoUs were "
    f"{original['selected']['iou']:.6f}/{original['selected']['macro_iou']:.6f} (3x3) and "
    f"{original['no_smoothing']['iou']:.6f}/{original['no_smoothing']['macro_iou']:.6f} (raw VH). "
    "The table above uses the common mask instead. If the 64 missing Otsu predictions "
    "were hypothetically assigned to non-water, adding 64 false negatives, its pooled IoU "
    f"would be {scenario['iou']:.6f} on the original mask, versus "
    f"{common_otsu['iou']:.6f} on the common mask. This scenario is not an observed prediction."))
print('Per-chip counts and per-event summaries:', OTSU_OUT.relative_to(ROOT))''')
add('md', '''Precision is the fraction of predicted water pixels that are labeled water; recall is the fraction of labeled water pixels recovered. F1 combines them as their harmonic mean. Undefined denominators remain missing, not zero. The table shows the number of defined chip IoUs so the macro denominator is visible. On the full test, Otsu has 85 defined chip IoUs and each fixed detector has 88: these macro means average different chip sets. Pooled scores use the identical pixel mask.

The following table is deliberately separated: training was available for development and validation chose the settings. Neither row estimates untouched-test performance.''')
add('code', '''display(summary_table(result['accuracy_summary'], ['train', 'valid']))
fig = event_figure(result['event_summary'])
fig.savefig(OUT / 'test_event_scores.png', dpi=150, bbox_inches='tight')
plt.show()''')
add('md', '''The event breakdown reveals heterogeneity that a single pooled score can obscure. It is descriptive: chips within events and neighboring pixels are correlated, event sample sizes differ, and these bars are not confidence intervals. A good aggregate result does not imply reliable detection in every event.''')
add('md', '''## 6. Numerical agreement precedes performance interpretation

Halo tiling adds a border around each 128 x 128 interior tile so its neighborhood calculation includes adjacent pixels. The halo radius is half the odd window width. Whole-array and tiled means are compared with absolute tolerance 1e-12 and relative tolerance 1e-10, including matching undefined-value masks. Final water predictions must agree exactly.

Small floating-point differences can occur because operations are accumulated in different orders. Close intermediate values alone do not guarantee identical predictions near a threshold, so prediction disagreement is checked separately. Tiling still holds complete chip input/output arrays in memory; this is not an out-of-core algorithm or evidence of lower peak memory.''')
add('code', '''agreement = result['cpu_equivalence']
assert all(r['numeric_allclose'] and r['nan_masks_equal'] and
           r['prediction_disagreements'] == 0 for r in agreement)
print('Whole/tiled checks passed:', len(agreement))
print('Prediction disagreements:', sum(r['prediction_disagreements'] for r in agreement))
print('Largest absolute power difference:',
      max((r['max_abs_power_error'] or 0) for r in agreement))''')
add('md', '''## 7. Optional warm CPU/GPU timing on this machine

Batches of 1, 4, and 16 repeat three fixed training chips. They test computational workload, not additional independent data. Nine timed repetitions follow warmup. CPU comparisons share this host and thread policy.

GPU kernel-only time includes synchronized launch overhead. Transfer-inclusive time includes host-to-device transfer, smoothing, and device-to-host transfer using reused buffers. These are **smoothing-stage timings**, excluding disk decoding, dB conversion, classification, scoring, first compilation, and explicit buffer allocation. CPU smoothing still allocates its internal temporary arrays. Transfer-inclusive timing is not end-to-end mapping latency.

CUDA failure leaves the CPU experiment usable and is reported with its reason. Resident-set size (RSS) is process-wide memory sampled in separate untimed passes. Existing arrays and allocator reuse affect the baseline, and the requested 1 ms sampling interval can be longer under operating-system scheduling, missing short peaks. Sample counts are therefore shown. The compact RSS table shows observations, not per-method peaks. Explicit device-buffer bytes exclude context/allocator overhead; neither measure establishes peak-memory savings.''')
add('code', '''benchmark = None
if RUN_BENCHMARK:
    from water_mapping.benchmark import run_benchmark
    benchmark = run_benchmark(ROOT, width=frozen['selected']['size'],
                              threshold_db=frozen['selected']['threshold_db'], repeats=9)
    print('CUDA available:', benchmark['cuda_available'])
    if not benchmark['cuda_available']:
        print('CUDA reason:', benchmark.get('cuda_unavailability', 'Unavailable'))
    for batch in benchmark['batches']:
        for key in ('cpu_tiled_equivalence', 'gpu_equivalence'):
            if key in batch:
                check = batch[key]
                assert check['allclose'] and check['same_nan_mask'] and check['prediction_disagreements'] == 0
        if 'cuda_error' in batch:
            print(f"Batch {batch['batch']} CUDA issue: {batch['cuda_error']}")
    fig = timing_figure(benchmark)
    fig.savefig(OUT / 'smoothing_timings.png', dpi=150, bbox_inches='tight')
    plt.show()
    observations = memory_table(benchmark)
    if not observations.empty:
        display(observations)
        print('Process RSS sampled in separate untimed passes; these are not per-method peaks.')
    print('Available whole/tiled/GPU agreement checks passed for the benchmark batches.')
else:
    print('Timing skipped; no GPU or speed claim is made by this run.')''')
add('md', '''## 8. Findings from this run

The next cell reports measured values from the outputs above. It does not infer statistical significance or generalization beyond the evaluated splits. The fixed search and prior test inspection remain part of the interpretation even if a rerun reproduces every number.''')
add('code', '''save_study(OUT, {'status': 'complete', 'protocol': run_protocol, 'environment': environment,
                 'input_manifest': input_manifest,
                 'validation_candidates': candidates, 'validation_details': validation_details,
                 'configuration': frozen, 'result': result, 'benchmark': benchmark,
                 'released_otsu_comparison': {'directory': 'released_otsu',
                     'status': otsu['status'],
                     'evidence_files': ['study.json', 'evaluation.csv']}})
s = frozen['selected']
test = result['accuracy_summary']['test_all_90']
new = result['accuracy_summary']['test_new_80']['selected']
selected, baseline = test['selected'], test['no_smoothing']
released = otsu['result']['accuracy_summary']['test_all_90']['released_otsu']
fmt = lambda x: 'undefined' if x is None else f'{x:.4f}'
display(Markdown(
    f"Validation selected **{s['size']} x {s['size']} at {s['threshold_db']:g} dB**. "
    f"On all 90 test chips, selected-detector macro IoU was **{fmt(selected['macro_iou'])}**, "
    f"compared with **{fmt(baseline['macro_iou'])}** for the independently tuned unsmoothed baseline. "
    f"Selected pooled IoU was **{fmt(selected['iou'])}**. "
    f"The complementary 80-chip group's macro IoU was **{fmt(new['macro_iou'])}**. "
    f"Released Otsu macro IoU was **{fmt(released['macro_iou'])}** and pooled IoU "
    f"was **{fmt(released['iou'])}** on the identical full-test scoring mask. "
    "These are observed comparisons, not evidence of statistical significance or unseen-event performance."))
print('Complete counts, settings, provenance, checks, and figures:', OUT.relative_to(ROOT))''')
add('md', '''### Limits and next steps

- **Target:** These are labeled surface-water observations. A flood-change claim needs a defensible pre-event reference and change definition. Low radar backscatter can also arise from non-water surfaces; rough water or vegetation can hide water.
- **Evaluation:** The 15-setting grid is small and validation-specific. Previously inspected test chips stay disclosed. Shared events and spatial dependence limit broad generalization; a future study could prespecify event-held-out evaluation and suitable uncertainty estimation.
- **Scale:** Evaluating 446 chips demonstrates a reproducible, sequential workflow, not distributed execution. Repeated-chip timing is a workload comparison. It does not establish operational throughput, lower peak memory, or performance on other hardware.
- **Geography:** EPSG:4326 grid pixel counts are not area estimates. Surface-area reporting requires appropriate geospatial treatment.
- **Reproducibility:** Matching hashes identifies the input bytes used; it is not independent proof that labels are correct. Saved environments, settings, per-chip confusion counts, and checks make the result inspectable. Hardware-sensitive timing should be rerun on the deployment machine.

## Sources and artifact map

Dataset and official split source: [cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).
Dataset paper: Bonafilia et al. (2020), *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, [CVPR Workshops open-access paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html). The dataset's learning applications do not imply that this notebook trains a model.

- `water_mapping/core.py`: shared numerical detector and metrics.
- `water_mapping/data.py`: manifests, retrieval, hashes, and input checks.
- `water_mapping/experiment.py`: calibration, freezing, and evaluation.
- `water_mapping/otsu.py`: released-map validation, frozen-reference verification, and harmonized scoring.
- `water_mapping/benchmark.py`: optional CUDA and same-host timing.
- `water_mapping/reporting.py`: writes the compact study bundle without changing numerical results.
- `runs/notebook/<UTC timestamp>/`: each run's separate bundle; use the output path printed above. `study.json` contains the complete nested evidence, `frozen_config.json` records settings saved before test scoring, and `evaluation.csv` provides flat per-chip results. Four PNG figures support visual review when benchmarking is enabled. A skipped benchmark produces three figures and is recorded as null in the study bundle. The `released_otsu/` subdirectory has its own `study.json` and `evaluation.csv` with all three methods, exact counts, metric denominators, per-event summaries, released-map checks and hashes.
- `reference/` contains only the frozen configuration and per-chip counts/fingerprints needed to check the released-map comparison. Reports and old execution outputs are not part of this release.

**Reuse and FAIR:** Public citations and manifests support findability and access; GeoTIFF, CSV, JSON, and notebook formats support interoperability; shared functions, declared conventions, environment records, and checks support reuse. These concrete practices are not a formal FAIR certification.

Data licensing has not been independently resolved here. This repository records source URLs and hashes rather than redistributing the TIFFs; check upstream terms before redistribution.''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'}})
nbf.validate(notebook)
path = ROOT / 'notebooks/surface_water_mapping.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(nbf.writes(notebook), encoding='utf-8', newline='\n')
print(f'Wrote {path.name}: {len(cells)} cells, validated, no execution.')
