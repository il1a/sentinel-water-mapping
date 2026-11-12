# Method and interpretation

## Question

Can a small, interpretable Sentinel-1 threshold identify labeled surface water,
and how does its smoothing stage behave on whole CPU arrays, CPU tiles and an
optional GPU? Accuracy and runtime are separate questions. No model is trained.

## Inputs and masks

Each input is a 512×512 chip with VV and VH radar bands in decibels. The paired
hand label is 1 for water, 0 for non-water and −1 for ignored pixels. The loader
checks shapes, types, classes, geographic reference and matching grids.

A SAR pixel is valid only when **both radar bands are finite**. This mask alone
controls smoothing. Scoring additionally requires a nonnegative reference label.
An ignored label therefore cannot affect the detector, but valid radar at that
location may contribute to a neighbor's average.

The official chip identities are disjoint across training (252), validation (89),
test (90) and Bolivia (15). Events are shared across ordinary splits, so these
are not independent event-held-out evaluations. No weak labels are used.

## Detector and selection

For smoothing widths 3 and 5, convert VH to float64 linear power:
`power = 10 ** (VH / 10)`. Average valid neighbors with a square box filter.
Edges use half-sample symmetric reflection, repeating the boundary pixel.
Round valid-neighbor counts to integers to avoid floating-point denominator
residue. A neighborhood with no valid samples stays undefined.

A valid center is water when its mean power is strictly less than
`10 ** (threshold_dB / 10)`. The width-1 baseline instead compares the original
VH dB value strictly with the threshold; this avoids a conversion-rounding
ambiguity for values exactly on that threshold.

Validation tests widths {1, 3, 5} crossed with thresholds
{−25, −22, −19, −17, −15} dB: **15 prespecified candidates**. It selects the
largest mean defined-chip IoU, retaining candidate order for ties. The raw-VH
baseline chooses its threshold independently from the width-1 candidates.
Settings and source hashes are written before test scoring.

Training scores are descriptive, not held-out accuracy. Ten test chips were
already inspected in the October pilot; their identities are retained in a
manifest. Full90, prior10 and additional80 are related summaries, not independent
experiments. Bolivia stays separate and is never used to select settings.

## Metrics

TP is correctly predicted water; FP is false water; FN is missed water; TN is
correctly predicted non-water.

- IoU = TP / (TP + FP + FN).
- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2 TP / (2 TP + FP + FN).

A zero denominator is undefined (`null`), not zero or a perfect score. Pooled
metrics add the counts before division. Macro IoU gives each chip with a defined
IoU equal weight and records how many chips contributed. Different methods may
have different defined-chip sets. A paired-defined-chip sensitivity comparison
therefore also reports the same narrower set for all methods.

Each full boolean prediction is packed in C order with NumPy `packbits`, then
SHA-256 hashed. Fingerprints and confusion counts allow exact comparison without
committing prediction rasters. Hash agreement checks consistency, not correctness
of the labels or algorithm.

## Released Otsu comparison

Otsu uses event-adaptive VH thresholds. This workflow **reads the authors' released
maps**, rather than generating them. The local detectors use the saved 3×3/−22 dB
and raw-VH/−22 dB settings from validation, with no new test-based search.

All 105 test/Bolivia maps must match the SAR grids and supported classes/nodata.
First, the fixed-detector original-mask counts and prediction fingerprints must
match `reference/evaluation.csv` for all 210 method/chip combinations. The checked
release source hashes must match the reference configuration as well.

The comparison uses `valid SAR AND valid hand label AND valid released map`
identically for all three methods. Missing Otsu values are not non-water. The
90-chip test has full coverage. Bolivia excludes 64 reference-water pixels,
leaving 2,824,370 rather than 2,824,434 scored pixels. This is only 0.00227%, but
the exclusion is not random with respect to class. Original-mask fixed-detector
results remain in the output bundle. Assigning the missing Otsu pixels to
non-water is a separate **hypothetical** sensitivity calculation, not a recovered
prediction.

This is a harmonized evaluation of released maps, not exact paper-score
reproduction. The paper's epsilon convention gives empty reference/prediction
pairs an IoU of 1; this project's macro convention excludes undefined ratios.
No Otsu runtime claim follows from reading its saved maps.

## Numerical agreement and timing

Evaluation streams one complete chip at a time. Tiled smoothing uses 128×128
interiors and an overlap border equal to the window radius. Whole/tiled means
must have identical NaN masks, agree within absolute tolerance 1e−12 and relative
tolerance 1e−10, and produce **zero prediction disagreements** on every chip.
Tiling retains the complete input/output chip arrays; it is not out-of-core.

The benchmark repeats three fixed training chips to form batches of 1, 4 and 16.
These are synthetic workloads, not additional independent observations. Nine
warmed timings use the same host, float64 data and thread policy. CUDA execution
is synchronized, and host-to-device, kernel, device-to-host and transfer-inclusive
times are recorded separately. GPU buffers are reused; CPU internals allocate
temporary arrays.

Timings exclude TIFF decoding, dB conversion, masks, classification, scoring,
first compilation and explicit buffer allocation. Transfer-inclusive smoothing
is not end-to-end mapping latency. GPU correctness tests use synthetic inputs
and repeated training chips; full-data whole/tiled checks are CPU checks.

Sampled process RSS may miss short peaks and includes prior allocations.
Explicit device-array bytes omit the CUDA context and allocator. Neither is a
per-method peak-memory measurement or evidence of lower peak memory.

## Reuse and limitations

Source links, split identities, hashes, standard GeoTIFF/CSV/JSON/notebook formats,
pinned packages, pure functions and tests make the experiment inspectable. They
support FAIR practice but do not establish formal FAIR certification or rights.
See [input provenance and rights](../manifests/README.md).

The detector is deliberately limited: low backscatter is not unique to water,
and rough water or vegetation may hide water. Surface-water labels do not prove
new inundation, flood damage or causality. Geographic-grid pixel counts are not
equal-area estimates. Shared events and spatial dependence limit statistical and
geographic generalization. Repeated execution cannot undo prior test inspection.

The release keeps only the reference evidence required by the runnable workflow.
Reports, the earlier pilot implementation and old execution bundles are omitted.
New runs write fresh output folders; they do not overwrite the reference inputs.
