# Reference inputs

These two files are the minimum saved evidence required by the Otsu comparison:

- `frozen_config.json` records the validation-selected 3×3 / −22 dB detector and
  independently selected raw-VH / −22 dB baseline.
- `evaluation.csv` records the original full-run per-chip counts and prediction
  fingerprints for the two detectors. The comparison uses its 90 test and
  15 Bolivia chips; the other rows allow a full-run equivalence check.

They contain no imagery or prediction rasters. A fingerprint is a SHA-256 hash
of a packed boolean prediction, not a downloadable prediction product.

The settings and counts originate from the verified 10 November 2026 experiment.
Release preparation keeps those numerical results unchanged. The configuration
retains its original source hashes and separately records the release source
hashes checked after refactoring file layout and download handling. This is not
a claim that the test was unseen during release preparation.

Treat this folder as read-only. New experiments write to `runs/`, never here.
The comparison fails if reference counts, predictions or checked release sources
differ. If you change the algorithm, make a new, explicitly labeled study rather
than editing these records to suppress a failed check.
