# Repository guidance

Do not publish, change repository visibility, merge, choose a license, or share
raw data without the author's authorization.

Keep only reusable source, tests, notebook source, provenance manifests and the
small verification inputs. Do not add reports, execution outputs, raw TIFFs,
credentials, private machine paths, copied coursework or unscreened logs.
`data/`, `runs/` and Python environments stay ignored.

Scientific contract:
- Water observed during an event is not necessarily newly flooded land.
- Use official disjoint chip identities: 252/89/90/15 train/valid/test/Bolivia.
- Select only on validation over the declared 15 candidates. Freeze before test
  scoring. Keep the independently calibrated raw-VH baseline.
- Disclose the ten earlier-inspected test chips; retain full90/prior10/new80 and
  Bolivia separately. Shared events prevent an independent event-level claim.
- Smooth float64 linear power, using only SAR validity, reflected edges and
  integer-rounded neighbor counts. Labels affect scoring, never smoothing.
- Keep strict threshold semantics, undefined ratios, confusion counts and both
  pooled and mean-defined-chip IoU. Check numeric AND prediction agreement.
- The Otsu workflow reads released maps and uses an explicit common scoring mask.
  Nodata is never non-water. Keep coverage exclusions and sensitivity visible.
- Do not change reference numerical evidence to make a failed check pass.
- Timing is warmed smoothing only. Tiling is not out-of-core and RSS sampling is
  not a true peak-memory measurement. GPU remains optional.

Before committing:
1. Run unit tests, compileall, notebook validation and pip check.
2. Regenerate the source notebook after changing its builder. Execute a fresh
   notebook when changing the runnable workflow, with verified data where possible.
3. Compare full-run counts, frozen settings and fingerprints with reference inputs.
4. Check diff whitespace, tracked files, secrets/private paths and local links.
5. Verify the pushed commit and its CI; disclose unrun checks. Do not merge.

Do not claim security is guaranteed. Do not invent code or data rights.
