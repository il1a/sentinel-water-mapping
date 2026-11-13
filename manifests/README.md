# Input provenance and rights

The code uses [Sen1Floods11 v1.1](https://github.com/cloudtostreet/Sen1Floods11),
created by Cloud to Street. It downloads public files directly from
`https://storage.googleapis.com/sen1floods11/v1.1/` without credentials.

- `splits/` contains the official hand-labeled split CSVs: 252 training,
  89 validation, 90 test and 15 separate Bolivia image/label pairs.
- `full_input_manifest.json` pins names, public URLs, byte sizes and SHA-256 hashes
  for all 892 S1Hand/LabelHand TIFFs.
- `released_otsu_manifest.json` pins the 105 S1OtsuLabelHand maps used in the
  test/Bolivia comparison.
- `illustration_and_prior_test.json` preserves three training examples and the
  ten test identities inspected in the earlier pilot. It is not another split.

Upstream documents the layers and splits in its
[dataset documentation](https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md).
Hashes were observed from the retrieved bytes; they identify the experiment's
inputs but do not independently prove authenticity, label quality or ownership.
Downloads fail if pinned sizes or hashes differ. Do not edit hashes simply to
make a failed download pass: establish why upstream bytes changed first.

## Data rights

The upstream README requests citation but does not itself provide a complete
reuse/redistribution grant. As checked on 13 November 2026, the upstream
repository has no root license file,
its [missing-license issue](https://github.com/cloudtostreet/Sen1Floods11/issues/18)
remains unresolved, and both the official
[source collection](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_source/collection.json)
and [hand-label collection](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_label/collection.json)
declare `license: proprietary`. These declarations do not establish terms for
all other assets; public availability alone is not a redistribution license.
Confirm the current terms for imagery, hand labels, released maps and copied
split metadata with the authoritative provider before redistribution or a use
that requires permission. This project does not assign a license to their work.

This repository contains file identities, source links and hashes, not raw dataset
files or embedded dataset preview images. Dataset rights are separate from any
license covering this project’s code. The project's [MIT License](../LICENSE)
does not cover these upstream materials, including the copied split CSVs.

Please cite Bonafilia, D., Tellman, B., Anderson, T. and Issenberg, E. (2020),
*Sen1Floods11: a georeferenced dataset to train and test deep learning flood
algorithms for Sentinel-1*, IEEE/CVF CVPR Workshops, pp. 210–211.
