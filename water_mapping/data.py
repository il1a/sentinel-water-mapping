"""Public-data provenance, split validation, and checked GeoTIFF loading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import time
import tempfile
import urllib.request

import numpy as np
import tifffile

BASE_URL = 'https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/'
SPLIT_COUNTS = {'train': 252, 'valid': 89, 'test': 90, 'bolivia': 15}


def sha256(path: Path) -> str:
    """Hash bytes in bounded blocks rather than reading a large file at once."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def load_splits(root: Path) -> dict[str, list[list[str]]]:
    """Enforce the official 446 distinct pairs, including a separate Bolivia split."""
    splits = {}
    seen = set()
    for split, expected in SPLIT_COUNTS.items():
        path = Path(root) / 'manifests' / 'splits' / f'flood_{split}_data.csv'
        rows = list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines()))
        if len(rows) != expected:
            raise ValueError(f'{split}: expected {expected} pairs, found {len(rows)}')
        for row in rows:
            if len(row) != 2 or not re.fullmatch(r'[A-Za-z-]+_[0-9]+_S1Hand\.tif', row[0]):
                raise ValueError(f'Invalid manifest row: {row}')
            if row[1] != row[0].replace('_S1Hand.tif', '_LabelHand.tif'):
                raise ValueError(f'Image/label identity mismatch: {row}')
            if row[0] in seen:
                raise ValueError(f'Duplicate or overlapping chip: {row[0]}')
            seen.add(row[0])
        splits[split] = rows
    return splits


def recorded_selection(root: Path) -> dict[str, list[list[str]]]:
    return read_json(Path(root) / 'manifests/illustration_and_prior_test.json')['selected']


def verified_download(path: Path, url: str, expected: dict, *, attempts: int = 3) -> None:
    """Accept only pinned bytes; stream to a unique temporary file before promotion.

    The published hash is a reproducibility pin, not independent authenticity proof.
    Failed downloads never become cached input and leave no temporary file behind.
    """
    if not isinstance(expected, dict):
        raise ValueError('Missing pinned input metadata')
    size, digest = expected.get('bytes'), expected.get('sha256')
    if (type(size) is not int or size <= 0 or not isinstance(digest, str)
            or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or expected.get('url') != url or not url.startswith(BASE_URL)):
        raise ValueError('Invalid pinned input metadata')
    path = Path(path)
    if path.exists():
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f'Cached data hash mismatch: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                    prefix=path.name + '.', suffix='.part', delete=False) as target:
                temporary = Path(target.name)
                received = 0
                checksum = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=90) as source:
                    while block := source.read(min(1024 * 1024, size - received + 1)):
                        received += len(block)
                        if received > size:
                            raise ValueError(f'Download exceeds pinned size: {path.name}')
                        checksum.update(block)
                        target.write(block)
                if received != size or checksum.hexdigest() != digest:
                    raise ValueError(f'Download hash mismatch: {path.name}')
            temporary.replace(path)
            return
        except (OSError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def contained_data_path(root: Path, relative: Path) -> Path:
    """Resolve filesystem links and check containment using equivalent Windows paths.

    On Windows, resolving a missing path while its parent is being created can
    retain the extended-length prefix. Normalize only that equivalent spelling;
    do not replace the resolved path with an unchecked lexical path.
    """
    def canonical(path):
        resolved = Path(path).resolve()
        text = str(resolved)
        if os.name == 'nt':
            if text.startswith('\\\\?\\UNC\\'):
                resolved = Path('\\\\' + text[8:])
            elif text.startswith('\\\\?\\') and len(text) >= 7 and text[5:7] == ':\\':
                resolved = Path(text[4:])
        return resolved

    root = canonical(root)
    path = canonical(root / relative)
    if not path.is_relative_to(root / 'data'):
        raise ValueError('Input path escapes data directory')
    return path


def acquire_pairs(root: Path, selected: dict, *, workers: int = 6) -> list[dict]:
    """Download or reuse public TIFFs only when pinned size and SHA256 match."""
    root = Path(root).resolve()
    pins = {}
    for filename in ['full_input_manifest.json']:
        path = root / 'manifests' / filename
        if path.exists():
            pins.update({item['name']: item for item in read_json(path)})

    def acquire(job):
        split, name = job
        if not re.fullmatch(r'[A-Za-z-]+_[0-9]+_(S1Hand|LabelHand)\.tif', name):
            raise ValueError(f'Invalid data filename: {name}')
        kind = 'S1Hand' if name.endswith('_S1Hand.tif') else 'LabelHand'
        relative = Path('data') / kind / name
        path = contained_data_path(root, relative)
        url = BASE_URL + kind + '/' + name
        verified_download(path, url, pins.get(name))
        return dict(split=split, name=name, relative_path=relative.as_posix(),
                    url=url, bytes=path.stat().st_size, sha256=sha256(path))

    jobs = [(split, name) for split, rows in selected.items() for row in rows for name in row]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(acquire, jobs))


@dataclass
class Chip:
    name: str
    vh_db: np.ndarray
    power: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    check: dict


def load_chip(root: Path, row: list[str]) -> Chip:
    """Read one pair; never use labels to construct the SAR smoothing mask."""
    root = Path(root)
    with tifffile.TiffFile(root / 'data/S1Hand' / row[0]) as image_file, \
            tifffile.TiffFile(root / 'data/LabelHand' / row[1]) as label_file:
        image, labels = image_file.asarray(), label_file.asarray()
        image_geo, label_geo = image_file.geotiff_metadata, label_file.geotiff_metadata
    if image.shape != (2, 512, 512) or labels.shape != (512, 512):
        raise ValueError(f'Unexpected dimensions: {row[0]}')
    if image.dtype != np.float32 or labels.dtype != np.int16:
        raise ValueError(f'Unexpected data types: {row[0]}')
    if not set(np.unique(labels)).issubset({-1, 0, 1}):
        raise ValueError(f'Unexpected label class: {row[1]}')
    if image_geo.get('GeographicTypeGeoKey') != 4326 or label_geo.get('GeographicTypeGeoKey') != 4326:
        raise ValueError(f'Expected EPSG:4326: {row[0]}')
    for key in ['ModelPixelScale', 'ModelTiepoint']:
        if not np.allclose(image_geo[key], label_geo[key], rtol=0, atol=1e-12):
            raise ValueError(f'Image/label grid mismatch ({key}): {row[0]}')
    vh_db = image[1].astype(np.float64)
    valid = np.isfinite(image).all(axis=0)
    with np.errstate(over='ignore', invalid='ignore'):
        power = np.power(10.0, vh_db / 10.0)
    if not np.isfinite(power[valid]).all():
        raise ValueError(f'Nonfinite converted SAR power: {row[0]}')
    check = dict(image=row[0], shape=list(image.shape),
                 label_ignore_pixels=int((labels < 0).sum()),
                 finite_sar_pixels=int(valid.sum()),
                 scored_pixels=int((valid & (labels >= 0)).sum()))
    return Chip(row[0], vh_db, power, valid, labels, check)
