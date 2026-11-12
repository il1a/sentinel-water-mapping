"""CPU-only numerical core; importing this module performs no experiment or I/O.

Smoothing operates in float64 linear power, never in decibels. Masks describe
SAR validity, not reference labels. SciPy's reflect mode is half-sample
symmetric (the edge sample is repeated). Invalid neighbors have zero weight.
"""
from numbers import Integral

import numpy as np
from scipy.ndimage import uniform_filter


def _positive_integer(value, name, odd=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0 or (odd and value % 2 == 0):
        raise ValueError(f"{name} must be positive" + (" and odd" if odd else ""))
    return int(value)


def _arrays(*arrays):
    result = [np.asarray(array) for array in arrays]
    if any(a.ndim != 2 or 0 in a.shape for a in result):
        raise ValueError("inputs must be nonempty 2-D arrays")
    if any(a.shape != result[0].shape for a in result[1:]):
        raise ValueError("input shapes must match")
    return result


def smooth_power(power, valid, width):
    """Return the masked reflected-window mean of linear power as float64.

    Valid-neighbor counts are rounded to integers before division, matching
    the historical experiment and avoiding tiny positive denominators in
    empty windows. Empty windows produce NaN. An invalid center can receive
    a finite mean from neighbors; classification separately excludes it.
    Width must be a positive odd integer. Even a 1x1 image is supported.
    """
    width = _positive_integer(width, "width", odd=True)
    power, valid = _arrays(power, valid)
    power = power.astype(np.float64, copy=False)
    valid = valid.astype(bool, copy=False)
    if np.any(valid & (~np.isfinite(power) | (power < 0))):
        raise ValueError("valid linear-power samples must be finite and nonnegative")
    if width == 1:
        return np.where(valid, power, np.nan)
    numerator = uniform_filter(np.where(valid, power, 0.), size=width, mode="reflect")
    area = width * width
    denominator = np.rint(uniform_filter(valid.astype(np.float64), size=width,
                                        mode="reflect") * area) / area
    result = np.full(power.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def tiled_mean(power, valid, width, tile=128):
    """Compute the same mean using radius-sized halos around CPU tiles.

    The complete input and output arrays remain resident in memory. This is
    halo-correct tiled computation, not an out-of-core implementation or a
    claim of lower peak memory. Floating-point running sums can differ from
    whole-array smoothing, so compare means with an explicit tolerance.
    """
    width = _positive_integer(width, "width", odd=True)
    tile = _positive_integer(tile, "tile")
    power, valid = _arrays(power, valid)
    result = np.empty(power.shape, dtype=np.float64)
    h, w = power.shape
    radius = width // 2
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            ey, ex = min(y + tile, h), min(x + tile, w)
            ya, xa = max(0, y - radius), max(0, x - radius)
            yb, xb = min(h, ey + radius), min(w, ex + radius)
            local = smooth_power(power[ya:yb, xa:xb], valid[ya:yb, xa:xb], width)
            result[y:ey, x:ex] = local[y-ya:ey-ya, x-xa:ex-xa]
    return result


def classify(vh_db, smoothed_power, valid, width, threshold_db, *, legacy_raw_power=False):
    """Predict water using a strict threshold and require a valid SAR center.

    New convention: width 1 compares VH dB directly with threshold_db. This
    avoids platform-dependent power-conversion rounding at exact thresholds.
    For archival reproduction, legacy_raw_power=True uses the provided power
    even at width 1. Widths above 1 always compare smoothed linear power to
    10**(threshold_db/10). Historical files/results are not rewritten.
    """
    width = _positive_integer(width, "width", odd=True)
    vh_db, smoothed_power, valid = _arrays(vh_db, smoothed_power, valid)
    if not np.isscalar(threshold_db) or not np.isfinite(threshold_db):
        raise ValueError("threshold_db must be a finite scalar")
    if width == 1 and not legacy_raw_power:
        predicted = vh_db < threshold_db
    else:
        predicted = smoothed_power < 10 ** (float(threshold_db) / 10)
    return predicted & valid.astype(bool)


def _counts(tp, fp, fn, tn):
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                scored_pixels=tp+fp+fn+tn)


def confusion(prediction, labels, valid):
    """Score labels 0/1 on valid SAR centers, ignoring negative labels.

    Counts and scored_pixels are Python integers; undefined metric ratios
    are None (including IoU for a chip with no true or predicted water).
    """
    prediction, labels, valid = _arrays(prediction, labels, valid)
    if np.any(~np.isin(labels, [-1, 0, 1])):
        raise ValueError("labels must contain only -1 (ignored), 0, or 1")
    prediction = prediction.astype(bool)
    scored = valid.astype(bool) & (labels >= 0)
    truth = labels == 1
    return _counts(int(np.count_nonzero(prediction & truth & scored)),
                   int(np.count_nonzero(prediction & ~truth & scored)),
                   int(np.count_nonzero(~prediction & truth & scored)),
                   int(np.count_nonzero(~prediction & ~truth & scored)))


def aggregate(rows):
    """Pool chip confusion counts and report equally weighted defined IoUs.

    Undefined chip IoUs are excluded from macro_iou and counted explicitly.
    Empty input returns zero counts and None for every ratio and macro_iou.
    """
    rows = list(rows)
    counts = []
    for row in rows:
        values = [row[k] for k in ("tp", "fp", "fn", "tn")]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
               or v < 0 for v in values):
            raise ValueError("confusion counts must be nonnegative integers")
        counts.append(_counts(*(int(v) for v in values)))
    pooled = _counts(*(sum(row[k] for row in counts) for k in ("tp", "fp", "fn", "tn")))
    ious = [row["iou"] for row in counts if row["iou"] is not None]
    pooled.update(macro_iou=float(np.mean(ious)) if ious else None,
                  defined_iou_chips=len(ious), chips=len(rows))
    return pooled
