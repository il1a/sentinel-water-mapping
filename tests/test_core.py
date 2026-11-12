"""Synthetic tests with an independent, direct reflected-index oracle."""
import unittest

import numpy as np
from water_mapping import aggregate, classify, confusion, smooth_power, tiled_mean


def reflected(index, length):
    index %= 2 * length
    return index if index < length else 2 * length - index - 1


def direct_mean(power, valid, width):
    result = np.full(power.shape, np.nan)
    radius = width // 2
    for y in range(power.shape[0]):
        for x in range(power.shape[1]):
            values = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = reflected(y + dy, power.shape[0])
                    xx = reflected(x + dx, power.shape[1])
                    if valid[yy, xx]:
                        values.append(power[yy, xx])
            if values:
                result[y, x] = sum(values) / len(values)
    return result


class SmoothingTests(unittest.TestCase):
    def test_oracle_and_tile_seams(self):
        rng = np.random.default_rng(2026)
        for shape in [(1, 1), (1, 9), (7, 1), (2, 3), (13, 17)]:
            power = np.exp(rng.normal(size=shape))
            valid = rng.random(shape) > .25
            for width in [1, 3, 5]:
                expected = direct_mean(power, valid, width)
                np.testing.assert_allclose(smooth_power(power, valid, width), expected,
                                           rtol=1e-12, atol=1e-12, equal_nan=True)
                for tile in [1, 2, 4, 128]:
                    with self.subTest(shape=shape, width=width, tile=tile):
                        actual = tiled_mean(power, valid, width, tile)
                        np.testing.assert_allclose(actual, expected, rtol=1e-12,
                                                   atol=1e-12, equal_nan=True)

    def test_holes_and_invalid_centers(self):
        power = np.ones((11, 11))
        valid = np.ones_like(power, dtype=bool)
        valid[2:9, 2:9] = False
        power[~valid] = np.nan
        for width in [1, 3, 5]:
            actual = smooth_power(power, valid, width)
            expected = direct_mean(power, valid, width)
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            self.assertTrue(np.isnan(actual[5, 5]))
            predicted = classify(np.full(power.shape, -30.), actual, valid, width, 10.)
            self.assertFalse(predicted[~valid].any())
        self.assertTrue(np.isfinite(smooth_power(power, valid, 3)[2, 2]))

    def test_singleton(self):
        for width in [1, 3, 5, 11]:
            np.testing.assert_array_equal(smooth_power([[7]], [[True]], width), [[7.]])
            self.assertTrue(np.isnan(tiled_mean([[7]], [[False]], width)[0, 0]))

    def test_validation(self):
        p, v = np.ones((2, 3)), np.ones((2, 3), dtype=bool)
        for width in [0, -1, 2, 3., True]:
            with self.assertRaises(ValueError):
                smooth_power(p, v, width)
        for tile in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                tiled_mean(p, v, 3, tile)
        for bad in [np.ones((3, 2)), np.ones(6), np.ones((0, 3))]:
            with self.assertRaises(ValueError):
                smooth_power(p, bad, 3)
        for value in [-1., np.inf, np.nan]:
            with self.assertRaises(ValueError):
                smooth_power([[value]], [[True]], 1)


class ClassificationTests(unittest.TestCase):
    def test_strict_db_boundary_and_legacy(self):
        threshold = -22.
        vh = np.array([[np.nextafter(threshold, -np.inf), threshold,
                        np.nextafter(threshold, np.inf)]])
        boundary = 10 ** (threshold / 10)
        power = np.full(vh.shape, np.nextafter(boundary, -np.inf))
        valid = np.ones(vh.shape, dtype=bool)
        np.testing.assert_array_equal(classify(vh, power, valid, 1, threshold),
                                      [[True, False, False]])
        np.testing.assert_array_equal(classify(vh, power, valid, 1, threshold,
                                              legacy_raw_power=True), [[True]*3])

    def test_strict_power_boundary(self):
        threshold = -22.
        boundary = 10 ** (threshold / 10)
        power = np.array([[np.nextafter(boundary, -np.inf), boundary,
                           np.nextafter(boundary, np.inf)]])
        for width in [1, 3, 5]:
            np.testing.assert_array_equal(
                classify(np.zeros_like(power), power, np.ones_like(power, bool),
                         width, threshold, legacy_raw_power=True), [[True, False, False]])
        self.assertFalse(classify([[-30.]], [[.001]], [[False]], 1, threshold)[0, 0])
        with self.assertRaises(ValueError):
            classify([[0.]], [[1.]], [[True]], 1, np.nan)


class MetricTests(unittest.TestCase):
    def test_all_counts_masks_and_ratios(self):
        result = confusion([[1, 1, 0, 0, 1, 1]], [[1, 0, 1, 0, -1, 0]],
                           [[1, 1, 1, 1, 1, 0]])
        self.assertEqual([result[k] for k in ['tp', 'fp', 'fn', 'tn']], [1]*4)
        self.assertTrue(all(type(result[k]) is int for k in ['tp', 'fp', 'fn', 'tn']))
        self.assertEqual(result['scored_pixels'], 4)
        self.assertEqual(result['iou'], 1/3)
        for key in ['precision', 'recall', 'f1']:
            self.assertEqual(result[key], .5)

    def test_empty_and_dry_metrics(self):
        for result in [confusion([[0]], [[-1]], [[1]]), aggregate([])]:
            for key in ['iou', 'precision', 'recall', 'f1']:
                self.assertIsNone(result[key])
            self.assertEqual(result['scored_pixels'], 0)
        dry = confusion([[0]], [[0]], [[1]])
        self.assertEqual(dry['tn'], 1)
        self.assertIsNone(dry['iou'])
        self.assertIsNone(aggregate([])['macro_iou'])
        self.assertEqual(aggregate([])['chips'], 0)

    def test_pooling_differs_from_macro(self):
        rows = [dict(tp=1, fp=0, fn=0, tn=0), dict(tp=0, fp=9, fn=0, tn=0),
                dict(tp=0, fp=0, fn=0, tn=5)]
        result = aggregate(iter(rows))
        self.assertEqual(result['iou'], .1)
        self.assertEqual(result['macro_iou'], .5)
        self.assertEqual(result['defined_iou_chips'], 2)
        self.assertEqual(result['chips'], 3)
        self.assertEqual(result['scored_pixels'], 15)
        with self.assertRaises(ValueError):
            aggregate([dict(tp=-1, fp=0, fn=0, tn=0)])
        with self.assertRaises(ValueError):
            confusion([[0]], [[2]], [[True]])


if __name__ == '__main__':
    unittest.main()
