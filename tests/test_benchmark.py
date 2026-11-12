"""The benchmark must remain useful without Numba or a CUDA driver."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from water_mapping import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_cpu_only_records_samples_and_equivalence(self):
        power = np.arange(35, dtype=float).reshape(5, 7) / 1000
        valid = np.ones_like(power, dtype=bool)
        valid[1:4, 2:5] = False
        chip = SimpleNamespace(name='fixture', power=power, valid=valid,
                               vh_db=np.full_like(power, -25.0))
        with patch.object(benchmark, 'recorded_selection', return_value={'train': [[], [], []]}), \
             patch.object(benchmark, 'load_chip', return_value=chip), \
             patch.object(benchmark, '_cuda_backend', side_effect=ImportError('fixture: no CUDA')):
            result = benchmark.run_benchmark('.', width=3, repeats=2)
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['cuda_available'])
        self.assertEqual(result['cuda_unavailability'], 'ImportError: fixture: no CUDA')
        self.assertEqual([row['batch'] for row in result['batches']], [1, 4, 16])
        for row in result['batches']:
            self.assertEqual(len(row['cpu']['samples_ms']), 2)
            self.assertEqual(len(row['cpu_tiled']['samples_ms']), 2)
            self.assertTrue(row['cpu_tiled_equivalence']['allclose'])
            self.assertTrue(row['cpu_tiled_equivalence']['same_nan_mask'])
            self.assertEqual(row['cpu_tiled_equivalence']['prediction_disagreements'], 0)
            self.assertNotIn('gpu_kernel', row)

    def test_invalid_arguments(self):
        for kwargs in ({'width': 2}, {'repeats': 0}, {'threshold_db': float('nan')}):
            with self.assertRaises(ValueError):
                benchmark.run_benchmark('.', **kwargs)



class OptionalCudaTests(unittest.TestCase):
    def test_reflect_invalid_windows_and_tiny_images(self):
        try:
            cuda, kernel = benchmark._cuda_backend()
        except Exception as error:
            self.skipTest(f'{type(error).__name__}: {error}')
        rng = np.random.default_rng(2026)
        for shape in [(1, 1), (11, 13)]:
            power = rng.random((1, *shape)).astype(np.float64)
            valid = np.ones(power.shape, dtype=bool)
            valid[:, :7, :7] = False
            power[~valid] = np.nan
            dp, dv = cuda.to_device(power), cuda.to_device(valid)
            output = cuda.device_array_like(dp)
            for width in [1, 3, 9, 29]:
                kernel[(1, 1, 1), (16, 16, 1)](dp, dv, output, width // 2)
                cuda.synchronize()
                actual = output.copy_to_host()[0]
                reference = benchmark.smooth_power(power[0], valid[0], width)
                np.testing.assert_array_equal(np.isnan(reference), np.isnan(actual))
                np.testing.assert_allclose(reference, actual, rtol=1e-10, atol=1e-12)



class MemorySamplingTests(unittest.TestCase):
    def test_missing_psutil(self):
        with patch.object(benchmark.importlib, 'import_module', side_effect=ImportError('no psutil')):
            result = benchmark._sampled_process_rss(lambda: None, 9)
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], 'ImportError: no psutil')
        json.dumps(result, allow_nan=False)

    def test_separate_process_samples(self):
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=12345))
        module = SimpleNamespace(Process=lambda: process)
        calls = []
        with patch.object(benchmark.importlib, 'import_module', return_value=module):
            result = benchmark._sampled_process_rss(lambda: calls.append(1), 9)
        self.assertEqual(len(calls), 9)
        self.assertEqual(result['baseline_bytes'], 12345)
        self.assertEqual(result['sampled_max_bytes'], 12345)
        self.assertEqual(result['sampled_delta_bytes'], 0)
        self.assertGreaterEqual(result['sample_count'], 2)
        self.assertTrue(result['separate_untimed_pass'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
