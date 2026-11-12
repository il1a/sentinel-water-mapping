"""Failure-path guards for experiment records."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_study import main
from water_mapping.benchmark import validate_benchmark
from water_mapping.reporting import save_study
from water_mapping.data import read_json


class WorkflowTests(unittest.TestCase):
    def test_nonempty_output_cannot_mix_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = output / 'completion.json'
            status.write_text('{"complete":true}')
            with patch('sys.argv', ['run_study', '--output', directory]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(status.read_text(), '{"complete":true}')

    def test_failed_benchmark_equivalence_rejected(self):
        check = dict(allclose=True, same_nan_mask=True, prediction_disagreements=1)
        result = {'batches': [{'batch': 1, 'cpu_tiled_equivalence': check}]}
        with self.assertRaisesRegex(ValueError, 'Invalid benchmark'):
            validate_benchmark(result)

    def test_compact_bundle_has_clear_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            save_study(out, {'status': 'declared'})
            self.assertEqual({p.name for p in out.iterdir()}, {'study.json'})
            payload = {'status': 'complete', 'result': {'evaluation': [{'image': 'a', 'tp': 1}]}}
            save_study(out, payload)
            self.assertEqual({p.name for p in out.iterdir()}, {'study.json', 'evaluation.csv'})
            self.assertEqual(read_json(out / 'study.json'), payload)

    def test_cuda_unavailable_preserves_valid_cpu_result(self):
        check = dict(allclose=True, same_nan_mask=True, prediction_disagreements=0)
        result = dict(cuda_available=False, batches=[dict(batch=1, cpu_tiled_equivalence=check)])
        self.assertIs(validate_benchmark(result), result)


if __name__ == '__main__':
    unittest.main()
