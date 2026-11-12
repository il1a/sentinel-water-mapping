"""Data-free checks for the compact release inputs and source notebook."""
import csv
import json
from pathlib import Path
import unittest

from water_mapping.data import BASE_URL, load_splits, recorded_selection, sha256

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_full_manifest_covers_every_official_pair_once(self):
        expected = {name for pairs in load_splits(ROOT).values() for pair in pairs for name in pair}
        manifest = json.loads((ROOT / 'manifests/full_input_manifest.json').read_text())
        self.assertEqual(len(manifest), len(expected))
        self.assertEqual({row['name'] for row in manifest}, expected)
        for row in manifest:
            self.assertGreater(row['bytes'], 0)
            self.assertRegex(row['sha256'], r'^[0-9a-f]{64}$')
            self.assertTrue(row['url'].startswith(BASE_URL))

    def test_example_and_prior_test_membership_is_explicit(self):
        selected, splits = recorded_selection(ROOT), load_splits(ROOT)
        self.assertEqual(len(selected['train']), 3)
        self.assertEqual(len(selected['test']), 10)
        for group in ('train', 'test'):
            self.assertTrue(all(pair in splits[group] for pair in selected[group]))

    def test_reference_has_exact_expected_rows(self):
        with (ROOT / 'reference/evaluation.csv').open(newline='') as source:
            rows = list(csv.DictReader(source))
        expected = {(split, pair[0], method) for split, pairs in load_splits(ROOT).items()
                    for pair in pairs for method in ('selected', 'no_smoothing')}
        self.assertEqual(len(rows), len(expected))
        self.assertEqual({(r['split'], r['image'], r['method']) for r in rows}, expected)

    def test_release_sources_match_checked_reference(self):
        frozen = json.loads((ROOT / 'reference/frozen_config.json').read_text())
        for name, digest in frozen['release_source_sha256'].items():
            self.assertEqual(sha256(ROOT / name), digest, name)

    def test_notebook_has_no_execution_outputs(self):
        notebook = json.loads((ROOT / 'notebooks/surface_water_mapping.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])


if __name__ == '__main__':
    unittest.main()
