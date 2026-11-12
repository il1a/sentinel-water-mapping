"""Split/provenance checks without network access or dataset files."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from water_mapping.data import load_splits, write_json, read_json


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'manifests/splits'
        self.directory.mkdir(parents=True)
        self.counts = dict(train=1, valid=1, test=1, bolivia=1)
        self.rows = {split: [f'Event_{i}_S1Hand.tif', f'Event_{i}_LabelHand.tif']
                     for i, split in enumerate(self.counts)}
        self.write_rows()

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self):
        for split, row in self.rows.items():
            with (self.directory / f'flood_{split}_data.csv').open('w', newline='') as stream:
                csv.writer(stream).writerow(row)

    def load(self):
        with patch('water_mapping.data.SPLIT_COUNTS', self.counts):
            return load_splits(self.root)

    def test_disjoint_split_identity(self):
        self.assertEqual(len(self.load()), 4)

    def test_duplicate_is_rejected(self):
        self.rows['test'] = self.rows['train']
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load()

    def test_pair_mismatch_is_rejected(self):
        self.rows['valid'][1] = 'Other_99_LabelHand.tif'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.load()

    def test_missing_pair_is_rejected(self):
        (self.directory / 'flood_train_data.csv').write_text('')
        with self.assertRaisesRegex(ValueError, 'expected 1 pairs'):
            self.load()

    def test_path_escape_is_rejected(self):
        self.rows['test'][0] = '../../secret_S1Hand.tif'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Invalid manifest'):
            self.load()

    def test_json_cannot_hide_nan(self):
        with self.assertRaises(ValueError):
            write_json(self.root / 'bad.json', {'score': float('nan')})
        write_json(self.root / 'good.json', {'score': None, 'tp': 3})
        self.assertEqual(read_json(self.root / 'good.json'), {'score': None, 'tp': 3})


if __name__ == '__main__':
    unittest.main()
