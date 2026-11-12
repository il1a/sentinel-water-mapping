"""Offline regression tests for pinned, bounded public-data acquisition."""
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from water_mapping.data import BASE_URL, acquire_pairs, contained_data_path, verified_download, write_json
from water_mapping.otsu import acquire_otsu


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'sample.tif'
        self.payload = b'valid synthetic bytes'
        self.url = BASE_URL + 'S1Hand/Event_1_S1Hand.tif'
        self.pin = dict(bytes=len(self.payload), url=self.url,
                        sha256=hashlib.sha256(self.payload).hexdigest())

    def assert_no_partial(self):
        self.assertEqual(list(self.root.rglob('*.part')), [])

    def test_successful_stream_is_verified_and_cached(self):
        with patch('water_mapping.data.urllib.request.urlopen',
                   return_value=io.BytesIO(self.payload)) as fetch:
            verified_download(self.path, self.url, self.pin)
            verified_download(self.path, self.url, self.pin)
        self.assertEqual(self.path.read_bytes(), self.payload)
        fetch.assert_called_once()
        self.assert_no_partial()

    def test_missing_or_invalid_pin_rejected_before_network(self):
        for pin in (None, {}, dict(self.pin, bytes=-1),
                    dict(self.pin, sha256='bad'), dict(self.pin, url='https://example.com')):
            with self.subTest(pin=pin), patch('water_mapping.data.urllib.request.urlopen') as fetch:
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    verified_download(self.path, self.url, pin)
                fetch.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assert_no_partial()

    def test_oversized_truncated_and_corrupt_downloads_never_promoted(self):
        for payload in (self.payload + b'x', self.payload[:-1], b'x' * len(self.payload)):
            with self.subTest(payload=payload), patch('water_mapping.data.urllib.request.urlopen',
                                                     return_value=io.BytesIO(payload)):
                with self.assertRaises(ValueError):
                    verified_download(self.path, self.url, self.pin)
            self.assertFalse(self.path.exists())
            self.assert_no_partial()

    def test_corrupt_cache_is_rejected_without_network_or_overwrite(self):
        self.path.write_bytes(b'bad')
        with patch('water_mapping.data.urllib.request.urlopen') as fetch:
            with self.assertRaisesRegex(ValueError, 'Cached'):
                verified_download(self.path, self.url, self.pin)
            fetch.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'bad')

    def test_transport_failure_cleans_partial_file(self):
        with patch('water_mapping.data.urllib.request.urlopen', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                verified_download(self.path, self.url, self.pin, attempts=1)
        self.assertFalse(self.path.exists())
        self.assert_no_partial()

    def test_pairs_require_pins(self):
        with patch('water_mapping.data.urllib.request.urlopen') as fetch:
            with self.assertRaisesRegex(ValueError, 'metadata'):
                acquire_pairs(self.root, {'test': [['Event_1_S1Hand.tif', 'Event_1_LabelHand.tif']]})
            fetch.assert_not_called()

    def test_parallel_fresh_directories(self):
        # Repeat fresh directory creation: the Windows resolve spelling race is
        # transient and does not appear when inspecting an existing cache.
        for trial in range(6):
            root = self.root / f'fresh-{trial}'
            rows = [[f'Event_{i}_S1Hand.tif', f'Event_{i}_LabelHand.tif'] for i in range(25)]
            pins = [dict(self.pin, name=name, url=BASE_URL +
                         ('S1Hand' if name.endswith('_S1Hand.tif') else 'LabelHand') + '/' + name)
                    for row in rows for name in row]
            write_json(root / 'manifests/full_input_manifest.json', pins)
            with patch('water_mapping.data.urllib.request.urlopen',
                       side_effect=lambda *a, **k: io.BytesIO(self.payload)):
                result = acquire_pairs(root, {'test': rows}, workers=12)
            self.assertEqual(len(result), 50)
            for entry in result:
                self.assertEqual((root / entry['relative_path']).read_bytes(), self.payload)
        self.assert_no_partial()

    @unittest.skipUnless(os.name == 'nt', 'Windows extended-length path spelling')
    def test_equivalent_windows_extended_path(self):
        root = self.root.resolve()
        relative = Path('data/S1Hand/Event_1_S1Hand.tif')
        extended = Path('\\\\?\\' + str(root / relative))
        with patch.object(Path, 'resolve', side_effect=[root, extended]):
            self.assertEqual(contained_data_path(root, relative), root / relative)

    def test_resolved_escape_is_still_rejected(self):
        root = self.root.resolve()
        outside = root.parent / 'outside' / 'Event_1_S1Hand.tif'
        with patch.object(Path, 'resolve', side_effect=[root, outside]):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                contained_data_path(root, Path('data/S1Hand/Event_1_S1Hand.tif'))

    def test_otsu_bad_download_never_becomes_cached(self):
        name = 'Event_1_S1OtsuLabelHand.tif'
        pin = dict(self.pin, name=name, url=BASE_URL + 'S1OtsuLabelHand/' + name)
        write_json(self.root / 'manifests/released_otsu_manifest.json', [pin])
        with patch('water_mapping.data.urllib.request.urlopen', return_value=io.BytesIO(b'bad')):
            with self.assertRaises(ValueError):
                acquire_otsu(self.root, {'test': [['Event_1_S1Hand.tif', 'Event_1_LabelHand.tif']]})
        self.assertFalse((self.root / 'data/S1OtsuLabelHand' / name).exists())
        self.assert_no_partial()


if __name__ == '__main__':
    unittest.main()
