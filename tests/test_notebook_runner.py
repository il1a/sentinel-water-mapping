"""Data-free fresh-kernel checks for the supported notebook execution path."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from nbclient.exceptions import CellExecutionError
from jupyter_client import KernelManager

from scripts.execute_notebook import execute


class NotebookRunnerTests(unittest.TestCase):
    def run_notebook(self, source, *, fails=False):
        managers = []

        def manager_factory(*args, **kwargs):
            manager = KernelManager(*args, **kwargs)
            managers.append(manager)
            return manager

        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.execute_notebook.KernelManager', side_effect=manager_factory):
                if fails:
                    with self.assertRaises(CellExecutionError):
                        execute(notebook, Path(directory), timeout=60)
                else:
                    execute(notebook, Path(directory), timeout=60)
        self.assertEqual(len(managers), 1)
        manager = managers[0]
        self.assertFalse(manager.has_kernel)
        self.assertEqual(manager.ip, '127.0.0.1')
        self.assertEqual(manager.transport_encryption, 'required')
        self.assertFalse(Path(manager.connection_file).exists())
        self.assertFalse(Path(manager.connection_file).parent.exists())
        return notebook

    def test_encrypted_loopback_execution_and_cleanup(self):
        notebook = self.run_notebook(
            "from ipykernel.kernelapp import IPKernelApp\n"
            "app = IPKernelApp.instance()\n"
            "assert app.ip == '127.0.0.1'\n"
            "assert app.curve_publickey is not None and app.curve_secretkey is not None\n"
            "print(6 * 7)"
        )
        output = ''.join(item.get('text', '') for item in notebook.cells[0].outputs)
        self.assertEqual(output.strip(), '42')
        self.assertEqual(notebook.cells[0].execution_count, 1)

    def test_cell_failure_still_stops_kernel_and_removes_connection(self):
        self.run_notebook("raise ValueError('intentional test failure')", fails=True)


if __name__ == '__main__':
    unittest.main()
