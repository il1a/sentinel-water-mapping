"""Execute the teaching notebook with this Python, without installing a kernel."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute(notebook, root: Path, *, timeout: int = 1800):
    """Run a fresh, encrypted loopback kernel and remove all temporary resources."""
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix='water-mapping-kernel-') as directory:
        kernel = Path(directory) / 'water-mapping'
        kernel.mkdir()
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python (current environment)', 'language': 'python',
            'metadata': {'supported_encryption': ['curve']}}), encoding='utf-8')
        manager = KernelManager(kernel_name='water-mapping', ip='127.0.0.1',
                                connection_file=str(Path(directory) / 'connection.json'),
                                transport_encryption='required',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        client = NotebookClient(notebook, km=manager, timeout=timeout,
                                resources={'metadata': {'path': str(root)}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            # Also handle startup failures before nbclient enters its cleanup block.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_connection_file()
    nbformat.validate(notebook)
    return notebook


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'runs/notebook/surface_water_mapping.executed.ipynb')
    args = parser.parse_args()
    notebook = nbformat.read(root / 'notebooks/surface_water_mapping.ipynb', as_version=4)
    nbformat.validate(notebook)
    try:
        execute(notebook, root)
    except Exception:
        failed = root / 'runs/notebook/failed.ipynb'
        failed.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, failed)
        raise
    nbformat.validate(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print('Executed', sum(cell.cell_type == 'code' for cell in notebook.cells), 'code cells without errors.')
    print('Output:', args.output.name)


if __name__ == '__main__':
    main()
