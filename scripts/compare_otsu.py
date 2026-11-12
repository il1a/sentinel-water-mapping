"""Compare frozen detectors with released Otsu maps on the same 105 chips."""
from pathlib import Path
import argparse
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from water_mapping.otsu import run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/released-otsu'))
    parser.add_argument('--coverage-policy', choices=('strict', 'intersection'), default='strict')
    args = parser.parse_args()
    evidence = run_comparison(Path(__file__).resolve().parents[1], args.output, progress=True, coverage_policy=args.coverage_policy)
    for group, methods in evidence['result']['accuracy_summary'].items():
        for method, metrics in methods.items():
            print(group, method, 'pooled IoU', metrics['iou'], 'mean chip IoU', metrics['macro_iou'])


if __name__ == '__main__':
    main()
