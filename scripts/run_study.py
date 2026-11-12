"""Run the full checked study into a fresh, compact evidence directory."""
from pathlib import Path
import argparse
import os
import sys
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from water_mapping.data import acquire_pairs, load_splits
from water_mapping.experiment import (calibrate, environment_record,
                                      evaluate, freeze, protocol)
from water_mapping.reporting import save_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/study'))
    parser.add_argument('--skip-benchmark', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to((root / 'reference').resolve()):
        parser.error('Reference evidence is immutable; choose a new output directory.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory is not empty; choose a fresh --output path to avoid mixed runs.')
    started = time.perf_counter()
    evidence = dict(status='declared', protocol=protocol(root), environment=environment_record())
    save_study(output, evidence)
    splits = load_splits(root)
    evidence['input_manifest'] = acquire_pairs(root, splits)
    print(f"Inputs verified: {len(evidence['input_manifest'])} TIFFs.", flush=True)
    candidates, detail = calibrate(root, splits['valid'], progress=True)
    evidence.update(validation_candidates=candidates, validation_details=detail)
    config = freeze(root, output, candidates)
    evidence.update(status='frozen', configuration=config)
    save_study(output, evidence)
    print('Frozen:', config['selected']['size'], config['selected']['threshold_db'], flush=True)
    evidence['result'] = evaluate(root, splits, config, progress=True)
    evidence['benchmark'] = None
    if not args.skip_benchmark:
        from water_mapping.benchmark import run_benchmark, validate_benchmark
        benchmark = run_benchmark(root, width=config['selected']['size'],
                                  threshold_db=config['selected']['threshold_db'])
        validate_benchmark(benchmark)
        evidence['benchmark'] = benchmark
        if not benchmark['cuda_available']:
            print('CUDA unavailable:', benchmark.get('cuda_unavailability'), flush=True)
        for batch in benchmark['batches']:
            if 'cuda_error' in batch:
                print('CUDA benchmark issue:', batch['cuda_error'], flush=True)
    evidence['completion'] = dict(complete=True, elapsed_seconds=time.perf_counter() - started,
                                 evaluated_pairs=sum(map(len, splits.values())),
                                 data_bytes=sum(row['bytes'] for row in evidence['input_manifest']),
                                 benchmark_requested=not args.skip_benchmark,
                                 note='Elapsed wall time includes hash verification and any missing-data downloads; it is not a comparative benchmark.')
    evidence['status'] = 'complete'
    save_study(output, evidence)
    print('Complete:', evidence['completion'], flush=True)
    for group in ['test_all_90', 'test_new_80', 'bolivia_separate_15']:
        print(group, evidence['result']['accuracy_summary'][group], flush=True)


if __name__ == '__main__':
    main()
