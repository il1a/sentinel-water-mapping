"""Optional, warm CPU/CUDA smoothing benchmark with no import-time GPU dependency.

Batches repeat three recorded training chips; they are not independent scenes.
All arrays stay resident. Device-buffer bytes are not total GPU memory usage.
"""
from pathlib import Path
import platform
import time
import threading
import importlib

import numpy as np

from .core import smooth_power, tiled_mean, classify, _positive_integer
from .data import load_chip, recorded_selection


def _cuda_backend():
    # Numba resolves cuda.grid through a module global when compiling the kernel.
    global cuda
    from numba import cuda

    if not cuda.is_available():
        # Initializing a context usually supplies a more useful driver error.
        cuda.current_context()
        raise RuntimeError("numba.cuda.is_available() returned False")

    @cuda.jit
    def mean_kernel(power, valid, output, radius):
        x, y, z = cuda.grid(3)
        if z < power.shape[0] and y < power.shape[1] and x < power.shape[2]:
            total = 0.0
            count = 0
            for dy in range(-radius, radius + 1):
                # Periodic folding implements half-sample reflect even when
                # a window exceeds the image dimensions.
                yy = (y + dy) % (2 * power.shape[1])
                if yy >= power.shape[1]:
                    yy = 2 * power.shape[1] - yy - 1
                for dx in range(-radius, radius + 1):
                    xx = (x + dx) % (2 * power.shape[2])
                    if xx >= power.shape[2]:
                        xx = 2 * power.shape[2] - xx - 1
                    if valid[z, yy, xx]:
                        total += power[z, yy, xx]
                        count += 1
            output[z, y, x] = total / count if count else float('nan')

    return cuda, mean_kernel


def _timed(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(samples_ms=samples, median_ms=float(np.median(samples)),
                min_ms=min(samples), max_ms=max(samples))


def _sampled_process_rss(function, repeats):
    """Observe process RSS in a separate untimed pass, not a true memory peak."""
    try:
        process = importlib.import_module('psutil').Process()
        baseline = int(process.memory_info().rss)
    except Exception as error:
        return dict(available=False, reason=f'{type(error).__name__}: {error}',
                    separate_untimed_pass=True)
    samples = [baseline]
    errors = []
    stop = threading.Event()

    def sample():
        while not stop.wait(0.001):
            try:
                samples.append(int(process.memory_info().rss))
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
                return

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        for _ in range(repeats):
            function()
    finally:
        stop.set()
        worker.join()
    background_count = len(samples) - 1
    try:
        samples.append(int(process.memory_info().rss))
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    result = dict(available=True, baseline_bytes=baseline,
                  sampled_max_bytes=max(samples), sampled_delta_bytes=max(samples) - baseline,
                  sample_count=len(samples), background_sample_count=background_count,
                  interval_ms=1.0, workload_repeats=repeats, separate_untimed_pass=True)
    if errors:
        result['sampling_errors'] = errors
    return result


def _equivalence(reference, actual, power_db, valid, width, threshold_db):
    finite = np.isfinite(reference) & np.isfinite(actual)
    disagreements = sum(int(np.count_nonzero(
        classify(db, a, mask, width, threshold_db) !=
        classify(db, b, mask, width, threshold_db)))
        for db, a, b, mask in zip(power_db, reference, actual, valid))
    return dict(allclose=bool(np.allclose(reference, actual, rtol=1e-10,
                                         atol=1e-12, equal_nan=True)),
                same_nan_mask=bool(np.array_equal(np.isnan(reference), np.isnan(actual))),
                max_abs_power_error=float(np.max(np.abs(reference[finite] - actual[finite])))
                if np.any(finite) else None,
                prediction_disagreements=disagreements,
                prediction_valid_centers=int(valid.sum()), rtol=1e-10, atol=1e-12)


def run_benchmark(root, width=5, threshold_db=-22, repeats=9):
    """Return JSON-safe timings, equivalence, and exact optional-CUDA failures.

    Warmup and explicit host/device buffer allocation are outside timed runs.
    CPU methods include the temporary arrays allocated internally by the core.
    Kernel timing is synchronized host wall time, including launch overhead.
    H2D/D2H use pageable host arrays; the inclusive pipeline serializes both
    transfers and the kernel. No disk I/O or JIT is included in timed samples.
    """
    width = _positive_integer(width, 'width', odd=True)
    repeats = _positive_integer(repeats, 'repeats')
    if not np.isscalar(threshold_db) or not np.isfinite(threshold_db):
        raise ValueError('threshold_db must be a finite scalar')
    chips = [load_chip(Path(root), row)
             for row in recorded_selection(Path(root))['train'][:3]]
    if len(chips) != 3:
        raise ValueError('benchmark requires three recorded training chips')
    result = dict(width=width, threshold_db=float(threshold_db), repeats=repeats,
                  dtype='float64', source_chips=[chip.name for chip in chips],
                  batch_sizes=[1, 4, 16], platform=platform.platform(),
                  numpy_version=np.__version__, cuda_available=False, batches=[],
                  timing_notes=run_benchmark.__doc__,
                  memory_notes='Full arrays remain resident; no out-of-core or memory-saving claim. '
                               'Explicit device-buffer bytes exclude CUDA context and allocator overhead. '
                               'RSS is sampled process-wide in separate untimed passes after timing, '
                               'with a start baseline, requested 1ms background sampling and an end sample. '
                               'Existing allocations and allocator reuse affect baselines and deltas; '
                               'OS scheduling can make actual intervals longer and short peaks may be missed. Sampled maxima are not true per-method peaks '
                               'and do not establish memory reduction. RSS is host memory, not GPU memory.')
    try:
        backend, kernel = _cuda_backend()
        device = backend.get_current_device()
        result.update(cuda_available=True, gpu_name=str(device.name),
                      compute_capability=list(device.compute_capability))
    except Exception as error:
        backend = None
        result['cuda_unavailability'] = f'{type(error).__name__}: {error}'

    for batch in result['batch_sizes']:
        selected = [chips[i % 3] for i in range(batch)]
        power = np.stack([chip.power for chip in selected]).astype(np.float64)
        valid = np.stack([chip.valid for chip in selected]).astype(bool)
        db = np.stack([chip.vh_db for chip in selected])
        whole, tiled = np.empty_like(power), np.empty_like(power)

        def cpu():
            for i in range(batch):
                whole[i] = smooth_power(power[i], valid[i], width)

        def cpu_tiled():
            for i in range(batch):
                tiled[i] = tiled_mean(power[i], valid[i], width, tile=128)

        cpu()
        cpu_tiled()
        record = dict(batch=batch, synthetic_repetition_of_three_train_chips=True,
                      shape=list(power.shape), cpu=_timed(cpu, repeats),
                      cpu_tiled=_timed(cpu_tiled, repeats),
                      cpu_tiled_equivalence=_equivalence(whole, tiled, db, valid, width, threshold_db))
        result['batches'].append(record)
        record['sampled_process_rss'] = {}
        if backend is None:
            record['sampled_process_rss']['cpu'] = _sampled_process_rss(cpu, max(9, repeats))
            record['sampled_process_rss']['cpu_tiled'] = _sampled_process_rss(cpu_tiled, max(9, repeats))
            continue
        try:
            dp = backend.to_device(power)
            dv = backend.to_device(valid)
            output = backend.device_array_like(dp)
            host = np.empty_like(power)
            block = (16, 16, 1)
            grid = ((power.shape[2] + 15) // 16, (power.shape[1] + 15) // 16, batch)

            def gpu_kernel():
                kernel[grid, block](dp, dv, output, width // 2)
                backend.synchronize()

            def h2d():
                dp.copy_to_device(power)
                dv.copy_to_device(valid)
                backend.synchronize()

            def d2h():
                output.copy_to_host(host)
                backend.synchronize()

            def pipeline():
                h2d()
                gpu_kernel()
                d2h()

            start = time.perf_counter()
            pipeline()
            record['excluded_first_pipeline_ms'] = (time.perf_counter() - start) * 1000
            record.update(gpu_kernel=_timed(gpu_kernel, repeats), h2d=_timed(h2d, repeats),
                          d2h=_timed(d2h, repeats), gpu_transfer_inclusive=_timed(pipeline, repeats),
                          explicit_device_buffer_bytes=int(dp.nbytes + dv.nbytes + output.nbytes),
                          gpu_equivalence=_equivalence(whole, host, db, valid, width, threshold_db))
            record['sampled_process_rss']['gpu_transfer_inclusive'] = _sampled_process_rss(pipeline, max(9, repeats))
            del dp, dv, output, host
        except Exception as error:
            record['cuda_error'] = f'{type(error).__name__}: {error}'
            # Preserve every CPU measurement even if compilation or execution fails.
        record['sampled_process_rss']['cpu'] = _sampled_process_rss(cpu, max(9, repeats))
        record['sampled_process_rss']['cpu_tiled'] = _sampled_process_rss(cpu_tiled, max(9, repeats))
    return result



def validate_benchmark(result):
    """Reject measured timing comparisons whose numerical checks did not pass.

    Unavailable CUDA or a recorded CUDA execution error is not a CPU failure.
    Available comparisons must pass numeric, NaN-mask, and prediction checks.
    """
    for batch in result['batches']:
        if 'cpu_tiled_equivalence' not in batch:
            raise ValueError('Missing CPU equivalence result')
        for name in ['cpu_tiled_equivalence', 'gpu_equivalence']:
            if name in batch:
                check = batch[name]
                if not (check['allclose'] and check['same_nan_mask']
                        and check['prediction_disagreements'] == 0):
                    raise ValueError(f"Invalid benchmark comparison: batch {batch['batch']}, {name}")
    return result
