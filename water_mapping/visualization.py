"""Small presentation helpers. All detector calculations live in core/experiment."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def example_figure(chip, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(np.ma.masked_where(~chip.valid, chip.vh_db), cmap='gray', vmin=-30, vmax=-5)
    axes[0].set_title('VH backscatter (dB)')
    scored = chip.valid & (chip.labels >= 0)
    colors = ListedColormap(['#eee9dc', '#1676a3'])
    axes[1].imshow(np.ma.masked_where(~scored, chip.labels), cmap=colors, vmin=0, vmax=1)
    axes[1].set_title('Reference: blue = water')
    axes[2].imshow(np.ma.masked_where(~scored, prediction), cmap=colors, vmin=0, vmax=1)
    axes[2].set_title('Illustrative detector: blue = water')
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(chip.name + ' | white: invalid SAR (left); outside scoring mask (right two)' )
    return fig


def calibration_figure(candidates):
    table = pd.DataFrame(candidates).pivot(index='size', columns='threshold_db', values='macro_iou')
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    im = ax.imshow(table.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(table.columns)), [f'{v:g}' for v in table.columns])
    ax.set_yticks(range(len(table.index)), [f'{v} x {v}' for v in table.index])
    ax.set(xlabel='Strict threshold (dB)', ylabel='Mean-filter window', title='89-chip validation: mean defined chip water IoU')
    for i, row in enumerate(table.values):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value > .6 else 'black')
    fig.colorbar(im, ax=ax, label='Macro IoU (0 to 1)')
    return fig


def summary_table(summary, groups=None):
    rows = []
    for group in groups or summary:
        for method, score in summary[group].items():
            rows.append(dict(group=group, method=method, chips=score['chips'],
                             defined_IoUs=score['defined_iou_chips'],
                             macro_IoU=score['macro_iou'], pooled_IoU=score['iou'],
                             precision=score['precision'], recall=score['recall'], F1=score['f1']))
    return pd.DataFrame(rows).set_index(['group', 'method']).round(4)


def event_figure(event_summary):
    table = pd.DataFrame(event_summary)
    table = table[table['split'].eq('test')].pivot(index='event', columns='method', values='macro_iou')
    ax = table.rename(columns={'selected': 'Validation winner', 'no_smoothing': 'Unsmoothed baseline'}).plot.barh(
        figsize=(9, 5), color=['#d89a32', '#1676a3'], xlim=(0, 1))
    ax.set(xlabel='Mean defined chip water IoU', ylabel='', title='Full 90-chip test: descriptive event breakdown')
    ax.legend(loc='lower right')
    ax.figure.tight_layout()
    return ax.figure


def timing_figure(benchmark):
    names = {'cpu': 'Whole CPU', 'cpu_tiled': 'Tiled CPU',
             'gpu_kernel': 'GPU kernel only', 'gpu_transfer_inclusive': 'GPU + transfers'}
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for key, label in names.items():
        points = [(r['batch'], r[key]['median_ms']) for r in benchmark['batches'] if key in r]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker='o', label=label)
    ax.set(xlabel='Batch size (repeated training chips)', ylabel='Median elapsed time (ms)',
           title='Warm smoothing-stage timing on this machine', xticks=benchmark['batch_sizes'])
    ax.legend()
    ax.grid(alpha=.2)
    return fig


def memory_table(benchmark):
    """Present observations without interpreting them as method peak memory."""
    rows = []
    for batch in benchmark['batches']:
        for method, sample in batch.get('sampled_process_rss', {}).items():
            if sample.get('available'):
                rows.append(dict(batch=batch['batch'], method=method,
                                 baseline_process_MiB=sample['baseline_bytes'] / 2**20,
                                 max_sampled_process_MiB=sample['sampled_max_bytes'] / 2**20,
                                 sampled_delta_MiB=sample['sampled_delta_bytes'] / 2**20,
                                 samples=sample['sample_count']))
    return pd.DataFrame(rows).round(2)
