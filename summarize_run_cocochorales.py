r"""
Summarize an Onsets and Frames training run in one picture and one table,
so you don't have to screenshot TensorBoard graph by graph.

Reads the TensorBoard log in a run folder and writes, to
R:\experiments\results\<run name>\:
  summary_plots.png   the key training and validation graphs on one page
  all_scalars.csv     every logged number (step, name, value), for later analysis
It also prints a table of the validation scores every 5,000 steps.

Run from the onsets-and-frames folder, oaf environment active:
    python summarize_run_cocochorales.py                 (newest oaf_cocochorales run)
    python summarize_run_cocochorales.py <run folder>    (a specific run)
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = r'R:\experiments\runs'
RESULTS = r'R:\experiments\results'
TABLE_EVERY = 5000

# (panel title, [(tag, label), ...])
PANELS = [
    ('Total training loss', [('loss', 'train')]),
    ('Training losses by part', [('loss/onset', 'onset'), ('loss/offset', 'offset'),
                                 ('loss/frame', 'frame')]),
    ('Train vs validation loss (onset)', [('loss/onset', 'train'),
                                          ('validation/loss/onset', 'validation')]),
    ('Train vs validation loss (frame)', [('loss/frame', 'train'),
                                          ('validation/loss/frame', 'validation')]),
    ('Validation note F1 (onset only)', [('validation/metric/note/precision', 'precision'),
                                         ('validation/metric/note/recall', 'recall'),
                                         ('validation/metric/note/f1', 'F1')]),
    ('Validation note-with-offsets F1', [('validation/metric/note-with-offsets/f1', 'F1')]),
    ('Validation frame F1', [('validation/metric/frame/f1', 'F1')]),
]
TABLE = [('validation/metric/note/f1', 'note F1'),
         ('validation/metric/note/precision', 'note prec'),
         ('validation/metric/note/recall', 'note rec'),
         ('validation/metric/note-with-offsets/f1', 'note+off F1'),
         ('validation/metric/frame/f1', 'frame F1')]


def smooth(values, weight=0.9):
    out, last = [], values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        out.append(last)
    return out


def main():
    if len(sys.argv) > 1:
        run = sys.argv[1].rstrip('\\/')
    else:
        runs = sorted(glob.glob(os.path.join(RUNS, 'oaf_cocochorales-*')), key=os.path.getmtime)
        if not runs:
            sys.exit(f'No oaf_cocochorales runs found in {RUNS}')
        run = runs[-1]
    name = os.path.basename(run)
    out_dir = os.path.join(RESULTS, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Reading {run} ...')

    acc = EventAccumulator(run, size_guidance={'scalars': 0})   # 0 = keep every point
    acc.Reload()
    tags = acc.Tags()['scalars']
    data = {t: ([e.step for e in acc.Scalars(t)], [e.value for e in acc.Scalars(t)]) for t in tags}
    last_step = max((s[-1] for s, _ in data.values() if s), default=0)
    print(f'  {len(tags)} logged values, last step {last_step}')

    with open(os.path.join(out_dir, 'all_scalars.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'name', 'value'])
        for t in sorted(tags):
            for s, v in zip(*data[t]):
                w.writerow([s, t, v])

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for ax, (title, series) in zip(axes.flat, PANELS):
        for tag, label in series:
            if tag not in data:
                continue
            steps, values = data[tag]
            many = len(values) > 200          # training losses: raw faint, smoothed bold
            if many:
                ax.plot(steps, values, alpha=0.15)
                ax.plot(steps, smooth(values), label=label, color=ax.lines[-1].get_color())
            else:
                ax.plot(steps, values, marker='.', label=label)
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
        ax.legend()
        if 'loss' in title.lower():
            ax.set_yscale('log')
    for ax in list(axes.flat)[len(PANELS):]:
        ax.axis('off')
    fig.suptitle(f'{name}  (last step {last_step})', fontsize=14)
    fig.tight_layout()
    png = os.path.join(out_dir, 'summary_plots.png')
    fig.savefig(png, dpi=80)
    print(f'  saved {png}')

    print(f'\nValidation scores every {TABLE_EVERY} steps (200-clip sample):')
    print(f'{"step":>8}' + ''.join(f'{label:>13}' for _, label in TABLE))
    lookup = {t: dict(zip(*data[t])) for t, _ in TABLE if t in data}
    steps = sorted(set().union(*[d.keys() for d in lookup.values()])) if lookup else []
    for s in steps:
        if s % TABLE_EVERY == 0 or s == steps[-1]:
            print(f'{s:>8}' + ''.join(f'{lookup[t].get(s, float("nan")):>13.3f}'
                                      if t in lookup else f'{"-":>13}' for t, _ in TABLE))
    for t, label in TABLE:
        if t in lookup:
            best = max(lookup[t].items(), key=lambda kv: kv[1])
            print(f'  best {label}: {best[1]:.3f} at step {best[0]}')
    print('\nSUMMARY DONE')


if __name__ == '__main__':
    main()
