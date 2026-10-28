r"""
Train several Onsets and Frames data variants one after another, so a queue of
runs can go overnight without restarting anything by hand.

Variants (pick any, in any order, repeats allowed; they run in the order given):
    none    the original baseline data
    pitch   random pitch shift (+/-5 semitones unless --max-shift says otherwise)
    stems   random stem subsets (half the clips use 1-3 instruments)
    both    pitch shift and stem subsets together

Full runs (50,000 steps each, same settings as the baseline):
    python run_trainings.py --variants none both pitch stems
Every run is a new training run with its own random seed and its own folder, e.g.
R:\experiments\runs\oaf_cocochorales-aug_both-<date>. Listing a variant twice
(--variants pitch pitch) gives two separate runs.
If one run fails, the queue moves on to the next. Ctrl+C stops the whole queue.

Pitch-shift ablation: --max-shift K sets the largest shift for pitch and both
(default 5). Runs with a size other than 5 get it in their folder name, e.g.
oaf_cocochorales-aug_both_s3-<date>, so they never mix with the +/-5 runs:
    python run_trainings.py --variants both both both --max-shift 3

Continue a stopped queue later with just:
    python run_trainings.py --resume
This finishes the queue that was running when you stopped it, and nothing else:
finished runs are skipped, the interrupted run continues from its last
checkpoint (at most 1,000 steps are lost), and runs that never started begin
fresh. It reads R:\experiments\runs\current_queue.txt, which every new queue writes.

See every run, its progress and its seed, and how many of each type you have:
    python run_trainings.py --status

A log of start/end times and results is kept in R:\experiments\runs\training_queue_log.txt.

Quick check before a long queue (a few minutes per variant):
    python run_trainings.py --variants none both pitch stems --check
For each augmented variant it saves 3 example clips (audio you can listen to, a
picture of the labels, and a note list) to R:\experiments\augment_check\<variant>,
then trains each variant for 300 steps in R:\experiments\runs_checks (kept apart
from the real runs). Including 'none' gives a speed reference: the summary then
estimates each variant's full 50,000-step time from the baseline's 4 hours.

Run from the onsets-and-frames folder, oaf environment active.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

VARIANTS = ['none', 'pitch', 'stems', 'both']
LOG = r'R:\experiments\runs\training_queue_log.txt'
CHECK_RUNS = r'R:\experiments\runs_checks'
CHECK_EXAMPLES = r'R:\experiments\augment_check'
BASELINE_HOURS = 4.0          # the baseline's full 50,000-step run
CHECK_STEPS = 300
RUNS = r'R:\experiments\runs'
QUEUE = os.path.join(RUNS, 'current_queue.txt')
DEFAULT_SHIFT = 5
TARGET_REPEATS = 3
HERE = os.path.dirname(os.path.abspath(__file__))


FOLDER_RE = re.compile(r'oaf_cocochorales-(?:aug_(pitch|stems|both)(?:_s(\d+))?-)?(\d{6}-\d{6})$')


def uses_shift(variant):
    return variant in ('pitch', 'both')


def run_name(variant, shift, stamp):
    """Folder name for a run. +/-5 runs keep the original names; other sizes add _s<K>."""
    if variant == 'none':
        tag = ''
    elif uses_shift(variant) and shift != DEFAULT_SHIFT:
        tag = f'aug_{variant}_s{shift}-'
    else:
        tag = f'aug_{variant}-'
    return f'oaf_cocochorales-{tag}{stamp}'


def last_step(folder):
    """Highest checkpoint step saved in a run folder (0 if none or missing)."""
    if not os.path.isdir(folder):
        return 0
    steps = [int(m.group(1)) for f in os.listdir(folder)
             for m in [re.fullmatch(r'model-(\d+)\.pt', f)] if m]
    return max(steps, default=0)


def run_seed(folder):
    """Seed of the run's first training session (sacred saves it in <folder>/1/config.json)."""
    try:
        with open(os.path.join(folder, '1', 'config.json')) as f:
            return json.load(f).get('seed')
    except (OSError, ValueError):
        return None


def write_queue(entries):
    with open(QUEUE, 'w') as f:
        f.write('# variant  max_shift  steps  folder   (written by run_trainings.py; used by --resume)\n')
        for e in entries:
            f.write(f"{e['variant']}\t{e['shift']}\t{e['steps']}\t{e['folder']}\n")


def read_queue():
    entries = []
    with open(QUEUE) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                v, shift, steps, folder = line.rstrip('\n').split('\t')
                entries.append(dict(variant=v, shift=int(shift), steps=int(steps), folder=folder))
    return entries


def status(iterations):
    """Print every run folder with its progress and seed, then counts per run type."""
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, 'oaf_cocochorales-*'))):
        m = FOLDER_RE.match(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        variant = m.group(1) or 'none'
        shift = (int(m.group(2)) if m.group(2) else DEFAULT_SHIFT) if uses_shift(variant) else 0
        rows.append((variant, shift, last_step(d), run_seed(d), os.path.basename(d)))
    queued = set()
    if os.path.exists(QUEUE):
        queued = {os.path.basename(e['folder']) for e in read_queue()}

    print(f'\nAll runs in {RUNS}:')
    print(f"  {'type':<10} {'steps':>7}  {'state':<9} {'seed':>10}  folder")
    for variant, shift, step, seed, name in rows:
        typ = variant + (f' +/-{shift}' if shift else '')
        state = 'finished' if step >= iterations else ('stopped' if step else 'no ckpt')
        mark = '  (in current queue)' if name in queued and step < iterations else ''
        print(f"  {typ:<10} {step:>7}  {state:<9} {seed if seed is not None else '?':>10}  {name}{mark}")

    print(f'\nFinished runs per type (target {TARGET_REPEATS}):')
    counts = {}
    for variant, shift, step, _, _ in rows:
        if step >= iterations:
            counts[(variant, shift)] = counts.get((variant, shift), 0) + 1
    for key in sorted(counts, key=lambda k: (VARIANTS.index(k[0]), k[1])):
        variant, shift = key
        typ = variant + (f' +/-{shift}' if shift else '')
        n = counts[key]
        print(f"  {typ:<10} {n}" + ('' if n >= TARGET_REPEATS else f'   ({TARGET_REPEATS - n} to go)'))
    stopped = [r for r in rows if 0 < r[2] < iterations]
    if stopped:
        print('\nStopped runs can be continued with: python run_trainings.py --resume'
              if any(r[4] in queued for r in stopped) else
              '\nSome runs stopped early and are not in the current queue; ask before deleting them.')


def log(line):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def train(variant, extra):
    cmd = [sys.executable, 'train_cocochorales.py', 'with', f'augment={variant}'] + extra
    print('\n' + '=' * 70)
    print(f'{datetime.now():%Y-%m-%d %H:%M}  starting variant: {variant}')
    print('  ' + ' '.join(cmd[1:]))
    print('=' * 70, flush=True)
    t0 = time.time()
    code = subprocess.call(cmd, cwd=HERE)
    return code, time.time() - t0


def save_examples(variant, shift=DEFAULT_SHIFT, n=3):
    """Write example clips for one augmented variant: wav, label picture, note list."""
    from PIL import Image
    import soundfile
    from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
    from onsets_and_frames.loader_augment import CocoChoralesBrassAug
    from onsets_and_frames.loader_cocochorales import INSTRUMENTS, N_KEYS

    out = os.path.join(CHECK_EXAMPLES, variant)
    os.makedirs(out, exist_ok=True)
    ds = CocoChoralesBrassAug('train', sequence_length=327680,
                              pitch_shift=shift if variant in ('pitch', 'both') else 0,
                              stem_prob=0.5 if variant in ('stems', 'both') else 0.0,
                              seed=int(time.time()) % 100000)
    colors = [(230, 75, 60), (240, 180, 40), (60, 160, 90), (60, 110, 220)]   # trumpet horn trombone tuba
    lines = [f'Example clips for variant "{variant}" (colors in the pictures: '
             'trumpet red, horn yellow, trombone green, tuba blue; darker = note start)']
    for k in range(1, n + 1):
        track = ds.tracks[ds.random.randint(len(ds.tracks))]
        audio, label, _, info = ds.make_clip(track)
        name = f'example{k}'
        soundfile.write(os.path.join(out, f'{name}.wav'), audio, SAMPLE_RATE)
        # picture: time left to right, pitch bottom to top, 3 pixels per key
        steps = label.shape[0]
        img = np.full((N_KEYS * 3, steps, 3), 255, dtype=np.uint8)
        notes = []
        for inst in range(len(INSTRUMENTS)):
            block = label[:, inst * N_KEYS:(inst + 1) * N_KEYS]
            for key in np.nonzero(block.any(axis=0))[0]:
                row = (N_KEYS - 1 - key) * 3
                col = block[:, key]
                c = np.array(colors[inst])
                img[row:row + 3, col >= 2] = c
                img[row:row + 3, col == 3] = (c * 0.5).astype(np.uint8)
                starts = np.nonzero(col == 3)[0]
                notes += [(s * HOP_LENGTH / SAMPLE_RATE, MIN_MIDI + key, INSTRUMENTS[inst]) for s in starts]
        Image.fromarray(img).resize((steps * 2, N_KEYS * 6), Image.NEAREST) \
            .save(os.path.join(out, f'{name}.png'))
        notes.sort()
        with open(os.path.join(out, f'{name}_notes.txt'), 'w') as f:
            f.write(f"track {info['track']}, from {info['begin_sec']:.1f} s, shift {info['shift']:+d} "
                    f"semitones, instruments: {', '.join(info['instruments'])}\n")
            f.write('start_sec\tpitch\tinstrument\n')
            f.writelines(f'{t:.2f}\t{p}\t{i}\n' for t, p, i in notes)
        lines.append(f"  {name}: {info['track']}, shift {info['shift']:+d}, "
                     f"instruments: {', '.join(info['instruments'])}, {len(notes)} note starts")
    print('\n'.join(lines))
    print(f'  saved to {out}')


def main():
    ap = argparse.ArgumentParser(description='Train several data variants one after another.')
    ap.add_argument('--variants', nargs='+', choices=VARIANTS)
    ap.add_argument('--max-shift', type=int, default=DEFAULT_SHIFT,
                    help='largest pitch shift in semitones for pitch and both (default 5)')
    ap.add_argument('--iterations', type=int, default=50000)
    ap.add_argument('--check', action='store_true', help='quick check: examples + 300-step runs')
    ap.add_argument('--resume', action='store_true',
                    help='finish the queue that was stopped (reads current_queue.txt)')
    ap.add_argument('--status', action='store_true', help='list all runs and counts per type')
    a = ap.parse_args()

    if a.status:
        status(a.iterations)
        return
    if a.resume:
        if a.variants or a.check:
            ap.error('--resume takes no --variants or --check: it finishes the stopped queue as it was')
        if not os.path.exists(QUEUE):
            ap.error(f'no queue to resume ({QUEUE} not found). Run --status to see where things stand.')
        entries = read_queue()
    else:
        if not a.variants:
            ap.error('give --variants (or use --resume or --status)')
        if a.max_shift < 1:
            ap.error('--max-shift must be at least 1')
        if a.max_shift != DEFAULT_SHIFT and not any(uses_shift(v) for v in a.variants):
            ap.error('--max-shift only affects pitch and both, and none of those is in --variants')
        entries = [dict(variant=v, shift=a.max_shift if uses_shift(v) else 0,
                        steps=a.iterations, folder='') for v in a.variants]
        if not a.check:
            write_queue(entries)
    stamp = datetime.now().strftime('%y%m%d-%H%M%S')

    results = []
    try:
        for e in entries:
            v, shift = e['variant'], e['shift']
            label = v + (f' +/-{shift}' if shift and shift != DEFAULT_SHIFT else '')
            extra = [f'pitch_shift_max={shift or DEFAULT_SHIFT}']
            if a.check:
                if v != 'none':
                    print(f'\nSaving example clips for "{v}" ...')
                    save_examples(v, shift or DEFAULT_SHIFT)
                extra += [f'iterations={CHECK_STEPS}', 'validation_interval=150',
                          f'checkpoint_interval={CHECK_STEPS}',
                          'logdir=' + os.path.join(CHECK_RUNS, f'{v}-{stamp}')]
            else:
                step = last_step(e['folder']) if e['folder'] else 0
                if e['folder'] and step >= e['steps']:
                    print(f"\n{label}: already finished ({e['folder']}), skipped")
                    results.append((label, 'skipped', 0.0))
                    continue
                extra.append(f"iterations={e['steps']}")
                if step > 0:
                    extra += [f"logdir={e['folder']}", f'resume_iteration={step}']
                    what = f'RESUME from step {step}:'
                    print(f"\n{label}: continuing {e['folder']} from step {step}")
                else:
                    # new run (or one stopped before its first checkpoint): fresh folder
                    while True:   # a fresh folder, even if the previous run failed within the same second
                        e['folder'] = os.path.join(RUNS, run_name(v, shift, datetime.now().strftime('%y%m%d-%H%M%S')))
                        if not os.path.exists(e['folder']):
                            break
                        time.sleep(1)
                    write_queue(entries)
                    extra.append(f"logdir={e['folder']}")
                    what = 'START '
                log(f"{datetime.now():%Y-%m-%d %H:%M}  {what} {label}  ({e['steps']} steps)  {e['folder']}")
            code, secs = train(v, extra)
            results.append((label, code, secs))
            if not a.check:
                log(f'{datetime.now():%Y-%m-%d %H:%M}  {"DONE " if code == 0 else "FAILED"} {label}  '
                    f'after {secs / 3600:.2f} h (exit code {code})')
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C. The queue will not continue.')
        print('Continue it later with:  python run_trainings.py --resume')
        if not a.check:
            log(f'{datetime.now():%Y-%m-%d %H:%M}  STOPPED by Ctrl+C')

    print('\n' + '=' * 70)
    print('Summary' + (' (quick check)' if a.check else ''))
    ref = next((s_ for v, c, s_ in results if v == 'none' and c == 0), None)
    for v, code, secs in results:
        if code == 'skipped':
            print(f'  {v:<10} already finished, skipped')
            continue
        st = 'ok' if code == 0 else f'FAILED (exit code {code})'
        line = f'  {v:<10} {st:<24} {secs / 60:6.1f} min'
        if a.check and ref and code == 0:
            line += f'   rough full-run estimate: about {BASELINE_HOURS * secs / ref:.1f} h'
        print(line)
    if a.check:
        print(f'\nExample clips: {CHECK_EXAMPLES}  (listen to the .wav files, look at the .png pictures)')
        print(f'Check runs (safe to delete afterwards): {CHECK_RUNS}')
    print('ALL DONE' if all(c in (0, 'skipped') for _, c, _ in results) and len(results) == len(entries)
          else 'FINISHED WITH PROBLEMS: see above')


if __name__ == '__main__':
    main()
