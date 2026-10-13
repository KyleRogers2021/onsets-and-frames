r"""
Train several Onsets and Frames data variants one after another, so a queue of
runs can go overnight without restarting anything by hand.

Variants (pick any, in any order; they run in the order given):
    none    the original baseline data
    pitch   random pitch shift (+/-5 semitones)
    stems   random stem subsets (half the clips use 1-3 instruments)
    both    pitch shift and stem subsets together

Full runs (50,000 steps each, same settings as the baseline):
    python run_trainings.py --variants both pitch stems
Each run gets its own folder, e.g. R:\experiments\runs\oaf_cocochorales-aug_both-<date>.
If one run fails, the queue moves on to the next. Ctrl+C stops the whole queue.

Continue a stopped queue later with the same command plus --resume:
    python run_trainings.py --variants both pitch stems --resume
For each variant it looks at that variant's newest run folder: finished runs
are skipped, an unfinished run continues from its last checkpoint (at most
1,000 steps are lost) with the same variant, and a variant with no run yet
starts fresh. Without --resume, every variant always starts a new run.
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
HERE = os.path.dirname(os.path.abspath(__file__))


def latest_run(variant):
    """(newest run folder of this variant, its last checkpoint step), or (None, 0)."""
    tag = '' if variant == 'none' else f'aug_{variant}-'
    pattern = re.compile(r'oaf_cocochorales-' + re.escape(tag) + r'\d{6}-\d{6}$')
    folders = [d for d in glob.glob(os.path.join(RUNS, 'oaf_cocochorales-*'))
               if os.path.isdir(d) and pattern.match(os.path.basename(d))]
    if not folders:
        return None, 0
    folder = max(folders, key=os.path.getmtime)
    steps = [int(m.group(1)) for f in os.listdir(folder)
             for m in [re.fullmatch(r'model-(\d+)\.pt', f)] if m]
    return folder, max(steps, default=0)


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


def save_examples(variant, n=3):
    """Write example clips for one augmented variant: wav, label picture, note list."""
    from PIL import Image
    import soundfile
    from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
    from onsets_and_frames.loader_augment import CocoChoralesBrassAug
    from onsets_and_frames.loader_cocochorales import INSTRUMENTS, N_KEYS

    out = os.path.join(CHECK_EXAMPLES, variant)
    os.makedirs(out, exist_ok=True)
    ds = CocoChoralesBrassAug('train', sequence_length=327680,
                              pitch_shift=5 if variant in ('pitch', 'both') else 0,
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
    ap.add_argument('--variants', nargs='+', required=True, choices=VARIANTS)
    ap.add_argument('--iterations', type=int, default=50000)
    ap.add_argument('--check', action='store_true', help='quick check: examples + 300-step runs')
    ap.add_argument('--resume', action='store_true',
                    help='skip finished runs and continue unfinished ones from their last checkpoint')
    a = ap.parse_args()
    stamp = datetime.now().strftime('%y%m%d-%H%M%S')

    results = []
    try:
        for v in a.variants:
            if a.check:
                if v != 'none':
                    print(f'\nSaving example clips for "{v}" ...')
                    save_examples(v)
                extra = [f'iterations={CHECK_STEPS}', 'validation_interval=150',
                         f'checkpoint_interval={CHECK_STEPS}',
                         'logdir=' + os.path.join(CHECK_RUNS, f'{v}-{stamp}')]
            else:
                extra = [f'iterations={a.iterations}']
                what = 'START '
                if a.resume:
                    folder, step = latest_run(v)
                    if folder and step >= a.iterations:
                        print(f'\n{v}: already finished ({folder}, step {step}), skipped')
                        results.append((v, 'skipped', 0.0))
                        continue
                    if folder and step > 0:
                        extra += [f'logdir={folder}', f'resume_iteration={step}']
                        what = f'RESUME from step {step}:'
                        print(f'\n{v}: continuing {folder} from step {step}')
                    elif folder:
                        print(f'\n{v}: {folder} has no checkpoint yet, starting a new run instead')
                log(f'{datetime.now():%Y-%m-%d %H:%M}  {what} {v}  ({a.iterations} steps)')
            code, secs = train(v, extra)
            results.append((v, code, secs))
            if not a.check:
                log(f'{datetime.now():%Y-%m-%d %H:%M}  {"DONE " if code == 0 else "FAILED"} {v}  '
                    f'after {secs / 3600:.2f} h (exit code {code})')
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C. The queue will not continue.')
        if not a.check:
            log(f'{datetime.now():%Y-%m-%d %H:%M}  STOPPED by Ctrl+C')

    print('\n' + '=' * 70)
    print('Summary' + (' (quick check)' if a.check else ''))
    ref = next((s for v, c, s in results if v == 'none' and c == 0), None)
    for v, code, secs in results:
        if code == 'skipped':
            print(f'  {v:<7} already finished, skipped')
            continue
        status = 'ok' if code == 0 else f'FAILED (exit code {code})'
        line = f'  {v:<7} {status:<24} {secs / 60:6.1f} min'
        if a.check and ref and code == 0:
            line += f'   rough full-run estimate: about {BASELINE_HOURS * secs / ref:.1f} h'
        print(line)
    if a.check:
        print(f'\nExample clips: {CHECK_EXAMPLES}  (listen to the .wav files, look at the .png pictures)')
        print(f'Check runs (safe to delete afterwards): {CHECK_RUNS}')
    print('ALL DONE' if all(c in (0, 'skipped') for _, c, _ in results) and len(results) == len(a.variants)
          else 'FINISHED WITH PROBLEMS: see above')


if __name__ == '__main__':
    main()
