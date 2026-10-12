r"""
Run the trained Onsets and Frames model on every track of a test set (full
length) and write its predictions as note lists, in the same format as the answer key, so
score_transcriptions.py can score them.

For each track: reads mix.wav, runs the model, turns its output into notes with
the original Onsets and Frames rule (extract_notes, thresholds 0.5, unchanged),
and splits the 352 output columns back into instrument + pitch:
    column = instrument * 88 + (MIDI pitch - 21)

Writes R:\experiments\predictions\<run name>\<dataset>\<track>.tsv:
    onset   offset   pitch   velocity   instrument
If interrupted, run it again: finished tracks are skipped.

Run from the onsets-and-frames folder, oaf environment active:
    python predict_oaf.py --dataset cocochorales      (CocoChorales brass test set)
    python predict_oaf.py --dataset choralebricks     (real recordings, from prepare_choralebricks.py)
    add --run <run folder> for a specific run (default: newest oaf_cocochorales run)
"""
import argparse
import glob
import os
import sys
import time

import soundfile
import torch

from onsets_and_frames.constants import DEFAULT_DEVICE, HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
from onsets_and_frames.decoding import extract_notes
from onsets_and_frames.mel import melspectrogram
from onsets_and_frames.loader_cocochorales import AUDIO_ROOT, INSTRUMENTS, N_KEYS

RUNS = r'R:\experiments\runs'
PREDICTIONS = r'R:\experiments\predictions'
TRACK_LIST = r'R:\splits\cocochorales_brass\test.txt'
CHECKPOINT = 'model-50000.pt'      # final checkpoint, as decided
ONSET_THRESHOLD = 0.5              # original Onsets and Frames defaults, not tuned
FRAME_THRESHOLD = 0.5
HEADER = 'onset\toffset\tpitch\tvelocity\tinstrument\n'
DATASETS = {   # name: (audio root containing <split folder>\<track>\mix.wav, split folder, track list)
    'cocochorales': (AUDIO_ROOT, 'test', TRACK_LIST),
    'choralebricks': (r'R:\datasets\choralebricks\prepared', 'test',
                      r'R:\splits\choralebricks_brass\all.txt'),
}


def load_model(path):
    try:                                    # newer PyTorch needs weights_only=False
        model = torch.load(path, map_location=DEFAULT_DEVICE, weights_only=False)
    except TypeError:                       # older PyTorch doesn't know the option
        model = torch.load(path, map_location=DEFAULT_DEVICE)
    return model.eval()


def transcribe(model, wav_path):
    """List of (onset_sec, offset_sec, midi_pitch, velocity, instrument), sorted."""
    audio, sr = soundfile.read(wav_path, dtype='int16')
    assert sr == SAMPLE_RATE, f'{wav_path}: sample rate {sr}'
    audio = torch.ShortTensor(audio).to(DEFAULT_DEVICE).float().div_(32768.0)
    with torch.no_grad():
        mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        onset, _offset, _, frame, velocity = model(mel)
    # drop the batch dimension and move to the CPU (decoding is a per-frame loop)
    onset, frame, velocity = (x.reshape(x.shape[-2], x.shape[-1]).cpu()
                              for x in (onset, frame, velocity))
    pitches, intervals, velocities = extract_notes(onset, frame, velocity,
                                                   ONSET_THRESHOLD, FRAME_THRESHOLD)
    scale = HOP_LENGTH / SAMPLE_RATE
    notes = []
    for col, (a, b), v in zip(pitches, intervals, velocities):
        inst, key = divmod(int(col), N_KEYS)
        notes.append((a * scale, b * scale, MIN_MIDI + key,
                      int(round(min(max(float(v), 0.0), 1.0) * 127)), inst))
    return sorted(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='run folder (default: newest oaf_cocochorales run)')
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    a = ap.parse_args()
    audio_root, split, track_list = DATASETS[a.dataset]
    if a.run:
        run_dir = a.run
    else:
        runs = sorted(glob.glob(os.path.join(RUNS, 'oaf_cocochorales-*')), key=os.path.getmtime)
        if not runs:
            sys.exit(f'No oaf_cocochorales run found in {RUNS}')
        run_dir = runs[-1]
    run_name = os.path.basename(os.path.normpath(run_dir))
    ckpt = os.path.join(run_dir, CHECKPOINT)
    out_dir = os.path.join(PREDICTIONS, run_name, a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with open(track_list) as f:
        tracks = f.read().split()
    todo = [t for t in tracks if not os.path.exists(os.path.join(out_dir, f'{t}.tsv'))]
    print(f'Model:  {ckpt}')
    print(f'Output: {out_dir}')
    print(f'{len(tracks)} tracks ({a.dataset}), {len(tracks) - len(todo)} already done, {len(todo)} to go')
    if not todo:
        print('PREDICTIONS DONE')
        return

    model = load_model(ckpt)
    t0 = time.time()
    per_inst = [0] * len(INSTRUMENTS)
    for k, track in enumerate(todo, 1):
        notes = transcribe(model, os.path.join(audio_root, split, track, 'mix.wav'))
        tmp = os.path.join(out_dir, f'{track}.tsv.part')
        with open(tmp, 'w') as f:
            f.write(HEADER)
            for on, off, p, v, inst in notes:
                f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{v}\t{inst}\n')
                per_inst[inst] += 1
        os.replace(tmp, os.path.join(out_dir, f'{track}.tsv'))   # never leaves half-written files
        if k in (10, 50) or k % 250 == 0 or k == len(todo):
            el = time.time() - t0
            eta = el / k * (len(todo) - k)
            print(f'  {k}/{len(todo)} tracks  ({el / 60:.1f} min, about {eta / 60:.0f} min left)')

    print('\nPredicted notes per instrument (this session):')
    for name, n in zip(INSTRUMENTS, per_inst):
        print(f'  {name:<9}{n:>9}')
    if a.dataset == 'cocochorales':
        print('\nNext, score them (Miniforge Prompt, oaf):')
        print(f'  python R:\\code\\evaluation\\score_transcriptions.py --pred {out_dir} '
              f'--name {run_name}_cocochorales')
    else:
        print('\nNext: python R:\\code\\evaluation\\score_choralebricks.py')
    print('PREDICTIONS DONE')


if __name__ == '__main__':
    main()
