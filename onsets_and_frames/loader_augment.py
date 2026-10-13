"""
Augmented CocoChorales brass training data for Onsets and Frames.

Same output as CocoChoralesBrass in loader_cocochorales.py (same 4 x 88 label
columns, same labeling rules: onset = 3, sounding = 2, offset = 1, silence = 0,
same 32 ms frames), plus two optional augmentations:

  pitch_shift = K   (0 = off)
      Each clip is shifted by a random whole number of semitones from -K to +K,
      by resampling: pitch and tempo change together (+5 semitones also plays
      about 33% faster, -5 about 25% slower). Note pitches and times in the labels
      are changed to match. The whole clip moves by the same amount, so the
      music stays in tune.

  stem_prob = p     (0 = off)
      With probability p, the clip is a mix of a random 1 to 3 of the four
      instrument stems instead of the full mix, and the labels keep only those
      instruments. The stems are scaled by the same factor that turns their full
      sum into mix.wav, so levels match the real mixes.

Unlike the original loader, only the needed slice of each file is read from
disk (the data itself is the same). Validation should keep using the
un-augmented CocoChoralesBrass class, so validation numbers stay comparable.
"""
import os
from fractions import Fraction

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .constants import (DEFAULT_DEVICE, HOP_LENGTH, HOPS_IN_OFFSET, HOPS_IN_ONSET,
                        MAX_MIDI, MIN_MIDI, SAMPLE_RATE)
from .loader_cocochorales import AUDIO_ROOT, INSTRUMENTS, LABEL_ROOT, N_KEYS, N_OUTPUTS, SPLIT_ROOT

STEM_FILES = [f'{i + 1}_{name}.wav' for i, name in enumerate(INSTRUMENTS)]


def shift_ratio(semitones):
    """(up, down) for resample_poly: playing input*up/down samples raises pitch by `semitones`.

    The exact ratio 2^(s/12) is irrational, so it is approximated by a fraction
    (error under 1 cent); the labels use the exact fraction that was applied.
    """
    if semitones == 0:
        return 1, 1
    f = Fraction(2 ** (-semitones / 12)).limit_denominator(64)
    return f.numerator, f.denominator


def build_labels(notes, n_steps, begin_sec, time_scale, pitch_shift, keep):
    """Label and velocity grids for one clip.

    notes:      array of rows (onset, offset, pitch, velocity, instrument) in track time
    begin_sec:  where the clip starts in track time (before resampling)
    time_scale: output seconds per input second (up / down: below 1 when shifting up)
    keep:       set of instrument numbers to keep
    Same rules as the original loader; notes that started before the clip show
    as 'sounding' from its first frame, exactly as if the full-track grid were sliced.
    """
    label = np.zeros((n_steps, N_OUTPUTS), dtype=np.uint8)
    velocity = np.zeros((n_steps, N_OUTPUTS), dtype=np.uint8)
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    for onset, offset, note, vel, inst in notes:
        inst = int(inst)
        if inst not in keep:
            continue
        on = (onset - begin_sec) * time_scale
        off = (offset - begin_sec) * time_scale
        if off < 0 or on * frames_per_sec >= n_steps:
            continue
        pitch = int(note) + pitch_shift
        if not MIN_MIDI <= pitch <= MAX_MIDI:
            continue
        left = int(round(on * frames_per_sec))
        onset_right = min(n_steps, left + HOPS_IN_ONSET)
        frame_right = min(n_steps, int(round(off * frames_per_sec)))
        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
        f = inst * N_KEYS + pitch - MIN_MIDI
        clip = lambda x: min(max(x, 0), n_steps)
        label[clip(left):clip(onset_right), f] = 3
        label[clip(onset_right):clip(frame_right), f] = 2
        label[clip(frame_right):clip(offset_right), f] = 1
        velocity[clip(left):clip(frame_right), f] = int(vel)
    return label, velocity


class CocoChoralesBrassAug(Dataset):
    def __init__(self, split='train', sequence_length=327680, pitch_shift=0, stem_prob=0.0,
                 seed=42, device=DEFAULT_DEVICE, audio_root=AUDIO_ROOT, label_root=LABEL_ROOT,
                 split_root=SPLIT_ROOT):
        assert sequence_length is not None, 'augmented loader needs a sequence_length'
        self.split, self.sequence_length, self.device = split, sequence_length, device
        self.pitch_shift, self.stem_prob = int(pitch_shift), float(stem_prob)
        self.random = np.random.RandomState(seed)
        self.audio_root, self.label_root = audio_root, label_root
        with open(os.path.join(split_root, f'{split}.txt')) as f:
            self.tracks = f.read().split()
        self.last = {}          # what was done to the most recent clip (for checks)
        print(f'CocoChoralesBrassAug: {len(self.tracks)} {split} tracks, '
              f'pitch shift +/-{self.pitch_shift} semitones, '
              f'stem subsets for {self.stem_prob:.0%} of clips')

    def __len__(self):
        return len(self.tracks)

    def make_clip(self, track):
        """Numpy audio (float32, -1..1), label grid, velocity grid, and a description."""
        folder = os.path.join(self.audio_root, self.split, track)
        mix_path = os.path.join(folder, 'mix.wav')
        audio_len = soundfile.info(mix_path).frames
        n_out = self.sequence_length

        # 1. pitch shift: pick one that fits in the track (shifting up needs more input)
        shift = self.random.randint(-self.pitch_shift, self.pitch_shift + 1) if self.pitch_shift else 0
        while True:
            up, down = shift_ratio(shift)
            n_in = int(np.ceil(n_out * down / up)) + down     # small margin for the filter
            if n_in <= audio_len or shift == 0:
                break
            shift -= int(np.sign(shift))
        n_in = min(n_in, audio_len)
        begin = self.random.randint(audio_len - n_in + 1)

        # 2. stems: full mix, or a random 1-3 of the four instruments
        if self.stem_prob and self.random.rand() < self.stem_prob:
            keep = set(int(i) for i in self.random.choice(4, self.random.randint(1, 4), replace=False))
        else:
            keep = {0, 1, 2, 3}

        read = lambda p: soundfile.read(p, start=begin, stop=begin + n_in, dtype='int16')[0] \
            .astype(np.float64) / 32768.0
        mix = read(mix_path)
        if len(keep) == 4:
            audio = mix
        else:
            stems = [read(os.path.join(folder, 'stems_audio', name)) for name in STEM_FILES]
            n = min(len(mix), *(len(s) for s in stems))
            total = sum(s[:n] for s in stems)
            energy = float(np.dot(total, total))
            gain = float(np.dot(mix[:n], total)) / energy if energy > 0 else 1.0
            audio = gain * sum(stems[i][:n] for i in sorted(keep))

        if shift:
            audio = resample_poly(audio, up, down)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        if len(audio) < n_out:                                   # very short track: pad with silence
            audio = np.pad(audio, (0, n_out - len(audio)))
        audio = audio[:n_out]

        notes = np.loadtxt(os.path.join(self.label_root, self.split, f'{track}.tsv'),
                           delimiter='\t', skiprows=1, ndmin=2)
        n_steps = n_out // HOP_LENGTH
        label, velocity = build_labels(notes, n_steps, begin / SAMPLE_RATE, up / down, shift, keep)
        info = dict(track=track, shift=shift, instruments=[INSTRUMENTS[i] for i in sorted(keep)],
                    begin_sec=begin / SAMPLE_RATE)
        return audio, label, velocity, info

    def __getitem__(self, index):
        audio, label, velocity, info = self.make_clip(self.tracks[index])
        self.last = info
        label = torch.from_numpy(label).to(self.device)
        result = dict(path=info['track'],
                      audio=torch.from_numpy(audio).to(self.device),
                      label=label,
                      velocity=torch.from_numpy(velocity).to(self.device).float().div_(128.0))
        result['onset'] = (label == 3).float()
        result['offset'] = (label == 1).float()
        result['frame'] = (label > 1).float()
        return result
