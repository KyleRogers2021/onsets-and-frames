"""
CocoChorales brass dataset for Onsets and Frames.

Does the same job as PianoRollAudioDataset in dataset.py, with the SAME
labeling rules (onset = 3, sounding = 2, offset = 1, silence = 0; same 32 ms
frames; same random clip selection), but:

  1. Reads each track from disk when it's needed, instead of loading the
     whole dataset into memory at startup (which would not fit in 32 GB),
     and never writes cached copies next to the data.
  2. Uses 4 x 88 label columns, one block of 88 keys per instrument:
         columns   0- 87  trumpet
         columns  88-175  horn
         columns 176-263  trombone
         columns 264-351  tuba
     Column = instrument * 88 + (MIDI pitch - 21).

Note lists come from R:\\datasets\\cocochorales\\labels (made by prep_labels_cocochorales.py);
track lists come from R:\\splits\\cocochorales_brass (made by make_splits_cocochorales.py).
"""
import os

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from .constants import (DEFAULT_DEVICE, HOP_LENGTH, HOPS_IN_OFFSET, HOPS_IN_ONSET,
                        MAX_MIDI, MIN_MIDI, SAMPLE_RATE)

INSTRUMENTS = ['trumpet', 'horn', 'trombone', 'tuba']
N_KEYS = MAX_MIDI - MIN_MIDI + 1                     # 88
N_OUTPUTS = len(INSTRUMENTS) * N_KEYS                # 352

AUDIO_ROOT = r'R:\datasets\cocochorales\extracted\main_dataset'
LABEL_ROOT = r'R:\datasets\cocochorales\labels'
SPLIT_ROOT = r'R:\splits\cocochorales_brass'


class CocoChoralesBrass(Dataset):
    def __init__(self, split='train', list_name=None, sequence_length=None, seed=42,
                 device=DEFAULT_DEVICE, audio_root=AUDIO_ROOT, label_root=LABEL_ROOT,
                 split_root=SPLIT_ROOT):
        """
        split:     'train', 'valid' or 'test' (which folder the tracks live in)
        list_name: which track list to use; defaults to the split's own list.
                   Use 'valid_sample' for the 200-track validation sample.
        """
        self.split = split
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.audio_root, self.label_root = audio_root, label_root
        with open(os.path.join(split_root, f'{list_name or split}.txt')) as f:
            self.tracks = f.read().split()
        print(f'CocoChoralesBrass: {len(self.tracks)} {split} tracks '
              f'(list: {list_name or split}), read from disk as needed')

    def __len__(self):
        return len(self.tracks)

    def load(self, track):
        """Full track: int16 audio, label grid, velocity grid (same as the original load())."""
        audio_path = os.path.join(self.audio_root, self.split, track, 'mix.wav')
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE, f'{audio_path}: sample rate {sr}'
        audio = torch.ShortTensor(audio)
        n_steps = (len(audio) - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, N_OUTPUTS, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, N_OUTPUTS, dtype=torch.uint8)
        notes = np.loadtxt(os.path.join(self.label_root, self.split, f'{track}.tsv'),
                           delimiter='\t', skiprows=1, ndmin=2)
        for onset, offset, note, vel, inst in notes:
            # --- identical to the original labeling rules ---
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = min(n_steps, int(round(offset * SAMPLE_RATE / HOP_LENGTH)))
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
            # --- only change: the column also depends on the instrument ---
            f = int(inst) * N_KEYS + int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = int(vel)
        return dict(path=audio_path, audio=audio, label=label, velocity=velocity)

    def __getitem__(self, index):
        # Below here matches the original __getitem__ line for line.
        data = self.load(self.tracks[index])
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        return result
