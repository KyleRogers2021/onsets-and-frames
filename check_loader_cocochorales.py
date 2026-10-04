r"""
Test the CocoChorales loader (loader_cocochorales.py) before any training.

  1. Builds a few training clips exactly as training will, and checks their
     shapes and contents.
  2. Batches them the way training does, to make sure they stack.
  3. Saves pictures (loader_check_*.png in R:\experiments\results\loader_check)
     of each clip's spectrogram with the answer-key notes drawn on top,
     colored by instrument. If the colored lines sit on the bright bands,
     the timing and pitches are right.

Run from the onsets-and-frames folder, oaf environment active:
    python check_loader_cocochorales.py
"""
import os

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from onsets_and_frames.loader_cocochorales import CocoChoralesBrass, INSTRUMENTS, N_KEYS, N_OUTPUTS
from onsets_and_frames.constants import (HOP_LENGTH, MEL_FMAX, MEL_FMIN, MIN_MIDI, N_MELS,
                                         SAMPLE_RATE, WINDOW_LENGTH)

OUT = r'R:\experiments\results\loader_check'
SEQUENCE_LENGTH = 327680 // 2          # what training uses on your 8 GB GPU (10.24 s)
COLORS = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
N_CLIPS = 4

os.makedirs(OUT, exist_ok=True)
data = CocoChoralesBrass('train', sequence_length=SEQUENCE_LENGTH, device='cpu')
expected_steps = SEQUENCE_LENGTH // HOP_LENGTH

print('\n1. Individual clips')
for i in range(N_CLIPS):
    clip = data[i]
    audio, label = clip['audio'], clip['label']
    ok = audio.shape == (SEQUENCE_LENGTH,) and label.shape == (expected_steps, N_OUTPUTS)
    onsets = [int((label[:, k * N_KEYS:(k + 1) * N_KEYS] == 3).sum()) for k in range(4)]
    print(f'  {os.path.basename(os.path.dirname(clip["path"]))}: audio {tuple(audio.shape)}, '
          f'label {tuple(label.shape)}, peak level {audio.abs().max():.2f}, '
          f'onsets per instrument {dict(zip(INSTRUMENTS, onsets))}  {"OK" if ok else "PROBLEM"}')

    # Picture: mel spectrogram (same settings as the model) + true notes
    mel = librosa.feature.melspectrogram(y=audio.numpy(), sr=SAMPLE_RATE, n_fft=WINDOW_LENGTH,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=MEL_FMIN,
                                         fmax=MEL_FMAX, htk=True)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(np.log(mel + 1e-6), origin='lower', aspect='auto', cmap='gray_r')
    for k in range(4):
        block = label[:, k * N_KEYS:(k + 1) * N_KEYS].numpy()
        frames, keys = np.nonzero(block >= 2)            # onset or still sounding
        freqs = 440 * 2 ** ((keys + MIN_MIDI - 69) / 12)
        bins = np.interp(freqs, mel_freqs, np.arange(N_MELS))
        ax.scatter(frames, bins, s=2, color=COLORS[k], label=INSTRUMENTS[k])
    ax.set_ylim(0, 120)
    ax.set_xlabel('time (32 ms frames)')
    ax.set_ylabel('mel bin (low to high pitch)')
    ax.set_title(f'clip {i}: true notes (fundamental) over the spectrogram')
    ax.legend(loc='upper right', markerscale=6)
    path = os.path.join(OUT, f'loader_check_{i}.png')
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'     picture saved: {path}')

print('\n2. Batching (4 clips stacked, as in training)')
batch = next(iter(DataLoader(data, batch_size=4, shuffle=True, drop_last=True)))
print(f'  audio {tuple(batch["audio"].shape)}, onset {tuple(batch["onset"].shape)}, '
      f'frame {tuple(batch["frame"].shape)}, velocity values {torch.unique(batch["velocity"]).tolist()}')

print('\n3. Validation sample list')
valid = CocoChoralesBrass('valid', list_name='valid_sample', sequence_length=SEQUENCE_LENGTH, device='cpu')
print(f'  first clip label shape {tuple(valid[0]["label"].shape)}')
print('\nLOADER CHECK DONE')
