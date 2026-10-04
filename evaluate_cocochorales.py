"""
Instrument-aware validation scoring for Onsets and Frames on CocoChorales brass.

The original evaluate() in evaluate.py expects 88 piano keys. With 352 outputs
(4 instruments x 88 keys) it fails, because it converts every column to a
frequency and columns past 88 become impossible frequencies.

This file leaves evaluate.py completely unchanged and calls it as published.
It only rearranges the grids before they reach it: the four instruments'
88-key blocks are laid end to end in time, with one silent frame between them:

    352 columns x T frames   ->   88 columns x 4*(T+1) frames
    [trumpet | horn | trombone | tuba]   ->   trumpet, gap, horn, gap, ...

Because each instrument occupies its own stretch of time, a predicted note can
only match a true note of the SAME instrument. So every original metric
(note F1, note-with-offsets F1, frame F1) becomes instrument-aware, using real
pitches. The silent gap stops a note at the end of one instrument's block from
running into the next block.
"""
import torch

from evaluate import evaluate as evaluate_original
from onsets_and_frames.loader_cocochorales import INSTRUMENTS, N_KEYS

GRIDS = ('onset', 'offset', 'frame', 'velocity')


def stack_instruments(x):
    """(T, 4*88) grid -> (4*(T+1), 88) grid: instruments end to end in time."""
    n_frames = x.shape[0]
    blocks = x.reshape(n_frames, len(INSTRUMENTS), N_KEYS).permute(1, 0, 2)   # (4, T, 88)
    gap = torch.zeros(len(INSTRUMENTS), 1, N_KEYS, dtype=x.dtype, device=x.device)
    return torch.cat([blocks, gap], dim=1).reshape(-1, N_KEYS)


class _StackedModel:
    """Runs the real model on the original clip, then rearranges its predictions."""
    def __init__(self, model):
        self.model = model

    def run_on_batch(self, item):
        predictions, losses = self.model.run_on_batch(item['_original'])
        return {key: stack_instruments(value) for key, value in predictions.items()}, losses


def _stacked_items(data):
    """Same clips, with the label grids rearranged the same way."""
    for item in data:
        stacked = dict(item)
        stacked['_original'] = item
        for key in GRIDS:
            stacked[key] = stack_instruments(item[key])
        yield stacked


def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    return evaluate_original(_stacked_items(data), _StackedModel(model),
                             onset_threshold, frame_threshold, save_path)
