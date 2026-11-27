"""LAFAN1 dataset for stage-1 (keyjoint) training.

Each sample is a random window of one of the training lengths, cut from a
long LAFAN1 sequence and converted to ``glo_joints`` features on the fly.
Run ``python -m LAFAN_process.preprocess_lafan`` first.
"""
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data

from LAFAN_process.lafan_skeleton import FPS
from LAFAN_process.representation import window_to_glo_joints

DATA_ROOT = './dataset/LAFAN1'
LENGTHS = (30, 73, 115, 158, 200)
# One epoch has one sample per this many frames of the split.
FRAMES_PER_SAMPLE = 60

# Stage-1 keyjoint features in the 67-dim glo_joints vector, same as
# Text2MotionDatasetV2: root (4), left/right toe, left/right wrist, head.
STAGE1_DIM = [0, 1, 2, 3, 31, 32, 33, 34, 35, 36, 61, 62, 63, 64, 65, 66]
HEAD_DIM = [46, 47, 48]


def load_split(split, data_root=DATA_ROOT):
    """Load world joint positions of every sequence in a split.

    Parameters
    ----------
    split : str
        ``'train'`` or ``'val'``.
    data_root : str
        Folder written by ``preprocess_lafan``.

    Returns
    -------
    names : list of str
        Sequence names.
    sequences : list of numpy.ndarray
        World positions in SMPL order, each of shape ``[T, 22, 3]``.
    """
    with open(pjoin(data_root, f'{split}.txt')) as f:
        names = [line.strip() for line in f if line.strip()]
    sequences = [np.load(pjoin(data_root, 'joints', n + '.npy'))
                 for n in names]
    return names, sequences


class LAFAN1(data.Dataset):
    """Random LAFAN1 windows as normalized stage-1 ``glo_joints`` features.

    Items match ``Text2MotionDatasetV2`` so ``t2m_collate`` can batch them,
    with empty text. The class is its own ``t2m_dataset`` because training
    code reads ``data.dataset.t2m_dataset``.

    Parameters
    ----------
    split : str
        ``'train'`` or ``'val'``.
    num_frames : int
        Length every sample is zero-padded to. Must be at least the longest
        training length.
    endeffector_only : bool
        Must be True; only stage 1 is supported.
    head : bool
        Include the head keyjoint (19 features instead of 16).
    data_rep : str
        Must be ``'glo_joints'``.
    lengths : tuple of int
        Window lengths to sample from, uniformly.
    data_root : str
        Folder written by ``preprocess_lafan``.

    Raises
    ------
    NotImplementedError
        For stage 2 or a data representation other than ``glo_joints``.
    ValueError
        If ``num_frames`` is shorter than the longest length.
    """

    def __init__(self, split='train', num_frames=224, endeffector_only=True,
                 head=True, data_rep='glo_joints', lengths=LENGTHS,
                 data_root=DATA_ROOT, **kwargs):
        if not endeffector_only or data_rep != 'glo_joints':
            raise NotImplementedError(
                'LAFAN1 currently supports only stage 1 '
                '(--endeffector_only --data_rep glo_joints).')
        if num_frames < max(lengths):
            raise ValueError(f'num_frames={num_frames} is shorter than the '
                             f'longest length {max(lengths)}.')
        self.t2m_dataset = self
        self.split = split
        self.max_motion_length = num_frames
        self.lengths = tuple(lengths)
        self.fps = FPS
        self.data_rep = data_rep
        self.endeffector_only = endeffector_only
        self.head = head
        self.endeffector_dim = STAGE1_DIM + (HEAD_DIM if head else [])
        # Only used by stage 2 (all_endeffectors), kept for TrainLoop.
        self.endeffector_joints = None

        self.mean = np.load(pjoin(data_root, 'Mean_glo_joints.npy'))
        self.std = np.load(pjoin(data_root, 'Std_glo_joints.npy'))

        self.names, self.sequences = load_split(split, data_root)
        too_short = [n for n, s in zip(self.names, self.sequences)
                     if len(s) <= max(self.lengths)]
        if too_short:
            raise ValueError(f'Sequences shorter than {max(self.lengths)} '
                             f'frames: {too_short}')
        seq_lens = np.array([len(s) for s in self.sequences])
        self.cum_frames = np.cumsum(seq_lens)
        self.num_samples = int(self.cum_frames[-1]) // FRAMES_PER_SAMPLE
        print(f'LAFAN1 {split}: {len(self.sequences)} sequences, '
              f'{self.cum_frames[-1]} frames, {self.num_samples} samples '
              f'per epoch, lengths {self.lengths}')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # Item i points at frame i * FRAMES_PER_SAMPLE of the concatenated
        # split, so sequences are drawn in proportion to their length.
        seq_idx = int(np.searchsorted(self.cum_frames,
                                      item * FRAMES_PER_SAMPLE,
                                      side='right'))
        positions = self.sequences[seq_idx]
        m_length = random.choice(self.lengths)
        start = random.randint(0, len(positions) - m_length - 1)
        window = positions[start:start + m_length + 1]

        motion = window_to_glo_joints(window, self.fps)
        motion = (motion - self.mean) / self.std
        motion = np.concatenate([
            motion,
            np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        ], axis=0)
        motion = motion[:, self.endeffector_dim]
        # Same layout as Text2MotionDatasetV2: word embeddings,
        # part-of-speech one-hots, caption, sentence length, motion,
        # length, tokens.
        return None, None, '', 0, motion, m_length, ''

    def inv_transform(self, data):
        """Undo normalization of full 67-dim features."""
        if isinstance(data, torch.Tensor):
            return (data * torch.from_numpy(self.std).to(data.device)
                    + torch.from_numpy(self.mean).to(data.device))
        return data * self.std + self.mean

    def inv_transform_stage1(self, data, traject_only=None):
        """Undo normalization of stage-1 keyjoint features."""
        std = self.std[self.endeffector_dim]
        mean = self.mean[self.endeffector_dim]
        if isinstance(data, torch.Tensor):
            return (data * torch.from_numpy(std).to(data.device)
                    + torch.from_numpy(mean).to(data.device))
        return data * std + mean
