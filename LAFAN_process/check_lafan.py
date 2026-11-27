"""Checks for the LAFAN1 stage-1 data pipeline.

1. Representation round trip: positions rebuilt from ``glo_joints`` match
   the canonicalized window.
2. Loader batch: shapes, lengths, mask, padding and normalization.
3. Stage-1 loss path: loader items, un-normalized with
   ``inv_transform_stage1`` and converted with ``recover_from_glo_joints``
   (what the ric_* losses use), match the true keyjoint positions. Also
   reports the foot-contact fraction at the training threshold.
4. A stick-figure video of one window.

Run from the repository root::

    python -m LAFAN_process.check_lafan --out_dir <folder for the video>
"""
import argparse
import os
import random

import numpy as np
import torch

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import \
    recover_from_glo_joints
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from LAFAN_process.dataset_lafan import (FRAMES_PER_SAMPLE, LENGTHS,
                                        load_split)
from LAFAN_process.lafan_skeleton import FPS
from LAFAN_process.representation import canonicalize, window_to_glo_joints

# Stage-1 keyjoints in feature order: root, left/right toe,
# left/right wrist, head.
KEYJOINTS = [0, 10, 11, 20, 21, 15]
# Same value train_sfcontrol sets: 1 m/s.
FC_THRESHOLD = 0.05 * 20 / FPS


def positions_from_features(feats):
    return recover_from_glo_joints(torch.from_numpy(feats), 22).numpy()


def check_round_trip(sequences, n_windows, rng):
    errors = []
    for _ in range(n_windows):
        pos = sequences[rng.integers(len(sequences))]
        length = int(rng.choice(LENGTHS))
        start = int(rng.integers(0, len(pos) - length - 1))
        window = pos[start:start + length + 1]
        rebuilt = positions_from_features(window_to_glo_joints(window, FPS))
        errors.append(np.abs(rebuilt - canonicalize(window)[:-1]).max())
    print(f'[1] round trip over {n_windows} windows: max error '
          f'{max(errors):.2e} m, median {np.median(errors):.2e} m')
    assert max(errors) < 1e-4


def check_loader(batch_size, n_batches):
    conf = DatasetConfig(name='lafan', batch_size=batch_size, num_frames=224,
                         endeffector_only=True, head=True,
                         data_rep='glo_joints')
    loader = get_dataset_loader(conf, num_workers=4)
    values, lengths_seen = [], set()
    for i, (motion, cond) in enumerate(loader):
        lengths = cond['y']['lengths']
        mask = cond['y']['mask']
        if i == 0:
            print(f'[2] batch motion {tuple(motion.shape)}, mask '
                  f'{tuple(mask.shape)}, text {cond["y"]["text"][:2]}')
            assert motion.shape == (batch_size, 19, 1, 224)
        frames = torch.arange(224)
        assert torch.equal(mask[:, 0, 0], frames[None] < lengths[:, None])
        assert torch.all(motion[~mask.expand_as(motion)] == 0)
        lengths_seen.update(lengths.tolist())
        values.append(motion.permute(0, 3, 1, 2)[mask[:, 0, 0]][..., 0])
        if i + 1 == n_batches:
            break
    values = torch.cat(values)
    assert lengths_seen <= set(LENGTHS)
    print(f'    lengths seen: {sorted(lengths_seen)}')
    print(f'    normalized valid frames ({len(values)}): mean '
          f'{values.mean(0).abs().max():.3f} (max |.|), std range '
          f'{values.std(0).min():.3f}-{values.std(0).max():.3f}')
    return loader.dataset


def item_with_window(dataset, item, seed):
    """Fetch a dataset item and the raw window it was cut from."""
    random.seed(seed)
    _, _, _, _, normed, length, _ = dataset[item]
    random.seed(seed)
    seq_idx = int(np.searchsorted(dataset.cum_frames,
                                  item * FRAMES_PER_SAMPLE,
                                  side='right'))
    random.choice(dataset.lengths)
    positions = dataset.sequences[seq_idx]
    start = random.randint(0, len(positions) - length - 1)
    return normed, length, positions[start:start + length + 1]


def check_stage1_loss_path(dataset, n_items):
    errors, contact = [], []
    for k in range(n_items):
        item = (k * 997) % len(dataset)
        normed, length, window = item_with_window(dataset, item, seed=k)
        # As in training_losses: un-normalize, then positions.
        x = dataset.inv_transform_stage1(torch.from_numpy(normed[:length]))
        loss_pos = recover_from_glo_joints(x.float(), joints_num=6).numpy()
        true_pos = canonicalize(window)[:-1][:, KEYJOINTS]
        errors.append(np.abs(loss_pos - true_pos).max())
        foot_speed = np.linalg.norm(np.diff(true_pos[:, 1:3], axis=0), axis=-1)
        contact.append((foot_speed <= FC_THRESHOLD).mean())
    print(f'[3] stage-1 loss positions vs true keyjoints over {n_items} '
          f'items: max error {max(errors):.2e} m')
    print(f'    foot-contact fraction at {FC_THRESHOLD:.4f} m/frame '
          f'(1 m/s): {np.mean(contact):.2f}')
    assert max(errors) < 1e-4


def write_video(sequences, out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)
    pos = sequences[0]
    length = max(LENGTHS)
    start = int(rng.integers(0, len(pos) - length - 1))
    feats = window_to_glo_joints(pos[start:start + length + 1], FPS)
    path = os.path.join(out_dir, f'lafan_window_{length}f.mp4')
    plot_3d_motion(path, t2m_kinematic_chain, positions_from_features(feats),
                   f'LAFAN1 {length} frames', 'humanml', fps=FPS)
    print(f'[4] wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out_dir', default='save/lafan_check')
    parser.add_argument('--n_windows', type=int, default=500)
    parser.add_argument('--n_batches', type=int, default=30)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    _, sequences = load_split('train')
    check_round_trip(sequences, args.n_windows, rng)
    dataset = check_loader(64, args.n_batches)
    check_stage1_loss_path(dataset, 200)
    write_video(sequences, args.out_dir, rng)


if __name__ == '__main__':
    main()
