"""Convert LAFAN1 (YBot) local transforms to world joint positions.

Writes to ``--out_dir`` (default ``dataset/LAFAN1``):

- ``joints/<name>.npy``: float32 world positions, SMPL joint order, metres.
- ``train.txt`` / ``val.txt``: sequence names from the split file.
- ``Mean_glo_joints.npy`` / ``Std_glo_joints.npy``: per-feature statistics
  of 67-dim ``glo_joints`` over training windows.
- ``meta.json``: settings used.

Run from the repository root::

    python -m LAFAN_process.preprocess_lafan
"""
import argparse
import json
import os
from os.path import join as pjoin

import numpy as np
import yaml
from tqdm import tqdm

from LAFAN_process import lafan_skeleton as sk
from LAFAN_process.dataset_lafan import DATA_ROOT, LENGTHS
from LAFAN_process.representation import window_to_glo_joints

SRC_ROOT = ('/NAS/Personal/HaeminKim/06_Research/00_Data/Geno_Holden/'
            'LAFAN1_resolved/to_YBot')


def convert_sequences(src_root, split_file, out_dir):
    """Run FK on every split sequence and save world positions.

    Parameters
    ----------
    src_root : str
        Folder with ``ybot_skin_mod.bvh`` and ``npy_total/``.
    split_file : str
        YAML file with ``train`` and ``val`` lists of npy file names.
    out_dir : str
        Output folder.

    Returns
    -------
    dict
        ``{split: [sequence name, ...]}``.

    Raises
    ------
    ValueError
        If a file's joint offsets differ from the BVH offsets.
    """
    names, parents, offsets = sk.parse_bvh_header(
        pjoin(src_root, 'ybot_skin_mod.bvh'))
    reorder = sk.smpl_reorder_index(names)
    sk.check_smpl_topology(parents, reorder)
    offsets_m = offsets / 100.0

    with open(split_file) as f:
        splits = yaml.safe_load(f)
    os.makedirs(pjoin(out_dir, 'joints'), exist_ok=True)

    split_names = {}
    for split in ('train', 'val'):
        split_names[split] = []
        for fname in tqdm(splits[split], desc=f'FK {split}'):
            local_tf = np.load(pjoin(src_root, 'npy_total', fname))
            offset_err = np.abs(local_tf[:, 1:, :3, 3]
                                - offsets_m[None, 1:]).max()
            if offset_err > 1e-6:
                raise ValueError(f'{fname}: joint offsets differ from the '
                                 f'BVH by {offset_err:.2e} m.')
            positions = sk.forward_kinematics(local_tf, parents)[:, reorder]
            name = os.path.splitext(fname)[0]
            np.save(pjoin(out_dir, 'joints', name + '.npy'),
                    positions.astype(np.float32))
            split_names[split].append(name)
        with open(pjoin(out_dir, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(split_names[split]) + '\n')
    return split_names


def report_sequences(out_dir, names):
    """Print floor height and bone-length spread of each sequence.

    Parameters
    ----------
    out_dir : str
        Output folder of :func:`convert_sequences`.
    names : list of str
        Sequence names to check.
    """
    print(f'{"sequence":40s} {"toe y p5 (m)":>12s} {"min y (m)":>10s} '
          f'{"bone len range (m)":>18s}')
    for name in names:
        pos = np.load(pjoin(out_dir, 'joints', name + '.npy'))
        bones = np.linalg.norm(
            pos[:, 1:] - pos[:, sk.SMPL_PARENTS[1:]], axis=-1)
        bone_spread = (bones.max(0) - bones.min(0)).max()
        toe_y = np.percentile(pos[:, [10, 11], 1].min(1), 5)
        print(f'{name:40s} {toe_y:12.3f} {pos[..., 1].min():10.3f} '
              f'{bone_spread:18.2e}')


def compute_stats(out_dir, names, fps, lengths, stride):
    """Per-feature mean and std of glo_joints over training windows.

    Windows start every ``stride`` frames and use every length in
    ``lengths``, so each length contributes equally.

    Parameters
    ----------
    out_dir : str
        Output folder of :func:`convert_sequences`.
    names : list of str
        Training sequence names.
    fps : int
        Frame rate.
    lengths : tuple of int
        Window lengths.
    stride : int
        Frames between window starts.

    Returns
    -------
    mean, std : numpy.ndarray
        float32 arrays of shape ``[67]``.
    """
    total = np.zeros(67)
    total_sq = np.zeros(67)
    count = 0
    for name in tqdm(names, desc='stats'):
        pos = np.load(pjoin(out_dir, 'joints', name + '.npy'))
        for length in lengths:
            for start in range(0, len(pos) - length - 1, stride):
                feats = window_to_glo_joints(
                    pos[start:start + length + 1], fps).astype(np.float64)
                total += feats.sum(0)
                total_sq += (feats ** 2).sum(0)
                count += len(feats)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--src_root', default=SRC_ROOT)
    parser.add_argument('--split_file', default=None,
                        help='default: <src_root>/file_splits_no_obs.yaml')
    parser.add_argument('--out_dir', default=DATA_ROOT)
    parser.add_argument('--stats_stride', type=int, default=60,
                        help='frames between window starts for statistics')
    args = parser.parse_args()
    split_file = args.split_file or pjoin(args.src_root,
                                          'file_splits_no_obs.yaml')

    split_names = convert_sequences(args.src_root, split_file, args.out_dir)
    report_sequences(args.out_dir,
                     split_names['train'] + split_names['val'])

    mean, std = compute_stats(args.out_dir, split_names['train'], sk.FPS,
                              LENGTHS, args.stats_stride)
    if (std < 1e-6).any():
        raise ValueError(f'Near-zero std at features '
                         f'{np.where(std < 1e-6)[0].tolist()}')
    np.save(pjoin(args.out_dir, 'Mean_glo_joints.npy'), mean)
    np.save(pjoin(args.out_dir, 'Std_glo_joints.npy'), std)

    meta = {
        'src_root': args.src_root,
        'split_file': split_file,
        'fps': sk.FPS,
        'joint_order': sk.SMPL_ORDER_YBOT_NAMES,
        'lengths': list(LENGTHS),
        'stats_stride': args.stats_stride,
        'n_sequences': {k: len(v) for k, v in split_names.items()},
    }
    with open(pjoin(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('mean', np.round(mean[:4], 4), 'std', np.round(std[:4], 4))
    print(f'Wrote {args.out_dir}')


if __name__ == '__main__':
    main()
