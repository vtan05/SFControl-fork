"""YBot (LAFAN1 retarget) skeleton: BVH header, joint reorder and FK.

The LAFAN1 npy files store per-joint local 4x4 transforms in BVH joint
order, in metres. After reordering, the 22 YBot joints have the same tree
as the 22 SMPL joints used by HumanML3D, so every HumanML3D joint index in
this repository (feet, wrists, head, face joints) can be used unchanged.
"""
import re

import numpy as np

FPS = 60

# SMPL / HumanML3D joint order, named with the YBot joint at that position.
SMPL_ORDER_YBOT_NAMES = [
    'Hips',           # 0  pelvis
    'LeftUpLeg',      # 1  left hip
    'RightUpLeg',     # 2  right hip
    'Spine',          # 3  spine1
    'LeftLeg',        # 4  left knee
    'RightLeg',       # 5  right knee
    'Spine1',         # 6  spine2
    'LeftFoot',       # 7  left ankle
    'RightFoot',      # 8  right ankle
    'Spine2',         # 9  spine3
    'LeftToeBase',    # 10 left foot (stage-1 keyjoint)
    'RightToeBase',   # 11 right foot (stage-1 keyjoint)
    'Neck',           # 12 neck
    'LeftShoulder',   # 13 left collar
    'RightShoulder',  # 14 right collar
    'Head',           # 15 head (stage-1 keyjoint)
    'LeftArm',        # 16 left shoulder
    'RightArm',       # 17 right shoulder
    'LeftForeArm',    # 18 left elbow
    'RightForeArm',   # 19 right elbow
    'LeftHand',       # 20 left wrist (stage-1 keyjoint)
    'RightHand',      # 21 right wrist (stage-1 keyjoint)
]

# Parent of each joint in SMPL order (same tree as t2m_kinematic_chain).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19]


def parse_bvh_header(bvh_path):
    """Read joint names, parents and offsets from a BVH hierarchy.

    End sites are skipped, so the result lists only real joints, in the
    order they appear in the file (the order of the npy joint axis).

    Parameters
    ----------
    bvh_path : str
        Path to a BVH file.

    Returns
    -------
    names : list of str
        Joint names in file order.
    parents : numpy.ndarray
        Parent index of each joint, ``-1`` for the root. Shape ``[J]``.
    offsets : numpy.ndarray
        Joint offsets from the parent, in the file's units (centimetres for
        YBot). Shape ``[J, 3]``.
    """
    names, parents, offsets = [], [], []
    stack = []  # joint index for each open brace, None for end sites
    pending = None
    with open(bvh_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('MOTION'):
                break
            m = re.match(r'(ROOT|JOINT)\s+(\S+)', s)
            if m:
                pending = m.group(2)
                continue
            if s.startswith('End Site'):
                pending = None
                continue
            if s == '{':
                if pending is None:
                    stack.append(None)
                else:
                    open_joints = [j for j in stack if j is not None]
                    parents.append(open_joints[-1] if open_joints else -1)
                    names.append(pending)
                    stack.append(len(names) - 1)
                    pending = None
            elif s == '}':
                stack.pop()
            elif s.startswith('OFFSET') and stack and stack[-1] is not None:
                if len(offsets) < len(names):
                    offsets.append([float(x) for x in s.split()[1:4]])
    return names, np.array(parents), np.array(offsets)


def smpl_reorder_index(bvh_names):
    """Index that maps SMPL joint positions to BVH joint indices.

    Parameters
    ----------
    bvh_names : list of str
        Joint names in BVH order.

    Returns
    -------
    numpy.ndarray
        ``idx`` with ``data_smpl = data_bvh[:, idx]``. Shape ``[22]``.

    Raises
    ------
    ValueError
        If a YBot joint name is missing from ``bvh_names``.
    """
    return np.array([bvh_names.index(n) for n in SMPL_ORDER_YBOT_NAMES])


def check_smpl_topology(bvh_parents, reorder_idx):
    """Raise if the reordered BVH tree differs from the SMPL tree.

    Parameters
    ----------
    bvh_parents : numpy.ndarray
        Parent index of each joint in BVH order. Shape ``[J]``.
    reorder_idx : numpy.ndarray
        Output of :func:`smpl_reorder_index`.

    Raises
    ------
    ValueError
        If any joint's parent does not match.
    """
    for j, p in enumerate(SMPL_PARENTS):
        expected = -1 if p == -1 else reorder_idx[p]
        if bvh_parents[reorder_idx[j]] != expected:
            raise ValueError(
                f'Joint {SMPL_ORDER_YBOT_NAMES[j]} has a different parent '
                'than in the SMPL tree.')


def forward_kinematics(local_tf, bvh_parents):
    """Chain local 4x4 transforms into world joint positions.

    Parameters
    ----------
    local_tf : numpy.ndarray
        Local transforms in BVH joint order. Shape ``[T, J, 4, 4]``.
    bvh_parents : numpy.ndarray
        Parent index of each joint in BVH order (parents come before
        children). Shape ``[J]``.

    Returns
    -------
    numpy.ndarray
        World joint positions in BVH order. Shape ``[T, J, 3]``.
    """
    world = np.empty_like(local_tf)
    for j, p in enumerate(bvh_parents):
        if p == -1:
            world[:, j] = local_tf[:, j]
        else:
            world[:, j] = world[:, p] @ local_tf[:, j]
    return world[:, :, :3, 3]
