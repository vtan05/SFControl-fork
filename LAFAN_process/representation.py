"""GMD absolute-root representation (stage-1 ``glo_joints``) for LAFAN1.

Follows ``process_file_abs_root`` in GMD's
``dataset/HumanML3D_abs/motion_representation.ipynb`` and the
``glo_joints`` step of ``Text2MotionDatasetV2.__getitem__``
(``data_loaders/humanml/data/dataset.py``), restricted to the first 67
features (root + joint positions), which are all that ``glo_joints`` uses.

Differences from GMD, all deliberate:

- No skeleton retargeting (LAFAN1 is a single character) and no per-clip
  floor shift (the LAFAN1 floor is already at y = 0).
- The heading smoothing width is 1 second (``fps`` frames) instead of 20
  frames, which is 1 second at HumanML3D's 20 fps.
- Heading quaternions are made sign-continuous (``qfix``) before taking
  frame-to-frame differences. Without it, a heading that passes +-180
  degrees flips the quaternion sign, the per-frame turn is read with the
  wrong sign, and the recovered root drifts (about 1 cm in a 158-frame
  walking window).
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from data_loaders.humanml.common.quaternion import (qbetween_np, qfix,
                                                    qinv_np, qmul_np,
                                                    qrot_np)
from data_loaders.humanml.scripts.motion_process import (recover_from_ric,
                                                         recover_root_rot_pos)

N_JOINTS = 22
# HumanML3D face joints, as passed to process_file and
# Skeleton.inverse_kinematics_np.
FACE_JOINT_INDX = [2, 1, 17, 16]
Y_AXIS = np.array([[0., 1., 0.]])
Z_AXIS = np.array([[0., 0., 1.]])


def _normalize(v):
    return v / np.sqrt((v ** 2).sum(axis=-1))[..., np.newaxis]


def canonicalize(positions):
    """Move frame 0 to the origin (xz) and rotate it to face +Z.

    Parameters
    ----------
    positions : numpy.ndarray
        World joint positions in SMPL order. Shape ``[T, 22, 3]``.

    Returns
    -------
    numpy.ndarray
        Canonicalized positions, float32. Shape ``[T, 22, 3]``.
    """
    root_pos_init = positions[0]
    positions = positions - root_pos_init[0] * np.array([1, 0, 1])

    # Same unpacking as GMD's process_file_abs_root.
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = (root_pos_init[r_hip] - root_pos_init[l_hip]
              + root_pos_init[sdr_r] - root_pos_init[sdr_l])
    forward_init = _normalize(np.cross(Y_AXIS, _normalize(across), axis=-1))
    root_quat_init = qbetween_np(forward_init, Z_AXIS)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    return qrot_np(root_quat_init, positions)


def root_heading(positions, fps):
    """Per-frame root heading quaternions, with frame 0 set to identity.

    Same as the root part of ``Skeleton.inverse_kinematics_np`` with
    ``smooth_forward=True``, including its face-joint unpacking order.

    Parameters
    ----------
    positions : numpy.ndarray
        Canonicalized joint positions. Shape ``[T, 22, 3]``.
    fps : int
        Frame rate; the forward direction is smoothed over ``fps`` frames.

    Returns
    -------
    numpy.ndarray
        Sign-continuous heading quaternions (w, x, y, z). Shape ``[T, 4]``.
    """
    l_hip, r_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = (positions[:, r_hip] - positions[:, l_hip]
              + positions[:, sdr_r] - positions[:, sdr_l])
    forward = np.cross(Y_AXIS, _normalize(across), axis=-1)
    forward = gaussian_filter1d(forward, fps, axis=0, mode='nearest')
    forward = _normalize(forward)
    root_quat = qbetween_np(forward, Z_AXIS.repeat(len(forward), axis=0))
    root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    return qfix(root_quat[:, None])[:, 0]


def window_to_glo_joints(positions, fps):
    """Convert a window of world positions to 67-dim ``glo_joints``.

    Parameters
    ----------
    positions : numpy.ndarray
        World joint positions in SMPL order, floor at y = 0.
        Shape ``[L + 1, 22, 3]``.
    fps : int
        Frame rate of ``positions``.

    Returns
    -------
    numpy.ndarray
        float32 features of shape ``[L, 67]``: heading angle, root x, root
        z, root y, then global xyz of joints 1-21. One frame shorter than
        the input because the representation is built from velocities.
    """
    positions = canonicalize(positions)
    r_rot = root_heading(positions, fps)

    '''Root linear and angular velocity'''
    velocity = qrot_np(r_rot[1:], positions[1:, 0] - positions[:-1, 0])
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    '''Root-relative, heading-free joint positions (get_rifke)'''
    ric = positions.copy()
    ric[..., 0] -= ric[:, 0:1, 0]
    ric[..., 2] -= ric[:, 0:1, 2]
    ric = qrot_np(np.repeat(r_rot[:, None], N_JOINTS, axis=1), ric)

    root_data = np.concatenate([
        np.arcsin(r_velocity[:, 2:3]),
        velocity[:, [0, 2]],
        ric[:-1, 0, 1:2],
    ], axis=-1)
    data = np.concatenate(
        [root_data, ric[:-1, 1:].reshape(len(root_data), -1)], axis=-1)

    '''Absolute root heading and xz (GMD's new_data)'''
    data = torch.from_numpy(data).float()
    _, r_pos, rot_ang = recover_root_rot_pos(data, return_rot_ang=True)
    data[:, 0] = rot_ang
    data[:, [1, 2]] = r_pos[:, [0, 2]]

    '''Global joint positions (glo_joints)'''
    glo = recover_from_ric(data, N_JOINTS, abs_3d=True)
    out = torch.cat([data[:, :4], glo[:, 1:].reshape(len(data), -1)], dim=1)
    return out.numpy()
