# LAFAN1 for stage-1 training

Stage-1 (keyjoint) training on LAFAN1 retargeted to the YBot character,
without text, at 60 fps, with window lengths 30, 73, 115, 158 and 200 frames.

## Steps

Run from the repository root in the `sfcontrol` environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /host_source/van/envs/sfcontrol

# 1. World joint positions and statistics -> dataset/LAFAN1/ (about 7 min)
python -m LAFAN_process.preprocess_lafan

# 2. Optional checks; writes a stick-figure video to --out_dir
python -m LAFAN_process.check_lafan --out_dir save/lafan_check

# 3. Train stage 1 -> save/lafan_stage1_glo_joints_random_joints/
python -m train.train_sfcontrol --endeffector_conditioned --endeffector_only \
  --endeffector_selection_scheme random_joints --head --diffusion_steps 50 \
  --latent_dim 128 --lambda_ric_pos 0.1 --lambda_ric_vel 10.0 \
  --lambda_ric_fc 10.0 --dataset lafan --data_rep glo_joints --num_frames 224
```

Losses are printed and written to `log.txt` and `progress.csv` in the save
folder every `--log_interval` steps (default 1000). Training runs for
`--num_steps` steps (default 1.2M) and saves every `--save_interval` steps.

## Source data

`preprocess_lafan.py` reads, by default,
`/NAS/Personal/HaeminKim/06_Research/00_Data/Geno_Holden/LAFAN1_resolved/to_YBot`:

- `npy_total/<name>.npy`: per-joint local 4x4 transforms `[T, 22, 4, 4]`,
  metres, in BVH joint order.
- `ybot_skin_mod.bvh`: hierarchy and offsets (centimetres).
- `file_splits_no_obs.yaml`: 100 train and 20 val sequences
  (6k-18k frames each).

## Representation

The same stage-1 features as HumanML3D in this repository
(`--data_rep glo_joints`, GMD's absolute-root representation), and the same
as the paper: global positions of 6 keyjoints plus the root heading.

- **Joint order:** YBot joints are reordered to the SMPL/HumanML3D order
  (`lafan_skeleton.SMPL_ORDER_YBOT_NAMES`). The tree is identical, so the
  repository's joint indices apply: feet = toe joints 10/11, wrists 20/21,
  head 15, face joints `[2, 1, 17, 16]`.
- **Per window:** frame 0 is moved to the origin (xz) and turned to face +Z.
  Features per frame: `[heading, root x, root z, root y]` + global xyz of
  joints 1-21 (67 values). Stage 1 keeps 19: root, both toes, both wrists,
  head.
- **Differences from GMD's script:** no retargeting (one character); no
  per-clip floor shift (LAFAN1's floor is at y = 0); heading smoothing over
  1 s (60 frames); heading quaternions made sign-continuous (see
  `docs/implementation_issues.md`, item 7).
- **Statistics:** per-feature mean/std over training windows starting every
  60 frames, at each of the 5 lengths (`Mean_glo_joints.npy`,
  `Std_glo_joints.npy`).

## Loader

`dataset_lafan.LAFAN1` cuts a random window for every item: the sequence is
chosen in proportion to its length, the length uniformly from the 5, and the
start uniformly. One epoch is (total frames / 60) items. Items are padded to
`--num_frames` (at most 224, the UNet's fixed size) and carry their true
length, so one model trains on all 5 lengths; losses ignore padded frames.

Only stage 1 is supported; `--data_rep glo_root` (stage 2) raises
`NotImplementedError`.

## Files

| File | Purpose |
|---|---|
| `lafan_skeleton.py` | BVH header parsing, SMPL reorder, forward kinematics |
| `representation.py` | window -> 67-dim `glo_joints` features |
| `preprocess_lafan.py` | writes `dataset/LAFAN1/` |
| `dataset_lafan.py` | `LAFAN1` dataset (`--dataset lafan`) |
| `check_lafan.py` | round trip, loader, loss-path and video checks |
