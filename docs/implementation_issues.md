# Implementation issues

Issues found while adapting stage-1 training to LAFAN1 (2026-11-26).
Severity: **high** = trains on wrong targets or crashes; **medium** = wrong
for some datasets/settings; **low** = logging or convenience.

## Fixed

### 1. Stage-1 position losses transformed global positions twice — high
- **Category:** loss correctness
- **Location:** `diffusion/gaussian_diffusion.py`, `training_losses` (ric_* losses)
- **Problem:** with `--data_rep glo_joints`, features 4+ already hold global
  joint positions, but `recover_from_ric(abs_3d=True)` rotated them by the
  root heading and added the root xz again. On a LAFAN1 batch the positions
  used by `ric_pos_loss`, `ric_vel_loss` and `ric_fc_loss` were off by a median
  of 0.94 m (max 6.2 m), and contact detection used the wrong foot speeds.
  Also affects HumanML3D stage-1 training with the README command.
- **Fix:** new `recover_from_glo_joints` in
  `data_loaders/humanml/scripts/motion_process.py` (same idea as the
  `glo_joints` branch in `diffusion/measurements.py`), used when
  `DiffusionConfig.glo_joints` is set (from `--data_rep glo_joints`).
  Loss positions now match the true keyjoints to 2e-6 m
  (`LAFAN_process/check_lafan.py`, check 3).

### 2. Foot-contact threshold assumed 20 fps — medium
- **Category:** frame-rate dependence
- **Location:** `foot_contact_loss_humanml3d` (`threshold=0.05` m/frame)
- **Problem:** 0.05 m/frame is 1 m/s at 20 fps but 3 m/s at 60 fps.
- **Fix:** `GaussianDiffusion.fc_vel_threshold`, set in
  `train/train_sfcontrol.py` to `0.05 * 20 / fps` from the dataset's `fps`
  attribute (20 if absent, so HumanML3D is unchanged).

### 3. Training losses were never written — low
- **Category:** logging
- **Location:** `train/training_loop.py`, `train/train_sfcontrol.py`
- **Problem:** losses were accumulated with `logkv_mean` but `dumpkvs` was
  never called; the default output format was wandb without `wandb.init`.
- **Fix:** `train_sfcontrol` configures the logger for stdout,
  `<save_dir>/log.txt` and `<save_dir>/progress.csv`; `run_loop` dumps every
  `--log_interval` steps.

### 4. `keyframes_mse` was NaN for some batches — low
- **Category:** logging
- **Location:** `masked_l2_weighted`; cause in `get_keyframes_mask('random_joints')`
- **Problem:** the sampler can observe zero joints for a sample, and the
  per-sample mean divided 0 by 0. Logged metric only; the training loss was
  unaffected.
- **Fix:** divide by `max(count, 1)`. Such samples now count as 0, which
  biases the logged mean slightly low.

### 5. `--num_steps` ran to the end of the epoch — low
- **Location:** `TrainLoop.run_loop`
- **Fix:** stop as soon as `num_steps` steps are done.

### 6. Training could not start: missing `eval/` package — high
- **Location:** `train/training_loop.py`
- **Fix:** `eval` is imported inside `evaluate()`, which only runs with
  `--eval_during_training` (HumanML3D only, and `eval/` is still missing).

### 7. GMD heading sign flip — medium (LAFAN1 pipeline only)
- **Location:** `LAFAN_process/representation.py`
- **Problem:** GMD's representation script takes frame-to-frame differences
  of heading quaternions that can flip sign when the heading passes ±180°,
  so the turn is read with the wrong sign and the recovered root drifts
  (about 1 cm in a 158-frame walking window). HumanML3D data built with GMD's
  script has the same flaw; it is not fixed there.
- **Fix:** `qfix` on the heading quaternions before differencing.

## Open

### 8. Sampling-time constants assume 20 fps and 5 keyjoints — medium
- **Location:** `diffusion/gaussian_diffusion.py` (`foot_phys` branch:
  `fix_foot_sliding(threshold=0.5/20.)`, `recover_from_ric(joints_num=5)`),
  `diffusion/measurements.py` (`data_rep` default `glo_root`)
- **Impact:** only for sampling, whose entry points are not in this
  repository. Check frame rate, keyjoint count (6 with `--head`) and
  `glo_joints` handling when sampling scripts are added.

### 9. README stage-1 losses differ from the paper — note
- The paper trains stage 1 with the plain L2 denoising loss only; the README
  command adds `--lambda_ric_pos 0.1 --lambda_ric_vel 10 --lambda_ric_fc 10`.
  This fork keeps the README terms.

### 10. HumanML3D heading formula unpacks face joints inconsistently — low
- **Location:** `Skeleton.inverse_kinematics_np`
  (`l_hip, r_hip, sdr_r, sdr_l = face_joint_idx`, versus
  `r_hip, l_hip, ...` in GMD's `process_file`)
- **Impact:** hip and shoulder vectors partly cancel; heading still points
  the right way because shoulders are wider than hips (true for SMPL and
  YBot). Kept in the LAFAN1 pipeline to match the representation.

### 11. Missing modules — high for sampling/evaluation
- `sample/`, `visualize/`, `eval/` and `utils/utils_transform` are imported
  by sampling and evaluation code but not in this repository.
