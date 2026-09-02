# UR5 + Robotiq 2F-85: Target-Aware Pick Task

Reinforcement learning for target-aware pick task with UR5 robot using **Isaac Lab 5.1.0 + RSL-RL PPO**.

## Project Goals

- Pick a specified color cube (red/blue)
- Extend to camera-based color-conditioned RL
- Final goal: Pick specified color cube then stack on another cube
- Target: PRESM 2026 (November 2026, Jeju)

## Hardware & Simulation

- **Robot**: UR5 + Robotiq 2F-85 gripper
- **DoF**: 5 (wrist_3_joint = PhysicsFixedJoint, forced top-down)
- **Simulation**: Isaac Sim 5.1, Isaac Lab 5.1.0
- **RL**: RSL-RL PPO (with NaN protection patches)
- **GPU**: RTX 5080 (16GB)

## Progress Summary

### Completed

- Baseline: Single cube pick success (Franka lift example ported to UR5)
- Wrist twist resolved (wrist_1/2 deviation penalty)
- Distractor added (blue cube) - target-only pick learning
- Target ID randomization implemented
- Vision RL pipeline implemented (Custom CNN + PPO)
- Camera orientation bug discovered and fixed (via code review)
- IK Action Space added (position-only, 5-DoF matching)
- Multiple training runs completed (128x128 to 540x540 resolution)

### Root Causes Identified (3-week debugging cycle)

Through systematic code-level review, five root causes were identified:

1. **Camera orientation error**: Initial quaternion `rot=(0.707, 0, 0, 0.707)` with `convention="world"` was pointing horizontally, not top-down. Fixed to `rot=(0, 1, 0, 0)` with `convention="ros"`.

2. **Perception not the bottleneck**: Policy was receiving ground-truth object position via `object_position_in_robot_root_frame`, meaning the CNN was learning features unnecessary for the task.

3. **IK rank deficiency**: 5-DoF arm with 6-DoF pose IK command produced approximate solutions with unpredictable orientation. Switched to position-only IK.

4. **Gripper effort_limit insufficient**: Set to 10.0 vs official Robotiq reference of 1650 (165x under-specified). Updated to reference value.

5. **Physics solver instability**: `solver_velocity_iteration_count=0` caused critic value function divergence (~1e29) during training. Set to 1 (matching UR10e reference).

6. **Reward gaming**: `object_is_lifted` allowed height-only detection without actual grasp, rewarding accidental cube bumping. Added `object_is_lifted_sustained` with proximity and duration conditions.

### Play Mode Results

Despite all fixes and 155x metric improvement (Bonus 0.04 -> 6.2 at peak):

- Deterministic Play mode shows the robot approaching cube but failing to grasp
- Peak Bonus 6.2 (Reward fix): approaches from side, cannot grasp
- Final fix (all combined): gripper contracts upward, cannot approach cube

### Conclusion

Training metrics improved substantially, but real-world grasp remains unreliable. The gap between training reward and deterministic execution indicates fundamental limitations of the current RL + control setup, not tuning issues.

## Known Issues

1. **5-DoF vs 6-DoF gap**: Fixed wrist_3 limits reachable grasp poses
2. **Isaac Sim rendering hangs**: DLSS minimum resolution (300px) causes hangs at lower camera resolutions
3. **Checkpoint disk usage**: CNN checkpoints ~770MB each; requires periodic cleanup

## File Structure

- `src/joint_pos_env_cfg.py`: UR5 task config (active)
- `src/ik_rel_env_cfg.py`: IK action space config
- `src/cnn_policy.py`: Custom CNN policy for vision input
- `reference/`: Base Isaac Lab files (reference only)
- `scripts/auto_train_lift_30k.sh`: Auto-restart training script

## How to Run

```bash
# Start training (with auto-restart)
bash scripts/auto_train_lift_30k.sh

# Evaluation with video output (headless, single environment)
cd ~/IsaacLab
timeout 90 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-UR5-Robotiq-IK-Play-v0 \
    --num_envs 1 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 200 \
    --checkpoint {checkpoint_path}
```

## Next Steps

- **VLA approach (OpenVLA)**: Leverage pretrained vision-language-action models
- **Behavior Cloning**: Collect teleop demonstrations for supervised learning
- **6-DoF robot reconsideration**: Address structural 5-DoF limitation
- **Sim-to-real transfer**: After grasp stability achieved in simulation

## Notes

- USD file: `/home/choi/ur5_robotiq_2f85.usd` (wrist_3 = PhysicsFixedJoint)
- PPO patches: NaN protection via finite mask in `ppo.py`
- Camera resolution: 540x540 (DLSS 300px minimum compliance)
