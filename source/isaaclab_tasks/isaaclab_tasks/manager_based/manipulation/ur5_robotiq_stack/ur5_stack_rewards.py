"""Custom reward functions for UR5 stacking task."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── DEBUG: nan/inf 발생 지점 격리용 ────────────────────────────────
_DEBUG_NAN_CHECK = True
_DEBUG_STEP_COUNTER = {"count": 0}

def _check_signal(name: str, tensor: torch.Tensor):
    """텐서에 nan/inf 있으면 처음 30회까지만 stdout에 출력."""
    if not _DEBUG_NAN_CHECK:
        return tensor
    bad = ~torch.isfinite(tensor)
    if bad.any():
        _DEBUG_STEP_COUNTER["count"] += 1
        if _DEBUG_STEP_COUNTER["count"] <= 30:
            n_bad = int(bad.sum().item())
            n_total = tensor.numel()
            finite_vals = tensor[~bad]
            if finite_vals.numel() > 0:
                fmin = finite_vals.min().item()
                fmax = finite_vals.max().item()
                rng = f"[{fmin:.3g}, {fmax:.3g}]"
            else:
                rng = "[all bad]"
            print(f"[NAN_DEBUG #{_DEBUG_STEP_COUNTER['count']}] "
                  f"{name}: {n_bad}/{n_total} bad, finite_range={rng}",
                  flush=True)
    return tensor



def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for reaching the object with the end-effector (Gaussian kernel)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[:, 0, :]
    _check_signal("object_ee_distance/obj_pos", obj_pos)
    _check_signal("object_ee_distance/ee_pos", ee_pos)
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)
    _check_signal("object_ee_distance/dist", dist)
    # ── DEBUG: 실제 위치 값 출력 (첫 5번만) ──────
    if not hasattr(env, "_pos_debug_count"):
        env._pos_debug_count = 0
    if env._pos_debug_count < 5:
        env._pos_debug_count += 1
        print(f"[POS_DEBUG #{env._pos_debug_count}]", flush=True)
        print(f"  env_origin[0]: {env.scene.env_origins[0].tolist()}", flush=True)
        print(f"  obj_pos[0]:    {obj_pos[0].tolist()}", flush=True)
        print(f"  ee_pos[0]:     {ee_pos[0].tolist()}", flush=True)
        print(f"  dist[0]:       {dist[0].item():.4f}", flush=True)
        print(f"  reward[0]:     {(1 - torch.tanh(dist[0] / 0.3)).item():.4f}", flush=True)
    result = 1 - torch.tanh(dist / std)
    _check_signal("object_ee_distance/result", result)
    return result


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward when object is lifted above minimal_height."""
    obj: RigidObject = env.scene[object_cfg.name]
    # height above table surface (table top ~= 0.0 in env-local frame)
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (obj_height > minimal_height).float()


def stacking_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upper_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    lower_object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    xy_threshold: float = 0.06,
    height_diff: float = 0.0406,   # cube height = 2 * init_z(0.0203)
    height_threshold: float = 0.015,
) -> torch.Tensor:
    """Reward when cube_2 is stacked on cube_1 and gripper is open."""
    upper: RigidObject = env.scene[upper_object_cfg.name]
    lower: RigidObject = env.scene[lower_object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)

    stacked = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    # gripper must be open
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_open = torch.isclose(
            robot.data.joint_pos[:, gripper_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, device=env.device),
            atol=0.05,
        )
        stacked = torch.logical_and(stacked, gripper_open)

    return stacked.float()


def cubes_stacked_termination(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    xy_threshold: float = 0.05,
    height_diff: float = 0.0406,   # must match stacking_reward
    height_threshold: float = 0.012,
) -> "torch.Tensor":
    """Terminate when cube_2 is stacked on cube_1 and released."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if robot_cfg is None:
        robot_cfg = SEC("robot")
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]
    robot = env.scene[robot_cfg.name]

    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)
    stacked = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_open = torch.abs(
            robot.data.joint_pos[:, gripper_ids[0]] - torch.tensor(env.cfg.gripper_open_val, device=env.device)
        ) < env.cfg.gripper_threshold
        stacked = torch.logical_and(stacked, gripper_open)

    return stacked


def approach_from_above(
    env,
    std: float = 0.3,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward approaching the object from above (ee z > object z)."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None:
        object_cfg = SEC("cube_2")
    if ee_frame_cfg is None:
        ee_frame_cfg = SEC("ee_frame")

    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]

    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[:, 0, :]

    # ee가 object보다 위에 있을수록 reward
    height_diff = ee_pos[:, 2] - obj_pos[:, 2]
    above_reward = torch.clamp(height_diff, min=0.0)

    # 수평 거리도 가까울수록 reward
    xy_dist = torch.linalg.vector_norm(ee_pos[:, :2] - obj_pos[:, :2], dim=1)
    proximity = torch.exp(-xy_dist / std)

    return above_reward * proximity


def object_goal_distance(
    env,
    std: float = 0.15,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    height_diff: float = 0.0406,
) -> "torch.Tensor":
    """Reward for moving cube_2 toward its stacking target (3D) when lifted."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]

    upper_pos = upper.data.root_pos_w
    lower_pos = lower.data.root_pos_w

    # cube_2가 들려있을 때만 reward
    is_lifted = (upper_pos[:, 2] - env.scene.env_origins[:, 2]) > 0.04

    # 3D 목표: cube_1 위치 + 높이 offset
    target_pos = lower_pos.clone()
    target_pos[:, 2] = target_pos[:, 2] + height_diff

    dist_3d = torch.linalg.vector_norm(upper_pos - target_pos, dim=1)
    goal_reward = torch.exp(-dist_3d / std)

    return goal_reward * is_lifted.float()


def object_xy_goal_distance(
    env,
    std: float = 0.10,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    min_height: float = 0.04,
) -> "torch.Tensor":
    """Reward carrying cube_2 above cube_1 in XY while cube_2 is lifted."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]

    upper_pos = upper.data.root_pos_w
    lower_pos = lower.data.root_pos_w
    is_lifted = (upper_pos[:, 2] - env.scene.env_origins[:, 2]) > min_height
    xy_dist = torch.linalg.vector_norm(upper_pos[:, :2] - lower_pos[:, :2], dim=1)

    return torch.exp(-xy_dist / std) * is_lifted.float()


def object_stack_height_distance(
    env,
    std: float = 0.025,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    height_diff: float = 0.0406,
    xy_threshold: float = 0.08,
) -> "torch.Tensor":
    """Reward lowering cube_2 to the stack height after XY alignment."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]

    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)
    xy_aligned = xy_dist < xy_threshold

    return torch.exp(-height_dist / std) * xy_aligned.float()


def open_gripper_when_aligned(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    xy_threshold: float = 0.08,
    height_diff: float = 0.0406,
    height_threshold: float = 0.025,
) -> "torch.Tensor":
    """Reward opening the gripper when cube_2 is close to the stacking pose."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if robot_cfg is None:
        robot_cfg = SEC("robot")
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    robot = env.scene[robot_cfg.name]
    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]

    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)
    aligned = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_open = torch.abs(
        robot.data.joint_pos[:, gripper_ids[0]] - torch.tensor(env.cfg.gripper_open_val, device=env.device)
    ) < env.cfg.gripper_threshold

    return torch.logical_and(aligned, gripper_open).float()


def grasp_reward(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    diff_threshold: float = 0.06,
) -> "torch.Tensor":
    """Reward for grasping the object — gripper closed + object close to ee."""
    from isaaclab.managers import SceneEntityCfg as SEC
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import FrameTransformer

    if robot_cfg is None:
        robot_cfg = SEC("robot")
    if object_cfg is None:
        object_cfg = SEC("cube_2")
    if ee_frame_cfg is None:
        ee_frame_cfg = SEC("ee_frame")

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]

    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[:, 0, :]
    dist = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    # gripper 닫혀있는지 확인
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_closed = torch.abs(
        robot.data.joint_pos[:, gripper_ids[0]] -
        torch.tensor(env.cfg.gripper_open_val, device=env.device)
    ) > env.cfg.gripper_threshold

    # 가까이 있고 gripper 닫혀있으면 reward
    grasped = torch.logical_and(dist < diff_threshold, gripper_closed)
    return grasped.float()


def lifting_height_reward(
    env,
    robot_cfg=None,
    object_cfg=None,
    ee_frame_cfg=None,
    max_height: float = 0.15,
    minimal_height: float = 0.04
) -> "torch.Tensor":
    """Dense lift signal: 물체가 minimal_height 이상 올라간 후, max_height까지의 진행도.
    가짜 파지면 물체가 안 올라가므로 이 신호도 자동으로 0."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None: object_cfg = SEC("cube_2")

    obj = env.scene[object_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    is_lifted = obj_height > minimal_height
    # minimal_height 이상 올라간 후, 얼마나 더 올라갔는지 정규화
    height_above_min = torch.clamp(obj_height - minimal_height, min=0.0)
    height_norm = torch.clamp(height_above_min / (max_height - minimal_height), 0.0, 1.0)

    _check_signal("lifting_height_reward/obj_height", obj_height)
    _check_signal("lifting_height_reward/height_norm", height_norm)
    result = height_norm * is_lifted.float()
    _check_signal("lifting_height_reward/result", result)
    return result



def real_grasp_reward(
    env,
    robot_cfg=None,
    object_cfg=None,
    ee_frame_cfg=None,
    minimal_height: float = 0.04
) -> "torch.Tensor":
    """Franka lift 스타일: 물체가 minimal_height 이상 올라갔으면 1.
    가짜 파지는 중력이 자동으로 걸러줌 (진짜 안 잡으면 물체가 안 올라감)."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None: object_cfg = SEC("cube_2")

    obj = env.scene[object_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    _check_signal("real_grasp_reward/obj_height", obj_height)
    result = torch.where(obj_height > minimal_height, 1.0, 0.0)
    _check_signal("real_grasp_reward/result", result)
    return result



def pick_success_reward(
    env,
    robot_cfg=None,
    object_cfg=None,
    ee_frame_cfg=None,
    goal_height: float = 0.15,
    height_threshold: float = 0.03
) -> "torch.Tensor":
    """Sparse success: 물체가 목표 높이에 도달."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None: object_cfg = SEC("cube_2")

    obj = env.scene[object_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (obj_height >= (goal_height - height_threshold)).float()



def pick_success_termination(
    env,
    robot_cfg=None,
    object_cfg=None,
    ee_frame_cfg=None,
    goal_height: float = 0.15,
    height_threshold: float = 0.03
) -> "torch.Tensor":
    """Terminate on goal height."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None: object_cfg = SEC("cube_2")

    obj = env.scene[object_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return obj_height >= (goal_height - height_threshold)



def wrist_1_deviation_penalty(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    target: float = -1.571,
) -> "torch.Tensor":
    """Penalize wrist_1 deviation (forward/backward tilt)."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if robot_cfg is None:
        robot_cfg = SEC("robot")
    robot = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(["wrist_1_joint"])
    return (robot.data.joint_pos[:, ids[0]] - target).pow(2)


def wrist_2_deviation_penalty(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    target: float = -1.571,
) -> "torch.Tensor":
    """Penalize wrist_2 deviation (lateral rotation — main cause of sideways grasp)."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if robot_cfg is None:
        robot_cfg = SEC("robot")
    robot = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(["wrist_2_joint"])
    return (robot.data.joint_pos[:, ids[0]] - target).pow(2)


def release_on_cube1(
    env,
    robot_cfg: "SceneEntityCfg" = None,
    upper_object_cfg: "SceneEntityCfg" = None,
    lower_object_cfg: "SceneEntityCfg" = None,
    xy_threshold: float = 0.06,
    height_diff: float = 0.0406,   # must match stacking_reward
    height_threshold: float = 0.015,
) -> "torch.Tensor":
    """Reward for releasing cube_2 on top of cube_1 (gripper open + cubes aligned)."""
    from isaaclab.managers import SceneEntityCfg as SEC
    from isaaclab.assets import Articulation, RigidObject

    if robot_cfg is None:
        robot_cfg = SEC("robot")
    if upper_object_cfg is None:
        upper_object_cfg = SEC("cube_2")
    if lower_object_cfg is None:
        lower_object_cfg = SEC("cube_1")

    robot = env.scene[robot_cfg.name]
    upper = env.scene[upper_object_cfg.name]
    lower = env.scene[lower_object_cfg.name]

    pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)

    # 위치가 맞는지
    aligned = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    # 그리퍼가 열려있는지
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_open = torch.abs(
        robot.data.joint_pos[:, gripper_ids[0]] -
        torch.tensor(env.cfg.gripper_open_val, device=env.device)
    ) < env.cfg.gripper_threshold

    return (aligned & gripper_open).float()


def object_is_lifted_env(
    env,
    minimal_height: float,
    object_cfg=None,
) -> "torch.Tensor":
    """object_is_lifted, but relative to env origin."""
    from isaaclab.managers import SceneEntityCfg as SEC
    if object_cfg is None:
        object_cfg = SEC("cube_2")
    obj = env.scene[object_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(obj_height > minimal_height, 1.0, 0.0)
