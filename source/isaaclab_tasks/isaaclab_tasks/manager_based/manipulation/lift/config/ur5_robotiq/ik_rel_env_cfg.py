from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class UR5RobotiqCubeLiftEnvCfg_IK(joint_pos_env_cfg.UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # IK Action (arm만 override, gripper 유지)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                # [FIX] "pose" 명령은 6D(pos+rot)를 요구하지만 wrist_3 제외 5-DOF 팔로는
                # 구조적으로 rank-deficient (Jacobian 6x5). "position"으로 바꾸면 action_dim이
                # 자동으로 6->3이 되고, 5-DOF로 3D 위치만 추적하므로 완전히 realizable해짐.
                # 방향은 명령하지 않으므로 IK는 현재 EE orientation을 그대로 유지함
                # (differential_ik.py: command_type=="position" 분기, ee_quat_des = 현재 ee_quat).
                command_type="position",
                use_relative_mode=True,
                ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.13]),
        )


@configclass
class UR5RobotiqCubeLiftEnvCfg_IK_PLAY(UR5RobotiqCubeLiftEnvCfg_IK):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
