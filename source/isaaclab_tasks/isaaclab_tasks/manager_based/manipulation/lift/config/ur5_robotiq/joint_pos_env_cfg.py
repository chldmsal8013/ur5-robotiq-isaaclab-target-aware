from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import joint_deviation_l1
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils


def raw_rgb_image(env, sensor_cfg):
    """Raw RGB image, normalized to [0, 1], as (N, H, W, 3). Fed to a learnable CNN in the policy."""
    camera = env.scene[sensor_cfg.name]
    rgb = camera.data.output["rgb"]  # (N, H, W, 3), uint8
    return rgb.float() / 255.0


@configclass
class ImageCfg(ObsGroup):
    """Raw camera image observation group (kept separate from the 1D 'policy' group)."""

    image = ObsTerm(func=raw_rgb_image, params={"sensor_cfg": SceneEntityCfg("tiled_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


UR5_ROBOTIQ_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="/home/choi/ur5_robotiq_2f85.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # [FIX] velocity_iteration_count 0 -> 1. IsaacLab 공식 UR10e_CFG(universal_robots.py) --
        # solver_position_iteration_count=16으로 우리와 정확히 동일한 값을 쓰는 같은 UR 계열
        # 레퍼런스 -- 도 velocity_iteration_count=1을 씀. 0이면 접촉 시 velocity-level 제약(반발/
        # 마찰) 해석이 생략돼 에너지가 소산되지 않고 오히려 증폭될 수 있음 -- gripper effort_limit을
        # 165배(10->1650) 올린 이후 critic이 iter 26 근처에서 폭주(value_function loss ~1e29)한
        # 사고와 같은 계열의 원인으로 지목된 항목.
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": -1.571,
            "wrist_2_joint": -1.571,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint"],
            stiffness=400.0,
            damping=60.0,
        ),
        # [FIX] effort/velocity limit + stiffness/damping ported from IsaacLab's own validated
        # Robotiq 2F-85 reference config (isaaclab_assets/robots/franka.py: FRANKA_ROBOTIQ_GRIPPER_CFG)
        # -- same physical gripper, different arm. Previous effort_limit_sim=10.0 was ~165x below
        # that reference, which likely capped achievable grip force well below what's needed to
        # hold a cube. Also splits out the inner-finger joints into their own low-PD "gripper_finger"
        # group instead of lumping them into fully-passive (0/0): the reference config's own comment
        # says this is "to enable the gripper to grasp in a parallel manner".
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_inner_finger_joint", "left_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=0.2,
            damping=0.001,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint",
                              "right_inner_finger_knuckle_joint",
                              "left_inner_finger_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class UR5RobotiqCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # reaching_object weight: 가까이 가기만으론 부족 (Kimi)
        self.rewards.reaching_object.weight = 0.8  # Kimi: 0.5 -> 0.8 (가까이 가기도 필요)
        # Reward 중복 제거 (Claude Code 진단)
        self.rewards.lifting_object.weight = 0.0  # object_is_lifted_bonus와 중복
        self.rewards.object_goal_tracking.weight = 0.0  # pick 안정화만
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.reaching_object.params["std"] = 0.3
        
        self.scene.robot = UR5_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint"],
            scale=0.5,
            use_default_offset=True,
        )
        # Continuous gripper action (Binary → JointPositionActionCfg)
        # 이전 결과: Binary로 잡기 시도했지만 안정 grip 어려움 (throwing 발생)
        # 목적: Continuous로 부분 close 가능하게 함 (0.0~0.8 사이)
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            scale=0.4,  # action [-1, 1] → joint pos [-0.4, 0.4], offset과 함께 [0.0, 0.8]
            use_default_offset=False,
            offset=0.4,  # neutral position (half-closed)
        )
        self.commands.object_pose.body_name = "wrist_3_link"
        # Target cube (red)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        # Distractor cube (blue) - fixed position
        self.scene.distractor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.25, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # TiledCamera for Vision RL (top-down view)
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 1.0),
                rot=(0.0, 1.0, 0.0, 0.0),
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            width=540,  # DLSS 300px 통과 (540*0.58=313)
            height=540,  # DLSS 300px 통과
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.13]),
                ),
            ],
        )
        self.curriculum = None
        self.rewards.action_rate.weight = -1e-4
        self.rewards.joint_vel.weight = -1e-4
        # wrist_3는 USD에서 fixed joint로 고정, deviation penalty 불필요
        self.rewards.reaching_object.params["std"] = 0.3
        
        # Object_is_lifted bonus reward (Kimi 조언 Step 3)
        # 목적: outcome-based reward - 실제 lift 성공에 강한 보상
        # 예상: 잡기 + 들어올리기 학습 유도 (fine-grained reach reward 보완)
        # [FIX] object_is_lifted -> object_is_lifted_sustained: object 초기/리셋 높이(0.055, 아래
        # RigidObjectCfg 참고)가 이 minimal_height(0.05)보다 이미 높아서, height 단독 조건은
        # 아무것도 안 해도 거의 항상 참이 될 수 있음. EE 근접 조건 + N스텝 연속 유지 조건을 추가해
        # "우연히 튕겨서 5cm 넘긴 것"과 "실제로 쥐고 버틴 것"을 구분함.
        self.rewards.object_is_lifted_bonus = RewTerm(
            func=mdp.object_is_lifted_sustained,
            weight=10.0,
            params={
                "minimal_height": 0.05,  # 15cm(unreachable) -> 5cm (Kimi 지적)
                # [FIX] 9시간/21320iter 실측 결과 Bonus가 사실상 0(1/21320회)이라 조건이 너무 엄격했음.
                # min_steps 10->5(0.2s->0.1s), max_ee_distance 0.06->0.12(2x)로 완화.
                # 주의: max_ee_distance를 너무 풀면 "안 잡았는데 근처를 지나가기만 해도 성공" 판정되는
                # 원래 버그(resting height 0.055 > minimal_height 0.05)의 위험이 다시 커짐 -- 이번에도
                # Bonus가 여전히 거의 안 뜨면 더 풀지 말고, height/proximity/sustain 각각을 분리해서
                # 로그를 찍어보는 게 다음 단계로 맞음.
                "min_steps": 5,
                "max_ee_distance": 0.12,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        
        # wrist_1, wrist_2 twist 방지 penalty (5DoF에서 잡은 후 자세 유지)
        self.rewards.wrist_1_deviation = RewTerm(
            func=joint_deviation_l1,
            weight=-0.1,  # Kimi: -0.5 -> -0.1 (grasp 자세 허용)
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_1_joint"])},
        )
        self.rewards.wrist_2_deviation = RewTerm(
            func=joint_deviation_l1,
            weight=-0.1,  # Kimi: -0.5 -> -0.1
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_2_joint"])},
        )
        
        # red_block의 body name이 "Cube"라서 base의 "Object" 참조 실패함
        # body_names 없이 root만 참조하도록 override
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")

        # Raw RGB image observation group, fed to a learnable CNN inside the policy (end-to-end PPO)
        self.observations.image = ImageCfg()


@configclass
class UR5RobotiqCubeLiftEnvCfg_PLAY(UR5RobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
