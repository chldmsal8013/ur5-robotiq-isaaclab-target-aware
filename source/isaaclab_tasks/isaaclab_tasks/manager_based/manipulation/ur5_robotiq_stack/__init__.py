"""
Isaac-Stack-UR5-Robotiq-v0
UR5 + Robotiq 2F-85 cube stacking with overhead RGB camera (64x64).
"""

import gymnasium as gym

from . import agents
from .ur5_robotiq_stack_env_cfg import UR5RobotiqStackEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-Stack-UR5-Robotiq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": UR5RobotiqStackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5RobotiqStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Eval variant (fewer envs, useful for visualization)
gym.register(
    id="Isaac-Stack-UR5-Robotiq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": UR5RobotiqStackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5RobotiqStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
