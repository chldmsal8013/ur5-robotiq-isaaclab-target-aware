from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class UR5RobotiqStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env   = 24
    max_iterations      = 30000
    save_interval       = 500
    experiment_name     = "ur5_robotiq_pick"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std      = 1.0,
        actor_hidden_dims   = [256, 128, 64],
        critic_hidden_dims  = [256, 128, 64],
        activation          = "elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef         = 1.0,
        use_clipped_value_loss  = True,
        clip_param              = 0.2,
        entropy_coef            = 0.005,
        num_learning_epochs     = 8,
        num_mini_batches        = 4,
        learning_rate           = 1.0e-4,
        schedule                = "fixed",
        gamma                   = 0.995,  # 0.99 → 0.995: 0.995^600 ≈ 0.05 (vs 0.99^600 ≈ 0.0025)
        lam                     = 0.95,
        desired_kl              = 0.05,
        max_grad_norm           = 1.0,
    )
