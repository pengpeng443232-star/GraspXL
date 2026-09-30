"""Configuration for the lightweight GraspXL JDF environment."""
from isaaclab.utils import configclass
from experiments.lightweight_validation_rl.env_PPO_cfg_multiobject import ShovelMultiObjectEnvCfg

@configclass
class GraspXLJDFEnvCfg(ShovelMultiObjectEnvCfg):
    observation_space: int = 282
    require_contact_for_lift: bool = True
    goal_height_offset: float = 0.10
    jdf_surface_samples: int = 256
    jdf_distance_temperature: float = 0.025
    jdf_accuracy_rew_scale: float = 2.0
    stable_contact_rew_scale: float = 2.0
    goal_distance_rew_scale: float = 20.0
    non_grasp_distance_rew_scale: float = 2.0
    heading_rew_scale: float = 1.0
    wrist_rew_scale: float = 1.0
    midpoint_rew_scale: float = 20.0
    positive_contact_rew_scale: float = 2.0
    negative_contact_rew_scale: float = 2.0
    positive_force_rew_scale: float = 0.5
    negative_force_rew_scale: float = 0.5
    hand_velocity_rew_scale: float = 0.01
    object_velocity_rew_scale: float = 0.05
    guidance_position_gain: float = 0.5
    guidance_rotation_gain: float = 0.1
    force_clip: float = 10.0
    # 1: fixed cup/JDF accuracy, 2: dynamic cup stable grasp,
    # 3: dynamic cup grasp + 10 cm lift.
    curriculum_stage: int = 3
