import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "218622274083",
        "wrist_2": "218622271526",
    }
    TARGET_POSE = np.array(
        [0.45744842827647436,0.0761972695608443,0.07560009685145638,3.136626282737411,-0.02836925376945154,1.596492563779626]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    RANDOM_RESET = False #Turn to true after basic task is finished
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
