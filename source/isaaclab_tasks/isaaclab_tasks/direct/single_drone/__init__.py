# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .single_drone_env import MAD3DEnv, MAD3DEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="MAD3D-v0",
    entry_point="isaaclab_tasks.direct.single_drone:MAD3DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MAD3DEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
