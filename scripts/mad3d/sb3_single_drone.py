# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from sb3_ppo_cus import PPO_Cus


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg


# RewardLoggingCallback definition
class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)   
        self.freq = 200 

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        current_step = self.num_timesteps

        aggregated_rewards = {}
        aggregated_train_cus = {}

        # Loop through each environment's info and aggregate data
        for info in infos:
            episode_info = info.get('episode', None)
            if episode_info:
                for key, value in episode_info.items():
                    # Aggregate "Episode Reward" values
                    if "Episode Reward" in key:
                        # Extract the specific reward type, e.g., "Reward"
                        reward_type = key.split('/')[1]
                        if reward_type not in aggregated_rewards:
                            aggregated_rewards[reward_type] = []
                        # Append the value to the list of the reward type
                        aggregated_rewards[reward_type].append(value.detach().cpu().item())
                    
                    # Aggregate "train_cus" values
                    elif "train_cus" in key:
                        # Extract the specific train_cus type, e.g., "train_cus_metric"
                        train_cus_type = key.split('/')[1]
                        if train_cus_type not in aggregated_train_cus:
                            aggregated_train_cus[train_cus_type] = []
                        # Append each value with its corresponding adjusted step
                        for j, v in enumerate(value):
                            aggregated_train_cus[train_cus_type].append(
                                (v, current_step - (len(value) - j - 1) * len(infos))
                            )
        # Log the aggregated "Episode Reward" information
        for reward_type, values in aggregated_rewards.items():
            mean_reward = sum(values) / len(values)
            self.writer.add_scalar(f"Aggregated Episode Reward/{reward_type}", mean_reward, current_step)

        # Log the aggregated "train_cus" information
        for train_cus_type, values in aggregated_train_cus.items():
            # Log each value with its adjusted step
            for v, step in values:
                self.writer.add_scalar(f"Aggregated train_cus/{train_cus_type}", v, step)

        if self.num_timesteps % self.freq == 0:
            try:
                x_s, y_s, z_s, pitch_s, yaw_s = torch.exp(self.model.policy.log_std).detach().cpu().numpy()
                self.writer.add_scalar("train_cus/x_std", x_s, current_step)
                self.writer.add_scalar("train_cus/y_std", y_s, current_step)
                self.writer.add_scalar("train_cus/z_std", z_s, current_step)
                self.writer.add_scalar("train_cus/pitch_std", pitch_s, current_step)
                self.writer.add_scalar("train_cus/yaw_std", yaw_s, current_step)
            except:
                x_s, y_s, z_s = torch.exp(self.model.policy.log_std).detach().cpu().numpy()
                self.writer.add_scalar("train_cus/x_std", x_s, current_step)
                self.writer.add_scalar("train_cus/y_std", y_s, current_step)
                self.writer.add_scalar("train_cus/z_std", z_s, current_step)
        return True
        
    def _on_training_end(self) -> None:
        self.writer.close()


def piecewise_step_lr_schedule(lr_list, step_list, total_timesteps):
    # Basic checks for consistency
    if len(lr_list) != len(step_list):
        raise ValueError("lr_list and step_list must have the same length.")
    if sorted(step_list) != step_list:
        raise ValueError("step_list must be sorted in ascending order.")

    def func(progress_remaining):
        # Calculate the current training step based on progress_remaining
        current_step = int((1 - progress_remaining) * total_timesteps)

        for i, boundary in enumerate(step_list):
            if current_step <= boundary:
                return lr_list[i]
        # If current_step >= the last boundary, use the last LR
        return lr_list[-1]

    return func


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg["seed"]

    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    # TODO: disable fast variant for extra logs
    env = Sb3VecEnvWrapper(env, fast_variant=False)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # lr schedule
    lr = 3e-4
    agent_cfg["learning_rate"] = piecewise_step_lr_schedule(lr_list=[lr, lr/2., lr/4., lr/8., lr/16.], step_list=[1500000, 2000000, 2500000, 3000000, 6000000], total_timesteps=n_timesteps)
    #agent_cfg["learning_rate"] = piecewise_step_lr_schedule(lr_list=[lr, lr/2., lr/4., lr/8.], step_list=[2000000, 2500000, 3000000, 4000000], total_timesteps=n_timesteps)

    agent = PPO_Cus(policy_arch, env, verbose=1, **agent_cfg)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)

    # Instantiate the RewardLoggingCallback
    reward_logging_callback = RewardLoggingCallback(log_dir, verbose=2)

    # Train the agent with both the checkpoint and reward logging callbacks
    agent.learn(total_timesteps=n_timesteps, callback=[checkpoint_callback, reward_logging_callback])

    agent.save(os.path.join(log_dir, "model"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
