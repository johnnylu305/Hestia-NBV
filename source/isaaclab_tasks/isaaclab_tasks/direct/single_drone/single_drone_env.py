# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import pickle
import gymnasium as gym


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

##
# Pre-defined configs
##
from isaacsim.core.utils.prims import delete_prim

from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_multiple_usd_env_id
from isaaclab.utils.math import convert_camera_frame_orientation_convention, euler_xyz_from_quat

from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors import TiledCamera
from .single_drone_env_cfg import MAD3DEnvCfg
import random
import glob
import os
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
import omni
from .mad3d_utils import check_building_collision, rescale_scene, compute_orientation, create_blocks_from_occupancy, OccupancyGrid, get_seen_face, remove_occluded_face, check_free, shift_gt_occs, shift_gt_faces, shift_occs, check_height, compute_weighted_centroid


class MAD3DEnv(DirectRLEnv):
    cfg: MAD3DEnvCfg

    def __init__(self, cfg: MAD3DEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.set_debug_vis(self.cfg.debug_vis)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "face_ratio", # positive reward
                "all_penalty", # negative reward
                "status coverage_ratio", # average coverage ratio
                "status face_coverage_ratio", # average face coverage ratio
            ]
        }

        self.episode_rec = dict()
        self.episode_rec["x"] = [[] for i in range(self.num_envs)]
        self.episode_rec["y"] = [[] for i in range(self.num_envs)]
        self.episode_rec["z"] = [[] for i in range(self.num_envs)]
        self.episode_rec["pitch"] = [[] for i in range(self.num_envs)]
        self.episode_rec["yaw"] = [[] for i in range(self.num_envs)]

    def _setup_scene(self):
        
        self.cfg.num_envs = self.num_envs

        # (n_env, xyz)
        self._actions = torch.zeros(self.cfg.num_envs, self.cfg.num_actions, device=self.device)


        # occ set for collision detection
        # a list of set from fill_occ_set.pkl
        self.occs = [set() for i in range(self.cfg.num_envs)]

        # ground truth occ grid for coverage ratio
        # a list of grid from hollow_occ.npy
        self.gt_occs = torch.zeros((self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size), device=self.device)


        # ground truth for face coverage ratio
        # a list of face grid from faces.npy
        # [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.gt_faces = torch.zeros((self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 6), device=self.device)


        # collision or not according to the fill_occ_set.pkl
        self.col = [False for i in range(self.cfg.num_envs)]
        # is free or not according to the probability occupancy grid
        self.not_free = [False for i in range(self.cfg.num_envs)]
        # exceed height limit or not
        self.not_height = [False for i in range(self.cfg.num_envs)]


        # x, y, z
        self.robot_pos = torch.zeros((self.cfg.num_envs, 3), device=self.device)
        # non-nearest neighbor xyz (truth xyz)
        self.real_xyz = torch.zeros((self.cfg.num_envs, 3), device=self.device)
        # pitch, yaw
        self.robot_ori = torch.zeros((self.cfg.num_envs, 2), device=self.device)


        # obv
        
        # robot pose across time
        # N,T,xyz + pitch + yaw
        self.obv_pose_history = torch.zeros(self.cfg.num_envs, self.cfg.total_img, 5, device=self.device)

        # current step for each environment
        # it should start from 0 instead of -1 because we need one obv for initial action
        self.env_step = torch.zeros(self.cfg.num_envs,)
        
        # current episode for each environment
        self.env_episode = np.ones(self.cfg.num_envs).astype(np.int32) * -1
        
        # rgb image for observation
        # N,T,H,W,3
        # T is 1
        self.obv_imgs = torch.zeros((self.cfg.num_envs, 1, self.cfg.camera_h, self.cfg.camera_w, 3), device=self.device)
                
        # probability occupancy grid and its coordinate for observation
        # N, x_size, y_size, z_size, label+xyz
        self.obv_occ = torch.ones(self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 4, device=self.device)*0.5
        # generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        y_coords = torch.linspace(-self.cfg.env_size/2.0, self.cfg.env_size/2.0, self.cfg.grid_size, device=self.device)
        z_coords = torch.linspace(0.0, self.cfg.env_size, self.cfg.grid_size, device=self.device)
        # create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1) / self.cfg.env_size


        # face grid for observation
        # N, x_size, y_size, z_size, 6 
        self.obv_face = torch.zeros(self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size, 6, device=self.device)

        # coverage ratio
        self.coverage_ratio = torch.zeros((self.cfg.num_envs, 1),  device=self.device)

        # previous face ratio
        self.last_face_ratio = torch.zeros(self.cfg.num_envs, device=self.device).reshape(-1, 1)
        # face ratio
        self.face_ratio = torch.zeros(self.cfg.num_envs,  device=self.device).reshape(-1, 1)

 
        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # make sure every scene has floor
        rescale_scene(scene_prim_root="/World/ground/terrain/Environment", max_len=13e4)
 
        # prevent mirror
        self.scene.clone_environments(copy_from_source=True)

        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # sensor
        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
 
        # occupancy grid
        grid_size = (self.cfg.num_envs, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size)
        self.grid = OccupancyGrid(self.cfg.env_size, grid_size, self.cfg.decrement, self.cfg.increment, 
                                  self.cfg.max_log_odds, self.cfg.min_log_odds, self.device)


        ### scene
        # load paths for scenes
        # (usd_path, hollow_occ_path, fill_occ_set_path, faces_path, usdfilecfg)
        self.scene_paths = []
        train_file = os.path.join(self.cfg.data_root, "preprocess", "train.txt")
        
        """
        self.files = []
        scenes_path = []
        for batch_num in range(1, 7):  # Range goes from 1 to 6 inclusive
            # Generate the path pattern for the glob function
            path_pattern = os.path.join(self.cfg.data_root, "preprocess", f'BATCH_{batch_num}', '**', '*[!_non_metric].usd')
            # Use glob to find all .usd files (excluding those ending with _non_metric.usd) and add to the list
            scenes_path.extend(sorted(glob.glob(path_pattern, recursive=True)))
        for s in scenes_path:
            d = os.path.dirname(s)
            self.files.append(f'{os.path.join(d, "hollow_occ.npy")} 0\n')
        """

        with open(train_file, "r") as file:
            for line in file:
            #for line in self.files:
                # Strip newline and split by space
                line = line.strip()
                if line:  # Ignore empty lines
                    *path_parts, _ = line.rsplit(maxsplit=1)  # Split by spaces, last part is the value
                    hollow_occ_path = " ".join(path_parts)  # Recombine the path parts with spaces

                    # Check if the path is a hollow_occ.npy
                    if hollow_occ_path.endswith("hollow_occ.npy"):
                        directory = os.path.dirname(hollow_occ_path)  # Extract directory from the path

                        # Search for the corresponding `.usd` file
                        usd_files = glob.glob(os.path.join(directory, "*.usd"))
                        usd_files = [file for file in usd_files if "non_metric" not in file]

                        # Search for `faces.npy` and `fill_occ_set.pkl` in the same directory
                        faces_path = os.path.join(directory, "faces.npy")
                        fill_occ_set_path = os.path.join(directory, "fill_occ_set.pkl")

                        # Ensure the required files exist
                        if usd_files and os.path.exists(faces_path) and os.path.exists(fill_occ_set_path):
                            self.scene_paths.append((usd_files[0], hollow_occ_path, fill_occ_set_path, faces_path, UsdFileCfg(usd_path=usd_files[0])))
        
        # object translation vector
        if self.cfg.random_trans_obj:     
            # random translation
            tx = torch.randint(self.cfg.trans_obj_x[0], self.cfg.trans_obj_x[1] + 1, (self.cfg.num_envs, 1))
            ty = torch.randint(self.cfg.trans_obj_y[0], self.cfg.trans_obj_y[1] + 1, (self.cfg.num_envs, 1))
            tz = torch.zeros((self.cfg.num_envs, 1), dtype=torch.int)
            self.txyz = torch.cat((tx, ty, tz), dim=1).to(self.device)
        else:
            # no translation
            self.txyz = torch.zeros((self.cfg.num_envs, 3), dtype=torch.int, device=self.device)
        
        # sample n objects from m objects
        if self.cfg.random_sample_obj:
            indices = torch.randint(0, len(self.scene_paths), (self.cfg.num_envs,))
            self.sampled_scene_lists = [self.scene_paths[i] for i in indices]
        else:
            self.sampled_scene_lists = self.scene_paths[:self.cfg.num_envs]

        # create scenes
        spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                       env_ids=torch.arange(self.cfg.num_envs), my_asset_list=[asset[4] for asset in self.sampled_scene_lists], translation=self.txyz)
 
        # load labels
        for i, scene in enumerate(self.sampled_scene_lists):            
            _, hollow_occ_path, fill_occ_path, faces_path, _ = scene
    
            # load hollow_occ.npy for coverage ratio computation
            self.gt_occs[i] = torch.tensor(np.load(hollow_occ_path)).to(self.device)

            # load fill_occ_set.pkl for collision detection
            with open(fill_occ_path, 'rb') as file:
                self.occs[i] = pickle.load(file)

            # load faces.npy for face coverage ratio computation
            self.gt_faces[i] = torch.tensor(np.load(faces_path)).to(self.device)

            # shift
            self.occs[i] = shift_occs(self.occs[i], self.txyz[i], self.cfg.grid_size, self.cfg.env_size)
            self.gt_occs[i] = shift_gt_occs(self.gt_occs[i], self.txyz[i], self.cfg.grid_size, self.cfg.env_size)
            self.gt_faces[i] = shift_gt_faces(self.gt_faces[i], self.txyz[i], self.cfg.grid_size, self.cfg.env_size)

        # current height limit
        self.h_limit = (torch.ones(self.cfg.num_envs).to(self.device)*10*self.cfg.grid_size/self.cfg.env_size).int()
        
        self.lookatxyz = torch.zeros((self.cfg.num_envs, 3))

        # settings for the occupancy grid
        self.org_x, self.org_y = self.cfg.env_size/2., self.cfg.env_size/2.
        self.org_z = 0
        self.cell_size = self.cfg.env_size/self.cfg.grid_size # meters per cell
        self.slice_height = self.cfg.env_size/self.cfg.grid_size  # height of each slice in meters



    def _pre_physics_step(self, actions: torch.Tensor):
        self.env_step += 1

        # action
        # N, 9 (nearnest free xyz, look at xyz, real xyz)
        self._actions = actions.clone()
        self._xyz = self._actions[:, :3]
        self._xyz = (self._xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)

        if self.cfg.used_nearest:
            self.real_xyz = self._actions[:, 6:9]
            self.real_xyz = (self.real_xyz + torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)

        # 0~1
        self.lookatxyz = self._actions[:, 3:6]
        # -1~1
        self.lookatxyz = self.lookatxyz*2-1
        # to real-world xyz
        self.lookatxyz = (self.lookatxyz+torch.tensor([0., 0., 1.]).to(self.device)) * torch.tensor([self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3, self.cfg.env_size/2.0-1e-3]).to(self.device)
        # compute yaw and pitch
        dxyz = self.lookatxyz - self._xyz + 1e-6
        # calculate yaw using torch functions
        # -pi~pi
        self._yaw = torch.atan2(dxyz[:, 1], dxyz[:, 0])
        # calculate pitch using torch functions
        # -pi/2~pi/2
        self._pitch = torch.atan2(dxyz[:, 2], torch.sqrt(dxyz[:, 0]**2 + dxyz[:, 1]**2))
        # to positive: downward, negative: upward
        self._pitch *= -1
        # normalize pitch as specified
        # -pi/3~pi/2 which is -60-90
        self._pitch = torch.clamp(self._pitch, min=-torch.pi/3, max=torch.pi/2)

        self.last_face_ratio = (torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))).reshape(-1, 1)


    def _apply_action(self):
        env_ids = torch.arange(self.num_envs).to(self.device)

        target_position = self._xyz
        pitch_radians = self._pitch
        yaw = self._yaw
        # roll, pitch, yaw
        target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(yaw.shape[0],1), torch.zeros(yaw.shape[0],1), yaw.cpu().unsqueeze(1)],dim=1).numpy(), degrees=False)
        target_orientation = torch.from_numpy(target_orientation)
        
        rotation_matrix = rot_utils.quats_to_rot_matrices(target_orientation)
        
        # record robot pos and pitch, yaw
        self.robot_pos = target_position + self._terrain.env_origins
        self.robot_ori[:, 0] = self._pitch
        self.robot_ori[:, 1] = self._yaw

        # setup robot position
        root_state = torch.ones((self.num_envs, 13)).to(self.device) * 0
        root_state[:, :3] = target_position
        root_state[:,3:7] = target_orientation
        self.target_orientation = target_orientation
        self.target_position = target_position
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        
        # setup camera position
        drone_euler = torch.cat([torch.zeros(yaw.shape[0],1).to(self.device), pitch_radians.unsqueeze(1), yaw.unsqueeze(1)], dim=1).cpu()
        pitch_quat = torch.from_numpy(rot_utils.euler_angles_to_quats(drone_euler, degrees=False)).float()
        orientation_camera = convert_camera_frame_orientation_convention(pitch_quat, origin="world", target="ros")
        x_new = root_state[:, 0] + self.cfg.camera_offset[0] * torch.cos(self._yaw) - self.cfg.camera_offset[1] * torch.sin(self._yaw)
        y_new = root_state[:, 1] + self.cfg.camera_offset[0] * torch.sin(self._yaw) + self.cfg.camera_offset[1] * torch.cos(self._yaw)
        z_new = root_state[:, 2] + self.cfg.camera_offset[2]
 
        new_positions = torch.stack([x_new, y_new, z_new], dim=1)

        self._camera.set_world_poses(new_positions, orientation_camera)

 
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_states = self.cfg.num_states

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Dict()
        # normalize input
        # N, 6 (xyz, yaw pitch, height)
        self.single_observation_space["policy"]["pose_step"] = gym.spaces.Box(low=-1., high=1., shape=(6,))
        # N, 1, H, W
        # TODO: isaaclab2.1 requires -1~1 instead of 0~1
        self.single_observation_space["policy"]["img"] = gym.spaces.Box(low=-1., high=1., shape=(1, self.cfg.camera_h, self.cfg.camera_w))
        # N, 10 (occ + coords + face occ), x, y, z
        self.single_observation_space["policy"]["occ"] = gym.spaces.Box(low=-1., high=1., shape=(10, self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size))
        # N, 1
        self.single_observation_space["policy"]["env_size"] = gym.spaces.Box(low=-100., high=100., shape=(1,))
        # N, 3
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_actions,))
        
        self.single_observation_space["policy"]["aux_center"] = gym.spaces.Box(low=0., high=1., shape=(3,))


        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

    def update_observations(self, env_ids=None):
        env_ids = [i for i in range(self.cfg.num_envs)] if env_ids is None else env_ids

        # get images
        depth_image = self._camera.data.output["distance_to_image_plane"].clone()[:, :, :, 0]
        rgb_image = self._camera.data.output["rgb"].clone()
        
        # obv image is 0~1 
        for i in env_ids:
            self.obv_imgs[i][0] = rgb_image[i][:, :, :3] / 255.

        # collision detection
        for i in env_ids:
            # collision detection
            self.col[i] = check_building_collision(self.occs, self.robot_pos[i].clone(), i, self.org_x, self.org_y, self.org_z, 
                                                   self.cell_size, self.slice_height, self._terrain.env_origins)
            # safe 
            if self.cfg.used_nearest:
                self.not_free[i] = check_free(self.obv_occ[i, :, :, :, 0], self.real_xyz[i].clone(), i, self.org_x, self.org_y, self.org_z, 
                                              self.cell_size, self.slice_height, self._terrain.env_origins, False)
                # height limit
                self.not_height[i] = check_height(self.h_limit[i], self.real_xyz[i].clone(), i, self.org_z, 
                                                  self.cell_size, self.slice_height, self._terrain.env_origins, False)
            else:
                self.not_free[i] = check_free(self.obv_occ[i, :, :, :, 0], self.robot_pos[i].clone(), i, self.org_x, self.org_y, self.org_z, 
                                              self.cell_size, self.slice_height, self._terrain.env_origins)
                # not height limit for non-nearest setting

        # robot pose
        for i in env_ids:
            self.obv_pose_history[i, self.env_step[i].int(), :3] = (self.robot_pos[i] - self._terrain.env_origins[i])/self.cfg.env_size
            self.obv_pose_history[i, self.env_step[i].int(), 3:] = self.robot_ori[i]/3.15

        # get camera intrinsic and extrinsic matrix
        intrinsic_matrix = self._camera.data.intrinsic_matrices.clone()
        camera_pos = self._camera.data.pos_w.clone()
        camera_quat = self._camera.data.quat_w_ros.clone()
        
        # prevent inf
        depth_image = torch.clamp(depth_image, 0, self.cfg.env_size*2)
        # make log odd occupancy grid
        points_3d_cam = unproject_depth(depth_image, intrinsic_matrix)
        points_3d_world = transform_points(points_3d_cam, camera_pos, camera_quat)
        
        for i in env_ids:
            mask_x = (points_3d_world[i,:, 0]-self._terrain.env_origins[i][0]).abs() < self.cfg.env_size/2 - 1e-3
            mask_y = (points_3d_world[i,:, 1]-self._terrain.env_origins[i][1]).abs() < self.cfg.env_size/2 - 1e-3
            mask_z = (points_3d_world[i,:, 2] < self.cfg.env_size - 1e-3) & (points_3d_world[i,:, 2] >=0) 
            # combine masks to keep points where both xyz are within the range
            mask = mask_x & mask_y & mask_z
            
            offset = torch.tensor([self.org_x, self.org_y, self.org_z]).to(self.device)
            # update grid
            if not self.col[i] and points_3d_world[i][mask].shape[0] > 0:
                ratio = 1/self.cell_size
                self.grid.trace_path_and_update(i, 
                                                torch.floor(self._camera.data.pos_w[i]-self._terrain.env_origins[i]+offset)*ratio, 
                                                torch.floor((points_3d_world[i]-self._terrain.env_origins[i]+offset))*ratio)
                self.grid.update_log_odds(i,
                                          torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio),
                                          occupied=True)
                self.obv_face[i] = torch.logical_or(self.obv_face[i], 
                                                    get_seen_face(torch.unique(torch.floor((points_3d_world[i][mask]-self._terrain.env_origins[i]+offset)*ratio).int(), dim=0), 
                                                    torch.floor((self._camera.data.pos_w[i]-self._terrain.env_origins[i]+offset)*ratio), self.cfg.grid_size, self.device))
 
        if points_3d_world.size()[0] > 0:
            # get probability grid
            self.probability_grid = self.grid.log_odds_to_prob(self.grid.grid)
            # N, x_size, y_size, z_size
            self.obv_occ[:, :, :, :, 0] = self.probability_grid.clone()
            
            # visualization
            if self.cfg.vis_occ:
                # 0: free, 1: unknown, 2: occupied
                # free: <=0.3, unknown: 0.3-0.6, occupied: >=0.6
                vis_occ = torch.where(self.probability_grid <= 0.3, 0, torch.where(self.probability_grid <= 0.6, 1, 2))
                vis_occ = vis_occ.cpu().numpy()
                for j in range(self.obv_occ.shape[0]):
                    for i in range(self.cfg.grid_size):
                        # vis occ
                        create_blocks_from_occupancy(j, self._terrain.env_origins[j].cpu().numpy(), 
                                                     vis_occ[j, :, :, i], self.cell_size, i*self.slice_height, i, self.cfg.env_size, 2, 60)


        hard_occ = torch.where(self.obv_occ[:, :, :, :, 0] >= 0.6, 1, 0)

        # remove the occluded face by checking current occ grid
        # the face amount may still larger than gt because current occ grid is incomplete
        #self.obv_face = remove_occluded_face(self.cfg.grid_size, hard_occ, self.obv_face, self.device)

        # compute coverage ratio
        hard_occ = hard_occ[:, :, :, 1:]
        num_match_occ = torch.sum(torch.logical_and(hard_occ, self.gt_occs[:, :, :, 1:]), dim=(1, 2, 3))
        total_occ = torch.sum(self.gt_occs[:, :, :, 1:], dim=(1, 2, 3))
        self.coverage_ratio = (num_match_occ/total_occ).reshape(-1, 1)
        # compute face coverage ratio
        self.face_ratio = (torch.sum(torch.logical_and(self.obv_face[:, :, :, 1:, :], self.gt_faces[:, :, :, 1:, :]),(1,2,3,4))/torch.sum(self.gt_faces[:, :, :, 1:, :], (1,2,3,4))).reshape(-1, 1)

        # save images
        if self.cfg.save_img:
            for i in self.cfg.save_env_ids:
                if i >= self.num_envs:
                    break
                if self.env_episode[i]%self.cfg.save_img_freq != 0:
                    continue
                root_path = os.path.join(self.cfg.camera_folder, f'{self.env_episode[i]}')
                os.makedirs(root_path, exist_ok=True)
                x, y, z = self.obv_pose_history[i, self.env_step[i].int(), :3] * self.cfg.env_size
                cv = self.coverage_ratio[i, 0]
                lx, ly, lz = self.lookatxyz[i]
                base_name = f'{self.env_step[i].long()}_{x:.1f}_{y:.1f}_{z:.1f}_{cv:.3f}_{self.face_ratio[i][0]:.3f}_{self.not_free[i]}_{self.not_height[i]}_{self.real_xyz[i][0]:.3f}_{self.real_xyz[i][1]:.3f}_{self.real_xyz[i][2]:.3f}_{self.h_limit[i]}_{self.robot_ori[i, 0]:.2f}_{lx:.1f}_{ly:.1f}_{lz:.1f}.png'
                plt.imsave(os.path.join(root_path, f'{i}_depth_'+base_name),
                           np.clip(depth_image[i].detach().cpu().numpy(),0,self.cfg.env_size).astype(np.uint8),
                           cmap='gray',
                           vmin=0,
                           vmax=self.cfg.env_size)
                plt.imsave(os.path.join(root_path, f'{i}_rgb_'+base_name),
                           rgb_image[i].detach().cpu().numpy().astype(np.uint8))

    def _get_observations(self) -> dic:
        # pose: N, T, 7 (xyz + pitch yaw + height limit)
        # current pose
        current_pose = self.obv_pose_history[torch.arange(self.cfg.num_envs), self.env_step.int()]
        # reshape and normalize env_step
        #env_step_normalized = (self.env_step.to(self.device) / self.cfg.total_img).unsqueeze(1)
        # expand h_limit to match batch dimensions
        h_limit_expanded = self.h_limit.unsqueeze(1) / self.cfg.grid_size
        # concatenate current pose, normalized step, and h_limit
        pose_step = torch.cat([current_pose.reshape(self.cfg.num_envs, -1), h_limit_expanded], dim=1)

        # occ: N, grid_size, grid_size, grid_size, occ + coordinate + face occ
        occ_face = torch.cat((self.obv_occ, self.obv_face), dim=-1)

        # img: N, H, W, 1
        gray_scale_img = torch.mean(self.obv_imgs[:, 0, :, :, :], (3))

        center = compute_weighted_centroid(occ_face[:, :, :, 1:, 4:], self.gt_faces[:, :, :, 1:, :])
        center /= occ_face.shape[1]
        

        obs = {"pose_step": pose_step,
               "img": gray_scale_img.reshape(-1, 1, self.cfg.camera_h, self.cfg.camera_w),
               "occ": occ_face.permute(0, 4, 1, 2, 3),
               "env_size": torch.ones((self.cfg.num_envs, 1)) * self.cfg.env_size,
               "aux_center": center
               }

        observations = {"policy": obs}
       
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        # collision or not
        col_mask = (torch.tensor(self.col)==False).float().to(self.device).reshape(-1, 1)
        # if collision, no information gain, not free, or exceeding height limit
        # all_mask is False
        all_mask = ((torch.tensor(self.col).bool() | (self.face_ratio==self.last_face_ratio).reshape(-1).bool().cpu() | torch.tensor(self.not_free).bool() | torch.tensor(self.not_height).bool())==False).float().to(self.device).reshape(-1, 1)

        rewards = {
            "face_ratio": (self.face_ratio-self.last_face_ratio) * col_mask * 30,
            "all_penalty": (1.0-all_mask)*-1,
        }

        for k in rewards.keys():
            rewards[k] /= 100.

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0).reshape(-1)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value.squeeze(1)
        self._episode_sums["status coverage_ratio"] = self.coverage_ratio.squeeze()
        self._episode_sums["status face_coverage_ratio"] = self.face_ratio.squeeze()
        for i in range(self.cfg.num_envs):
            self.episode_rec["x"][i].append(self.robot_pos[i, 0].clone()-self._terrain.env_origins[i][0])
            self.episode_rec["y"][i].append(self.robot_pos[i, 1].clone()-self._terrain.env_origins[i][1])
            self.episode_rec["z"][i].append(self.robot_pos[i, 2].clone()-self._terrain.env_origins[i][2])
            self.episode_rec["pitch"][i].append(self.robot_ori[i, 0].clone())
            self.episode_rec["yaw"][i].append(self.robot_ori[i, 1].clone())

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # update observation here because _get_dones is the first call after _apply_action
        self.update_observations()
        time_out = self.episode_length_buf >= self.max_episode_length-1

        # done when exceeding the max steps or reaching goal face coverage ratio
        done = torch.logical_or(self.env_step.to(self.device) >= self.cfg.total_img - 1, self.face_ratio.squeeze() >= self.cfg.goal)

        return done, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)


        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            if "status" not in key:
                extras["Episode Reward/" + key] = self._episode_sums[key] / self.env_step.to(self.device)
                self._episode_sums[key][env_ids] = 0.0
        extras["Episode Reward/status coverage ratio"] = self._episode_sums["status coverage_ratio"].clone()
        self._episode_sums["status coverage_ratio"][env_ids] = 0.0
        extras["Episode Reward/status face coverage ratio"] = self._episode_sums["status face_coverage_ratio"].clone()
        self._episode_sums["status face_coverage_ratio"][env_ids] = 0.0

        extras["train_cus/x"] = self.episode_rec["x"].copy()
        extras["train_cus/y"] = self.episode_rec["y"].copy()
        extras["train_cus/z"] = self.episode_rec["z"].copy()
        extras["train_cus/pitch"] = self.episode_rec["pitch"].copy()
        extras["train_cus/yaw"] = self.episode_rec["yaw"].copy()
        for i in env_ids:
            self.episode_rec["x"][i] = []
            self.episode_rec["y"][i] = []
            self.episode_rec["z"][i] = []
            self.episode_rec["pitch"][i] = []
            self.episode_rec["yaw"][i] = []

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # reset step
        self.env_step[env_ids] = 0
        # reset pose history
        self.obv_pose_history[env_ids] = torch.zeros(self.cfg.total_img, 5).to(self.device)
        # reset imgs obv
        self.obv_imgs[env_ids] = torch.zeros((1, self.cfg.camera_h, self.cfg.camera_w, 3)).to(self.device)
        # reset occ obv
        self.obv_occ[env_ids, :, :, :, 0] = 0.5
        # reset face obv
        self.obv_face[env_ids] = 0
        # reset last face ratio
        self.last_face_ratio[env_ids] = 0
        # reset actions
        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # reset occupancy grid
        self.grid.grid[env_ids] = 0
        
        # clear building
        for env_id in env_ids:
            delete_prim(f'/World/envs/env_{env_id}/Scene')
                                
        # object translation vector
        if self.cfg.random_trans_obj:     
            # random translation
            tx = torch.randint(self.cfg.trans_obj_x[0], self.cfg.trans_obj_x[1] + 1, (len(env_ids), 1))
            ty = torch.randint(self.cfg.trans_obj_y[0], self.cfg.trans_obj_y[1] + 1, (len(env_ids), 1))
            tz = torch.zeros((len(env_ids), 1), dtype=torch.int)
            self.txyz[env_ids] = torch.cat((tx, ty, tz), dim=1).to(self.device)
        
        # sample n objects from m objects
        asset_list = []
        if self.cfg.random_sample_obj:
            indices = torch.randint(0, len(self.scene_paths), (len(env_ids),))
            for i, env_id in enumerate(env_ids):
                self.sampled_scene_lists[env_id] = self.scene_paths[indices[i]]
                asset_list.append(self.sampled_scene_lists[env_id][4])
        else:
            for i, env_id in enumerate(env_ids):
                asset_list.append(self.sampled_scene_lists[env_id][4])


        # create scenes
        spawn_from_multiple_usd_env_id(prim_path_template="/World/envs/env_.*/Scene", 
                                       env_ids=env_ids, my_asset_list=asset_list, translation=self.txyz[env_ids])

        # load labels
        for env_id in env_ids:
            _, hollow_occ_path, fill_occ_path, faces_path, _ = self.sampled_scene_lists[env_id]

            # load hollow_occ.npy for coverage ratio computation
            self.gt_occs[env_id] = torch.tensor(np.load(hollow_occ_path)).to(self.device)

            # load fill_occ_set.pkl for collision detection
            with open(fill_occ_path, 'rb') as file:
                self.occs[env_id] = pickle.load(file)

            # load faces.npy for face coverage ratio computation
            self.gt_faces[env_id] = torch.tensor(np.load(faces_path)).to(self.device)

            # shift
            self.occs[env_id] = shift_occs(self.occs[env_id], self.txyz[env_id], self.cfg.grid_size, self.cfg.env_size)
            self.gt_occs[env_id] = shift_gt_occs(self.gt_occs[env_id], self.txyz[env_id], self.cfg.grid_size, self.cfg.env_size)
            self.gt_faces[env_id] = shift_gt_faces(self.gt_faces[env_id], self.txyz[env_id], self.cfg.grid_size, self.cfg.env_size)


        if not self.cfg.random_initial_view:
            target_position = torch.tensor(self.cfg.default_init_pos).unsqueeze(0).to(self.device)

            # get look at vector
            yaw, pitch = compute_orientation(self.cfg.default_init_pos)
            # assume pitch is 0 for this version
            target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
            target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
            orientation_camera = convert_camera_frame_orientation_convention(target_orientation.float(), origin="world", target="ros") 

            # set robot position and orientation
            default_root_state[:,:3] = target_position
            default_root_state[:,3:7] = target_orientation        
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self.default_root_state = default_root_state

            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            
            # get camera position
            x_new = default_root_state[:, 0] + self.cfg.camera_offset[0] * np.cos(yaw) - self.cfg.camera_offset[1] * np.sin(yaw)
            y_new = default_root_state[:, 1] + self.cfg.camera_offset[0] * np.sin(yaw) + self.cfg.camera_offset[1] * np.cos(yaw)
            z_new = default_root_state[:, 2] + self.cfg.camera_offset[2]

            new_positions = torch.stack([x_new, y_new, z_new], dim=1)
        
            orientation_camera = orientation_camera.repeat(len(env_ids), 1)

            self._camera.set_world_poses(new_positions, orientation_camera, env_ids)

            # record robot pos and pitch, yaw
            self.robot_pos[env_ids] = target_position + self._terrain.env_origins[env_ids]
            self.real_xyz[env_ids] = target_position
            self.robot_ori[env_ids, 0] = 0
            self.robot_ori[env_ids, 1] = yaw
        else:
            # 26 neighbors 
            random_position_tensor = torch.empty((0, 3), dtype=torch.int)
            neighbor_offsets = [ (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                 (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                                 (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                                 (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                                 (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                                 (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

            for k, env_id in enumerate(env_ids):
                random_position = None
                vox_pos = None
                flag = True
                # find free position
                while flag:
                    # randomly sample a point within the range
                    x = random.randint(-self.cfg.env_size//2+1, self.cfg.env_size//2-1)
                    y = random.randint(-self.cfg.env_size//2+1, self.cfg.env_size//2-1)
                    z = random.randint(1, int(self.cfg.env_size/2*0.6))
                    pos = (x, y, z)

                    target_x, target_y, target_z = self.txyz[env_id].cpu().numpy()
                    # calculate the distance from the sampled point to the target point
                    distance = np.sqrt((x - target_x)**2 + (y - target_y)**2 + (z - target_z)**2)

                    # check if the distance is within the allowed range
                    if distance < 5.5:
                        continue  # if not, sample again

                    # get voxel coordinate
                    x += self.org_x
                    y += self.org_y
                    z += self.org_z
                    # to voxel id
                    x = np.floor(x/self.cell_size).astype(np.int32)
                    y = np.floor(y/self.cell_size).astype(np.int32)
                    z = np.floor(z/self.slice_height).astype(np.int32)
                    vox_pos = (x, y, z)
                
                    # check if the point and its neighbors are all free voxels
                    if vox_pos not in self.occs[env_id]:
                        random_position = pos
                        flag = False 

                # from initial h to max h
                # in voxel coordinate
                self.h_limit[env_id] = random.randint(int(random_position[2] * self.cfg.grid_size/self.cfg.env_size), 10)
                if random.random() <= 0.9: 
                    self.h_limit[env_id] = random.randint(9, 10)

                # set robot position and orientation
                random_position_tensor = torch.tensor(random_position).unsqueeze(0)
                yaw, pitch = compute_orientation(random_position, self.txyz[env_id].cpu().numpy())
                target_orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, yaw]), degrees=False)
                target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)

                default_root_state[k:k+1, :3] =  random_position_tensor
                default_root_state[k:k+1, 3:7] = target_orientation        
                default_root_state[k:k+1, :3] += self._terrain.env_origins[env_id]
                self.default_root_state = default_root_state
                
                # set camera position
                target_orientation = rot_utils.euler_angles_to_quats(np.array([0, -pitch, yaw]), degrees=False)
                target_orientation = torch.from_numpy(target_orientation).unsqueeze(0)
                orientation_camera = convert_camera_frame_orientation_convention(target_orientation.float(), origin="world", target="ros")
    
                x_new = default_root_state[k:k+1, 0] + self.cfg.camera_offset[0] * np.cos(yaw) - self.cfg.camera_offset[1] * np.sin(yaw)
                y_new = default_root_state[k:k+1, 1] + self.cfg.camera_offset[0] * np.sin(yaw) + self.cfg.camera_offset[1] * np.cos(yaw)
                z_new = default_root_state[k:k+1, 2] + self.cfg.camera_offset[2]
                new_positions = torch.stack([x_new, y_new, z_new], dim=1)

                self._camera.set_world_poses(new_positions, orientation_camera, [env_id])
                self.robot_pos[env_id] = random_position_tensor.to(self.device) + self._terrain.env_origins[env_id]
                self.real_xyz[env_id] = random_position_tensor.to(self.device)
                self.robot_ori[env_id, 0] = -pitch
                self.robot_ori[env_id, 1] = yaw
            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env_episode[env_ids.cpu().numpy()] += 1
        # still need this to stablize the initial viewpoint
        for i in range(3):
            self.sim.step()
            self.scene.update(dt=0)

        self.update_observations(env_ids)
