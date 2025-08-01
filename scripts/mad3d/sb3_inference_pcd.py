import argparse
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("--input", type=str, required=True, help="Path to the txt list.")
parser.add_argument("--vis", action="store_true", help="If set, visualize occupancy grid.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--trans", type=int, nargs=3, default=[0, 0, 0], help="Translation vector [x, y, z].")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import omni
import isaaclab.sim as sim_utils
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
import sys
import csv
import time
import open3d as o3d
#from pytorch3d.loss import chamfer_distance
from PIL import Image
from isaacsim.core import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import transform_points, unproject_depth
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from isaaclab.utils.math import convert_camera_frame_orientation_convention, euler_xyz_from_quat
from isaacsim.core.utils.prims import delete_prim


from stable_baselines3.common.vec_env import VecNormalize
sys.path.append("/home/dsr/Documents/mad3d/isaac-sim-4.2.0/home/IsaacLab")
from sb3_ppo_cus import PPO_Cus


# Env
NUM_STEPS = 30
NUM_ENVS = 1
GRID_SIZE = 20
ENV_SIZE = 20
TRANS = args_cli.trans
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_OFFSET = [0, 0, 0]
# camera initial position
INITIAL_POSE = [0, -9, 6] #[10, 10, 2]
# Sensor"
CAMERA_HEIGHT = 900
CAMERA_WIDTH = 900
TARGET_HEIGHT = 300
TARGET_WIDTH = 300


def _bresenhamline_nslope(slope, device):
    scale = torch.amax(torch.abs(slope), dim=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = torch.ones(1, dtype=torch.long).to(device)
    normalizedslope = slope / scale
    normalizedslope[zeroslope] = torch.zeros(slope[0].shape).to(device)
    return normalizedslope

def _bresenhamlines(start, end, max_iter, device):
    if max_iter == -1:
        max_iter = torch.amax(torch.amax(torch.abs(end - start), dim=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start, device)

    # steps to iterate on
    stepseq = torch.arange(1, max_iter + 1).to(device)
    stepmat = stepseq.repeat(dim, 1) #np.tile(stepseq, (dim, 1)).T
    stepmat = stepmat.T

    # some hacks for broadcasting properly
    bline = start[:, None, :] + nslope[:, None, :] * stepmat

    # Approximate to nearest int
    bline_points = torch.round(bline).to(start.dtype)

    return bline_points

def bresenhamline(start, end, max_iter=5, device='cpu'):
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter, device).reshape(-1, start.shape[-1])

class OccupancyGrid:
    def __init__(self, env_size, grid_size, device="cpu"):
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.grid_size = grid_size
        self.resolution = env_size / grid_size[1]
        self.device = device

    def update_log_odds(self, i, indices, occupied=True):
        indices = indices.long()
        if occupied:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] += 1.0
        else:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] -= 0.01
        self.grid.clamp_(min=-10.0, max=10.0)

    def log_odds_to_prob(self):
        return 1 / (1 + torch.exp(-self.grid))

    def trace_path_and_update(self, i, camera_position, points):
        camera_position = torch.tensor(camera_position).to(DEVICE)
        end_pts = (camera_position).unsqueeze(0).long().to(DEVICE)
        start_pts = (points.to(DEVICE)).long()
        #start_pts = start_pts.repeat(end_pts.shape[0],1)
        bresenham_path = bresenhamline(start_pts, end_pts, max_iter=-1, device=DEVICE)
        mask = (bresenham_path[:,0]>=0) & (bresenham_path[:,1]>=0) & (bresenham_path[:,2]>=0) &\
            (bresenham_path[:,0]<self.grid_size[1]) & (bresenham_path[:,1]<self.grid_size[1]) & (bresenham_path[:,2]<self.grid_size[1])
        if bresenham_path[mask] is not None:
            self.update_log_odds(i, bresenham_path[mask], occupied=False)

def setup_scene(world, scene_path, index, scene_prim_root="/World/Scene"):
    # floor
    world.scene.add_ground_plane(size=40.0, color=torch.tensor([52.0 / 255.0, 195.0 / 255.0, 235.0 / 255.0]))
    # light
    UsdLux.DomeLight.Define(world.scene.stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(500)
    
    # add usd into the stage
    scene = add_reference_to_stage(usd_path=scene_path, prim_path=scene_prim_root)

    # define the property of the stage
    scene_prim = XFormPrim(prim_path=scene_prim_root, translation=[0, 0, 0])

    # activate the stage
    world.scene.add(scene_prim)

    sim_dt = world.get_physics_dt()

    cameraCfg = CameraCfg(
        prim_path=f"/World/Camera_0",
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), convention="world"),
        update_period=sim_dt,  # Update every physical step
        data_types=["distance_to_image_plane","rgb"],
        spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.8, # in cm default 24, dji 1.38
                #focus_distance=1.0, # in m 
                horizontal_aperture=24., # in mm 
                clipping_range=(0.01, 60.0) # near and far plane in meter
            ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT
    )
 
    scene_entities = {}
    
    scene_entities[f"camera_0"] = Camera(cameraCfg)

    return scene_entities


def create_blocks_from_occupancy(env_id, env_origin, occupancy_grid, cell_size, base_height, z,env_size, target=0, h_off=60):
    stage = omni.usd.get_context().get_stage()
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == target:
                # Calculate position based on cell coordinates
                cube_pos = Gf.Vec3f((x*cell_size)+env_origin[0]-env_size/2, (y*cell_size)+env_origin[1]-env_size/2, base_height+h_off)

                # Define the cube's USD path
                cube_prim_path = f"/World/OccupancyBlocks/Block_{env_id}_{x}_{y}_{z}_{target}"

                # Create a cube primitive or get the existing one
                cube_prim = UsdGeom.Cube.Define(stage, Sdf.Path(cube_prim_path))
                cube_prim.GetPrim().GetAttribute("size").Set(cell_size)

                # Manage the transformation
                xform = UsdGeom.Xformable(cube_prim.GetPrim())
                xform_ops = xform.GetOrderedXformOps()
                if not xform_ops:
                    # If no transform ops exist, add a new translate op
                    xform_op = xform.AddTranslateOp()
                    xform_op.Set(cube_pos)
                else:
                    # If transform ops exist, modify the existing translate op
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(cube_pos)
                            break
                    else:
                        # If no translate op found, add it
                        xform.AddTranslateOp().Set(cube_pos)


def point_coverage_ratio(acc_points, pcd_gt, thres=0.01):
    # Compute pairwise distances between acc_points and pcd_gt
    distances = torch.cdist(acc_points, pcd_gt, p=2)

    # Mark points in pcd_gt within thres as covered
    occupied_mask = (distances <= thres).any(dim=0)

    # Compute the number of covered points
    num_occupied = occupied_mask.sum().item()

    # Compute the coverage ratio
    total_gt_points = pcd_gt.shape[0]
    coverage_ratio = num_occupied / total_gt_points

    return coverage_ratio


# Compute Chamfer Distance
def chamfer_distance(pcd1, pcd2):
    # Compute pairwise distances
    dist1 = torch.cdist(pcd1, pcd2, p=2)  # Pairwise Euclidean distances

    # For each point in pcd1, find the nearest point in pcd2
    min_dist1, _ = torch.min(dist1, dim=1)

    # For each point in pcd2, find the nearest point in pcd1
    min_dist2, _ = torch.min(dist1, dim=0)

    # Average the nearest neighbor distances
    cd = torch.mean(min_dist1) + torch.mean(min_dist2)
    print(torch.mean(min_dist1), torch.mean(min_dist2))
    return cd

def subsample_point_cloud(pcd, num_samples):
    if pcd.shape[0] <= num_samples:
        return pcd  # If already smaller, return as is
    indices = torch.randperm(pcd.shape[0])[:num_samples]  # Randomly sample indices
    return pcd[indices]

def farthest_point_sampling(pcd, num_samples):
    n, _ = pcd.shape
    if n <= num_samples:
        return pcd  # If already smaller, return as is

    # Initialize an empty set of sampled points
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=pcd.device)
    distances = torch.full((n,), float('inf'), device=pcd.device)  # Initialize distances to infinity

    # Start with a random point
    sampled_indices[0] = torch.randint(0, n, (1,), device=pcd.device)
    for i in range(1, num_samples):
        # Compute distances to the latest sampled point
        dist_to_new_point = torch.sum((pcd - pcd[sampled_indices[i - 1]]) ** 2, dim=1)
        # Update the minimum distances to the set of sampled points
        distances = torch.min(distances, dist_to_new_point)
        # Choose the point farthest away from the current set
        sampled_indices[i] = torch.argmax(distances)

    return pcd[sampled_indices]

def run_simulator(sim, scene_entities, agent, hollow_occ_path, gt_pcd_path, coverage_ratio_rec, cd_rec, acc_rec, scene_id, end, dataset_name):
    # env 
    grid_size = GRID_SIZE
    env_size = ENV_SIZE
    # sim time step
    sim_dt = sim.get_physics_dt()
    
    # camera
    camera = scene_entities[f"camera_0"]   
     
    # initial pose
    target_position = torch.tensor([INITIAL_POSE], dtype=torch.float32)
    # look at (0, 0, 2)
    camera.set_world_poses_from_view(target_position, torch.tensor(TRANS, dtype=torch.float32).unsqueeze(0))  

    # obv grid
    grid = OccupancyGrid(env_size, (1, grid_size, grid_size, grid_size), device=DEVICE)
    # obv face
    obv_face = torch.zeros(NUM_ENVS, GRID_SIZE, GRID_SIZE, GRID_SIZE, 6, device=DEVICE) 

    # ground truth
    occ_gt = torch.tensor(np.load(hollow_occ_path))
    shifted_occ_gt = torch.zeros(occ_gt.shape)
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                # Calculate the new coordinates after applying TRANS
                new_x = x + TRANS[0]
                new_y = y + TRANS[1]
                new_z = z + TRANS[2]

                # Check if the new coordinates are within the bounds of the grid
                if 0 <= new_x < grid_size and 0 <= new_y < grid_size and 0 <= new_z < grid_size:
                    # Shift the value to the new coordinates
                    shifted_occ_gt[new_x, new_y, new_z] = occ_gt[x, y, z]

    occ_gt = shifted_occ_gt

    # ground truth pcd
    # Load the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(gt_pcd_path)
    # Convert the point cloud to a NumPy array
    pcd_np = np.asarray(pcd.points)  # Shape: (n_points, 3)
    # Convert the NumPy array to a PyTorch tensor
    pcd_gt = torch.tensor(pcd_np, dtype=torch.float32)
    pcd_gt = pcd_gt + torch.tensor([TRANS])
    print(pcd_gt.shape)

    # save path
    save_img_folder = os.path.join(os.path.dirname(args_cli.checkpoint), f"{dataset_name}_{TRANS[0]}_{TRANS[1]}_data")
    folder_name = os.path.basename(os.path.dirname(hollow_occ_path))
    os.makedirs(os.path.join(save_img_folder, folder_name), exist_ok=True)

    all_points = []
    all_colors = []
    import time
    old_time = time.perf_counter()
    # start scanning
    for index in range(NUM_STEPS):
        print("")
        # simulate
        for _ in range(6):
            sim.step()
            camera.update(dt=sim_dt)

        camera_ori = camera.data.quat_w_ros.clone()
        camera_ori = convert_camera_frame_orientation_convention(camera_ori, origin="ros", target="world")
        roll, pitch, yaw = euler_xyz_from_quat(camera_ori)

        roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
        pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
        yaw = (yaw + torch.pi) % (2 * torch.pi) - torch.pi

        
        if index==0:
            pose = torch.tensor([INITIAL_POSE + [pitch, yaw]]).to(DEVICE)
        else:
            pose[0, :3] = new_positions
            pose[0, 3] = pitch
            pose[0, 4] = yaw

        # depth image
        depth_image = torch.clamp(camera.data.output["distance_to_image_plane"], 0, env_size * 4)
        # depth image to local point cloud
        points_3d_cam = unproject_depth(depth_image, camera.data.intrinsic_matrices)
        # local point cloud to global point cloud
        points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)
        # filter out out of boundary points
        mask_x = (points_3d_world[:, :, 0]).abs() < env_size / 2 - 1e-3
        mask_y = (points_3d_world[:, :, 1]).abs() < env_size / 2 - 1e-3
        mask_z = (points_3d_world[:, :, 2] < env_size - 1e-3) & (points_3d_world[:, :, 2] >= 0)
        mask = mask_x & mask_y & mask_z    
        
        # update grid
        offset = torch.tensor([env_size / 2, env_size / 2, 0]).to(points_3d_world.device)
        if points_3d_world[mask].shape[0] > 0:
            ratio = grid_size / env_size
            grid.trace_path_and_update(0,
                                       torch.floor((camera.data.pos_w[0]+offset)*ratio),
                                       torch.floor((points_3d_world[0] + offset) * ratio))
            grid.update_log_odds(0, torch.floor((points_3d_world[mask] + offset) * ratio), occupied=True)
            obv_face = torch.logical_or(obv_face,
                                        get_seen_face(torch.unique(torch.floor((points_3d_world[mask]+offset)*ratio).int(), dim=0),
                                        torch.floor((camera.data.pos_w+offset)*ratio), GRID_SIZE, DEVICE))
        # probability grid
        probability_grid = grid.log_odds_to_prob().cpu()

        # coverage ratio
        hard_occ = torch.where(probability_grid[0, :, :, :] >= 0.6, 1, 0)
        hard_occ = hard_occ[:, :, 1:]
        num_match_occ = torch.sum(torch.logical_and(hard_occ, occ_gt[:, :, 1:]), dim=(0, 1, 2))
        total_occ = torch.sum(occ_gt[:, :, 1:], dim=(0, 1, 2))
        coverage_ratio = (num_match_occ/total_occ).reshape(-1, 1)

        # record coverage ratio
        coverage_ratio_rec[scene_id, index] = coverage_ratio.cpu().numpy()[0][0]
        print("Cov:", coverage_ratio[0, 0].item())

        # get rgb image
        rgb_image = camera.data.output["rgb"].clone()     

        # pcd
        valid_points = points_3d_world[mask].reshape(-1, 3).cpu().numpy()
        valid_colors = torch.transpose(rgb_image[0], 0, 1)[mask.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)].reshape(-1, 3).cpu().numpy()

        floor_mask = valid_points[:, 2]>0.1 #(env_size/grid_size)
        valid_points = valid_points[floor_mask]
        valid_colors = valid_colors[floor_mask]

        floor_mask = pcd_gt[:, 2]>0.1 #(env_size/grid_size)
        pcd_gt = pcd_gt[floor_mask]

        # Append valid points and colors to lists
        all_points.append(valid_points)
        all_colors.append(valid_colors)
        acc_points = np.vstack(all_points)
        #if index >=NUM_STEPS-1:
        # this may make cd and point cv decrease
        #    acc_points = subsample_point_cloud(torch.tensor(acc_points), 100000)
        #    cd = chamfer_distance(acc_points, pcd_gt).item()
        #    cd_rec[scene_id, index] = cd
        #    print("CD:", cd)
        #    accuracy = point_coverage_ratio(acc_points, pcd_gt, 0.1)
        #    acc_rec[scene_id, index] = accuracy
        #    print("Acc:", accuracy)

        if index <= 20:
            hl = 10
        elif index <= 25:
            hl = 5
        elif index <= 30:
            hl = 2

        # get observation 
        obs = _get_observations(probability_grid, rgb_image, pose.clone(), obv_face, hl=hl)
        # torch tensor to numpy
        for key, value in obs.items():
            obs[key] = value.cpu().numpy()
        
        # get prediction
        # (nearest xyz, lookat xyz, real xyz)
        actions, _  = agent.predict(obs, deterministic=True)
        print(time.perf_counter()-old_time)
        # rescale actions
        new_positions, orientation_camera, yaw, pitch = process_action(actions)
                
        # save data
        x, y, z = pose[0, :3]
        _yaw, _pitch = pose[0, 3:]
        coverage_ratio = coverage_ratio.cpu().numpy()[0][0]
        suffix = f"_{index}_{x:.2f}_{y:.2f}_{z:.2f}_{_yaw:.2f}_{_pitch:.2f}_{coverage_ratio:.2f}.png"
        #plt.imsave(os.path.join(save_img_folder, folder_name, 'depth'+suffix),
        #           np.clip(depth_image[0,:,:,0].detach().cpu().numpy(), 0, ENV_SIZE*2).astype(np.uint8),
        #                   cmap='gray',
        #                   vmin=0,
        #                   vmax=ENV_SIZE)
        #plt.imsave(os.path.join(save_img_folder, folder_name, 'rgb'+suffix), 
        #           rgb_image[0].detach().cpu().numpy().astype(np.uint8))
        print(os.path.join(save_img_folder, folder_name, 'rgb'+suffix))
        # save occ grid
        #np.save(os.path.join(save_img_folder, folder_name, f'prob_occ_{index}.npy'), probability_grid[0, :, :, :].cpu().numpy())

        # save pcd
        # Create Open3D point cloud
        #point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(valid_points)
        #point_cloud.colors = o3d.utility.Vector3dVector(valid_colors/255.)  # Normalize colors to [0, 1]
        # Save to a PLY file
        #o3d.io.write_point_cloud(os.path.join(save_img_folder, folder_name, f'pcd_{index}.ply'), point_cloud)

        # this may slow down the process
        #org_x, org_y, org_z = env_size/2., env_size/2., 0
        #cell_size = env_size/grid_size  # meters per cell
        #slice_height = env_size / grid_size  # height of each slice in meters
        #if args_cli.vis:
        #    for i in range(grid_size):
        #        occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
        #        create_blocks_from_occupancy(0, [0, 0, 0], occupancy_grid.cpu().numpy(), cell_size, i * cell_size, i, env_size, 1, 30)
        
        # apply action
        camera.set_world_poses(new_positions, orientation_camera)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Calculate the average values up to the current scene_id
    # Plot the averaged values
    coverage_values = coverage_ratio_rec[scene_id]
    cd_values = cd_rec[scene_id]
    accuracy_values = acc_rec[scene_id]
    plots = [
        {"values": coverage_values, "label": "Coverage Ratio", "color": "b", "marker": "o", "filename": f'coverage_ratio_plot_scene_{scene_id + 1}.png'},
        {"values": cd_values, "label": "Chamfer Distance", "color": "g", "marker": "s", "filename": f'chamfer_distance_plot_scene_{scene_id + 1}.png'},
        {"values": accuracy_values, "label": "Accuracy", "color": "r", "marker": "^", "filename": f'accuracy_plot_scene_{scene_id + 1}.png'}
    ]

    # Save each plot as a separate figure
    for plot in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(plot["values"], marker=plot["marker"], color=plot["color"], label=plot["label"])
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{plot["label"]} Over Steps')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(save_img_folder, folder_name, plot["filename"])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # High resolution
        plt.close()

    # Save pcd
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors / 255.0)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(os.path.join(save_img_folder, folder_name, "pcd_final.ply"), point_cloud)

    # Open the CSV file in append mode
    csv_file = "coverage_ratios.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + coverage_values.tolist())

    # Open the CSV file in append mode
    csv_file = "cd.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + cd_values.tolist())

    # Open the CSV file in append mode
    csv_file = "accuracy.csv"
    with open(os.path.join(save_img_folder, csv_file), mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the scene_id and coverage ratios
        writer.writerow([folder_name] + accuracy_values.tolist())

    if scene_id+1 == end:
        # Calculate the average values up to the current scene_id
        average_values = np.mean(coverage_ratio_rec, axis=0)
        print("Average Values:", average_values)
        # Plot the averaged values
        plt.figure(figsize=(10, 6))  # Optional: Specify figure size for better resolution
        plt.plot(average_values, marker='o', color='b', label='Average Curve')
        plt.xlabel('Step')
        plt.ylabel('Average Value')
        plt.title('Average Line Curve of 20 Steps')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file instead of showing it
        plot_filename = f'average_curve_plot_scene.png'  # Dynamic filename based on scene_id
        plt.savefig(os.path.join(save_img_folder, plot_filename), dpi=300, bbox_inches='tight')  # dpi=300 for high-resolution
        plt.close()
        

def process_action(actions):
    actions = torch.tensor(actions).to(DEVICE)
    # nearest xyz
    _xyz = actions[:, :3]
    _xyz = (_xyz + torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)
    
    # non-nearest xyz
    real_xyz = actions[:, 6:9]
    real_xyz = (real_xyz + torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)

    # 0~1
    lookatxyz = actions[:, 3:6]
    # -1~1
    lookatxyz = lookatxyz*2-1
    # to real-world xyz
    lookatxyz = (lookatxyz+torch.tensor([0., 0., 1.]).to(DEVICE)) * torch.tensor([ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3, ENV_SIZE/2.0-1e-3]).to(DEVICE)
    # compute yaw and pitch
    dxyz = lookatxyz - _xyz + 1e-6
    # calculate yaw using torch functions
    # -pi~pi
    _yaw = torch.atan2(dxyz[:, 1], dxyz[:, 0])
    # calculate pitch using torch functions
    # -pi/2~pi/2
    _pitch = torch.atan2(dxyz[:, 2], torch.sqrt(dxyz[:, 0]**2 + dxyz[:, 1]**2))
    # to positive: downward, negative: upward
    _pitch *= -1
    # normalize pitch as specified
    # -pi/3~pi/2 which is -60-90
    _pitch = torch.clamp(_pitch, min=-torch.pi/3, max=torch.pi/2)

    target_position = _xyz #_xyz
    pitch_radians = _pitch
    yaw = _yaw
    
    # roll, pitch, yaw
    target_orientation = rot_utils.euler_angles_to_quats(torch.cat([torch.zeros(yaw.shape[0],1), pitch_radians.unsqueeze(1).cpu(), yaw.cpu().unsqueeze(1)],dim=1).numpy(), degrees=False)
    # setup camera position
    orientation_camera = convert_camera_frame_orientation_convention(torch.tensor(target_orientation).float(), origin="world", target="ros") 
    x_new = _xyz[:, 0] + CAMERA_OFFSET[0] * torch.cos(_yaw) - CAMERA_OFFSET[1] * torch.sin(_yaw)
    y_new = _xyz[:, 1] + CAMERA_OFFSET[0] * torch.sin(_yaw) + CAMERA_OFFSET[1] * torch.cos(_yaw)
    z_new = _xyz[:, 2] + CAMERA_OFFSET[2]

    new_positions = torch.stack([x_new, y_new, z_new], dim=1)

    return new_positions, orientation_camera, _yaw, _pitch

def get_seen_face(occ_grid_xyz, camera_xyz, grid_size, device):
    rays = camera_xyz - occ_grid_xyz

    # Normalize rays
    rays_norm = rays.to(device) / (torch.norm(rays, dim=-1, keepdim=True)+1e-10).to(device)

    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).to(device)
    face_grid = torch.zeros((grid_size, grid_size, grid_size, 6), dtype=torch.bool).to(device)

    # Check visibility for each face
    for i, face in enumerate(faces):
        dot_product = torch.sum(rays_norm * face, dim=-1)
        face_grid[occ_grid_xyz[:, 0], occ_grid_xyz[:, 1], occ_grid_xyz[:, 2], i] = dot_product > 0 #0.1736

    return face_grid

def _get_observations(probability_grid, rgb_image, pose, obv_face, hl=9):
        
        # pose
        current_pose = pose
        # norm
        current_pose[:3] /= ENV_SIZE
        current_pose /= 3.15
        h_limit = (torch.ones(NUM_ENVS).to(DEVICE)*hl*GRID_SIZE/ENV_SIZE).int()
        h_limit_expanded = h_limit.unsqueeze(1) / GRID_SIZE
        pose_step = torch.cat([current_pose.reshape(NUM_ENVS, -1), h_limit_expanded], dim=1)


        obv_occ = torch.ones(NUM_ENVS, GRID_SIZE, GRID_SIZE, GRID_SIZE, 4, device=DEVICE)*0.5
        # generate the linearly spaced values for each dimension
        x_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        y_coords = torch.linspace(-ENV_SIZE/2.0, ENV_SIZE/2.0, GRID_SIZE, device=DEVICE)
        z_coords = torch.linspace(0.0, ENV_SIZE, GRID_SIZE, device=DEVICE)
        
        # create a meshgrid of the coordinates
        x_mesh, y_mesh, z_mesh = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        # xyz
        obv_occ[:, :, :, :, 1:] = torch.stack((x_mesh, y_mesh, z_mesh), dim=-1) / ENV_SIZE
        # occ grid
        obv_occ[:, :, :, :, 0] = probability_grid.clone()
        # face
        occ_face = torch.cat((obv_occ, obv_face), dim=-1)
        # image
        # Normalize to [0, 1] and resize
        resized_rgb_image = F.interpolate(rgb_image.permute(0, 3, 1, 2)/255.0, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
        resized_rgb_image = resized_rgb_image.permute(0, 2, 3, 1)
        #rgb_image = rgb_image/ 255.
        # img: N, H, W, 1
        gray_scale_img = torch.mean(resized_rgb_image[0, :, :, :], (2))

        center = torch.zeros((NUM_ENVS, 3), device=DEVICE)

        obs = {"pose_step": pose_step,
               "img": gray_scale_img.reshape(-1, 1, TARGET_HEIGHT, TARGET_WIDTH),
               "occ": occ_face.permute(0, 4, 1, 2, 3),
               "env_size": torch.ones((NUM_ENVS, 1)) * ENV_SIZE,
               "aux_center": center
               }

        return obs

def make_env():
    """Play with stable-baselines agent."""
    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    agent = PPO_Cus.load(checkpoint_path, print_system_info=True) 
    return agent

def delete_prim_and_children(prim_path, stage):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        for child in prim.GetChildren():
            delete_prim_and_children(child.GetPath().pathString, stage)
        delete_prim(prim_path)

def main():
    # Define the input file
    input_file = args_cli.input
    # load data path    
    scene_paths = []
    with open(input_file, "r") as file:
        for i, line in enumerate(file):
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
                    #print(os.path.join(directory.replace('preprocess', 'PCD_RF'), "*.ply"))
                    pcd_path = glob.glob(os.path.join(directory.replace('preprocess', 'PCD100K_RF'), "*.ply"))[0]

                    # Ensure the required files exist
                    if usd_files and os.path.exists(faces_path) and os.path.exists(fill_occ_set_path):
                        #scene_paths.append((usd_files[0], hollow_occ_path, fill_occ_set_path, faces_path, UsdFileCfg(usd_path=usd_files[0])))
                        scene_paths.append((usd_files[0], hollow_occ_path, fill_occ_set_path, faces_path, pcd_path))
        
    # coverage ratio record
    coverage_ratio_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)
    # chamfer distance record
    cd_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)
    # accuracy record
    acc_rec = np.zeros((len(scene_paths), NUM_STEPS), dtype=np.float32)

    # world initialization
    world = World(stage_units_in_meters=1.0, backend="torch", device="cpu")
    # user viewport
    set_camera_view(eye=np.array([40, 40, 60]), target=np.array([-15, 15, 8]))
    # create agent
    agent = make_env()
    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"Total number of parameters: {total_params}")
    #exit()
    # floor
    world.scene.add_ground_plane(size=40.0, color=torch.tensor([52.0 / 255.0, 195.0 / 255.0, 235.0 / 255.0]))
    # light
    UsdLux.DomeLight.Define(world.scene.stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(500)    
    
    sim_dt = world.get_physics_dt()
    cameraCfg = CameraCfg(
        prim_path=f"/World/Camera_0",
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), convention="world"),
        update_period=sim_dt,  # Update every physical step
        data_types=["distance_to_image_plane","rgb"],
        spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.8, # in mm 13.8
                horizontal_aperture=24., # in mm 24
                clipping_range=(0.01, 60.0) # near and far plane in meter
            ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT
    )
    scene_entities = {}
    scene_entities[f"camera_0"] = Camera(cameraCfg)


    # start scan the shapes
    for i, (scene_path, hollow_occ_path, fill_occ_set_path, faces_path, pcd_path) in enumerate(scene_paths):
        print(f"{i}: {scene_path}")
        
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(args_cli.input)))

        # add usd into the stage
        scene_prim_root=f"/World/Scene_{i}"
        scene = add_reference_to_stage(usd_path=scene_path, prim_path=scene_prim_root)
        # define the property of the stage
        scene_prim = XFormPrim(prim_path=scene_prim_root, name=f"Scene_{i}", translation=TRANS)
        # activate the stage
        world.scene.add(scene_prim)
        scene_prim_path = scene_prim.prim_path

        # output dir
        output_dir = os.path.dirname(scene_path)
        world.reset()

        # TODO: update this
        gt_pcd_path = pcd_path
        run_simulator(world, scene_entities, agent, hollow_occ_path, gt_pcd_path, coverage_ratio_rec, cd_rec, acc_rec, i, len(scene_paths), dataset_name)

        # remove building
        delete_prim(scene_prim_path)
        if args_cli.vis:
            delete_prim("/World/OccupancyBlocks")

if __name__ == "__main__":
    main()
    simulation_app.close()
