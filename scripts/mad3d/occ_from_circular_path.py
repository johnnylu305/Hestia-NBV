import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to create occupancy grid")
parser.add_argument("--input", type=str, required=True, help="Path to input USD files.")
parser.add_argument("--vis", action="store_true", help="If set, visualize occupancy grid.")

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
import omni
import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import XFormPrim
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import transform_points, unproject_depth
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from PIL import Image


# Env
GRID_SIZE = 20
ENV_SIZE = 20 #10
# Sensor
CameraCfg = CameraCfg(
    prim_path="/World/Camera",
    offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), convention="world"),
    update_period=0,  # Update every physical step
    data_types=["distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        clipping_range=(0.1, ENV_SIZE*2)  # Near and far plane in meter
    ),
    width=2000,
    height=2000,
    update_latest_camera_pose=True
)


class OccupancyGrid:
    def __init__(self, env_size, grid_size, device="cpu"):
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.grid_size = grid_size
        self.resolution = env_size / grid_size[1]
        self.device = device

    def update_log_odds(self, i, indices, occupied=True):
        indices = indices.long()
        if occupied:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] += 0.84
        else:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] -= 0.4
        self.grid.clamp_(min=-3.5, max=3.5)

    def log_odds_to_prob(self):
        return 1 / (1 + torch.exp(-self.grid))


def setup_scene(world, scene_path, scene_prim_root="/World/Scene"):
    # floor
    world.scene.add_ground_plane(size=40.0, color=torch.tensor([52.0 / 255.0, 195.0 / 255.0, 235.0 / 255.0]))
    # light
    UsdLux.DomeLight.Define(world.scene.stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(500)
    
    # add usd into the stage
    scene = add_reference_to_stage(usd_path=scene_path, prim_path=scene_prim_root)

    # define the property of the stage
    scene_prim = XFormPrim(prim_paths_expr=scene_prim_root, translations=[[0, 0, 0]])
    # activate the stage
    world.scene.add(scene_prim)
    
    scene_entities = {}
    scene_entities["camera"] = Camera(CameraCfg)

    return scene_entities


def save_occupancy_grid_as_image(occupancy_grid, filename):
    colors = {1: (0, 0, 0), 0: (255, 255, 255)}
    image_data = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        image_data[occupancy_grid == value] = color
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(image_data, "RGB").save(filename)
    print(f"Saved occupancy map as image: {filename}")


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


def run_simulator(sim, scene_entities, output):
    # env 
    grid_size = GRID_SIZE
    env_size = ENV_SIZE
    # grid
    grid = OccupancyGrid(env_size, (1, grid_size, grid_size, grid_size))
    # sim time step
    sim_dt = sim.get_physics_dt()
    # camera
    camera = scene_entities["camera"]   
    # circular path
    index = 0
    theta = np.linspace(0, 6 * np.pi, 20)
    x = env_size / 2 * 3.5 * np.cos(theta)
    y = env_size / 2 * 3.5 * np.sin(theta)
    z = np.linspace(2, env_size / 2 * 3.0, len(theta))
    # start scanning
    while index < len(theta):
        # set camera to the target position
        target_position = torch.tensor([[x[index], y[index], z[index]]], dtype=torch.float32)
        # look at (0, 0, 2)
        camera.set_world_poses_from_view(target_position, torch.tensor([0, 0, 2], dtype=torch.float32).unsqueeze(0))
        # simulate
        for _ in range(10):
            sim.step()
            camera.update(dt=sim_dt)
        # next step
        index += 1
        # depth image
        if "distance_to_image_plane" in camera.data.output:
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
                grid.update_log_odds(0, torch.floor((points_3d_world[mask] + offset) * ratio), occupied=True)

        probability_grid = grid.log_odds_to_prob()

        # this may slow down the process
        org_x, org_y, org_z = env_size/2., env_size/2., 0
        cell_size = env_size/grid_size  # meters per cell
        slice_height = env_size / grid_size  # height of each slice in meters
        if args_cli.vis:
            for i in range(grid_size):
                occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
                create_blocks_from_occupancy(0, [0, 0, 0], occupancy_grid.cpu().numpy(), cell_size, i * cell_size, i, env_size, 1, 25)


    if args_cli.vis:
        for i in range(grid_size):
            occupancy_grid = (probability_grid[0, :, :, i] > 0.5).int()
            save_occupancy_grid_as_image(occupancy_grid.cpu().numpy(), os.path.join(output, f"circular_occupancy_map_slice_{i}.png"))
    occupancy_grid = (probability_grid[0] > 0.5).int()
    np.save(os.path.join(output, "circular_occ.npy"), occupancy_grid.cpu().numpy())
    # screenshot
    vp_api = get_active_viewport()
    capture_viewport_to_file(vp_api, os.path.join(output, "vis.png"))


def main():
    # load non-metric usd
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, "**", "*[!_non_metric].usd"), recursive=True))

    # world initialization
    world = World(stage_units_in_meters=1.0, backend="torch", device="cpu")
    # user viewport
    set_camera_view(eye=np.array([40, 40, 60]), target=np.array([-15, 15, 8]))

    # start scan the shapes
    for i, scene_path in enumerate(scenes_path):
        print(f"{i}: {scene_path}")
        # setup ground, light, and camera
        scene_entities = setup_scene(world, scene_path)

        # output dir
        output_dir = os.path.dirname(scene_path)

        # run simulation (capture)
        world.reset()
        run_simulator(world, scene_entities, output_dir)
        world.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
