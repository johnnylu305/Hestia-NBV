import torch
import isaaclab.sim as sim_utils
import os
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg


# env
GRID_SIZE = 20
ENV_SIZE = 20
USED_NEAREST = True
DEFAULT_INIT_POS = [ENV_SIZE/2-1, ENV_SIZE/2-1, ENV_SIZE/4-1]
# TODO: Change this back later
CAMERA_FOLDER = 'camera_image_corl_final_test' #'camera_image_envsize20_obja_final_oldlrsch_h3k'


# define crazyflie
CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, # won't drop
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
        copy_from_source=False, # mirror
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


# define MAD3D environment
@configclass
class MAD3DEnvCfg(DirectRLEnvCfg):

    # data
    data_root = os.path.join(os.path.abspath(os.sep), "home", "dsr", "Documents", "mad3d", "New_Dataset20", "objaverse") #"house3k")
    camera_folder = CAMERA_FOLDER

    # env
    env_size = ENV_SIZE
    grid_size = GRID_SIZE
    env_spacing = 60 # in meter, 2 cells is one unit
    
    # randomly initialize the first viewpoint
    random_initial_view = True
    default_init_pos = DEFAULT_INIT_POS    

    # randomly translate the object
    random_trans_obj = True
    # x, y translation range in meter
    trans_obj_x = [-4, 4] #[-5, 5] 
    trans_obj_y = [-4, 4] #[-5, 5]
    
    # randomly sample n object
    random_sample_obj = True

    # this is a placeholder
    num_envs = 2 # this might be overwrote by parser

    # rl
    num_actions = 3 # x, y, z
    used_nearest = USED_NEAREST
    num_states = 0 
    episode_length_s = 20000 # timeout
    total_img = 50 # timeout
    goal = 0.9 # coverage ratio

    # probability occupancy grid
    decrement = 0.01
    increment = 1.0
    max_log_odds = 10.
    min_log_odds = -10.

    # sensor
    # TODO: tune offset?
    camera_offset = [0.1, 0.0, 0.0]
    camera_w, camera_h = 300, 300

    # obv
    # these are placeholder
    # check _configure_gym_env_spaces
    observation_space = 0
    action_space = 0
    state_space = 0


    # save image during training
    save_img = True
    # visualize occupancy grid during training
    vis_occ = False
    # save images for these environments
    save_env_ids = [0, 1]
    # save frequency (episode)
    save_img_freq = 30

    # simulator
    decimation = 2 # _apply_action will run decimation time
    debug_vis = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/10, # physical simulation step
        gravity=(0.0, 0.0, 0.0), # disable gravity
        enable_scene_query_support=False, # disable collision query
        render_interval=decimation
    )

    # ground	
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane"
    )
    
    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG

    # tiled camera
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_offset, convention="world"),
        update_period=0, # update every physical step
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.8, # in cm default 24, dji 13.8
            #focus_distance=1.0, # in m 
            horizontal_aperture=24., # in mm 
            clipping_range=(0.1, 60.0) # near and far plane in meter
        ),
        width=camera_w,
        height=camera_h,
    )

    # setup interactive scene for rl training
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, 
                                                     env_spacing=env_spacing,
                                                     lazy_sensor_update=False, # update sensor data every time
                                                     replicate_physics=True)
