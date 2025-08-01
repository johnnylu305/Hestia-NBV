import argparse

from isaaclab.app import AppLauncher
import shutil

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("input", type=str, help="The path to the dataset txt file.")
parser.add_argument("output", type=str, help="The root to store the USD file.")

parser.add_argument(
    "--filter_input",
    type=str,
    default=None,
    help="Only load the mesh specified by the txt file.",
)


parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="none",
    choices=["convexDecomposition", "convexHull", "none", "meshSimplification"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import glob
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MeshConverterMAD3DPointCloud, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

# normalize the max side of the 3D shape to max_len
MAX_LEN = 8
# env and grid size
ENV_SIZE = 20
GRID_SIZE = 20


def run_convert(mesh_path, dest_path):
    # check valid file path
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.abspath(mesh_path)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid mesh file path: {mesh_path}")

    # create destination path
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    print(dest_path)

    # Mass properties
    if args_cli.mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=args_cli.make_instanceable,
        collision_approximation=args_cli.collision_approximation,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input Mesh file: {mesh_path}")
    print("Mesh importer config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Mesh converter and import the file
    mesh_converter = MeshConverterMAD3DPointCloud(mesh_converter_cfg, max_len=MAX_LEN, env_size=ENV_SIZE, grid_size=GRID_SIZE)
    # print output
    print("Mesh importer output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)


def main():
    # Define the input file
    filter_input_file = args_cli.filter_input
    
    if filter_input_file is not None:
        scene_paths = []
        with open(filter_input_file, "r") as file:
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
                        
                        batch_name = os.path.basename(directory)
                        
                    scene_paths.append(batch_name + '.') # avoid 1 10 is the same

        
    # load all 3D model paths  
    # list of file extensions to search for
    extensions = ['*.glb', '*.obj', '*.fbx']
    meshes_path = []
    for ext in extensions:
        meshes_path.extend(glob.glob(os.path.join(args_cli.input, '**', ext), recursive=True))
    print(meshes_path) 
    if filter_input_file is not None:
        # Filter mesh_paths
        meshes_path = [
            mesh for mesh in meshes_path if any(scene in mesh for scene in scene_paths)
        ]
    
    start = 0
    # start conversion 3D model (glb, obj, fbx) to rescaled, shifted usd and occupancy grid (occ.npy)
    for i, mesh_path in enumerate(meshes_path[start:]):
        print(i+start)
        relative_path = os.path.relpath(mesh_path, args_cli.input)
        dest_path = os.path.join(args_cli.output, relative_path)
        # save path
        dest_path = os.path.join(dest_path[:-4], os.path.split(relative_path)[-1][:-3]+'usd')
        run_convert(mesh_path, dest_path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
