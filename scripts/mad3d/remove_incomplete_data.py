import os
import shutil
import numpy as np
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Script to remove invalid folders based on specified conditions.")

# Add custom arguments
parser.add_argument("--root_directory", type=str, required=True, help="Path to the root directory to process.")
parser.add_argument("--mode", type=str, choices=["objaverse", "houses3k", "omniobject3d", "other"], required=True, help="Mode to specify the dataset type.")

# Import AppLauncher and append its CLI arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# Parse all arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import additional modules after initializing the simulation app
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils


def remove_objaverse_invalid_folders(root_dir):
    count = 0
    # Iterate through top-level folders (e.g., 000-000)
    for top_folder in os.listdir(root_dir):
        top_folder_path = os.path.join(root_dir, top_folder)
        if not os.path.isdir(top_folder_path):
            continue
        # Iterate through instance folders (e.g., 0a0f1b107acb4b94a8211e11ab69a67f)
        for instance_folder in os.listdir(top_folder_path):
            instance_folder_path = os.path.join(top_folder_path, instance_folder)
            if not os.path.isdir(instance_folder_path):
                continue

            usd_file = os.path.join(instance_folder_path, f"{instance_folder}.usd")
            occ_file = os.path.join(instance_folder_path, "occ.npy")

            # Condition 1: Check if the folder has the .usd file
            if not os.path.exists(usd_file):
                print(f"Removing {instance_folder_path} due to missing .usd file")
                count += 1
                shutil.rmtree(instance_folder_path)
                continue

            # Condition 2: Check if the folder has the occ.npy file
            if not os.path.exists(occ_file):
                print(f"Removing {instance_folder_path} due to missing occ.npy")
                count += 1
                shutil.rmtree(instance_folder_path)
                continue

            # Condition 3: Check if there is only floor
            occ_grid = np.load(occ_file)
            if np.sum(occ_grid[:, :, 1:]) == 0:
                print(f"Removing {instance_folder_path} due to empty occupancy grid")
                count += 1
                shutil.rmtree(instance_folder_path)
                continue

            # Condition 4: remove usd with invalid nodes
            stage = Usd.Stage.Open(usd_file)
            # Add USD to stage cache
            stage_id = UsdUtils.StageCache.Get().Insert(stage)
            # Get the default prim (which is the root prim)
            xform_prim = stage.GetDefaultPrim()

            # Check for invalid nodes under /geometry
            # valid node: Mesh, Look (Scope, Material, Shader)
            geometry_prim = stage.GetPrimAtPath(f"/{xform_prim.GetName()}/geometry")
            if geometry_prim:
                invalid_nodes = [
                    prim.GetPath().pathString for prim in geometry_prim.GetChildren()
                    if prim.GetTypeName() not in ["Mesh", "Scope", "Material", "Shader"]
                ]
                if invalid_nodes:
                    print(f"Removing {instance_folder_path} due to invalid nodes under geometry: {invalid_nodes}")
                    count += 1
                    shutil.rmtree(instance_folder_path)
                    continue            
    print(count)


def remove_house3k_invalid_folders(root_dir):
    count = 0
    # Iterate through batch folders (e.g., Batch_12)
    for batch_folder in os.listdir(root_dir):
        batch_folder_path = os.path.join(root_dir, batch_folder)
        if not os.path.isdir(batch_folder_path):
            continue
        # Iterate through set folders (e.g., SetA)
        for set_folder in os.listdir(batch_folder_path):
            set_folder_path = os.path.join(batch_folder_path, set_folder)
            if not os.path.isdir(set_folder_path):
                continue
            # Iterate through house folders (e.g., BAT12_SETA_HOUSE1)
            for house_folder in os.listdir(set_folder_path):
                house_folder_path = os.path.join(set_folder_path, house_folder)
                if not os.path.isdir(house_folder_path):
                    continue

                nested_folder_path = os.path.join(house_folder_path, house_folder)  # Add nested folder
                if not os.path.isdir(nested_folder_path):
                    continue
                
                usd_file = os.path.join(nested_folder_path, f"{house_folder}.usd")
                occ_file = os.path.join(nested_folder_path, "occ.npy")
                print(occ_file)
                # Condition 1: Missing .usd file
                if not os.path.exists(usd_file):
                    print(f"Removing {house_folder_path} due to missing .usd file")
                    count += 1
                    shutil.rmtree(house_folder_path)
                    continue

                # Condition 2: Missing occ.npy file
                if not os.path.exists(occ_file):
                    print(f"Removing {house_folder_path} due to missing occ.npy")
                    count += 1
                    shutil.rmtree(house_folder_path)
                    continue

                # Condition 3: Check if occupancy grid is empty
                occ_grid = np.load(occ_file)
                if np.sum(occ_grid[:, :, 1:]) == 0:
                    print(f"Removing {house_folder_path} due to empty occupancy grid")
                    count += 1
                    shutil.rmtree(house_folder_path)
                    continue

                # Condition 4: Check for invalid nodes in USD file
                try:
                    stage = Usd.Stage.Open(usd_file)
                    xform_prim = stage.GetDefaultPrim()
                    geometry_prim = stage.GetPrimAtPath(f"/{xform_prim.GetName()}/geometry")

                    if geometry_prim:
                        invalid_nodes = [
                            prim.GetPath().pathString for prim in geometry_prim.GetChildren()
                            if prim.GetTypeName() not in ["Mesh", "Scope", "Material", "Shader"]
                        ]
                        if invalid_nodes:
                            print(f"Removing {house_folder_path} due to invalid nodes: {invalid_nodes}")
                            count += 1
                            shutil.rmtree(house_folder_path)
                            continue
                except Exception as e:
                    print(f"Error processing USD file {usd_file} in {house_folder_path}: {e}")
                    count += 1
                    shutil.rmtree(house_folder_path)
                    continue
    print(f"Total invalid folders removed: {count}")


def remove_omniobject3d_invalid_folders(root_dir):
    count = 0
    # Iterate through top-level object folders (e.g., anise_001, apple_009, etc.)
    for object_folder in os.listdir(root_dir):
        object_folder_path = os.path.join(root_dir, object_folder)
        print(object_folder_path)
        if not os.path.isdir(object_folder_path):
            continue

        # The expected nested path is object_folder/Scan/Scan
        obj = os.path.basename(object_folder_path)
        nested_scan_path = os.path.join(object_folder_path, obj, obj)
        if not os.path.isdir(nested_scan_path):
            print(f"Removing {object_folder_path} due to missing nested Scan/Scan directory.")
            shutil.rmtree(object_folder_path)
            count += 1
            continue

        # Condition 1: Check for the main .usd file (Scan.usd)
        usd_file = os.path.join(nested_scan_path, f"{obj}.usd")
        if not os.path.exists(usd_file):
            print(f"Removing {object_folder_path} due to missing Scan.usd file.")
            shutil.rmtree(object_folder_path)
            count += 1
            continue

        # Condition 2: Check if the folder has the occ.npy file
        occ_file = os.path.join(nested_scan_path, "occ.npy")
        if not os.path.exists(occ_file):
            print(f"Removing {object_folder_path} due to missing occ.npy.")
            shutil.rmtree(object_folder_path)
            count += 1
            continue

        # Condition 3: Check if occupancy grid is empty
        try:
            occ_grid = np.load(occ_file)
            if np.sum(occ_grid[:, :, 1:]) == 0:
                print(f"Removing {object_folder_path} due to empty occupancy grid.")
                shutil.rmtree(object_folder_path)
                count += 1
                continue
        except Exception as e:
            print(f"Error loading occupancy file {occ_file} in {object_folder_path}: {e}")
            shutil.rmtree(object_folder_path)
            count += 1
            continue

        # Condition 4: Check for invalid USD nodes
        try:
            stage = Usd.Stage.Open(usd_file)
            xform_prim = stage.GetDefaultPrim()  # e.g., /Scan
            # Typically geometry is under /Scan/geometry
            geometry_prim = stage.GetPrimAtPath(f"/{xform_prim.GetName()}/geometry")

            if geometry_prim:
                invalid_nodes = [
                    prim.GetPath().pathString
                    for prim in geometry_prim.GetChildren()
                    if prim.GetTypeName() not in ["Mesh", "Scope", "Material", "Shader"]
                ]
                if invalid_nodes:
                    print(f"Removing {object_folder_path} due to invalid nodes under geometry: {invalid_nodes}")
                    shutil.rmtree(object_folder_path)
                    count += 1
                    continue
        except Exception as e:
            print(f"Error processing USD file {usd_file} in {object_folder_path}: {e}")
            shutil.rmtree(object_folder_path)
            count += 1
            continue

    print(f"Total invalid folders removed from OmniObject3D dataset: {count}")


def get_all_mesh_prim_path(stage, root):
    root_prim = stage.GetPrimAtPath(root)
    stack = [root_prim]
    mesh_prim_path = []
    # Traverse the scene graph
    while stack:
        node = stack.pop()
        if node.GetTypeName() == "Mesh":
            mesh_prim_path.append(node.GetPath().pathString)
        # Add children to the stack
        stack.extend(node.GetChildren())
    return mesh_prim_path


def main():
    # Call the function only if mode is 'objaverse'
    if args_cli.mode == "objaverse":
        remove_objaverse_invalid_folders(args_cli.root_directory)
    elif args_cli.mode == "houses3k":
        remove_house3k_invalid_folders(args_cli.root_directory)
    elif args_cli.mode == "omniobject3d":
        remove_omniobject3d_invalid_folders(args_cli.root_directory)
    else:
        print("Mode 'other' is not implemented yet.")


if __name__ == "__main__":
    main()
    simulation_app.close()
