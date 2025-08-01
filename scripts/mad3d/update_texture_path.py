import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Recursively update all .usd files in a directory, overwriting them if any Sdf.AssetPath starts with old_prefix."
)
parser.add_argument(
    "--root_dir", 
    type=str, 
    required=True, 
    help="Path to the root directory containing .usd files."
)
parser.add_argument(
    "--old_prefix", 
    type=str, 
    required=True, 
    help="Old prefix in file paths."
)
parser.add_argument(
    "--new_prefix", 
    type=str, 
    required=True, 
    help="New prefix to replace old prefix."
)

# Add any additional Isaac Sim app arguments (like --enable-extensions, etc.)
AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, Sdf


def update_asset_paths(stage, old_prefix, new_prefix):
    """
    For every prim in the stage, checks all authored attributes.
    If an attribute is an Sdf.AssetPath whose .path starts with old_prefix,
    replace old_prefix with new_prefix.
    """
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.HasAuthoredValue():
                old_value = attr.Get()
                
                # Check if it's an Sdf.AssetPath
                if isinstance(old_value, Sdf.AssetPath):
                    old_path = old_value.path
                    if old_path.startswith(old_prefix):
                        new_path = old_path.replace(old_prefix, new_prefix, 1)
                        attr.Set(Sdf.AssetPath(new_path))
                        print(f"Updated path: {old_path} -> {new_path}")


def main():
    # Walk through root_dir to find .usd files
    for root, dirs, files in os.walk(args.root_dir):
        for file_name in files:
            if file_name.endswith(".usd"):
                usd_path = os.path.join(root, file_name)
                
                # Open the USD stage
                stage = Usd.Stage.Open(usd_path)
                if not stage:
                    print(f"Warning: Could not open stage from {usd_path}, skipping.")
                    continue

                # Update Sdf.AssetPath attributes
                update_asset_paths(stage, args.old_prefix, args.new_prefix)

                # Overwrite the same USD file
                stage.Export(usd_path)
                print(f"Overwrote USD file: {usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
