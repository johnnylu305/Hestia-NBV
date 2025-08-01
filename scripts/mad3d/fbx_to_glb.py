import os
import argparse
import subprocess


def convert_fbx_to_glb(input_fbx, output_glb):
    """
    Convert an FBX file to GLB format using Blender in headless mode.

    Args:
        input_fbx (str): Path to the FBX file.
        output_glb (str): Path to save the GLB file.
    """
    blender_script = f"""
import bpy
import os

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX file
bpy.ops.import_scene.fbx(filepath=r"{input_fbx}")

# Export as GLB
output_path = r"{output_glb}"
bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')

bpy.ops.wm.quit_blender()
"""

    try:
        # Run Blender in background mode
        command = ["blender", "--background", "--python-expr", blender_script]
        subprocess.run(command, check=True)
        print(f"Converted: {input_fbx} -> {output_glb}")
    except Exception as e:
        print(f"Error converting {input_fbx}: {e}")


def process_directory(root_path, output_root):
    """
    Recursively search for FBX files in the root directory, convert them to GLB,
    and save them in the output root.

    Args:
        root_path (str): Root directory containing FBX files.
        output_root (str): Root directory to save converted GLB files.
    """
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".fbx"):
                input_fbx = os.path.join(root, file)
                
                # Maintain folder structure and create a folder per FBX file
                relative_path = os.path.relpath(root, root_path)
                instance_name = os.path.splitext(file)[0]  # FBX file name without extension
                output_folder = os.path.join(output_root, relative_path, instance_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # Output GLB path
                output_glb = os.path.join(output_folder, f"{instance_name}.glb")

                # Convert FBX to GLB
                convert_fbx_to_glb(input_fbx, output_glb)


def main():
    parser = argparse.ArgumentParser(description="Convert FBX files to GLB using Blender.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing FBX files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save GLB files.")
    args = parser.parse_args()

    # Ensure Blender is available
    if not shutil.which("blender"):
        print("Error: Blender is not installed or not added to PATH.")
        return

    print(f"Processing FBX files in: {args.root}")
    print(f"Saving GLB files to: {args.output}")

    process_directory(args.root, args.output)


if __name__ == "__main__":
    import shutil
    main()

