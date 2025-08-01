import numpy as np
import pickle
import open3d as o3d
import argparse
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt



def visualize_voxel_grid_and_pointcloud(occupancy_grid, point_cloud, grid_shift=(10, 10, 0)):
    """
    Visualizes the voxel grid and point cloud together.

    Args:
        occupancy_grid (np.ndarray): Binary 3D occupancy grid.
        point_cloud (np.ndarray): Nx3 array of points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxel grid
    x, y, z = np.where(occupancy_grid == 1)
    ax.scatter(x - grid_shift[0], y - grid_shift[1], z, c='red', label='Voxel Grid')

    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='blue', s=1, label='Point Cloud')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# Functions to be tested
def filter_point_cloud(point_cloud, occupancy_grid, grid_shift=(10,10,0)):
    """
    Filters out points in the point cloud that are within the occupancy grid.

    Args:
        point_cloud (np.ndarray): Original point cloud (Nx3 array).
        occupancy_grid (np.ndarray): Occupancy grid points (Mx3 array).

    Returns:
        np.ndarray: Filtered point cloud.
    """
    occupancy_grid[:,:,0] = 0  

    # Define a 3x3x3 structuring element for 26-connectivity
    structuring_element = np.ones((3, 3, 3), dtype=bool)

    # Apply binary dilation to the occupancy grid
    dilated_grid = binary_dilation(occupancy_grid, structure=structuring_element)

    # Map point cloud to grid indices
    filtered_points = np.floor(point_cloud).astype(int)

    dilated_grid[:,:,0] = 1

    filtered_indices = [i for i, point in enumerate(filtered_points) if dilated_grid[point[0]+grid_shift[0], point[1]+grid_shift[1], point[2]]==1]


    print(f"Number of points removed: {len(point_cloud) - len(filtered_indices)}")

    #visualize_voxel_grid_and_pointcloud(dilated_grid, point_cloud)
    
    return point_cloud[filtered_indices]


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a point cloud based on an occupancy grid.")
    parser.add_argument("--pointcloud", required=True, help="Path to the folder containing the point cloud PLY files.")
    parser.add_argument("--occ", required=True, help="Path to the folder containing the occupancy grid files.")
    parser.add_argument("--output", required=True, help="Path to the folder to save the filtered point cloud PLY files.")
    parser.add_argument("--mode", type=str, choices=["objaverse", "houses3k", "omniobject3d", "other"], required=True, help="Mode to specify the dataset type.")


    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Find all PLY files in the pointcloud directory and subdirectories
    ply_files = glob.glob(os.path.join(args.pointcloud, "**", "*.ply"), recursive=True)

    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in directory: {args.pointcloud}")

    for ply_file in ply_files:
        # Derive corresponding PKL directory using glob to find matching subdirectory
        relative_path = os.path.relpath(ply_file, args.pointcloud)
        base_subdir = os.path.basename(os.path.dirname(relative_path))
        occ_dirs = glob.glob(os.path.join(args.occ, "**", base_subdir), recursive=True)
        if not occ_dirs:
            print(f"Warning: No matching directory found for {base_subdir} in {args.occ}. Skipping {ply_file}.")
            continue

        if args.mode != "objaverse" and args.mode != "omniobject3d":
            # Locate the fill_occ_set.pkl in the matched directory
            occ_file = os.path.join(occ_dirs[0], base_subdir, "hollow_occ.npy")
        elif args.mode == "omniobject3d":
            occ_file = os.path.join(occ_dirs[0], base_subdir, base_subdir, "hollow_occ.npy")
        else:
            occ_file = os.path.join(occ_dirs[0], "hollow_occ.npy")

        if not os.path.isfile(occ_file):
            print(f"Warning: No fill_occ_set.pkl found in {occ_file}. Skipping {ply_file}.")
            continue

        # Load point cloud from PLY file
        pcd = o3d.io.read_point_cloud(ply_file)
        point_cloud = np.asarray(pcd.points)

        # Load occupancy grid from PKL file
        occupancy_grid = np.load(occ_file)
        # Run filter
        filtered_point_cloud = filter_point_cloud(point_cloud, occupancy_grid)

        # Prepare output file path
        output_file_path = os.path.join(args.output, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save filtered point cloud back to PLY file
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud)
        o3d.io.write_point_cloud(output_file_path, filtered_pcd)

        print(f"Filtered point cloud saved to {output_file_path}")
