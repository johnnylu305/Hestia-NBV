import os
import glob
import numpy as np
import argparse
from scipy.spatial import cKDTree
import open3d as o3d


def load_hollow_occ(path):
    """
    Load the hollow occupancy grid from a .npy file.
    Occupied voxels have value 1, empty voxels have value 0.
    """
    hollow_occ = np.load(path)
    hollow_occ[:, :, 0] = 0
    return hollow_occ


def occupancy_to_pointcloud(occ_grid):
    """
    Convert the 3D occupancy grid into a point cloud of occupied voxels.
    Each occupied voxel (value == 1) yields a point at (x, y, z).
    """
    coords = np.argwhere(occ_grid == 1)
    return coords


def generate_sphere_points_from_grid(occ_grid, thickness=1.0):
    """
    Generate a spherical shell point cloud within the dimensions of the occupancy grid.
    The sphere is centered at the grid center, and the radius is half of the smallest grid dimension.

    Parameters:
        occ_grid (np.ndarray): A 3D array of shape (X, Y, Z).
        thickness (float): How "thick" the spherical shell should be.
                           Points whose distance to center is in [radius - thickness, radius] will be included.

    Returns:
        np.ndarray: N x 3 array of points approximating a spherical shell within the grid.
    """
    X, Y, Z = occ_grid.shape
    # Compute the radius as half of the smallest dimension
    radius = (min(X, Y, Z) / 2.0)*0.8
    
    # Compute center
    cx, cy, cz = X / 2.0, Y / 2.0, Z / 2.0
    
    sphere_points = []
    r_min = radius - thickness
    r_max = radius

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r_min <= dist <= r_max:
                    sphere_points.append([x, y, z])
    
    return np.array(sphere_points)


def chamfer_distance(pcA, pcB):
    """
    Compute the Chamfer distance between two point clouds pcA and pcB.

    Chamfer distance is defined as:
    CD(A, B) = mean_{a in A}(min_{b in B}||a-b||^2) + mean_{b in B}(min_{a in A}||b-a||^2)
    """
    if len(pcA) == 0 or len(pcB) == 0:
        # If either point set is empty, chamfer distance is not well-defined.
        return np.inf
    
    treeA = cKDTree(pcA)
    treeB = cKDTree(pcB)

    dist_A_to_B, _ = treeB.query(pcA)
    dist_B_to_A, _ = treeA.query(pcB)

    dist_A_to_B_sq = dist_A_to_B**2
    dist_B_to_A_sq = dist_B_to_A**2

    cd = np.mean(dist_A_to_B_sq) + np.mean(dist_B_to_A_sq)
    return cd


def vis_hollow_and_sphere(hollow_pc, sphere_pc, output_path):
    """
    Visualize both the hollow occupancy points and the sphere points,
    then save a screenshot as an image file.
    """
    hollow_pcd = o3d.geometry.PointCloud()
    hollow_pcd.points = o3d.utility.Vector3dVector(hollow_pc)
    hollow_pcd.paint_uniform_color([1, 0, 0])  # Red for hollow

    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_pc)
    sphere_pcd.paint_uniform_color([0, 1, 0])  # Green for sphere

    # Shift the sphere point cloud along the x-axis
    sphere_pcd.translate((25, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(hollow_pcd)
    vis.add_geometry(sphere_pcd)

    # Rotate the viewpoint along the x-axis by 90 degrees
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 90.0)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Compute Chamfer distance from hollow_occ to a spherical shell in grid space.")
    parser.add_argument("--input", type=str, required=True, help="Path to the root directory containing hollow_occ.npy files.")
    args = parser.parse_args()

    # Find all hollow_occ.npy files in the given directory
    hollow_occ_files = glob.glob(os.path.join(args.input, "**", "hollow_occ.npy"), recursive=True)

    for hollow_occ_path in hollow_occ_files:
        hollow_occ = load_hollow_occ(hollow_occ_path)
        hollow_pc = occupancy_to_pointcloud(hollow_occ)
        sphere_pc = generate_sphere_points_from_grid(hollow_occ)
        
        # Compute Chamfer distance
        cd = chamfer_distance(hollow_pc, sphere_pc)
        print(f"{hollow_occ_path}: Chamfer Distance = {cd}")

        # Save visualization as an image named by the Chamfer Distance
        # We'll truncate or format the CD value for filename safety
        cd_str = f"{cd:.4f}"  # four decimal places
        output_dir = os.path.split(hollow_occ_path)[0]
        output_path = os.path.join(output_dir, f"CD_{cd_str}.png")

        vis_hollow_and_sphere(hollow_pc, sphere_pc, output_path)
        print(f"Saved screenshot to {output_path}")


if __name__ == "__main__":
    main()

