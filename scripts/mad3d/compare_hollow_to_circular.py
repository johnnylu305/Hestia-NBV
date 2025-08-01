import os
import glob
import argparse
import numpy as np
import open3d as o3d

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and visualize point clouds from occupancy grids.")
    parser.add_argument("--root", type=str, required=True, help="Root path to search for hollow_occ.npy and circular_occ.npy pairs.")
    parser.add_argument("--vis", action="store_true", help="If set, visualize the point clouds.")
    return parser.parse_args()

def load_grids(hollow_path, circular_path):
    """Load hollow and circular occupancy grids."""
    hollow_occ_grid = np.load(hollow_path)
    circular_occ_grid = np.load(circular_path)
    return hollow_occ_grid, circular_occ_grid

def grid_to_point_cloud_with_overlap(hollow_grid, circular_grid):
    """
    Create a point cloud where:
    - Red indicates unique points in `hollow_grid`.
    - Blue indicates unique points in `circular_grid`.
    - Black indicates overlapping points.
    """
    # Identify unique and overlapping points
    hollow_only = np.logical_and(hollow_grid == 1, circular_grid == 0)
    circular_only = np.logical_and(circular_grid == 1, hollow_grid == 0)
    overlap = np.logical_and(hollow_grid == 1, circular_grid == 1)

    # Create point clouds for each category
    hollow_points = np.argwhere(hollow_only)
    circular_points = np.argwhere(circular_only)
    overlap_points = np.argwhere(overlap)

    # Combine all points and assign colors
    all_points = np.vstack([hollow_points, circular_points, overlap_points])
    all_colors = np.vstack([
        np.tile([1, 0, 0], (hollow_points.shape[0], 1)),  # Red for `hollow`
        np.tile([0, 0, 1], (circular_points.shape[0], 1)),  # Blue for `circular`
        np.tile([0, 0, 0], (overlap_points.shape[0], 1))  # Black for overlap
    ])

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    return point_cloud

def process_grids_in_directory(root_path, visualize):
    """Process all hollow_occ.npy and circular_occ.npy pairs in the given root path."""
    pairs = sorted(glob.glob(os.path.join(root_path, '**', 'hollow_occ.npy'), recursive=True))
    overlap_ratios = []

    for hollow_path in pairs:
        if "hollow_occ.npy" in hollow_path:
            circular_path = hollow_path.replace("hollow_occ.npy", "circular_occ.npy")

            if os.path.exists(circular_path):
                print(f"Processing pair: {hollow_path} and {circular_path}")
                hollow_occ_grid, circular_occ_grid = load_grids(hollow_path, circular_path)
                # Ensure grids have the correct shape
                assert hollow_occ_grid.shape == (20, 20, 20), f"Unexpected shape for hollow_occ_grid: {hollow_occ_grid.shape}"
                assert circular_occ_grid.shape == (20, 20, 20), f"Unexpected shape for circular_occ_grid: {circular_occ_grid.shape}"

                # Compute overlap ratio
                overlap = np.logical_and(hollow_occ_grid == 1, circular_occ_grid == 1)
                hollow_nonzero = np.sum(hollow_occ_grid == 1)
                overlap_ratio = np.sum(overlap) / hollow_nonzero if hollow_nonzero > 0 else 0
                overlap_ratios.append(overlap_ratio)

                print(f"Overlap ratio for {hollow_path}: {overlap_ratio:.4f}")

                # Generate and visualize point cloud if requested
                if visualize:
                    point_cloud = grid_to_point_cloud_with_overlap(hollow_occ_grid, circular_occ_grid)
                    o3d.visualization.draw([point_cloud])

    # Print mean overlap ratio
    if overlap_ratios:
        print(f"Mean overlap ratio: {np.mean(overlap_ratios):.4f}")

def main():
    args = parse_arguments()
    process_grids_in_directory(args.root, args.vis)

if __name__ == "__main__":
    main()

