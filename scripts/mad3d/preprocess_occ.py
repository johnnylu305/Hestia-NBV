import os
import glob
import numpy as np
import argparse
import pickle
import open3d as o3d
from collections import deque



# add argparse arguments
parser = argparse.ArgumentParser(description="Preprocess occ.npy into faces.npy, fill_occ_set.pkl and hollow_occ.npy.")
parser.add_argument("input", type=str, help="The root to the input occupancy grid.")
parser.add_argument("--vis", action="store_true", help="If set, visualize results using vis_face_and_voxel.")
# parse the arguments
args_cli = parser.parse_args()


def add_neighbors(occupied_voxels, grid_shape):
    # Directions for 26-connected neighbors in a 3D grid
    directions = [ (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                     (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                     (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                     (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                     (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                     (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

    x_dim, y_dim, z_dim = grid_shape
    neighbors = set()

    for x, y, z in occupied_voxels:
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                neighbors.add((nx, ny, nz))
    
    # Combine original occupied voxels with their neighbors
    occupied_voxels.update(neighbors)
    
    return occupied_voxels


def find_occupied_voxels(grid):
    # Ensure grid is a numpy array
    grid = np.array(grid)
    x_dim, y_dim, z_dim = grid.shape
    
    # Directions for 6-connected neighbors in a 3D grid
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    
    # BFS to mark all reachable empty voxels from the exterior
    def bfs(start):
        queue = deque([start])
        visited = set()
        visited.add(start)
        
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                    if grid[nx, ny, nz] == 0 and (nx, ny, nz) not in visited:
                        visited.add((nx, ny, nz))
                        queue.append((nx, ny, nz))
        
        return visited
    
    # Find exterior empty voxels starting from occ[-1, -1, -1]
    exterior_empty_voxels = bfs((x_dim-1, y_dim-1, z_dim-1))
    
    occupied_voxels = set() 
    # Collect all occupied voxels and mark all non-reachable empty voxels as occupied
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                if grid[x, y, z] == 1 or (grid[x, y, z] == 0 and (x, y, z) not in exterior_empty_voxels):
                    occupied_voxels.add((x, y, z))
    
    return occupied_voxels


def fill_grid(path):

    occ = np.load(path)
    assert sorted(np.unique(occ)) == [0, 1]
    
    grid_size, _, _ = occ.shape

    # we assume this point is empty
    assert occ[-1, -1, -1] == 0
    # put floor
    occ[:, :, 0] = 1
    # occupancy grid set 
    occ_set = find_occupied_voxels(occ)

    # make the occupancy grid set larger for collision detection
    occ_set_dilated = add_neighbors(occ_set.copy(), occ.shape)

    output = os.path.join(os.path.split(path)[0], "fill_occ_set.pkl")
    # Save the occupied voxels to a file
    with open(output, 'wb') as file:
        pickle.dump(occ_set_dilated, file)
    return occ_set, occ_set_dilated, grid_size


def is_hollow(x, y, z, occ_set, grid_size):
    # List of the 6 surrounding voxel coordinates
    surrounding_voxels = [
        (x+1, y, z), (x-1, y, z), 
        (x, y+1, z), (x, y-1, z),
        (x, y, z+1), (x, y, z-1),
    ]
    
    # Check if all surrounding voxels are occupied and within bounds
    for vx, vy, vz in surrounding_voxels:
        # if surrounding voxel is boundary voxel, then the target is surface
        # except z = 0 which is floor
        if (vx < 0 or vx >= grid_size or
            vy < 0 or vy >= grid_size or
            vz >= grid_size):
            return False
        # if surrounding voxel is empty voxel, then the target is surface
        if vz >=0 and (vx, vy, vz) not in occ_set:
            return False
    # Otherwise, the target is not surface voxel
    return True


def convert_to_hollow(path, occ_set, grid_size):
    hollow_grid_set = set()
    
    for voxel in occ_set:
        x, y, z = voxel
        if not is_hollow(x, y, z, occ_set, grid_size):
            hollow_grid_set.add(voxel)

    # Initialize the grid with zeros
    hollow_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    
    # Map the hollow set to the grid
    for (x, y, z) in hollow_grid_set:
        hollow_grid[x, y, z] = 1

    output = os.path.join(os.path.split(path)[0], "hollow_occ.npy")
    np.save(output, hollow_grid)

    return hollow_grid_set, hollow_grid


def save_face(scene_path, hollow_occ, occ_set):

    assert sorted(np.unique(hollow_occ)) == [0, 1]
    
    grid_size, _, _ = hollow_occ.shape
    face_visibility = np.zeros((grid_size, grid_size, grid_size, 6), dtype=bool)
    # face represents by normal
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    
    def is_visible(x, y, z, dx, dy, dz):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nz < grid_size and 0 <= nx < grid_size and 0 <= ny < grid_size:
            return (nx, ny, nz) not in occ_set
        return True
    
    for x, y, z in occ_set:
        if hollow_occ[x, y, z] == 1:
            for n, (dx, dy, dz) in enumerate(faces):
                face_visibility[x, y, z, n] = is_visible(x, y, z, dx, dy, dz)

    # save face occupancy grid
    output = os.path.join(os.path.split(scene_path)[0], "faces.npy")
    np.save(output, face_visibility)

    return face_visibility


def vis_face_and_voxel(hollow_occ, face_vis):
    
    # remove floor
    hollow_occ[:, :, 0] = 0
    
    X, Y, Z = hollow_occ.shape
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    
    voxel_list = []
    normal_list = []

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if hollow_occ[x, y, z] == 1:
                    voxel_list.append([x, y, z])
                    for n, (dx, dy, dz) in enumerate(faces):
                        if face_vis[x, y, z, n]:
                            #normal_list.append([x + 0.5 * dx, y + 0.5 * dy, z + 0.5 * dz, dx, dy, dz])
                            normal_list.append([x, y, z, dx, dy, dz])
    
    
    # Create Open3D voxel grid
    voxel_points = o3d.utility.Vector3dVector(voxel_list)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        o3d.geometry.PointCloud(points=voxel_points),
        voxel_size=1.0,
        min_bound=[0, 0, 0],
        max_bound=[X, Y, Z]
    )
    
    # Create Open3D lines for normals
    line_set = o3d.geometry.LineSet()
    line_points = []
    lines = []
    colors = []

    for i, (px, py, pz, nx, ny, nz) in enumerate(normal_list):
        px += 0.5
        py += 0.5
        pz += 0.5
        line_points.append([px, py, pz])
        line_points.append([px + nx*0.7, py + ny*0.7, pz + nz*0.7])
        lines.append([2 * i, 2 * i + 1])
        colors.append([1, 0, 0])  # red color for normals

    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize using Open3D
    #o3d.visualization.draw_geometries([voxel_grid, line_set])
    o3d.visualization.draw([voxel_grid, line_set])


def main():
    # load target paths
    scenes_path = sorted(glob.glob(os.path.join(args_cli.input, '**', 'occ.npy'), recursive=True))
    for i, scene_path in enumerate(scenes_path):
        print(f"Start processing {i} {scene_path}")

        # solid occupancy grid in set formati for collision detection
        print("Start generating fill_occ_set.pkl")
        occ_set, occ_set_dilated, grid_size = fill_grid(scene_path)
        print(f"Occupied voxels of fill_occ_set.pkl: {len(occ_set_dilated)}")
            
        # hollow occupancy grid without faces
        print("Start generating hollow_occ.npy")
        hollow_set, hollow_grid = convert_to_hollow(scene_path, occ_set, grid_size)
        print(f"Occupied voxels of hollow_occ.npy: {np.sum(hollow_grid)}")
        
        # hollow occupancy grid with six faces
        print("Start generating faces.npy")
        face_vis = save_face(scene_path, hollow_grid, occ_set)
        print(f"Faces of faces.npy: {np.sum(face_vis)}")
        # visualization for debug purpose
        if args_cli.vis:
            print("Visualization enabled. Launching Open3D window...")
            vis_face_and_voxel(hollow_grid, face_vis)
        print("")

if __name__=="__main__":
    main()
