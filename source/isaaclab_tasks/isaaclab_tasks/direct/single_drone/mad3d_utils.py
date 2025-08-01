import torch
import numpy as np
import omni
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, Usd, Gf, Sdf
import open3d as o3d


def compute_weighted_centroid(obv_occ, gt_occ):
    """
    Compute the weighted centroid of XYZ from gt_occ, ignoring faces already observed in obv_occ.

    Parameters:
        obv_occ (torch.Tensor): Observed occupancy grid of shape [n_envs, grid_x, grid_y, grid_z, 6].
        gt_occ (torch.Tensor): Ground truth occupancy grid of shape [n_envs, grid_x, grid_y, grid_z, 6].

    Returns:
        torch.Tensor: Weighted centroid of shape [n_envs, 3].
    """
    # Validate inputs
    assert obv_occ.shape == gt_occ.shape, "obv_occ and gt_occ must have the same shape"
    assert obv_occ.ndim == 5 and obv_occ.shape[-1] == 6, "Input dimensions must match [n_envs, grid_x, grid_y, grid_z, 6]"

    n_envs, grid_x, grid_y, grid_z, _ = obv_occ.shape

    # Mask gt_occ to ignore faces already observed in obv_occ
    valid_occ = gt_occ * (1 - obv_occ)

    # Compute the weights for each voxel (sum of valid faces)
    weights = torch.sum(valid_occ, dim=-1)

    # Create a grid of XYZ coordinates
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(grid_x, device=obv_occ.device),
        torch.arange(grid_y, device=obv_occ.device),
        torch.arange(grid_z, device=obv_occ.device),
        indexing="ij"
    )

    # Expand coordinates to match the shape of weights
    x_coords = x_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]
    y_coords = y_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]
    z_coords = z_coords.unsqueeze(0).float()  # Shape [1, grid_x, grid_y, grid_z]

    # Compute weighted sums for each axis
    total_weights = torch.sum(weights, dim=(1, 2, 3), keepdim=True)
    total_weights = torch.clamp(total_weights, min=1e-6)  # Avoid division by zero

    weighted_x = torch.sum(weights * x_coords, dim=(1, 2, 3)) / total_weights.squeeze()
    weighted_y = torch.sum(weights * y_coords, dim=(1, 2, 3)) / total_weights.squeeze()
    weighted_z = torch.sum(weights * z_coords, dim=(1, 2, 3)) / total_weights.squeeze()

    # Combine weighted coordinates into centroids
    centroids = torch.stack((weighted_x, weighted_y, weighted_z), dim=-1)

    return centroids



def merge_point_clouds(pc1, pc2):
    if np.asarray(pc1.points).shape[0] == 0:
        return pc2
    # Merge points
    merged_points = np.vstack((np.asarray(pc1.points), np.asarray(pc2.points)))
    
    # Initialize a new point cloud object for the merged cloud
    merged_cloud = o3d.geometry.PointCloud()
    merged_cloud.points = o3d.utility.Vector3dVector(merged_points)
    
    # Check if both point clouds have colors
    if pc1.colors and pc2.colors:
        # Merge colors
        merged_colors = np.vstack((np.asarray(pc1.colors), np.asarray(pc2.colors)))
        merged_cloud.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_cloud

def _bresenhamline_nslope(slope, device):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = torch.amax(torch.abs(slope), dim=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = torch.ones(1, dtype=torch.long).to(device)
    normalizedslope = slope / scale
    normalizedslope[zeroslope] = torch.zeros(slope[0].shape).to(device)
    return normalizedslope

def _bresenhamlines(start, end, max_iter, device):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = torch.amax(torch.amax(torch.abs(end - start), dim=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start, device)

    # steps to iterate on
    stepseq = torch.arange(1, max_iter + 1).to(device)
    stepmat = stepseq.repeat(dim, 1) #np.tile(stepseq, (dim, 1)).T
    stepmat = stepmat.T

    # some hacks for broadcasting properly
    bline = start[:, None, :] + nslope[:, None, :] * stepmat

    # Approximate to nearest int
    bline_points = torch.round(bline).to(start.dtype)
    
    return bline_points

def bresenhamline(start, end, max_iter=5, device='cpu'):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter, device).reshape(-1, start.shape[-1])

# project distance to z axis
def dis_to_z(dist_images, intrinsic_matrix):
    # Extract parameters from the intrinsic matrix
    f_x = intrinsic_matrix[:, 0, 0]  # Focal length in x
    f_y = intrinsic_matrix[:, 1, 1]  # Focal length in y
    c_x = intrinsic_matrix[:, 0, 2]  # Principal point x
    c_y = intrinsic_matrix[:, 1, 2]  # Principal point y
    
    # Determine the number of images, height, and width
    n, img_height, img_width = dist_images.shape
    
    # Create arrays representing the x and y coordinates of each pixel
    y_indices, x_indices = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing='ij')
    y_indices, x_indices = y_indices.cuda(), x_indices.cuda()
    
    x_indices = x_indices[None, :, :] - c_x[:, None, None]
    y_indices = y_indices[None, :, :] - c_y[:, None, None]
    
    # Calculate the distance from each pixel to the principal point in the image plane
    d = torch.sqrt(x_indices**2 + y_indices**2 + f_x[:, None, None]**2)
    
    # Calculate Z-component using the cosine of the angle
    depth_images = dist_images * (f_x[:, None, None] / d)
    
    return depth_images

# TODO: implement unknown and free voxels
class OccupancyGrid:
    def __init__(self, env_size, grid_size, decrement=0.4, increment=0.84, max_log_odds=3.5, min_log_odds=-3.5, device='cpu'):
        """
        Initialize the occupancy grid on the specified device (CPU or GPU).
        """
        self.grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        self.grid_size = grid_size
        self.max_log_odds = max_log_odds
        self.min_log_odds = min_log_odds
        self.occupied_increment = increment
        self.free_decrement = decrement
        assert grid_size[1]==grid_size[2]
        assert grid_size[2]==grid_size[3]
        self.resolution = env_size/grid_size[1]
        self.device = device

    def prob_to_log_odds(self, probability):
        """
        Convert probability to log odds.
        """
        return torch.log(probability / (1 - probability))
    
    def log_odds_to_prob(self, l):
        """ Convert log-odds to probability. """
        return 1 / (1 + torch.exp(-l))

    def update_log_odds(self, i, indices, occupied=True):
        """
        Update log odds of the grid at specified indices.
        - indices: A 2D tensor of shape [N, 3] containing x, y, z indices to update.
        - occupied: Whether the points are occupied (True) or free (False).
        """
        indices = indices.long()
        if occupied:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] += self.occupied_increment
        else:
            self.grid[i, indices[:, 0], indices[:, 1], indices[:, 2]] -= self.free_decrement

        # Clamping the values
        self.grid.clamp_(min=self.min_log_odds, max=self.max_log_odds)
    
    def trace_path_and_update(self, i, camera_position, points):
        """
        Trace the path from the camera to each point using Bresenham's algorithm and update the grid.
        """
        camera_position = torch.tensor(camera_position).cuda()

        end_pts = (camera_position).unsqueeze(0).long()

        start_pts = (points).long()
        #start_pts = start_pts.repeat(end_pts.shape[0],1)
        bresenham_path = bresenhamline(start_pts, end_pts, max_iter=-1, device=self.device)
        mask = (bresenham_path[:,0]>=0) & (bresenham_path[:,1]>=0) & (bresenham_path[:,2]>=0) &\
            (bresenham_path[:,0]<self.grid_size[1]) & (bresenham_path[:,1]<self.grid_size[1]) & (bresenham_path[:,2]<self.grid_size[1])
        if bresenham_path[mask] is not None:
            self.update_log_odds(i, bresenham_path[mask], occupied=False)

def get_all_mesh_prim_path(root):
    #print("Root", root)
    root_prim = get_prim_at_path(prim_path=root)
    #print("Root Prim", root_prim)
    stack = [root_prim]
    mesh_prim_path = []
    # While there are nodes in the stack
    while stack:
        # Pop the last node from the stack
        node = stack.pop()
        #print("node", node)
        if node.GetTypeName() == "Mesh":
            mesh_prim_path.append(node.GetPath().pathString)
        # Get the children of the current node
        children = node.GetChildren()
    
        # Iterate over each child
        for child in children:
            # Add the child to the stack for further traversal
            stack.append(child)
    return mesh_prim_path

def get_minmax_mesh_coordinates(mesh_prim):
    # Access the mesh's point positions in local space
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    # Get the world transformation matrix for the mesh
    xformable = UsdGeom.Xformable(mesh_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Transform each point to world coordinates
    transformed_points = [world_transform.Transform(point) for point in points]

    # Calculate the maximum coordinates
    max_coords = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))
    min_coords = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
    for point in transformed_points:
        max_coords[0] = max(max_coords[0], point[0])
        max_coords[1] = max(max_coords[1], point[1])
        max_coords[2] = max(max_coords[2], point[2])

        min_coords[0] = min(min_coords[0], point[0])
        min_coords[1] = min(min_coords[1], point[1])
        min_coords[2] = min(min_coords[2], point[2])

    return max_coords, min_coords

def get_scale(mesh_prim_path, desired_len):
    
    max_x, max_y, max_z = -1e10, -1e10, -1e10
    min_x, min_y, min_z = 1e10, 1e10, 1e10

    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        max_coords, min_coords = get_minmax_mesh_coordinates(mesh_prim)

        max_x = max(max_x, max_coords[0])
        max_y = max(max_y, max_coords[1])
        max_z = max(max_z, max_coords[2])
        min_x = min(min_x, min_coords[0])
        min_y = min(min_y, min_coords[1])
        min_z = min(min_z, min_coords[2])
    extent = (max_x-min_x, max_y-min_y, max_z-min_z)
    max_side = max(extent)
    print(f"Max Side: {max_side} meters")
    return desired_len/max_side

def rescale_scene(scene_prim_root="/World/envs/env_0/Scene", max_len=15):

    mesh_prim_path = get_all_mesh_prim_path(scene_prim_root)
    scale_factor = get_scale(mesh_prim_path, max_len)
    print(f"Scaling factor: {scale_factor}")

    # Apply the scaling to the mesh
    for prim_path in mesh_prim_path:
        mesh_prim = get_prim_at_path(prim_path=prim_path)
        xform = UsdGeom.Xformable(mesh_prim)
        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
        xform.ClearXformOpOrder()  # Clear any existing transformations
        xform.AddTransformOp().Set(scale_transform)

def check_building_collision(occs, xyz, env_ids, org_x, org_y, org_z, cell_size, slice_height, env_origins):
    # remove offset
   
    xyz -= env_origins[env_ids]

    x, y, z = xyz.detach().cpu().numpy()

    x_, y_, z_ = x, y, z

    # smallest point to (0, 0, 0)
    x += org_x
    y += org_y
    z += org_z

    # to voxel id
    # TODO: this may have bug at boundary
    # we should try x/env_size*(grid_size-1)? or clamping
    x = np.floor(x/cell_size).astype(np.int32)
    y = np.floor(y/cell_size).astype(np.int32)
    z = np.floor(z/slice_height).astype(np.int32)

    #col = (z, x, y) in occs[env_ids]
    col = (x, y, z) in occs[env_ids]
    #print(f"Env: {env_ids} Map zxy: {z_} {x_} {y_} to voxel_zxy: {z} {x} {y}, Col: {col}")
    return col


def check_free(occs, xyz, env_ids, org_x, org_y, org_z, cell_size, slice_height, env_origins, shift=True):
    # remove offset

    if shift:
        xyz -= env_origins[env_ids]

    x, y, z = xyz.detach().cpu().numpy()

    x_, y_, z_ = x, y, z

    # smallest point to (0, 0, 0)
    x += org_x
    y += org_y
    z += org_z

    # to voxel id
    # TODO: this may have bug at boundary
    # we should try x/env_size*(grid_size-1)? or clamping
    x = np.floor(x/cell_size).astype(np.int32)
    y = np.floor(y/cell_size).astype(np.int32)
    z = np.floor(z/slice_height).astype(np.int32)

    not_free = occs[x, y, z] >= 0.5

    return not_free


def check_height(hlimit, xyz, env_ids, org_z, cell_size, slice_height, env_origins, shift=True):
    # remove offset

    if shift:
        xyz -= env_origins[env_ids]

    x, y, z = xyz.detach().cpu().numpy()

    x_, y_, z_ = x, y, z

    z += org_z

    # to voxel id
    z = np.floor(z/slice_height).astype(np.int32)

    not_height = z > hlimit

    return not_height


def get_robot_scale(robot_prim_root, desired_max_size):
    crazyflie_prim = get_prim_at_path(prim_path=robot_prim_root)
    # Calculate the bounding box (extents) of the robot
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(crazyflie_prim)
    min_extent = bbox.GetRange().GetMin()
    max_extent = bbox.GetRange().GetMax()

    # Calculate the current dimensions
    current_size_x = max_extent[0] - min_extent[0]
    current_size_y = max_extent[1] - min_extent[1]
    current_size_z = max_extent[2] - min_extent[2]

    # Find the maximum dimension
    current_max_size = max(current_size_x, current_size_y, current_size_z)
    print(min_extent, max_extent)
    print(f"Robot Current max size: {current_max_size} meters")

    # Calculate the scaling factor
    scale_factor = desired_max_size / current_max_size
    print(f"Robot Scaling factor: {scale_factor}")
    return scale_factor

def rescale_robot(robot_prim_root, scale_factor):
    crazyflie_prim = get_prim_at_path(prim_path=robot_prim_root)
    # Apply the scaling to the geometry itself, keeping the transformation matrix intact
    for child in crazyflie_prim.GetChildren():
        geom_xform = UsdGeom.Xformable(child)
        geom_xform.ClearXformOpOrder()  # Clear any existing transformations
        scale_op = geom_xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(scale_factor, scale_factor, scale_factor))
        print(f"Applied scaling factor to child: {child.GetPath()}")

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_orientation(current_position, target_position=np.array([0, 0, 0])):
    # Compute the direction vector from current position to target (origin)
    direction_vector = target_position - current_position
    # Normalize the direction vector
    direction_vector = normalize_vector(direction_vector)
    
    # Compute the yaw angle (in radians)
    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    
    # Compute the pitch angle (in radians)
    pitch = np.arcsin(direction_vector[2])
    return yaw, pitch

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

def create_blocks_from_occ_set(env_id, env_origin, occ_set, cell_size, slice_height, env_size):
    stage = omni.usd.get_context().get_stage()
    for (z, x, y) in occ_set:
        # Calculate position based on cell coordinates
        cube_pos = Gf.Vec3f((x*cell_size)-env_size/2.+env_origin[0], (y*cell_size)-env_size/2.+env_origin[1], slice_height*z+30)

        #cube_pos = Gf.Vec3f((x*cell_size), (y*cell_size), slice_height*z)

        # Define the cube's USD path
        cube_prim_path = f"/World/OccupancyBlocks/BlockGt_{env_id}_{x}_{y}_{z}"

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
                
def create_blocks_from_occ_list(env_id, env_origin, occ_set, cell_size, slice_height, env_size):
    stage = omni.usd.get_context().get_stage()
    i = 0
    for (x, y, z) in occ_set:
        # Calculate position based on cell coordinates
        cube_pos = Gf.Vec3f((x*cell_size)-env_size/2.+env_origin[0], (y*cell_size)-env_size/2.+env_origin[1], slice_height*z-env_size)

        #cube_pos = Gf.Vec3f((x*cell_size), (y*cell_size), slice_height*z)

        # Define the cube's USD path
        cube_prim_path = f"/World/OccupancyBlocks/BlockEnd_{env_id}_{i}"
        i += 1
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

def extract_foreground(p3d_world, floor_z, h, w, bound_mask):
    # N, H*W, 3
    mask = torch.logical_and(((torch.abs(p3d_world[:, 2]) - floor_z) > 0.1), bound_mask).int().reshape(-1, h, w).transpose(1, 2)
    return mask

def get_seen_face(occ_grid_xyz, camera_xyz, grid_size, device):
    #print(occ_grid_xyz.shape)
    #print(camera_xyz.shape)

    rays = camera_xyz - occ_grid_xyz
    
    # Normalize rays
    rays_norm = rays / (torch.norm(rays, dim=-1, keepdim=True)+1e-10)
    
    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).to(device)
    face_grid = torch.zeros((grid_size, grid_size, grid_size, 6), dtype=torch.bool).to(device)
    
    # Check visibility for each face
    for i, face in enumerate(faces):
        dot_product = torch.sum(rays_norm * face, dim=-1)
        face_grid[occ_grid_xyz[:, 0], occ_grid_xyz[:, 1], occ_grid_xyz[:, 2], i] = dot_product > 0. # this is around 80 degree
    
    #print(face_grid)
    #print(face_grid.shape)
    #exit()
    return face_grid

def remove_occluded_face(grid_size, obv_grid, face_grid, device):

    faces = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).to(device)


    face_grid_bool = face_grid.bool()
    # Mask out faces where the obv_grid is non-occupied (obv_grid == 0)
    non_occupied_mask = obv_grid == 0
    face_grid_bool[non_occupied_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 6)] = False

    # remove occluded face
    # Identify visible faces in the grid
    visible_indices = torch.nonzero(face_grid_bool, as_tuple=True)

    # Get indices and directions for occupied voxels
    env_idx, x, y, z, face_idx = visible_indices

    # Shift directions based on face index to find adjacent voxel positions
    shifts = faces[face_idx]  # Extract the shift for each visible face
    adj_x = x + shifts[:, 0].long()
    adj_y = y + shifts[:, 1].long()
    adj_z = z + shifts[:, 2].long()

    # Ensure adjacent positions are within bounds
    valid_mask = (
        (adj_x >= 0) & (adj_x < grid_size) &
        (adj_y >= 0) & (adj_y < grid_size) &
        (adj_z >= 0) & (adj_z < grid_size)
    )
    env_idx = env_idx[valid_mask]
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    adj_x = adj_x[valid_mask]
    adj_y = adj_y[valid_mask]
    adj_z = adj_z[valid_mask]
    face_idx = face_idx[valid_mask]

    # Check if the adjacent voxel is occupied, making the face occluded
    occluded_mask = obv_grid[env_idx, adj_x, adj_y, adj_z] == 1

    face_grid_bool = face_grid_bool.bool()

    # Update the face_grid to mark occluded faces as False
    face_grid_bool[env_idx, x, y, z, face_idx] = ~occluded_mask 
    
    return face_grid_bool

def compute_distance_to_center_distance(fg_masks, w, h):
    image_center = torch.tensor([w / 2, h / 2], device=fg_masks.device)
    max_distance = torch.norm(image_center)

    # Initialize a list to store centroids
    centroids = []

    # Loop through each mask in the batch
    for i in range(fg_masks.size(0)):
        # Get the non-zero coordinates for the current mask
        y_coords, x_coords = torch.nonzero(fg_masks[i], as_tuple=True)

        if y_coords.numel() == 0:  # If there are no foreground pixels
            centroids.append(torch.tensor([w, h], device=fg_masks.device))
        else:
            # Compute the centroid by averaging the non-zero coordinates
            y_mean = y_coords.float().mean()
            x_mean = x_coords.float().mean()
            centroids.append(torch.tensor([y_mean, x_mean], device=fg_masks.device))

    # Convert the centroids list into a tensor
    centroids = torch.stack(centroids)

    # Calculate the Euclidean distance from each centroid to the image center
    distances = torch.norm(centroids - image_center, dim=1)

    return distances / max_distance


def shift_gt_occs(gt_occs, txyz, grid_sizes, env_sizes):
    # Calculate the scaling factor from environment to grid for each axis
    scale_factors = grid_sizes / env_sizes  # Assuming element-wise division

    # Convert translation from environment to grid coordinates
    tx, ty, tz = (txyz * scale_factors).int()


    # Apply torch.roll on each axis (x, y, z) for the current environment
    gt_occs = torch.roll(gt_occs, shifts=(tx, ty, tz), dims=(0, 1, 2))

    return gt_occs


def shift_gt_faces(gt_faces, txyz, grid_sizes, env_sizes):
    # Calculate the scaling factor from environment to grid for each axis
    scale_factors = grid_sizes / env_sizes  # Assuming element-wise division

    # Convert translation from environment to grid coordinates
    tx, ty, tz = (txyz * scale_factors).int()

    # Apply torch.roll on each axis (x, y, z) for the current environment
    # Note: dims=(0, 1, 2) excludes the last dimension for faces
    gt_faces = torch.roll(gt_faces, shifts=(tx, ty, tz), dims=(0, 1, 2))

    return gt_faces


def shift_occs(occs, txyz, grid_sizes, env_sizes):
    # Convert the set of (x, y, z) coordinates into a NumPy array
    occs_array = np.array(list(occs), dtype=int)

    # Calculate the scaling factor from environment to grid for each axis
    scale_factors = grid_sizes / env_sizes  # Assuming element-wise division

    # Convert translation from environment to grid coordinates
    txyz_scaled = (txyz * scale_factors).cpu().int().numpy()

    # Apply the translation only where z > 0
    shifted_occs_array = np.where(
        occs_array[:, 2:] > 0,  # Check if z > 0
        occs_array + txyz_scaled,  # Apply the translation
        occs_array  # Keep original coordinates if z <= 0
    )

    # Convert the result back to a set of tuples
    shifted_occs_set = set(map(tuple, shifted_occs_array))

    return shifted_occs_set
