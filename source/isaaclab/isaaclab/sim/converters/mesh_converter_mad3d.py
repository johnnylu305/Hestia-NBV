# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import time
import omni
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file
from isaacsim.core.utils.prims import get_prim_at_path
import isaacsim.core.utils.prims as prims_utils
from pxr import Sdf, UsdLux, UsdGeom, Gf, Usd
from pxr import Gf, Vt

import pxr
import numpy as np
import open3d as o3d
from PIL import Image


VOXEL_SCALE = 0.12


class MeshConverterMAD3D(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg, max_len: 8, env_size: 10, grid_size: 20):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        self.max_len = max_len
        self.env_size = env_size
        self.grid_size = grid_size
        super().__init__(cfg=cfg)


    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        It stores the asset in the following format:

        /file_name (default prim)
          |- /geometry <- Made instanceable if requested
            |- /Looks
            |- /???

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        # must start from alphabet
        mesh_file_basename = 'a' + mesh_file_basename

        
        # Convert USD
        # this will save initial .usd and non_metric.usd
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(
                in_file=cfg.asset_path, out_file=self.usd_path, prim_path=f"/{mesh_file_basename}"
            )
        )
        # Open converted USD stage
        # note: This opens a new stage and does not use the stage created earlier by the user
        # create a new stage
        stage = Usd.Stage.Open(self.usd_path)

        # add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")

        # Make sure the prim is an Xformable type (which can have transformations like scale)
        xformable = UsdGeom.Xformable(geom_prim)

        # Check if the prim has a scale transformation
        scale_attr = xformable.GetXformOpOrderAttr().Get()

        # convert usd to open3d mesh for rescale and shift
        # update .usd 
        rescale_scene(stage, self.usd_path, scene_prim_root=f"/{mesh_file_basename}/geometry", max_len=self.max_len, 
                      env_size=self.env_size, grid_size=self.grid_size)

        # Move all meshes to underneath new Xform
        stack = [geom_prim]
        # While there are nodes in the stack
        while stack:
            # Pop the last node from the stack
            node = stack.pop()
            if node.GetTypeName() == "Mesh":
                # Apply collider properties to mesh
                if cfg.collision_props is not None:
                    # -- Collision approximation to mesh
                    # TODO: https://github.com/isaac-orbit/orbit/issues/163 Move this to a new Schema
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(node)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    # -- Collider properties such as offset, scale, etc.
                    schemas.define_collision_properties(
                        prim_path=node.GetPath(), cfg=cfg.collision_props, stage=stage
                    )

            # Get the children of the current node
            children = node.GetChildren()
    
            # Iterate over each child
            for child in children:
                # Add the child to the stack for further traversal
                stack.append(child)

        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)
        
        # Save changes to USD stage
        
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(
        in_file: str, out_file: str, prim_path: str = "/World", load_materials: bool = True
    ) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to meters.

        The asset hierarchy is arranged as follows:

        .. code-block:: none
            prim_path (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            prim_path: The prim path of the mesh.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        
        enable_extension("omni.kit.asset_converter")
        enable_extension("omni.usd.metrics.assembler")

        import omni.kit.asset_converter
        import omni.usd
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True #True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        out_file_non_metric = out_file.replace(".usd", "_non_metric.usd")
        task = instance.create_converter_task(in_file, out_file_non_metric, None, converter_context)
        # Start conversion task and wait for it to finish
        success = True
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break

        temp_stage = Usd.Stage.CreateInMemory()
        # z up for IsaacLab
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        base_prim = temp_stage.DefinePrim(prim_path, "Xform")
        prim = temp_stage.DefinePrim(f"{prim_path}/geometry", "Xform")
        prim.GetReferences().AddReference(out_file_non_metric)
        cache = UsdUtils.StageCache.Get()
        cache.Insert(temp_stage)
        stage_id = cache.GetId(temp_stage).ToLongInt()
        get_metrics_assembler_interface().resolve_stage(stage_id)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(out_file)
        return success

def get_all_mesh_prim_path(stage, root):
    root_prim = stage.GetPrimAtPath(root)
    stack = [root_prim]
    mesh_prim_path = []
    # While there are nodes in the stack
    while stack:
        # Pop the last node from the stack
        node = stack.pop()
        if node.GetTypeName() == "Mesh":
            mesh_prim_path.append(node.GetPath().pathString)
        # Get the children of the current node
        children = node.GetChildren()
        # Iterate over each child
        for child in children:
            # Add the child to the stack for further traversal
            stack.append(child)
    return mesh_prim_path

def convert_to_open3d_mesh(mesh_data):
    vertices, face_vertex_counts, face_vertex_indices = mesh_data
    vertices = np.array(vertices)
    triangles = []

    # Convert face indices
    index = 0
    for fvc in face_vertex_counts:
        if fvc == 3:  # Assuming all faces are triangles
            triangles.append(face_vertex_indices[index:index+3])
        index += fvc

    triangles = np.array(triangles)

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()  # Optional: for better visualization
    return o3d_mesh

def save_occupancy_grid_as_image(occupancy_grid, filename):
    # Define the color mapping for the occupancy grid
    colors = {
        1: (0, 0, 0),       # Black for occupied
        0: (255, 255, 255), # White for unoccupied
        2: (255, 255, 255)  # Assume it is free
    }

    # Create an RGB image from the occupancy grid
    image_data = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        image_data[occupancy_grid == value] = color

    # Create and save the image
    image = Image.fromarray(image_data, 'RGB')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)
    print(f"Saved occupancy map as image: {filename}")

def rescale_scene(stage, outfile, scene_prim_root="/World/Scene", max_len=8, env_size=10, grid_size=20):
        
    # only one mesh
    mesh_prim_path = get_all_mesh_prim_path(stage, scene_prim_root)
    if len(mesh_prim_path) != 1:
        return
    assert len(mesh_prim_path) == 1
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path[0])
    mesh = UsdGeom.Mesh(mesh_prim)
    # get vertex from usd
    points = mesh.GetPointsAttr().Get()
   
    # root prim contains geometry (root), geometry/looks (texture), geometry/mesh (mesh)
    root_prim = stage.GetPrimAtPath(scene_prim_root)
    # traverse all descendants of the root prim
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Xform):  # Check if the prim has transform attributes
            xform = UsdGeom.Xform(prim)
            #get all transformation operations on the prim
            xform_ops = xform.GetOrderedXformOps()
            # iterate through each operation and clear translation and scale transformations
            for op in xform_ops:
                op_type = op.GetOpType()
                if op_type == UsdGeom.XformOp.TypeScale or op_type == UsdGeom.XformOp.TypeOrient:
                    op.GetAttr().Clear()  # Clear the specific transformation operation

            # Update xformOpOrder to remove cleared operations, keeping only rotations
            new_xform_op_order = [op for op in xform.GetOrderedXformOps() if op.GetOpType() not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale)]
            xform.SetXformOpOrder(new_xform_op_order)

    # skip empty mesh
    if points is None or len(points) < 1:
        return

    assert points is not None
    assert len(points)>0
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    mesh_data = (points, face_vertex_counts, face_vertex_indices)

    # get o3d mesh
    o3d_mesh = convert_to_open3d_mesh(mesh_data)
    
    #center = o3d_mesh.get_center() 
    # use minmax instead of mean of the vertex
    center = (o3d_mesh.get_max_bound()+o3d_mesh.get_min_bound())/2
    # to origin
    o3d_mesh.translate(-center)

    # scale max side to max len
    scale_factor = max_len/max(o3d_mesh.get_max_bound()-o3d_mesh.get_min_bound())
    o3d_mesh.scale(scale_factor, center=(0, 0, 0))

    # get min max xyz
    x_min, y_min, z_min = o3d_mesh.get_min_bound()
    x_max, y_max, z_max = o3d_mesh.get_max_bound()
    # x: x, y: -z, z: y
    o3d_mesh.translate((0, -y_min, 0))

    #scaled_points =  Vt.Vec3fArray([tuple((1, 2, 3)) for vertex in np.asarray(o3d_mesh.vertices)]) 
    scaled_points = Vt.Vec3fArray([tuple(vertex) for vertex in np.asarray(o3d_mesh.vertices)])

    assert o3d_mesh.get_min_bound()[1] == 0

    vertices = np.asarray(o3d_mesh.vertices)
    transformed_vertices = vertices.copy()
    transformed_vertices[:, 2] = vertices[:, 1]  # z = y
    transformed_vertices[:, 1] = -vertices[:, 2]  # y = -z
    o3d_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    
    #o3d.visualization.draw([o3d_mesh])
    voxel_size = (env_size/grid_size)*VOXEL_SCALE # make the voxel resolution higher to create accurate occupancy grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)
    
    # real xyz
    real_coordinates = np.array([voxel_grid.origin+voxel.grid_index*voxel_size+np.array([voxel_size/2, voxel_size/2, voxel_size/2]) for voxel in voxel_grid.get_voxels()])
    # real xyz to voxel xyz
    voxel_coordinates = (real_coordinates + np.array([[env_size/2, env_size/2, 0]]))*grid_size/env_size
    voxel_centers = np.floor(voxel_coordinates).astype(int)
    
    # initialize occupancy grid
    occ_grid = np.zeros((grid_size, grid_size, grid_size))
    for i in range(len(voxel_centers)):
        # prevent boundary situation
        occ_grid[min(grid_size-1, voxel_centers[i][0]), min(grid_size-1, voxel_centers[i][1]), min(grid_size-1, voxel_centers[i][2])] = 1

    folder = os.path.dirname(outfile)
    for i in range(occ_grid.shape[2]):
        # vis occ
        image_filename = f"occupancy_map_slice_{i}.png"
        save_occupancy_grid_as_image(occ_grid[:, :, i], os.path.join(folder, image_filename))
        # save as npy file
        np.save(os.path.join(folder, "occ.npy"), occ_grid[:, :, :])
    

    # update usd
    assert(len(mesh_prim_path)==1)    
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path[0])
    mesh = UsdGeom.Mesh(mesh_prim)    
    mesh.GetPointsAttr().Set(scaled_points)
