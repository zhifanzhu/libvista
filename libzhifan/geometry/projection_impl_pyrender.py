""" The projection implementations using PyRender """

import os
import numpy as np
from numpy.typing import NDArray
import pyrender

from .mesh import SimpleMesh, AnyMesh
from .pyrender_api import pyrender_call

# set EGL is not set by environment variables
os.environ['PYOPENGL_PLATFORM'] = os.environ.get('PYOPENGL_PLATFORM', 'egl')

_R = np.eye(3)
_T = np.zeros(3)

def pr_perspective_projection(mesh_data: AnyMesh, 
                              cam_f,
                              cam_p,
                              coor_sys='pytorch3d',
                            #   R=_R,
                            #   T=_T,
                              image=None,
                              **kwargs) -> NDArray[np.uint8]:
    """ Perspective projection using PyRender

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]
        cam_f: Tuple, (2,)
        cam_p: Tuple, (2,)
        # R: (3, 3) Camera rotation matrix
        # T: (3,) Camera translation vector

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate system.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

    Returns:
        (H, W, 3) image
    """
    if 'R' in kwargs or 'T' in kwargs:
        raise NotImplementedError(
            'R and T arguments are not test yet in pr_perspective_projection')

    img_h, img_w = image.shape[0], image.shape[1]
    meshes_to_render = []

    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # white surface
        metallicFactor=0.0,                    # dielectric, not metallic
        roughnessFactor=0.2,                   # smooth (shininess ≈ 64)
        emissiveFactor=[0, 0, 0]         # no self-emission
    )

    if isinstance(mesh_data, list):
        for mesh in mesh_data:
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh,  smooth=True)
            for prim in pyrender_mesh.primitives:
                prim.material = mat
            meshes_to_render.append(pyrender_mesh)
    else:
        mesh = pyrender.Mesh.from_trimesh(mesh_data,  smooth=True)
        for prim in mesh.primitives:
            prim.material = mat
        meshes_to_render.append(mesh)
    
    fx, fy = cam_f
    cx, cy = cam_p
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=cx, cy=cy,
        znear=0.01, zfar=1000.0)

    if coor_sys == 'pytorch3d':
        # When the coordinate system is pytorch3d,
        # The object will be placed at +z loations
        # but the camera looks at -z direction (see projection.py fig-1.b)
        # so we need to rotate the object 180 degree around y axis
        camera_pose = Rot_Y = np.float32([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])  
    elif coor_sys in ['neural_renderer', 'nr']:
        # In this case, 
        # we need to rotate the camera along +x by 180 degree,
        camera_pose = Rot_X = np.float32([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
    elif coor_sys in ['open3d', 'openGL', 'pyrender']:
        # If the input meshes are already in Open3D/OpenGL/PyRender coordinate system,
        # nothing needs to be done.
        camera_pose = np.eye(4)
    else:
        raise ValueError(f'Unknown coor_sys: {coor_sys}')

    light = pyrender.DirectionalLight(
        color=np.ones(3), intensity=3.0
    )

    color, depth = pyrender_call(
        meshes=meshes_to_render,
        cam=(camera, camera_pose),
        lights=[(light, camera_pose)],
        size=(img_w, img_h),
        ambient_light=False,
    )
    is_bg = (depth == 0)
    out = image.copy()
    out[~is_bg] = color[..., :3][~is_bg]

    return out