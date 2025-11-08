""" The projection implementations using PyRender """

import os
import numpy as np
from typing import Union, List
from numpy.typing import NDArray
import trimesh
import pyrender

from .coor_utils import rotmat_Rx, rotmat_Ry
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
    assert image.max() <= 1.0 and image.min() >= 0.0, \
        'Input image should have values in [0, 1]'
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

    #TODO: handles face_colors for pyrender
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
    image_uint8 = (image * 255).astype(np.uint8)
    out = image_uint8
    out[~is_bg] = color[..., :3][~is_bg]

    return out


""" Helper function """
def _init_trimesh_no_facecolors(mesh: Union[SimpleMesh, trimesh.Trimesh]) -> trimesh.Trimesh:
    if issubclass(type(mesh), trimesh.Trimesh):
        return trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
            vertex_colors=mesh.visual.vertex_colors)
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

def pr_project_standardized(mesh_data: AnyMesh,
                            direction='+z',
                            image_size=200,
                            pad=0.2,
                            coor_sys='nr',
                            centering=True,
                            manual_dmax : float = None,
                            show_axis=False,
                            print_dmax=False,
                            **kwargs) -> np.ndarray:
    """
    Given any mesh(es), this function renders the zoom-in images.
    The meshes are proecessed to be in [-0.5, 0.5]^3 space,
    then a weak-perspective camera is applied.

    Args:
        pad: the fraction to be padded around rendered image.
        manual_dmax: set dmax manually
        print_dmax: This helps determine manual_dmax
        centering: if True, look at (xc, yc, zc); otherwise, look at (0, 0, 0)
        show_axis: if True, show the mesh w.r.t the original (un-centered) reference frame
        **kwargs: other kwargs passed to projection function

    Returns:
        (H, W, 3)
    """
    # Note: Don't use trimesh.util.concatenate, because the meshes loses vertex color
    # Convert into a list of trimesh.Trimesh copies
    if isinstance(mesh_data, list):
        _mesh_data = []
        for mesh in mesh_data:
            _mesh_data.append(_init_trimesh_no_facecolors(mesh))
    else:
        _mesh_data = [_init_trimesh_no_facecolors(mesh_data)]

    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for mesh in _mesh_data:
        vmin = mesh.vertices.min(axis=0)
        vmax = mesh.vertices.max(axis=0)
        bmin = np.minimum(bmin, vmin)
        bmax = np.maximum(bmax, vmax)

    xc, yc, zc = (bmin + bmax) / 2.0
    dmax = float((bmax - bmin).max())
    if manual_dmax is not None:
        dmax = manual_dmax
    if print_dmax:
        print(f"dmax = {dmax}")

    if show_axis:
        origin_size = kwargs.get('axis_origin_size', 0.01)
        axis_radius = kwargs.get('axis_radius', 0.004)
        axis_length = dmax * 0.6
        _axis = trimesh.creation.axis(
            origin_size=origin_size, axis_radius=axis_radius, axis_length=axis_length)
        _mesh_data.append(_init_trimesh_no_facecolors(_axis))

    # Centering
    if centering:
        _mesh_data = [
            m.apply_translation([-xc, -yc, -zc]) for m in _mesh_data
        ]

    # Scaling
    _mesh_data = [
        m.apply_scale(1./dmax) for m in _mesh_data
    ]

    if direction == '+z':
        transf_mat = np.eye(4)  # nothing
    elif direction == '-z':
        transf_mat = rotmat_Ry(180, to_homo=True)
    elif direction == '+x':
        transf_mat = rotmat_Ry(-90, to_homo=True)
    elif direction == '-x':
        transf_mat = rotmat_Ry(+90, to_homo=True)
    elif direction == '+y':
        transf_mat = rotmat_Rx(+90, to_homo=True)
    elif direction == '-y':
        transf_mat = rotmat_Rx(-90, to_homo=True)
    else:
        raise ValueError("direction not understood.")
    _mesh_data = [
        m.apply_transform(transf_mat) for m in _mesh_data
    ]

    # Put the mesh to a large z location to approximate weak-perspective projection
    large_z = 20  # can be arbitrary large value >> 1
    _mesh_data = [
        m.apply_translation([0, 0, large_z]) for m in _mesh_data
    ]

    fx_ndc = fy_ndc = 2*large_z / (1+pad)
    fx = fx_ndc * (image_size / 2)
    fy = fy_ndc * (image_size / 2)
    cx = cy = image_size / 2

    # The callee required image value range from 0 to 1
    image = np.ones([image_size, image_size, 3], dtype=np.float32)

    return pr_perspective_projection(
        mesh_data=_mesh_data,
        cam_f=(fx, fy),
        cam_p=(cx, cy),
        coor_sys=coor_sys,
        image=image,
        **kwargs
    )