from __future__ import annotations
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import torch

from . import _HAS_NR, _HAS_PYTORCH3D
from .mesh import AnyMesh
from .camera_manager import CameraManager
from .projection_impl_naive import naive_perspective_projection
from .projection_impl_pyrender import (
    pr_perspective_projection, pr_project_standardized
)

if _HAS_NR:
    from .projection_impl_neural_renderer import neural_renderer_perspective_projection

if _HAS_PYTORCH3D:
    from .projection_impl_pytorch3d import *

"""
Dealing with vertices projections, possibly using pytorch3d

Pytorch3D has two Perspective Cameras:

- pytorch3d.render.FovPerspectiveCameras(znear, zfar, ar, fov)

- pytorch3d.render.PerspectiveCameras(focal_length, principal_point)
    - or pytorch3d.render.PerspectiveCameras(K: (4,4))


1. Coordinate System.

    In the following drawing, we fix the the camera to look into the screen,
    and show the coordinate system from the eye's perspective.

    We can read out the default camera direction as follows:
        +z (pytorch3d)
        -z (OpenGL, Open3D, PyRender, naive_implementation)
        +z (OpenCV, neural_renderer)

a. pytorch3d / pytorch3d-NDC

            ^ Y
            |
            |   / Z
            |  /
            | /
    X <------

    - this will be projected into:

            ^ Y
            |
            |
        <----
        X

b. OpenGL, Open3D, PyRender, naive_implementation:

            ^ Y              Y ^
            |                  |  / Z
            |                  | /
            |                  |/
            /------> X          ------> X
           /                     (NDC)
          /
       Z /

    - this will be projected into:

            ----> X
            |
            |
            v Y

c. OpenCV, neural_renderer:

             / Z
            /
           /
          /
         ----------> X
         |
         |
         |
         v Y


2. Projection transforms.

In pytorch3d, the transforms are as follows:
model -> view -> ndc -> screen
In pinhole camera model, i.e. with simple 3x3 matrix K, the transforms is:
model -> screen


3. Rendering configuration

To render a cube [-1, 1]^3, on a W x W = (200, 200) image

naive method:
    - fx=fy=cx=cy=W/2, image_size=(W, W)

pytorch3d in_ndc=True:
    - fx=fy=1, cx=cy=0, image_size=(W, W)

pytorch3d in_ndc=False:
    - fx=fy=cx=cy=W/2, image_size=(W, W)

neural_renderer.git:
    - fx=fy=cx=cy=W/2, image_size=W, orig_size=W
    or,
    - fx=fy=cx=cy=1/2, image_size=W, orig_size=1

`naive` == `pytorch3d in_ndc=False` == `neural_renderer w/ orig_size=image_size`


Ref:
[1] https://medium.com/maochinn/%E7%AD%86%E8%A8%98-camera-dee562610e71https://medium.com/maochinn/%E7%AD%86%E8%A8%98-camera-dee562610e71

"""


def perspective_projection(mesh_data: AnyMesh,
                           cam_f: Tuple[float],
                           cam_p: Tuple[float],
                           method=dict(
                               name='pytorch3d',
                               ),
                           image=None,
                           img_h=None,
                           img_w=None,
                           device='cuda',
                           **kwargs) -> NDArray:
    """ Project verts/mesh by Perspective camera.

    Args:
        mesh_data: one of
            - SimpleMesh
            - pytorch3d.Meshes
            - list of SimpleMeshes
            - list of pytorch3d.Meshes
        cam_f: focal length (2,)
        cam_p: principal points (2,)
        method: dict
            - name: one of {
                'naive', 
                'pyrender', 
                'pytorch3d', 'pytorch3d_instance', 
                'neural_renderer'}.

            - coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
                In the Camera by default located at (0, 0, 0) and looks at +z.

                Set the input coordinate system.
                - 'pytorch3d': render using pytorch3d coordinate system,
                    i.e. X-left, Y-top, Z-in
                - 'neural_renderer'/'nr':
                        X-right, Y-down, Z-in.

        in_ndc: bool
        R: (3, 3) camera extrinsic matrix.
        T: (3,) camera extrinsic matrix.
        image: (H, W, 3), if `image` is None,
            will render a image with size (img_h, img_w).

    Returns:
        (H, W, 3) image
    """
    method_name = method.pop('name')
    if image is None:
        assert img_h is not None and img_w is not None
        image = np.ones([img_h, img_w, 3], dtype=np.uint8) * 255

    if method_name == 'naive':
        return naive_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p, image=image,
            **method,
        )
    elif method_name == 'pyrender':
        image = np.asarray(image, dtype=np.uint8) / 255.  # for api consistency
        return pr_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p, image=image,
            **method,
        )
    elif method_name == 'pytorch3d':
        assert _HAS_PYTORCH3D, "Please install pytorch3d to use this function."
        image = torch.as_tensor(image, dtype=torch.float32) / 255.
        img = pytorch3d_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image, device=device, **kwargs
        )
        return img
    elif method_name == 'pytorch3d_silhouette':
        assert _HAS_PYTORCH3D, "Please install pytorch3d to use this function."
        image = torch.as_tensor(image, dtype=torch.float32) / 255.
        img = pth3d_silhouette_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image)
        return img
    elif method_name == 'pytorch3d_instance':
        assert _HAS_PYTORCH3D, "Please install pytorch3d to use this function."
        blur_radius = method.pop('blur_radius', 1e-7)
        img = pth3d_instance_perspective_projection(
            meshes=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, img_h=img_h, img_w=img_w, blur_radius=blur_radius)
        return img
    elif method_name == 'neural_renderer' or method_name == 'nr':
        assert _HAS_NR, "Please install neural_renderer to use this function."
        img = neural_renderer_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image
        )
        return img
    else:
        raise ValueError(f"method_name: {method_name} not understood.")


def perspective_projection_by_camera(mesh_data: AnyMesh,
                                     camera: CameraManager,
                                     method=dict(
                                         name='pytorch3d',
                                         in_ndc=False,
                                     ),
                                     image=None,
                                     device='cuda',
                                     **kwargs) -> NDArray[np.uint8]:
    """
    Similar to perspective_projection() but with CameraManager as argument.
    """
    fx, fy, cx, cy, img_h, img_w = camera.unpack()
    assert method.get('in_ndc', False) == False, "in_ndc Must be False for CamaraManager"
    img = perspective_projection(
        mesh_data,
        cam_f=(fx, fy),
        cam_p=(cx, cy),
        method=method.copy(),  # Avoid being optimized by python
        image=image,
        img_h=int(img_h),
        img_w=int(img_w),
        device=device,
        **kwargs,
    )
    return img


def project_standardized(mesh_data: AnyMesh,
                         direction='+z',
                         image_size=200,
                         pad=0.2,
                         method=dict(
                             name='pytorch3d',
                             coor_sys='nr'
                         ),
                         centering=True,
                         manual_dmax : float = None,
                         show_axis=False,
                         print_dmax=False,
                         device='cuda',
                         **kwargs) -> NDArray[np.uint8]:
    """
    Given any mesh(es), this function renders the zoom-in images.
    The meshes are proecessed to be in [-0.5, 0.5]^3 space,
    then a weak-perspective camera is applied.

    Args:
        direction: the camera will be looking at this direction
            e.g. '+z' means camera is looking at infinity along +z axis.
                in other words, '-z' is facing the camera.
        pad: the fraction to be padded around rendered image.
        manual_dmax: set dmax manually
        print_dmax: This helps determine manual_dmax
        centering: if True, look at (xc, yc, zc); otherwise, look at (0, 0, 0)
        **kwargs: other kwargs passed to projection function

    Returns:
        (H, W, 3)
    """
    method_name = method.get('name', 'pyrender')
    assert 'in_ndc' not in method, \
        "in_ndc should not be set in method for project_standardized."

    if method_name == 'pytorch3d':
        assert _HAS_PYTORCH3D, "Please install pytorch3d to use this function."
        return pth3d_project_standardized(
            mesh_data=mesh_data,
            direction=direction,
            image_size=image_size,
            pad=pad,
            **method,
            centering=centering,
            manual_dmax=manual_dmax,
            show_axis=show_axis,
            print_dmax=print_dmax,
            device=device,
            **kwargs
        )
    elif method_name == 'pyrender':
        return pr_project_standardized(
            mesh_data=mesh_data,
            direction=direction,
            image_size=image_size,
            pad=pad,
            **method,
            centering=centering,
            manual_dmax=manual_dmax,
            show_axis=show_axis,
            print_dmax=print_dmax,
            **kwargs
        )
    else:
        raise NotImplementedError(f"method {method_name} not implemented for project_standardized.")