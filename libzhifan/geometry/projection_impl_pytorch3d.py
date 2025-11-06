""" The projection implementations using PyTorch3D """
from typing import List, Union
from numpy.typing import NDArray
import numpy as np
import trimesh
import torch

from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                PerspectiveCameras, PointLights,
                                RasterizationSettings, SoftPhongShader,
                                SoftSilhouetteShader, TexturesVertex)
from pytorch3d.structures import Meshes, join_meshes_as_scene

from ..numeric import numpize
from .mesh import SimpleMesh, AnyMesh
from .camera_manager import CameraManager
from . import coor_utils_pytorch3d
from .instance_id_rendering import InstanceIDRenderer



_R = torch.eye(3)
_T = torch.zeros(3)


def _to_th_mesh(m: AnyMesh) -> Meshes:
    if isinstance(m, list):
        l = [v for v in m if v is not None]
        return join_meshes_as_scene(list(map(_to_th_mesh, l)))
    elif isinstance(m, Meshes):
        return m
    elif isinstance(m, SimpleMesh):
        return m.synced_mesh
    else:
        raise ValueError(f"type {type(m)} not understood.")


def pytorch3d_perspective_projection(mesh_data: AnyMesh,
                                     cam_f,
                                     cam_p,
                                     in_ndc: bool,
                                     coor_sys='pytorch3d',
                                     R=_R,
                                     T=_T,
                                     image=None,
                                     flip_canvas_xy=False,
                                     device='cuda',
                                     **kwargs) -> NDArray[np.uint8]:
    """
    TODO
    flip issue: https://github.com/facebookresearch/pytorch3d/issues/78

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]
        cam_f: Tuple, (2,)
        cam_p: Tuple, (2,)
        R: (3, 3)
        T: (3,)

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

        flip_canvas_xy: see flip issue. Note the issue doesn't happen
            if coor_sys == 'nr'
    """
    image_size = image.shape[:2]
    _mesh_data = _to_th_mesh(mesh_data)
    _mesh_data = _mesh_data.to(device)

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_transform_matrix(
            _mesh_data, _Rz_mat)
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )

    blend_params = kwargs.pop('blend_params', BlendParams())
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0, faces_per_pixel=1,
        bin_size=kwargs.get('bin_size', None))
    lights = PointLights(location=[[0, 0, 0]])
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(
        cameras=cameras, lights=lights, blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=rasterizer, shader=shader).to(device)

    rendered = renderer(_mesh_data)

    # Add background image
    if image is not None:
        image = image.to(device)
        frags = renderer.rasterizer(_mesh_data)
        is_bg = frags.pix_to_face[..., 0] < 0
        dst = rendered[..., :3]
        mask = is_bg[..., None].repeat(1, 1, 1, 3)
        out = dst.masked_scatter(
            mask, image[None][mask])
        out = numpize(out.squeeze_(0))
    else:
        out = numpize(rendered.squeeze_(0))[..., :3]
    out_img = (out * 255).astype(np.uint8)
    return out_img


def pth3d_silhouette_perspective_projection(mesh_data: AnyMesh,
                                            cam_f,
                                            cam_p,
                                            in_ndc: bool,
                                            coor_sys='pytorch3d',
                                            R=_R,
                                            T=_T,
                                            image=None,
                                            **kwargs) -> np.ndarray:
    """

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

    """
    device = 'cuda'
    image_size = image.shape[:2]
    _mesh_data = _to_th_mesh(mesh_data)
    _mesh_data = _mesh_data.to(device)

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_transform_matrix(
            _mesh_data, _Rz_mat)
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )
    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-9, gamma=1e-9)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=1,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    renderer = silhouette_renderer.to(device)

    # raster_settings = RasterizationSettings(
    #     image_size=image_size, blur_radius=0, faces_per_pixel=1)
    # lights = PointLights(location=[[0, 0, 0]])
    # rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # shader = SoftPhongShader(cameras=cameras, lights=lights)
    # renderer = MeshRenderer(
    #     rasterizer=rasterizer, shader=shader).to(device)

    rendered = renderer(_mesh_data)

    # Add background image
    out = numpize(rendered.squeeze())[..., -1]
    return out


def pth3d_instance_perspective_projection(meshes: List[Meshes],
                                          cam_f,
                                          cam_p,
                                          in_ndc: bool,
                                          img_h,
                                          img_w,
                                          coor_sys='pytorch3d',
                                          R=_R,
                                          T=_T,
                                          device='cuda',
                                          **kwargs) -> np.ndarray:
    """ Instance ID

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

    Returns:
        instance_id_mask: (H, W) int32
    """
    image_size = (img_h, img_w)
    assert type(meshes) == list, "Must be list of meshes"
    meshes = [_to_th_mesh(v).to(device) for v in meshes]

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        meshes = [
            coor_utils_pytorch3d.torch3d_apply_transform_matrix(v,_Rz_mat)
            for v in meshes]
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )
    blur_radius = kwargs.pop('blur_radius', 1e-7)
    max_faces_per_bin = kwargs.pop('max_faces_per_bin', None)
    bin_size = kwargs.pop('bin_size', None)
    renderer = InstanceIDRenderer(
        cameras=cameras, image_size=image_size, 
        blur_radius=blur_radius, max_faces_per_bin=max_faces_per_bin,
        bin_size=bin_size).to(device)
    mesh_to_id = kwargs.pop('mesh_to_id', None)  # default assignment is [1, 2, 3, ...]
    rendered = renderer(meshes, mesh_to_id=mesh_to_id)

    out = numpize(rendered)
    return out



def pth3d_project_standardized(mesh_data: AnyMesh,
                               direction='+z',
                               image_size=200,
                               pad=0.2,
                               in_ndc=False,
                               coor_sys='nr',
                               centering=True,
                               manual_dmax : float = None,
                               show_axis=False,
                               print_dmax=False,
                               device='cuda',
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
        **kwargs: other kwargs passed to projection function

    Returns:
        (H, W, 3)
    """
    _mesh_data = _to_th_mesh(mesh_data)
    xmin, ymin, zmin = torch.min(_mesh_data.verts_packed(), 0).values
    xmax, ymax, zmax = torch.max(_mesh_data.verts_packed(), 0).values
    xc, yc, zc = map(lambda x: x/2, (xmin+xmax, ymin+ymax, zmin+zmax))
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
    dmax = max(dx, max(dy, dz)).item()
    if manual_dmax is not None:
        dmax = manual_dmax
    if print_dmax:
        print(f"dmax = {dmax}")

    if show_axis:
        device = _mesh_data.device
        _axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=dmax * 0.6)

        _ax_verts = torch.as_tensor(_axis.vertices, device=device, dtype=torch.float32)
        _ax_faces = torch.as_tensor(_axis.faces, device=device)
        _ax_verts_rgb = torch.as_tensor(_axis.visual.vertex_colors[:, :3], device=device)
        _ax_verts_rgb = _ax_verts_rgb / 255.
        textures = TexturesVertex(verts_features=_ax_verts_rgb[None].to(device))
        _axis = Meshes(
            verts=[_ax_verts],
            faces=[_ax_faces],
            textures=textures)
        _mesh_data = _to_th_mesh([_mesh_data, _axis])

    large_z = 20  # can be arbitrary large value >> 1
    if centering:
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_translation(
            _mesh_data, (-xc, -yc, -zc))
    _mesh_data = coor_utils_pytorch3d.torch3d_apply_scale(_mesh_data, 1./dmax)
    if direction == '+z':
        pass  # Nothing need to be changed
    elif direction == '-z':
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_Ry(_mesh_data, 180)
    elif direction == '+x':
        # I guarantee you it's +90, not -90
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_Ry(_mesh_data, +90)
    elif direction == '-x':
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_Ry(_mesh_data, -90)
    elif direction == '+y':
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_Rx(_mesh_data, +90)
    elif direction == '-y':
        _mesh_data = coor_utils_pytorch3d.torch3d_apply_Rx(_mesh_data, -90)
    else:
        raise ValueError("direction not understood.")
    _mesh_data = coor_utils_pytorch3d.torch3d_apply_translation(
        _mesh_data, (0, 0, large_z))

    fx = fy = 2*large_z / (1+pad)
    image = np.ones([image_size, image_size, 3], dtype=np.uint8) * 255
    # camera = CameraManager(
    #     fx=fx, fy=fy,
    #     cx=0, cy=0, img_h=image_size, img_w=image_size,
    #     in_ndc=True,
    # )
    # return perspective_projection_by_camera(
    #     _mesh_data,
    #     camera,
    #     method=method,
    #     device=device,
    #     **kwargs)
    return pytorch3d_perspective_projection(
        mesh_data=_mesh_data,
        cam_f=(fx, fy),
        cam_p=(0, 0),
        in_ndc=in_ndc,
        coor_sys=coor_sys,
        image=image,
        device=device,
        **kwargs
    )  # TODO: unittest this function