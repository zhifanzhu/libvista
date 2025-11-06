""" The projection implementations using Neural Renderer """
import torch
import neural_renderer as nr

from ..numeric import numpize
from .mesh import SimpleMesh

def neural_renderer_perspective_projection(mesh_data: SimpleMesh,
                                           cam_f,
                                           cam_p,
                                           R=torch.eye(3),
                                           T=torch.zeros(3),
                                           image=None,
                                           orig_size=None,
                                           device='cuda',
                                           **kwargs):
    """
    TODO(low priority): add image support, add texture render support.

    Args:
        orig_size: int or None.
            if None, orig_size will be set to image_size.
            It's recommended to keep it as None.
            See above "3." for explanation.
    """
    if isinstance(mesh_data, list):
        raise NotImplementedError

    verts = torch.as_tensor(
        mesh_data.vertices, device=device, dtype=torch.float32).unsqueeze(0)
    faces = torch.as_tensor(mesh_data.faces, device=device).unsqueeze(0)
    image_size = image.shape
    fx, fy = cam_f
    cx, cy = cam_p

    K = torch.as_tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
        ], dtype=torch.float32, device=device)
    K = K[None]
    R = torch.eye(3, device=device)[None]
    t = torch.zeros([1, 3], device=device)

    if orig_size is None:
        orig_size = image_size[0]
    renderer = nr.Renderer(
        image_size=image_size[0],
        K=K,
        R=R,
        t=t,
        orig_size=orig_size
    )

    img = renderer(
        verts,
        faces,
        mode='silhouettes'
    )
    return numpize(img)
