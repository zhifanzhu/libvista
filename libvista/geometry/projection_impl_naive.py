""" The naive projection implementations """
from .mesh import SimpleMesh
from . import coor_utils
from .visualize_2d import draw_dots_image


def naive_perspective_projection(mesh_data: SimpleMesh,
                                 cam_f,
                                 cam_p,
                                 image,
                                 color='green',
                                 thickness=4,
                                 **kwargs):
    """
    Given image size, naive calculation of K should be

    fx = cx = img_w/2, fy = cy = img_h/2

    """
    if isinstance(mesh_data, list):
        raise NotImplementedError

    verts = mesh_data.vertices
    fx, fy = cam_f
    cx, cy = cam_p
    fx, fy, cx, cy = map(float, (fx, fy, cx, cy))
    points = coor_utils.project_3d_2d(
        verts.T, fx=fx, fy=fy, cx=cx, cy=cy).T
    img = draw_dots_image(
        image, points, color=color, thickness=thickness)
    return img
