""" Utility function for coordinate system. """

from typing import Union

import numpy as np
import torch

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Transform3d


""" Pytorch3d transforms
"""

def torch3d_get_verts(geom: Union[Meshes, Pointclouds]) -> torch.Tensor:
    if isinstance(geom, Meshes) or isinstance(geom, Meshes):
        view_points = geom.verts_padded()
    elif isinstance(geom, Pointclouds):
        view_points = geom.points_padded()
    elif isinstance(geom, torch.Tensor):
        view_points = geom
    else:
        raise NotImplementedError(type(geom))
    return view_points


def torch3d_apply_transform(
        geom: Union[Meshes, Pointclouds, torch.Tensor],
        trans: Transform3d):
    """
    Returns:
        tranformed geometry object.
    """
    verts = torch3d_get_verts(geom)
    verts = trans.transform_points(verts)
    if hasattr(geom, 'update_padded'):
        geom = geom.update_padded(verts)
    else:
        geom = verts
    return geom


def torch3d_apply_transform_matrix(
        geom: Union[Meshes, Pointclouds, torch.Tensor],
        trans,
        convert_trans_col_to_row=True):
    """
    Note: transformation is implemented as right-multiplication,
    hence geom is row-vector.

    Args:
        trans: transformation. Either
            - matrix of (4, 4)
            - matrix of (1, 4, 4)
            - Transform3d

        convert_trans_col_to_rw: bool, i.e. transpose

    Returns:
        tranformed geometry object.
    """
    if isinstance(trans, Transform3d):
        return torch3d_apply_transform(geom, trans)

    trans = torch.as_tensor(trans).reshape(1, 4, 4)
    if convert_trans_col_to_row:
        trans = Transform3d(
            matrix=trans.transpose(1, 2), device=geom.device)
    else:
        trans = Transform3d(
            matrix=trans, device=geom.device)

    return torch3d_apply_transform(geom, trans)


def torch3d_apply_scale(geom: Union[Meshes, Pointclouds, torch.Tensor],
                        scale: float):
    device = geom.device
    trans = torch.eye(4).reshape(1, 4, 4).to(device)
    trans[..., [0,1,2], [0,1,2]] = scale
    return torch3d_apply_transform_matrix(
        geom,
        Transform3d(matrix=trans, device=device),
        convert_trans_col_to_row=True)


def torch3d_apply_translation(geom: Union[Meshes, Pointclouds, torch.Tensor],
                              translation):
    """
    Args:
        translation: (3,)
    """
    device = geom.device
    trans = torch.eye(4).reshape(1, 4, 4).to(device)
    trans[..., :3, -1] = torch.as_tensor(translation, device=device)
    return torch3d_apply_transform_matrix(
        geom, trans, convert_trans_col_to_row=True)


def torch3d_apply_Rx(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Rx_mat = torch.as_tensor([[
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Rx_mat, convert_trans_col_to_row=True)


def torch3d_apply_Ry(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Ry_mat = torch.as_tensor([[
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Ry_mat, convert_trans_col_to_row=True)


def torch3d_apply_Rz(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Rz_mat = torch.as_tensor([[
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Rz_mat, convert_trans_col_to_row=True)


# def rot6d_to_matrix(rot_6d):
#     """
#     TODO, finalize
#     Convert 6D rotation representation to 3x3 rotation matrix.
#     Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
#     Networks", CVPR 2019

#     Args:
#         rot_6d (B x 6): Batch of 6D Rotation representation.

#     Returns:
#         Rotation matrices (B x 3 x 3).
#     """
#     rot_6d = rot_6d.view(-1, 3, 2)
#     a1 = rot_6d[:, :, 0]
#     a2 = rot_6d[:, :, 1]
#     b1 = F.normalize(a1)
#     b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
#     b3 = torch.cross(b1, b2)
#     return torch.stack((b1, b2, b3), dim=-1)


# def rotation_xyz_from_euler(x_rot, y_rot, z_rot):
#     rz = np.float32([
#         [np.cos(z_rot), np.sin(z_rot), 0],
#         [-np.sin(z_rot), np.cos(z_rot), 0],
#         [0, 0, 1]
#     ])
#     ry = np.float32([
#         [np.cos(y_rot), 0, -np.sin(y_rot)],
#         [0, 1, 0],
#         [np.sin(y_rot), 0, np.cos(y_rot)],
#     ])
#     rx = np.float32([
#         [1, 0, 0],
#         [0, np.cos(x_rot), np.sin(x_rot)],
#         [0, -np.sin(x_rot), np.cos(x_rot)],
#     ])
#     return rz @ ry @ rx