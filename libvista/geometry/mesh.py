from __future__ import annotations
from typing import Union, List, Any, TYPE_CHECKING
try:  # Python 3.10+
    from typing import TypeAlias
except ImportError:  # Python < 3.10
    from typing_extensions import TypeAlias
from collections.abc import Sized

import numpy as np
import torch
from trimesh import Trimesh

from libvista.numeric import numpize
from . import _HAS_PYTORCH3D

# This block is ONLY read by static type checkers.
if TYPE_CHECKING:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex

if _HAS_PYTORCH3D:
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes
else:
    TexturesVertex = Any  # runtime placeholder so annotations still evaluate
    Meshes = Any  # runtime placeholder so annotations still evaluate


_COLORS = dict(
    light_blue=(0.65, 0.74, 0.86),
    yellow=(.85, .85, 0),
    grey=(0.8, 0.8, 0.8),
    red=(1.0, 0, 0),
    green=(0, 1.0, 0),
    purple=(0.63, 0.12, 0.94),
)


def _drop_dim0(tensor):
    if not hasattr(tensor, 'shape'):
        return tensor
    if len(tensor.shape) == 2:
        return tensor
    elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
        return tensor[0]
    else:
        raise ValueError("Input shape must be (1, ?, ?) or (?, ?)")


class SimpleMesh(Trimesh):

    """
    A unoptimized wrapper class that simplifies the conversion to pytorch3d.Meshes.

    When the underlying pytorch3d.Meshes is need,
    one should call self.synced_mesh to generate the pytorch3d.Meshes dynamically.

    self.synced_mesh will always be on CUDA.

    """

    def __init__(self,
                 verts: Union[np.ndarray, torch.Tensor],
                 faces: Union[np.ndarray, torch.Tensor],
                 process=False,
                 tex_color='light_blue',
                 vertex_colors_to_keep=None,
                 device='cuda'):
        """
        Args:
            verts: (V, 3) float32
            faces: (F, 3) int
            process : bool
                if True, Nan and Inf values will be removed
                immediately and vertices will be merged
            device: 'cpu' or 'cuda'. Used for self.synced_mesh
        """
        verts = numpize(_drop_dim0(verts))
        faces = numpize(_drop_dim0(faces))
        self.device = device

        if isinstance(tex_color, str) and tex_color in _COLORS:
            self.tex_color = _COLORS[tex_color]
        elif isinstance(tex_color, Sized) and len(tex_color) >= 3 \
            and (0.0 <= tex_color[0] <= 1.0):
            self.tex_color = tex_color
        elif tex_color is None and vertex_colors_to_keep is not None:
            # will not throw error in this case
            pass
        else:
            raise ValueError(f"tex_color {tex_color} not understood.")

        if vertex_colors_to_keep is not None:
            vertex_colors = vertex_colors_to_keep
        else:
            vertex_colors = (np.ones([len(verts), len(self.tex_color)]) * \
                self.tex_color * 255).astype(np.uint8)

        super().__init__(
            vertices=verts,
            faces=faces,
            process=process,
            vertex_colors=vertex_colors)

    @classmethod
    def from_trimesh(cls, mesh: Trimesh) -> SimpleMesh:
        """ This preserves vertex colors and/or face colors if any.
        """
        vertex_colors = mesh.visual.vertex_colors if mesh.visual else None
        return cls(
            verts=mesh.vertices,
            faces=mesh.faces,
            tex_color=None,
            vertex_colors_to_keep=vertex_colors
        )

    def as_trimesh(self):
        return super().copy()

    def copy(self):
        copied = super().copy()
        return SimpleMesh(
            copied.vertices, copied.faces, tex_color=None,
            vertex_colors_to_keep=copied.visual.vertex_colors)

    @property
    def synced_mesh(self):
        if not _HAS_PYTORCH3D:
            raise ImportError(
                "The '.synced_mesh' property requires PyTorch3D. "
                "Please install `pytorch3d`.")

        device = self.device
        verts = torch.as_tensor(self.vertices, device=device, dtype=torch.float32)
        faces = torch.as_tensor(self.faces, device=device)
        verts_rgb = torch.as_tensor(
            self.visual.vertex_colors[:, :3],
            dtype=torch.float32, device=device) / 255.
        textures = TexturesVertex(verts_features=verts_rgb[None].to(device))
        return Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )

    def apply_translation_(self, translation):
        """ Copied from Trimesh

        In-place translate the current mesh.

        Parameters
        ----------
        translation : (3,) float
          Translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape == (2,):
            raise NotImplementedError
        elif translation.shape != (3,):
            raise ValueError('Translation must be (3,) or (2,)!')

        # manually create a translation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        self.apply_transform(matrix)
        return self

    def apply_translation(self, translation):
        """
        Args:
            translation : (3,) float
                Translation in XYZ

        Returns:
            a copy of translated mesh.
        """
        out = self.copy()
        out.apply_translation_(translation)
        return out


AnyMesh: TypeAlias = Union[
    SimpleMesh, "Meshes", Trimesh,
    List[SimpleMesh], List["Meshes"], List[Trimesh]]
