from importlib.util import find_spec

_HAS_PYTORCH3D = find_spec("pytorch3d") is not None
_HAS_OPEN3D = find_spec("open3d") is not None
_HAS_NR = find_spec("neural_renderer") is not None

from .coor_utils import *
from .example_meshes import *
from .mesh import *
from .visualize import *
from .visualize_2d import *
from .camera_manager import *
from .pyrender_api import pyrender_call
from .projection import *
from .projection_impl_pyrender import *

if _HAS_PYTORCH3D:
    from .instance_id_rendering import InstanceIDRenderer
    from .projection_impl_pytorch3d import *

if _HAS_OPEN3D:
    from .open3d_utils import *

if _HAS_NR:
    from .projection_impl_neural_renderer import (
        neural_renderer_perspective_projection
    )