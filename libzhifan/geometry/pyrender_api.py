""" Create the renderer on first use and reuse it. Clean up automatically on exit. """

import atexit
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import pyrender

# Cache renderers by (width, height) so we can call with different sizes
_RENDERERS: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

DEFAULT_FLAGS = pyrender.RenderFlags.RGBA \
    | pyrender.RenderFlags.SHADOWS_DIRECTIONAL


def _get_renderer(size: tuple[int, int]) -> pyrender.OffscreenRenderer:
    if size not in _RENDERERS:
        _RENDERERS[size] = pyrender.OffscreenRenderer(*size)
    return _RENDERERS[size]


@atexit.register
def _cleanup_renderers():
    for r in _RENDERERS.values():
        try:
            r.delete()
        except Exception:
            pass
    _RENDERERS.clear()


def pyrender_call(meshes: List[pyrender.Mesh], 
                  cam: tuple[pyrender.Camera, np.ndarray],
                  lights: list[tuple[pyrender.Light, np.ndarray]]|None=None,
                  size: tuple[int, int]=(400, 400),
                  flags: pyrender.RenderFlags = DEFAULT_FLAGS,
                  ambient_light=False,
                  ) -> tuple[NDArray, NDArray]:
    """
    Render a single frame. You don't manage the OffscreenRenderer.

    Args:
        meshes: List[pyrender.Mesh]
        cam:  (camera_obj, camera_pose)
        lights: [(light_obj, light_pose), ...]
        size: (w, h)
    
    Returns:
        color: (H, W, 4) RGBA image
        depth: (H, W) depth image
    """
    if ambient_light:
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    else:
        scene = pyrender.Scene()
    for mesh in meshes:
        scene.add(mesh)
    camera_obj, camera_pose = cam
    scene.add(camera_obj, pose=camera_pose)

    if lights:
        for light_obj, light_pose in lights:
            scene.add(light_obj, pose=light_pose)

    r = _get_renderer(size)
    color, depth = r.render(scene, flags=flags)
    return color, depth