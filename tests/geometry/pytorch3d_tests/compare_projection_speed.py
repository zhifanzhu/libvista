"""
Comparing projection speed between pytorch3d and pyrenderer

Conclusion: 
For fix size image, in sequential rendering,
both render ~7 seconds for 5000 projections of a hand mesh.
"""

import trimesh
from PIL import Image
import numpy as np
import tqdm

from libvista.geometry import projection, SimpleMesh, CameraManager

mesh_path = 'tests/examples/P01_01_42273_right_hand.obj'
img_path = 'tests/examples/P01_01_42273.png'

pytorch3d_output_path = 'tests/geometry/outputs/pytorch3d_hand_projection.png'
pyrender_output_path = 'tests/geometry/outputs/pyrender_hand_projection.png'

def test_hand_projection():
    cam_params = dict(
        fx=5000, fy=5000, cx=228, cy=128,
        img_h=256, img_w=456, in_ndc=False)
    global_cam = CameraManager(**cam_params)

    img_pil = Image.open(img_path)

    mesh = trimesh.load_mesh(mesh_path)
    mesh = SimpleMesh(mesh.vertices, mesh.faces, tex_color='purple')
    meshes = []
    # slighly alter mesh to avoid caching effects
    for i in range(3000):
        v_offset = np.random.randn(*mesh.vertices.shape) * 1e-6
        meshes.append(SimpleMesh(mesh.vertices + v_offset, mesh.faces, tex_color='purple'))

    for i in tqdm.tqdm(range(len(meshes)), desc='pyrender projection'):
        rend = projection.perspective_projection_by_camera(
            meshes[i], global_cam,
            method=dict(name='pyrender', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        rend = None  # free memory

    for i in tqdm.tqdm(range(len(meshes)), desc='pytorch3d projection'):
        rend = projection.perspective_projection_by_camera(
            meshes[i], global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        rend = None  # free memory



if __name__ == '__main__':
    test_hand_projection()