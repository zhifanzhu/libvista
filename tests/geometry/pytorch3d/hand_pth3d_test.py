import unittest
import os
import trimesh
from PIL import Image
import numpy as np

from libvista.geometry import projection, SimpleMesh, CameraManager

mesh_path = 'tests/examples/P01_01_42273_right_hand.obj'
img_path = 'tests/examples/P01_01_42273.png'

pytorch3d_output_path = 'tests/geometry/outputs/pytorch3d_hand_projection.png'

class HandProjectionTest(unittest.TestCase):

    def test_hand_projection(self):
        cam_params = dict(
            fx=5000, fy=5000, cx=228, cy=128,
            img_h=256, img_w=456, in_ndc=False)
        global_cam = CameraManager(**cam_params)

        mesh = trimesh.load_mesh(mesh_path)
        mesh = SimpleMesh(mesh.vertices, mesh.faces, tex_color='purple')
        img_pil = Image.open(img_path)

        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))

        os.makedirs('tests/geometry/outputs', exist_ok=True)
        Image.fromarray(rend).save(pytorch3d_output_path)


if __name__ == '__main__':
    unittest.main()