import unittest

import os
import numpy as np
from PIL import Image
from libvista.geometry import example_meshes
from libvista.geometry import perspective_projection_by_camera
from libvista.geometry import CameraManager, BatchCameraManager


output_dir = 'tests/geometry/outputs'
expected_dir = 'tests/geometry/outputs_expected'


class CubePyRenderTest(unittest.TestCase):

    def test_crop_and_resize_pyrender(self):
        H, W = 200, 400

        global_cam = CameraManager(
            fx=10, fy=20, cx=0, cy=0, img_h=H, img_w=W,
            in_ndc=True)

        H1, W1 = 200, 200
        local_box_1 = np.asarray([0, 0, H1, W1]) # xywh
        local_cam_1_exp = CameraManager(
            fx=20, fy=20, cx=1, cy=0, img_h=H1, img_w=W1,
            in_ndc=True)

        H2, W2 = 100, 100
        local_box_2 = np.asarray([200, 100, H2, W2]) # xywh
        local_cam_2_exp = CameraManager(
            fx=40, fy=40, cx=-1, cy=-1, img_h=H2, img_w=W2,
            in_ndc=True)

        cube_1 = example_meshes.canonical_cuboids(
            x=0.5, y=0, z=10.25,
            w=0.5, h=0.5, d=0.5,
            convention='pytorch3d'
        )
        cube_2 = example_meshes.canonical_cuboids(
            x=-0.375, y=-0.125, z=10.125,
            w=0.25, h=0.25, d=0.25,
            convention='pytorch3d'
        )

        np.testing.assert_allclose(
            local_cam_1_exp.get_K(),
            global_cam.crop(local_box_1).get_K())
        np.testing.assert_allclose(
            local_cam_2_exp.get_K(),
            global_cam.crop(local_box_2).get_K())
        
        """ image rendered by Local camera 1 
        x=0.5 => x_pix=100
        x=0.75 => x_pix=150 (=50 after flip)
        """
        method = dict(
            name='pyrender',
            in_ndc=False,
            coor_sys='pytorch3d',
        )

        img_global = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam, method=method)
        img_1 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_1),
            method=method)
        img_2 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_2), method=method)

        def save_img(name, rend):
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(rend).save(
                f'{output_dir}/{name}.png')

        save_img('pyrender_test_crop_and_resize_global', img_global)
        save_img('pyrender_test_crop_and_resize_local_1', img_1)
        save_img('pyrender_test_crop_and_resize_local_2', img_2)

        # def compare_to_expected(name, rend):
        #     expected_path = f'tests/geometry/outputs_expected/{name}.png'
        #     expected = np.asarray(Image.open(expected_path))
        #     np.testing.assert_allclose(rend, expected, atol=2)
        
        # compare_to_expected('test_crop_and_resize_global', img_global)
        # compare_to_expected('test_crop_and_resize_local_1', img_1)
        # compare_to_expected('test_crop_and_resize_local_2', img_2)

if __name__ == '__main__':
    unittest.main()
        