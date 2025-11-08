import unittest

import os
import trimesh
import numpy as np
from libvista.geometry import projection, SimpleMesh
from libvista.geometry import example_meshes
from PIL import Image

mesh_path = 'tests/examples/P01_01_42273_right_hand.obj'

axis_out_fmt = 'tests/geometry/outputs/pytorch3d_axis_standardized.{coor_sys}.png'
cube_out_fmt = 'tests/geometry/outputs/pytorch3d_cube_standardized.{coor_sys}.png'
hand_out_fmt = 'tests/geometry/outputs/pytorch3d_hand_standardized.{coor_sys}.png'
multi_hand_out_fmt = 'tests/geometry/outputs/pytorch3d_multihand_standardized.{coor_sys}.png'

class P3DProjectionStandardizedTest(unittest.TestCase):

    # TODO: check why border is slightly different
    def do_projection(self, mesh_data, coor_sys, out_path):
        outs = []
        method = dict(
            name='pytorch3d',
            coor_sys=coor_sys,
        )
        for direction in ['+z', '-z', '+x', '-x', '+y', '-y']:
            out = projection.project_standardized(
                mesh_data=mesh_data,
                direction=direction,
                image_size=200,
                pad=0.2,
                method=method,
                centering=True,
                show_axis=True,
                print_dmax=False,
            )
            outs.append(out)
        
        out = np.vstack(outs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(out).save(out_path)

    def test_axis_projection(self):
        
        axis_mesh = trimesh.creation.axis(
            origin_size=0.1, axis_radius=0.01, axis_length=0.6)
        axis_mesh = SimpleMesh.from_trimesh(axis_mesh)
        self.do_projection(axis_mesh, 'pytorch3d', axis_out_fmt.format(coor_sys='pytorch3d'))
        # 'pyrender' expected to be empty for pytorch3d implementation
        self.do_projection(axis_mesh, 'pyrender', axis_out_fmt.format(coor_sys='pyrender'))
        self.do_projection(axis_mesh, 'nr', axis_out_fmt.format(coor_sys='nr'))

    def test_cube_projection(self):

        cube_1 = example_meshes.canonical_cuboids(
            x=0.5, y=0, z=10.25,
            w=0.5, h=0.5, d=0.5,
            convention='pytorch3d'
        )
        mesh = SimpleMesh(cube_1.vertices, cube_1.faces, tex_color='purple')
        vertex_color = mesh.visual.vertex_colors.copy()
        vertex_color[0, :] = [0, 255, 0, 255]
        mesh.visual.vertex_colors = vertex_color

        self.do_projection(mesh, 'pytorch3d', cube_out_fmt.format(coor_sys='pytorch3d'))
        self.do_projection(mesh, 'pyrender', cube_out_fmt.format(coor_sys='pyrender'))
        self.do_projection(mesh, 'nr', cube_out_fmt.format(coor_sys='nr'))

    def test_hand_projection(self):
        mesh = trimesh.load_mesh(mesh_path)
        mesh = SimpleMesh(mesh.vertices, mesh.faces, tex_color='purple')
        self.do_projection(mesh, 'pytorch3d', hand_out_fmt.format(coor_sys='pytorch3d'))
        self.do_projection(mesh, 'pyrender', hand_out_fmt.format(coor_sys='pyrender'))
        self.do_projection(mesh, 'nr', hand_out_fmt.format(coor_sys='nr'))

    def test_multiple_hand_projection(self):
        mesh = trimesh.load_mesh(mesh_path)
        mesh = SimpleMesh(mesh.vertices, mesh.faces, tex_color='purple')

        v_offset = np.array([0.05, 0.05, 0.1]) \
            + np.random.randn(*mesh.vertices.shape) * 2e-3
        mesh2 = SimpleMesh(
            mesh.vertices + v_offset, mesh.faces, tex_color='yellow')
        multi_mesh = [mesh, mesh2]
        self.do_projection(multi_mesh, 'pytorch3d', multi_hand_out_fmt.format(coor_sys='pytorch3d'))
        self.do_projection(multi_mesh, 'pyrender', multi_hand_out_fmt.format(coor_sys='pyrender'))
        self.do_projection(multi_mesh, 'nr', multi_hand_out_fmt.format(coor_sys='nr'))
    

if __name__ == '__main__':
    unittest.main()