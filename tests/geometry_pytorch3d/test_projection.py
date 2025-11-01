import trimesh
from PIL import Image
import numpy as np
from libzhifan.geometry_pytorch3d import projection, SimpleMesh, CameraManager


mesh_path = 'tests/data/P01_01_42273_right_hand.obj'
img_path = 'tests/data/P01_01_42273.png'

cam_params = dict(
    fx=5000, fy=5000, cx=228, cy=128,
    img_h=256, img_w=456, in_ndc=False)
global_cam = CameraManager(**cam_params)

mesh = trimesh.load_mesh(mesh_path)
mesh = SimpleMesh(mesh.vertices, mesh.faces)
img_pil = Image.open(img_path)

rend = projection.perspective_projection_by_camera(
    mesh, global_cam,
    method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
    image=np.asarray(img_pil))

Image.fromarray((rend * 255).astype(np.uint8)).save('tests/geometry_pytorch3d/test_projection.png')