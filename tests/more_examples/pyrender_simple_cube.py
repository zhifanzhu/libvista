""" temp code to migrate geometry module to use pyrender instead of pytorch3d """


import numpy as np
import trimesh
import pyrender

from PIL import Image
import matplotlib.pyplot as plt
import os

from libvista.geometry import pyrender_call
from libvista.geometry import CameraManager, SimpleMesh
from libvista.geometry.example_meshes import canonical_cuboids

# set EGL is not set by environment variables
os.environ['PYOPENGL_PLATFORM'] = os.environ.get('PYOPENGL_PLATFORM', 'egl')


cube_1 = canonical_cuboids(
    x=0.5, y=0, z=10.25,
    w=0.5, h=0.5, d=0.5,
    convention='pytorch3d'
)
mesh = SimpleMesh(verts=cube_1.vertices, faces=cube_1.faces, tex_color='yellow')
vertex_colors = mesh.visual.vertex_colors.copy()
vertex_colors[:1, :] = np.array([255, 0, 0, 255])  # make one vertex red
mesh.visual.vertex_colors = vertex_colors

# mesh.fix_normals()
mat = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # white surface
    metallicFactor=0.0,                    # dielectric, not metallic
    roughnessFactor=0.2,                   # smooth (shininess ≈ 64)
    emissiveFactor=[0, 0, 0]         # no self-emission
)


mesh = pyrender.Mesh.from_trimesh(
    mesh,  smooth=True,
)
for prim in mesh.primitives:
    prim.material = mat
print(mesh.primitives, len(mesh.primitives))


# scene = pyrender.Scene()
# scene.add(mesh)

H, W = 200, 400
cam_params = dict(
    fx=10, fy=20, cx=0, cy=0,
    img_h=H, img_w=W, in_ndc=True)
global_cam = CameraManager(**cam_params)
fx, fy, cx, cy, img_h, img_w = global_cam.unpack()
camera = pyrender.IntrinsicsCamera(
    fx=fx, fy=fy,
    cx=cx, cy=cy,
    znear=0.01, zfar=1000.0)
camera_pose = np.float32([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])  # because camera is looking into -z direction w.r.t. the camera coord, hence we need to rotate around y axis

light = pyrender.DirectionalLight(
    color=np.ones(3), intensity=3.0
)

color, depth = pyrender_call(
    meshes=[mesh],
    cam=(camera, camera_pose),
    lights=[(light, camera_pose)],
    size=(img_w, img_h)
)

Image.fromarray((color).astype(np.uint8)).save('quick_cube.png')