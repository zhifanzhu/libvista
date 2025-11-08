# LibVista

Previously [libzhifan](https://github.com/zhifanzhu/libzhifan), `libvista` is a collection of utility tools for 3D computer vision.

Aims to provide easy-to-use 3D visualization and rendering functionalities for research prototyping.

Example:

```python
from libvista.geometry import projection, SimpleMesh, CameraManager

global_cam = CameraManager(
    fx=5000, fy=5000, cx=228, cy=128,
    img_h=256, img_w=456, in_ndc=False)

mesh = trimesh.load_mesh(mesh_path)
mesh = SimpleMesh(mesh.vertices, mesh.faces, tex_color='purple')

img_pil = Image.open(img_path)

rend = projection.perspective_projection_by_camera(
    mesh, global_cam,
    method=dict(name='pyrender', coor_sys='nr', in_ndc=False),
    image=np.asarray(img_pil))

Image.fromarray(rend).save(pyrender_output_path)
```

Example output:

![pyrender_output](tests/geometry/outputs_expected/pyrender_hand_projection.png)

# Doc

Utility tools.

modules:
- *io*: txt json pickle reading/writing
- *odlib*: Visualization for object detection bounding boxes
- *epylab*: matplotlib extension
- *geometry*: 3D geometries, camera projections, mesh visualization.
    - The default renderer backend is `pyrender`.
    - `pytorch3d` is also supported if installed.
    - The develelopment of this module started with `pytorch3d` in [libzhifan](https://github.com/zhifanzhu/libzhifan); but `libvista` makes `pytorch3d` an optional backend, since `pytorch3d` is hard to install e.g. on Isambard or Mac OS.
- *numeric*: Utilities for numeric libraries (numpy, pytorch)
- *cvtools*: video / frame header and subtitles. Mask overlay.

# Install

Best practice is to specify a commit-hash or tag:

`pip install git+https://github.com/zhifanzhu/libvista.git@v0.2-2025`
