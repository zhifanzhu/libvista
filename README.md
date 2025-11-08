# LibZhifan

# Doc

Utility tools.

modules:
- *io*: txt json pickle reading/writing
- *odlib*: Visualization for object detection bounding boxes
- *epylab*: matplotlib extension
- *geometry*: 3D geometries, camera projections, mesh visualization.
    - The default renderer backend is `pyrender`.
    - The develelopment of this module started with `pytorch3d`, the aim is to reproduce the functionalities without using `pytorch3d` as default backend, which is hard to install e.g. on Isambard or Mac OS.
- *numeric*: Utilities for numeric libraries (numpy, pytorch)
- *cvtools*: video / frame header and subtitles. Mask overlay.

# Install

`pip install git+https://github.com/zhifanzhu/libzhifan.git`
