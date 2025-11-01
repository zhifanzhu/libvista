import trimesh 

from libzhifan import geometry_pytorch3d
from libzhifan.geometry_pytorch3d import SimpleMesh

verts, faces = geometry_pytorch3d.canonical_cuboids(
    x=0, y=0, z=3,
    w=2, h=2, d=2,
    convention='opencv',
    return_mesh=False
)


mesh = SimpleMesh(verts, faces)
scene = trimesh.scene.Scene([mesh])
a = mesh.synced_mesh

# pcd = SimplePCD(verts)
# scene = trimesh.scene.Scene([pcd])