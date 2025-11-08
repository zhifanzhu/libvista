from setuptools import find_packages, setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

setup(
    name='libvista',
    author="Zhifan Zhu",
    version="0.2.0",
    author_email="zhifan.zhu@bristol.ac.uk",
    packages=find_packages(exclude=("tests",)),
    # package_dir = {'libvista': 'libvista'},
    install_requires=[
        'numpy>=1.16.3,<2.0',
        'trimesh>=3.10.2',
        'matplotlib>=2.1.0',
        'pyrender>=0.1.45',
        'pillow>=6.0.0',
        # 'pytorch3d==0.6.2', suggested
    ],
)
