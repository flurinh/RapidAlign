from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set up paths to source files
cuda_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sources = [
    'src/pybind.cpp',
    os.path.join(cuda_src_dir, 'batch_alignment.cu')
]

# Set up extra compile flags
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80'
    ]
}

# Set up the extension
setup(
    name='rapidalign',
    ext_modules=[
        CUDAExtension(
            name='rapidalign',
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[os.path.join(cuda_src_dir, 'include')]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.7.0',
    ],
    python_requires='>=3.6',
    description='Batched Point Cloud Alignment for Graph Neural Networks using CUDA',
    author='Claude AI',
    author_email='noreply@anthropic.com',
    version='0.1.0',
    keywords=['deep learning', 'point cloud', 'alignment', 'icp', 'procrustes', 'chamfer', 'pytorch', 'cuda'],
)