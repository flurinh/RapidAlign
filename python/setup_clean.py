"""
Clean setup.py for RapidAlign
Builds the CUDA extension with modular kernel files
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building with CUDA {cuda_version}")

# Source files for the CUDA extension
cuda_sources = [
    'rapidalign/cuda_extension.cpp',
    'cuda_kernels/common.cu',
    'cuda_kernels/procrustes.cu', 
    'cuda_kernels/icp.cu',
    'cuda_kernels/dpcr.cu',
    'cuda_kernels/chamfer.cu'
]

# CUDA extension
ext_modules = [
    CUDAExtension(
        name='rapidalign_cuda',
        sources=cuda_sources,
        include_dirs=[
            'cuda_kernels',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '-std=c++14',
                '--expt-relaxed-constexpr',
                '-arch=sm_50',
                '-gencode=arch=compute_50,code=sm_50',
                '-gencode=arch=compute_60,code=sm_60', 
                '-gencode=arch=compute_70,code=sm_70',
                '-gencode=arch=compute_75,code=sm_75',
                '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86',
                '-gencode=arch=compute_80,code=compute_80',
            ]
        },
        libraries=['cublas', 'cusolver', 'curand'],
        language='c++'
    )
]

# Simple description
long_description = "Fast batch point cloud and graph alignment for PyTorch with CUDA acceleration"

setup(
    name="rapidalign",
    version="0.1.0",
    author="RapidAlign Team",
    description="Fast batch point cloud and graph alignment for PyTorch with CUDA acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False, max_jobs=4)
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "torch-geometric>=2.0.0",
            "matplotlib>=3.5.0", 
            "pandas>=1.3.0",
            "tabulate>=0.8.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "torch-geometric>=2.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)