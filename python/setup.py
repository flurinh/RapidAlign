from setuptools import setup
import os
import torch
import sys
import subprocess
import re

# Override PyTorch's CUDA check to allow building with mismatched CUDA versions
# This is useful when you have PyTorch compiled with one CUDA version, but
# system has a different CUDA version installed
os.environ['TORCH_ALLOW_CUDA_VERSION_MISMATCH'] = '1'

# Set up paths to source files
cuda_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Import CUDA extension classes
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

# Detect CUDA version
def get_cuda_version():
    """Detect CUDA version from nvcc or environment variables"""
    try:
        if CUDA_HOME:
            # Try using nvcc to get version
            cuda_version_raw = subprocess.check_output(
                [os.path.join(CUDA_HOME, 'bin', 'nvcc'), '--version']
            ).decode('utf-8')
            
            # Extract version from nvcc output
            version_match = re.search(r'release (\d+\.\d+)', cuda_version_raw)
            if version_match:
                return version_match.group(1)
        
        # Fallback: try nvidia-smi
        try:
            nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi_output)
            if version_match:
                return version_match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        # Fallback: Use torch.version.cuda
        if torch.version.cuda:
            return torch.version.cuda
    
    except (subprocess.SubprocessError, FileNotFoundError, AttributeError):
        pass
    
    # Default to 11.0 if detection fails
    print("Warning: Could not detect CUDA version. Defaulting to 11.0")
    return "11.0"

# Create a custom BuildExtension class that overrides CUDA version check
class BuildExtension(_BuildExtension):
    """Custom build extension that bypasses CUDA version check"""
    def build_extensions(self):
        # Skip CUDA version check in torch.utils.cpp_extension
        if hasattr(self, '_check_cuda_version'):
            self._check_cuda_version = lambda *args, **kwargs: None
        
        # Print a warning about version mismatch
        cuda_version = get_cuda_version()
        torch_cuda_version = torch.version.cuda
        if cuda_version and torch_cuda_version and cuda_version != torch_cuda_version:
            print(f"WARNING: CUDA version mismatch detected!")
            print(f"  - System CUDA version: {cuda_version}")
            print(f"  - PyTorch CUDA version: {torch_cuda_version}")
            print(f"  - Building anyway due to TORCH_ALLOW_CUDA_VERSION_MISMATCH=1")
            print(f"  - Note: This might lead to compatibility issues, but often works fine.")
        
        # Continue with regular build
        return super().build_extensions()

# Source files
sources = [
    'src/pybind.cpp',
    os.path.join(cuda_src_dir, 'batch_alignment.cu')
]

cuda_version = get_cuda_version()
print(f"Detected CUDA version: {cuda_version}")

# Define supported CUDA architectures based on CUDA version
def get_cuda_arch_flags():
    # Default architectures for all CUDA versions
    arch_flags = [
        '-gencode=arch=compute_60,code=sm_60',  # Pascal
        '-gencode=arch=compute_70,code=sm_70',  # Volta
        '-gencode=arch=compute_75,code=sm_75',  # Turing
    ]
    
    # Add Ampere for CUDA 11+
    if float(cuda_version.split('.')[0]) >= 11:
        arch_flags.append('-gencode=arch=compute_80,code=sm_80')  # Ampere
    
    # Add Hopper for CUDA 12+
    if float(cuda_version.split('.')[0]) >= 12:
        arch_flags.append('-gencode=arch=compute_89,code=sm_89')  # Hopper (H100)
        arch_flags.append('-gencode=arch=compute_90,code=sm_90')  # Ada/Blackwell/Hopper Next
    
    return arch_flags

# Base compiler flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
]

# Add architecture flags
nvcc_flags.extend(get_cuda_arch_flags())

# Set up extra compile flags
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': nvcc_flags
}

# Additional include directories
include_dirs = [os.path.join(cuda_src_dir, 'include')]

# Add CUDA include directory if available
if CUDA_HOME:
    include_dirs.append(os.path.join(CUDA_HOME, 'include'))

# Set up the extension
setup(
    name='rapidalign',
    ext_modules=[
        CUDAExtension(
            name='rapidalign',
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.17.0',
    ],
    extras_require={
        'visualize': ['matplotlib>=3.0.0'],
        'pyg': ['torch-geometric>=2.0.0'],
    },
    python_requires='>=3.6',
    description='Fast batch point cloud and graph alignment for PyTorch with CUDA acceleration',
    author='RapidAlign Authors',
    author_email='hidberf@gmail.com',
    version='0.1.0',
    keywords=['deep learning', 'point cloud', 'alignment', 'icp', 'procrustes', 'chamfer', 'pytorch', 'cuda', 'graph neural networks'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)