import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()

extension_sources = [
    ROOT / "csrc" / "bindings.cpp",
    ROOT / "csrc" / "kde_similarity.cu",
]

setup(
    name="rapidalign",
    version="0.2.0",
    author="RapidAlign Team",
    description="Differentiable KDE/MMD loss for point clouds",
    packages=find_packages(include=["rapidalign", "rapidalign.*"]),
    ext_modules=[
        CUDAExtension(
            name="rapidalign._cuda",
            sources=[str(src) for src in extension_sources],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "-gencode", "arch=compute_60,code=sm_60",
                    "-gencode", "arch=compute_70,code=sm_70",
                    "-gencode", "arch=compute_75,code=sm_75",
                    "-gencode", "arch=compute_80,code=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.9",
)
