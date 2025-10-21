# Repository Guidelines

## Project Structure & Module Organization
Core C++ prototypes sit in `main.cpp`, `visualize_graph_alignment.cpp`, and CUDA entry points `pointcloud_alignment.cu` / `batch_alignment.cu`, all driven by the root `CMakeLists.txt`. The `python/rapidalign` package wraps the GPU bindings, while design notes and references live in `python/docs/`, `docs/`, and `ref/`. Benchmarks, utilities, and archived experiments in `python/benchmarks/`, `dev/`, and `deprecated/` provide context; treat them as read-only unless you refresh datasets.

## Build, Test, and Development Commands
Configure C++ targets via `cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`, then rebuild with `cmake --build build -j`. Use `./build/CPUTest`, `./build/MyCudaProject`, or `./build/BatchedAlign` for quick checks. The Python extension expects the `rapidalign` conda environment; run `bash python/build.sh` for a clean in-place rebuild or `pip install -e python` during iterative development. CUDA-specific executables can also be compiled through `python/Makefile` targets when testing kernels in isolation.

## Coding Style & Naming Conventions
Maintain 4-space indentation, brace-on-same-line for C++, and prefer `snake_case` for files, functions, and CUDA kernels (`batched_dpcr.cu`). Python modules follow PEP 8 and type-aware tensor naming (`src_points`, `batch_idx`). Keep kernel launches near helper utilities and document custom memory layouts with concise comments aligned with the tone in `core.py` and `main.cpp`.

## Testing Guidelines
Python API tests live in `python/tests/python`; run them with `cd python/tests/python && python -m pytest`. CUDA validation sits in `python/tests/cuda`; compile with `make all` and execute `./run_all_tests.sh` on a compatible GPU. Lightweight import checks (`python/test_import.py`) should pass before opening a PR, and stress benchmarks in `python/benchmarks/` are recommended when touching performance-critical kernels.

## Commit & Pull Request Guidelines
Commit messages stay short and imperative (`Add automatic CUDA version detection`). Squash local fixups and mention performance evidence when relevant. Pull requests should note affected modules, list verification commands, link tracking issues, and attach metrics or screenshots for visualization updates. Flag any new CUDA requirements or environment adjustments so maintainers can reproduce your setup quickly.

## Environment & Configuration Tips
Align your CUDA toolkit with the architectures declared in `CMakeLists.txt` (`60;70;75;80`) to avoid mismatches. The Python build script expects `conda` and updates `LD_LIBRARY_PATH` using the active PyTorch install; mirror that logic if you customize the environment.
