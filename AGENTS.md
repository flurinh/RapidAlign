# Repository Guidelines

RapidAlign is a hybrid C++/CUDA core with Python bindings; follow the sections below to keep contributions consistent and reproducible.

## Project Structure & Module Organization
- Core CPU/CUDA prototypes live in `main.cpp`, `visualize_graph_alignment.cpp`, `pointcloud_alignment.cu`, and `batch_alignment.cu`, orchestrated by the root `CMakeLists.txt`.
- Python bindings sit in `python/rapidalign`, while docs/reference material lives under `python/docs/`, `docs/`, and `ref/`.
- Benchmarks and utilities (`python/benchmarks/`, `dev/`, `deprecated/`) provide historical experiments; treat datasets there as read-only unless you regenerate them.
- Tests reside in `python/tests/python` for API coverage and `python/tests/cuda` for kernel validation.

## Build, Test, and Development Commands
- Configure native targets: `cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- Rebuild: `cmake --build build -j` and run spot checks via `./build/CPUTest`, `./build/MyCudaProject`, or `./build/BatchedAlign`.
- Rebuild Python extension: `bash python/build.sh` for a clean CUDA extension refresh, or `pip install -e python` when iterating in-place.
- CUDA-only experiments can be compiled in `python/` with `make all`; run test harnesses via `./run_all_tests.sh`.

## Coding Style & Naming Conventions
Use 4-space indentation and same-line braces for C++. Prefer `snake_case` filenames, functions, and CUDA kernels (`batched_dpcr.cu`). Python modules follow PEP 8 with descriptive tensor names (`src_points`, `batch_idx`). Keep kernel launches near helper utilities and add concise comments documenting memory layouts, mirroring `core.py` and `main.cpp`.

## Testing Guidelines
Run Python API tests with `cd python/tests/python && python -m pytest`. CUDA validation sits in `python/tests/cuda`; `make all` builds fixtures and `./run_all_tests.sh` executes them on a GPU. Ensure `python/test_import.py` passes before submitting and trigger benchmarks in `python/benchmarks/` when touching performance-sensitive kernels.

## Commit & Pull Request Guidelines
Write short, imperative commit subjects (e.g., `Add automatic CUDA version detection`). Each PR should call out affected modules, linked issues, verification commands, and performance evidence or screenshots for visualization updates. Mention any new CUDA requirements or environment tweaks so maintainers can reproduce results quickly.

## Environment & Configuration Tips
Align your CUDA toolkit with the architectures declared in `CMakeLists.txt` (`60;70;75;80`). Use the `rapidalign` conda environment; `python/build.sh` updates `LD_LIBRARY_PATH` to match the active PyTorch install, so mirror that logic when customizing shells.
