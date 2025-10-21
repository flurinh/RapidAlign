# RapidAlign Baseline Roadmap

## Synthetic Data Overhaul (highest priority)
- [ ] Re-implement a lightweight `synthetic_data` module under `python/` that mirrors the legacy generator capabilities (shape library, random SE(3) transforms).
- [ ] Add configurable noise models (Gaussian, uniform, structured) with strengths validated against the old `SyntheticDataGenerator.add_noise` logic.
- [ ] Support correspondence-free scenarios by subsampling/oversampling targets (node deletion/addition) and partial-overlap masks.
- [ ] Create fixtures for three evaluation tiers: single pair, equal-size batched pairs, and mismatched batched pairs.
- [ ] Document the dataset knobs and default distributions inside `docs/`.

## CUDA Kernels & Python Integration
- [ ] Design a minimal CUDA extension layout under `python/csrc` for shared headers, device utilities, and pybind bindings.
- [ ] Implement Procrustes alignment kernel(s) with batching support (rigid SVD per graph).
- [ ] Implement SE(3) kernel-correlation loss in CUDA, including MM update loop on device.
- [ ] Provide CUDA helpers for pairwise distance spectrum baseline (or justify CPU fallback if infeasible).
- [ ] Expose kernels through PyTorch extension modules callable from Python baselines.
- [ ] Set up CMake or setup.py build scripts to compile the extension within the `rapidalign` package.

## Algorithm Baselines (on top of CUDA kernels)
### Procrustes
- [ ] Wire Python API to call CUDA implementation for single and batched inputs.
- [ ] Surface rotation/translation error metrics for regression testing.
- [ ] Extend API with optional weighting to match future graph attributes.

### SE(3) Kernel Loss
- [ ] Benchmark device MM iterations vs. convergence using the new datasets (single + batched).
- [ ] Explore annealing schedule for `sigma` to handle large noise / partial overlaps.
- [ ] Implement gradient checks (finite differences) on representative batches via PyTorch autograd.

### Pairwise Distance Loss
- [ ] Evaluate sensitivity to node insertion/deletion; consider trimmed weighting when spectra lengths diverge.
- [ ] Decide whether to normalise distances per-instance to avoid scale bias.

## Evaluation Harness
- [ ] Build a unified `python/tests/test_baselines.py` suite that consumes synthetic fixtures for all three objectives (CPU vs CUDA parity checks).
- [ ] Add pytest markers for slow / noisy cases and GPU-dependent tests to keep CI fast.
- [ ] Produce summary metrics (alignment error, loss values) for each tier and store expected thresholds.
- [ ] Add visualization scripts (matplotlib/plotly) that consume CUDA results for qualitative review.

## Tooling & Environment
- [ ] Record conda environment requirements (PyTorch, typing_extensions, pytest) in `python/requirements.txt` or `environment.yml`.
- [ ] Provide a short `make test` wrapper that activates the env and runs pytest.
- [ ] Investigate re-compiling archived CUDA kernels once PyTorch headers are present (optional legacy verification).

- [ ] Evaluate structured KDE/MMD on synthetic dataset

## Segmented KDE Backward Pass (new plan)
- [x] **Stage 1:** CPU analytic + finite-difference gradients for identical pairs (sanity) — adds `test_identical_pair_gradients`.
- [x] **Stage 2:** Extend CPU helper to noisy equal-size pairs; verify gradients vs. finite differences.
- [x] **Stage 3:** Support variable node counts (ragged ptr) in CPU helper; add tests for mismatched sizes.
- [x] **Stage 4:** Wrap CPU forward/backward in `torch.autograd.Function`; run `gradcheck` on single/variable pairs.
- [x] **Stage 5:** Implement CUDA backward kernel + bindings; ensure parity with CPU via tests and `gradcheck` on GPU.
