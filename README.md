# RapidAlign Baselines

RapidAlign provides differentiable, correspondence-free similarity measures for
graphs/point clouds (with classic baselines bundled in). Implemented objectives:

- **Procrustes alignment** – rigid pose recovery with known correspondences.
- **Pairwise distance loss** – structure-only baseline built from intra-graph distance spectra.
- **SE(3) kernel loss** – correspondence-free objective using a Gaussian kernel correlation.
- **KDE/MMD similarity** – fast, differentiable, correspondence-free RKHS distance. Loss = 0 iff identical sets.

The Python package now lives at the repository root under `rapidalign/`; CUDA
sources are in `csrc/`. Examples are under `examples/` and tests under `tests/`.

## Quickstart

```bash
pip install -e .
```

### Running Tests

The test suite requires PyTorch and pytest:

```bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate rapidalign
pip install torch pytest
python -m pytest tests
```

### Kernel Similarity (MMD) baseline

Use the pure-PyTorch baseline for quick experiments (fully differentiable):

```python
import torch
from rapidalign import kde_mmd_loss

x = torch.randn(128, 3)
y = x.clone()  # identical

loss, Kxx, Kyy, Kxy = kde_mmd_loss(x, y, sigma=0.2, center=True)
print('Loss≈0:', loss.item())
```

For speed at scale use the packaged CUDA extension (`rapidalign._cuda`), which
provides forward/backward kernels for batched, ragged inputs.
