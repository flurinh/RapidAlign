# RapidAlign

RapidAlign delivers differentiable, correspondence-free similarity metrics for point clouds and graph embeddings. The core of the project is a KDE/MMD kernel implemented in CUDA 12.8 with a seamless PyTorch autograd wrapper, plus a pure-PyTorch fallback for CPU-only workflows.

## At a Glance

- Minimal API: `kde_mmd_loss`, `kde_mmd_loss_dense`, and `pyg_kde_mmd_loss` cover individual graphs, padded batches, and ragged PyG-style batches.
- CUDA extension targets PyTorch 2.8.0+cu128 and supports segmented forward/backward passes for large batches.
- CPU path mirrors the math for fast experimentation when a compatible GPU is not available.
- Companion scripts in `examples/` illustrate benchmarking, PyG integration, and visualization workflows.

## Installation & Requirements

### Prerequisites

- Python 3.11 (matches the packaged wheels)
- NVIDIA GPU with drivers compatible with CUDA 12.8
- PyTorch 2.8.0 built for `cu128`
- Optional PyG stack: `torch_geometric`, `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`

### Install via PyPI

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install rapidalign
```

> **Note:** PyTorch publishes CUDA wheels from a dedicated index. Install the desired CUDA build before `pip install rapidalign` to avoid falling back to the CPU-only wheel.

### Optional: PyG Ecosystem Support

```bash
python -m pip install torch_geometric
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### From Source (editable)

```bash
git clone https://github.com/flurinh/RapidAlign.git
cd RapidAlign
python -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python setup.py build_ext --inplace  # compile the CUDA extension
python -m pytest tests              # optional regression check
```

The build script respects `TORCH_CUDA_ARCH_LIST` and targets compute `60;70;75;80` by default.

## Example Usage

The pure PyTorch API is ideal for quick sanity checks:

```python
import torch
from rapidalign import kde_mmd_loss

src = torch.randn(128, 3)
tgt = src + 0.05 * torch.randn_like(src)

loss, Kxx, Kyy, Kxy = kde_mmd_loss(src, tgt, sigma=0.2, center=True)
print(f"MMD^2: {loss.item():.6f}")
```

For ragged batches using PyTorch Geometric (similar to `examples/kde_segmented_demo.py`):

```python
import torch
from torch_geometric.data import Batch, Data
from rapidalign import pyg_kde_mmd_loss


def make_graph(n_nodes: int, noise: float, seed: int) -> Data:
    torch.manual_seed(seed)
    coords = torch.randn(n_nodes, 3)
    if noise > 0:
        coords = coords + noise * torch.randn_like(coords)
    return Data(pos=coords)


sources = [make_graph(64, noise=0.0, seed=0), make_graph(48, noise=0.0, seed=1)]
targets = [make_graph(64, noise=0.05, seed=2), make_graph(52, noise=0.08, seed=3)]

batch_src = Batch.from_data_list(sources).cuda()
batch_tgt = Batch.from_data_list(targets).cuda()

losses, Kxx, Kyy, Kxy = pyg_kde_mmd_loss(
    batch_src.pos,
    batch_tgt.pos,
    src_batch=batch_src.batch,
    tgt_batch=batch_tgt.batch,
    sigma=0.25,
    center=True,
)
print("Per-graph KDE losses:", losses.tolist())
```

Check the scripts in `examples/` for full workflows (benchmarks, visualization, and training loops). The CUDA extension is optional—if tensors remain on CPU, the pure PyTorch implementation is used automatically.
