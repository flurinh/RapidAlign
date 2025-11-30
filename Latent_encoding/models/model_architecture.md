# Graph Autoencoder Architecture Review

## 1. Model Snapshot

```
GraphAutoencoder(
  (encoder): EquivariantGraphSlotEncoder(
    (backbone): EquivariantBackbone(
      (layers): ModuleList(
        (0): EquivariantMPNNLayer(
          (conv): FullyConnectedTensorProductConv(
            (tp): FullyConnectedTensorProduct(
              shared_weights=False, internal_weights=False, weight_numel=32
              (transpose_in1): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (transpose_in2): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (f): ╭ a=[32:2⨯(4,1,4)] b=[4:1⨯(1,4)] c=[4:(1,1)+(3,1)] -> D=[16:(1,4)+(3,4)]
              ╰─ [ijk]·a[uvw]·b[iu]·c[jv]➜D[kw] ─ num_paths=2 i=1 j={1, 3} k={1, 3} u=4 v=1 w=4
            )
            (batch_norm): BatchNorm(4x0e+4x1o, layout=(irrep,mul), eps=1e-05, momentum=0.1)
          )
          (edge_mlp): Sequential(
            (0): Linear(in_features=1, out_features=64, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=64, out_features=32, bias=True)
          )
          (activation): GELU(approximate='none')
        )
        (1-2): 2 x EquivariantMPNNLayer(
          (conv): FullyConnectedTensorProductConv(
            (tp): FullyConnectedTensorProduct(
              shared_weights=False, internal_weights=False, weight_numel=64
              (transpose_in1): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (transpose_in2): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (transpose_out): TransposeIrrepsLayout((irrep,mul) -> (irrep,mul))
              (f): ╭ a=[64:4⨯(4,1,4)] b=[16:(1,4)+(3,4)] c=[4:(1,1)+(3,1)] -> D=[16:(1,4)+(3,4)]
              ╰─ [ijk]·a[uvw]·b[iu]·c[jv]➜D[kw] ─ num_paths=4 i={1, 3} j={1, 3} k={1, 3} u=4 v=1 w=4
            )
            (batch_norm): BatchNorm(4x0e+4x1o, layout=(irrep,mul), eps=1e-05, momentum=0.1)
          )
          (edge_mlp): Sequential(
            (0): Linear(in_features=1, out_features=64, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=64, out_features=64, bias=True)
          )
          (activation): GELU(approximate='none')
        )
      )
      (sh_module): SphericalHarmonics(
        (f): ╭ a=[3:3⨯()] -> B=[4:4⨯()]
        │  []➜B[] ───── num_paths=1
        ╰─ []·a[]➜B[] ─ num_paths=3
      )
      (input_proj): Linear(in_features=1, out_features=4, bias=True)
    )
    (slot_layers): ModuleList(
      (0-2): 3 x SlotAttention(
        (score_mlp): Sequential(
          (0): Linear(in_features=136, out_features=128, bias=True)
          (1): ReLU()
          (2): Linear(in_features=128, out_features=1, bias=True)
        )
        (node_proj): Linear(in_features=8, out_features=128, bias=True)
        (slot_self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
      )
    )
  )
  (decoder): DiffusionDecoder(
    (time_embed): SinusoidalTimeEmbedding()
    (mpn): SlotConditionedMPNN(
      (node_embed): Embedding(64, 256)
      (slot_proj): Linear(in_features=1024, out_features=256, bias=True)
      (time_proj): Linear(in_features=256, out_features=256, bias=True)
      (rbf): RadialRBF()
      (msg_mlp): Sequential(
        (0): Linear(in_features=528, out_features=256, bias=True)
        (1): SiLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): SiLU()
      )
      (update_mlp): Sequential(
        (0): Linear(in_features=512, out_features=256, bias=True)
        (1): SiLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): SiLU()
      )
      (out_head): Linear(in_features=256, out_features=3, bias=True)
    )
  )
)
```

**Quick read:** The encoder is an SE(3)-equivariant MPNN whose per-layer invariants feed a 3-layer SlotAttention stack, yielding \(K=8\) slot latents per graph. The decoder is a diffusion-style, slot-conditioned MPNN that iteratively refines a learnable template of point coordinates using dynamic kNN edges.

---

## 2. Encoder — Equivariant Backbone + SlotAttention

### Strengths
- Fully equivariant tensor-product convolutions (via `FullyConnectedTensorProductConv` + spherical harmonics) respect O(3) symmetry without hand-crafted features.
- Mixed scalar (`0e`) and vector (`1o`) channels (`4x0e + 4x1o`) correctly separate invariant and equivariant information; scalar/vector norms are later distilled into SE(3)-invariant descriptors.
- SlotAttention compresses variable-size graphs into a fixed \(K \times D\) latent, making downstream conditioning straightforward.

### Bottlenecks & Risks
- **cuequivariance fallback:** The warning `cuequivariance_ops_torch is not available. Falling back to naive implementation.` means tensor products execute in Python/PyTorch instead of the fused CUDA kernels. For proteins or other large-N graphs this quickly becomes the runtime bottleneck.
- **Irrep width ceiling:** `scalar_width = vector_width = 4` (12 DOF per node). Adequate for small synthetic graphs, but chemistry-scale workloads may need something like `8x0e + 8x1o` to avoid under-capacity.
- **SlotAttention loop:** The current `forward` loops over graphs and performs attention per graph, so cost scales \(O(B \cdot N \cdot K \cdot D)\). With \(K=8\) it is manageable, yet the loop becomes hot for large batches or K>16.
- **Latent bottleneck:** Slots (8×128) intentionally create a narrow information channel. For graphs with \(N \gg 100\) nodes, consider scaling \(K\) or `slot_dim` to maintain reconstruction quality.

---

## 3. Decoder — DiffusionDecoder + SlotConditionedMPNN

### Behaviour
- Starts from a learned coordinate template `[num_nodes, 3]` for every graph.
- Every denoising step builds dynamic kNN edges via `torch.cdist`, embeds node indices, injects slot/time context, runs an MPNN, and predicts coordinate deltas.

### Bottlenecks & Risks
- **Quadratic neighbor search:** `torch.cdist(coords, coords)` is \(O(B \cdot N^2)\) per diffusion step. This is fine for \(N \lesssim 128\); beyond that it dominates runtime. Consider approximate kNN or radius graphs for larger systems.
- **Index-tied embeddings:** `node_embed = nn.Embedding(num_nodes, hidden_dim)` pins the decoder to a fixed maximum number of nodes. For variable-sized inputs you’ll need either shared embeddings, modulo schemes, or structural positional signals.
- **Lack of equivariance:** The decoder MPNN is not SE(3)-equivariant. Training with invariant losses (Chamfer, RFF, etc.) keeps outputs aligned in expectation, but the learned dynamics themselves do not guarantee equivariance — call this out when documenting limitations.

---

## 4. Encoder Walkthrough for Documentation

### 4.1 High-Level Flow

Given a graph \(G = (X, R, E)\) with node features \(x_i\), positions \(r_i \in \mathbb{R}^3\), and edges \(E\):

1. Project raw node features to 4 scalar channels (`4x0e`).
2. Run three O(3)-equivariant message-passing layers that mix scalars & vectors using tensor products with spherical harmonics.
3. After each layer, convert equivariant features to invariant node descriptors.
4. Feed these per-node invariants into SlotAttention to obtain \(K\) latent slots per graph.

### 4.2 Irreps Primer

- **Scalar (0e):** unchanged under rotations/reflections (e.g., charge). `4x0e` means four independent scalar channels.
- **Vector (1o):** rotates like coordinates and flips under reflection. `4x1o` means four vector channels (12 floating-point numbers, but tightly coupled as vectors).
- Tracking irreps ensures every operation knows how features should transform. Rotating the input rotates the vector blocks identically, keeping equivariance exact.

### 4.3 Equivariant Message Passing

For each `EquivariantMPNNLayer`:

1. **Edge geometry**
   ```python
   edge_vec  = pos[row] - pos[col]      # direction
   edge_dist = edge_vec.norm(dim=-1)    # length
   edge_sh   = sh_module(edge_vec)      # Y_l(m)(v/||v||)
   ```
2. **Edge conditioning** — pass distances through `edge_mlp` to obtain tensor-product weights.
3. **Tensor-product convolution** — `FullyConnectedTensorProductConv` combines source node irreps, spherical harmonics, and learned weights so the outputs remain `4x0e + 4x1o`.
4. **Activation** — GELU applied within each irrep block keeps transformation rules intact.

### 4.4 Extracting Invariants

Scalars remain as-is; each vector channel is reduced to its norm:

```python
def extract_invariants(features, irreps):
    invariants, offset = [], 0
    for mul, ir in irreps:
        block_dim = mul * ir.dim
        block = features[:, offset : offset + block_dim]
        offset += block_dim
        if ir.is_scalar():
            invariants.append(block)
        else:
            reshaped = block.view(B, mul, ir.dim)
            invariants.append(reshaped.norm(dim=-1))
    return torch.cat(invariants, dim=1)
```

The result is a rotation-invariant descriptor per node, perfect for global pooling.

### 4.5 SlotAttention Mechanics

For each backbone layer:

1. Initialize \(K\) learnable slots per graph.
2. Compute attention scores between each slot and each node invariant (per graph) via a small MLP and `softmax` over nodes.
3. Project node descriptors into slot space, take attention-weighted sums, and optionally run a tiny self-attention across the \(K\) slots.
4. Feed updated slots into the next layer’s attention, so later layers can refine the same latent set.

Because the inputs to SlotAttention are invariant, the resulting `[B, K, D]` latents are also SE(3)-invariant summaries.

---

## 5. README-Ready Summary

> **Encoder: SE(3)-aware graph to fixed K latent slots**
> - Equivariant message passing keeps track of scalar (`0e`) and vector (`1o`) channels, guaranteeing the same response under any rotation/translation.
> - Edge directions are expanded in spherical harmonics and combined with node features through tensor-product convolutions.
> - After each layer we convert equivariant features into invariants by keeping scalar channels and taking the norms of vector channels.
> - SlotAttention pools these invariants into K latent slots per graph. Each slot learns to attend to a different structural motif, resulting in a compact, SE(3)-invariant latent representation `[B, K, D]`.

```
(x, pos, edges)
      │
      ▼
Equivariant MPNN layers → node invariants → SlotAttention stack → K slots
      │                                                       │
      └────────────── slots condition the diffusion decoder ──┘
```

Feel free to copy the block above (and the ASCII diagram) directly into project docs or slides when introducing the model.
