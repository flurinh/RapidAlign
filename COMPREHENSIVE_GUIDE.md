# Learnable Kernel-Based Graph Similarity: Comprehensive Guide

## Objective

Learn a task-specific, correspondence-free similarity that is:
1. Translation-invariant by default; rotation sensitivity optional (via features)
2. Permutation-invariant over nodes
3. Fully differentiable (usable as a training loss)
4. Fast on GPU (O(N²) dense; near-linear with cutoffs)
5. Optionally learnable (kernel parameters/features)

---

## Problem Statement

### Input
- **Query Graph**: `G_q = (V_q, E_q, X_q)` with `N_q` nodes
- **Target Graph**: `G_t = (V_t, E_t, X_t)` with `N_t` nodes
- Node features: `X_q ∈ ℝ^(N_q × d)`, `X_t ∈ ℝ^(N_t × d)`
- Node positions: `P_q ∈ ℝ^(N_q × 3)`, `P_t ∈ ℝ^(N_t × 3)`

### Output
- **Similarity score**: `s ∈ ℝ` (higher = more similar)
- **No explicit correspondence required**: Different node counts OK
- **Differentiable**: Gradients flow to node features and positions

---

## Approach: RKHS Distance (MMD) Similarity

Represent graphs as weighted point sets with optional features and compare them directly—no pose estimation. Let X = {(x_i, f_i, q_i)} and Y = {(y_j, g_j, p_j)}.

Kernel (coordinates × features):
```
K((x_i, f_i), (y_j, g_j)) = exp(-||x_i - y_j||² / (2σ²)) · κ_f(f_i, g_j)
```

RKHS distance (MMD²):
```
KXX = Σ_i Σ_k q_i q_k K((x_i,f_i),(x_k,f_k))
KYY = Σ_j Σ_l p_j p_l K((y_j,g_j),(y_l,g_l))
KXY = Σ_i Σ_j q_i p_j K((x_i,f_i),(y_j,g_j))

MMD²(X, Y) = KXX + KYY - 2·KXY
```

Losses:
```
L_mmd = MMD²  # zero iff identical multisets (with same weights/features)
L_cos = 1 - KXY / sqrt((KXX+ε)(KYY+ε))  # scale-robust variant
```

---

## Mathematical Framework

### 1. Pre-Processing: Centering

Remove translation ambiguity:

```
μ_q = Σᵢ wᵢ pᵢ^q / Σᵢ wᵢ
μ_t = Σⱼ wⱼ pⱼ^t / Σⱼ wⱼ

p̃ᵢ^q = pᵢ^q - μ_q
p̃ⱼ^t = pⱼ^t - μ_t
```

**Result**: Translation-invariant metric

---

### 2. Learnable Kernel Function

**Standard Gaussian RBF:**
```
K_σ(r) = exp(-r²/(2σ²))
```

**Learnable variants:**

#### a) Isotropic (single bandwidth)
```python
σ = exp(θ_σ)  # Learnable log-bandwidth
K(r) = exp(-r²/(2σ²))
```

#### b) Anisotropic (per-dimension bandwidth)
```python
Σ = diag(exp(θ_x), exp(θ_y), exp(θ_z))
K(d) = exp(-½ dᵀΣ⁻¹d)
```

#### c) Mahalanobis (full metric)
```python
M = LLᵀ  # L is lower-triangular, learnable
K(d) = exp(-½ dᵀM⁻¹d)
```

#### d) Mixture of Gaussians
```python
K(r) = Σₖ wₖ exp(-r²/(2σₖ²))
```
where `{wₖ, σₖ}` are learnable

#### e) Neural Kernel
```python
K(r) = MLP(r; θ)  # Full neural network
```

---

### 3. Alignment via EM Algorithm

**Goal**: Find optimal rotation `R*` and translation `t*` that maximize `κ(R, t)`

#### E-Step: Compute Soft Correspondences

```
ŵᵢⱼ = wᵢ wⱼ exp(-||p̃ᵢ^t - R·p̃ⱼ^q - t||²/(2σ²))

κ = Σᵢⱼ ŵᵢⱼ

wᵢⱼ = ŵᵢⱼ / κ  # Normalized weights
```

#### M-Step: Update Pose

**Weighted centroids:**
```
x̄ = Σᵢⱼ wᵢⱼ p̃ᵢ^t
ȳ = Σᵢⱼ wᵢⱼ p̃ⱼ^q
```

**Cross-covariance:**
```
S = Σᵢⱼ wᵢⱼ (p̃ᵢ^t - x̄)(p̃ⱼ^q - ȳ)ᵀ
```

**Rotation via SVD:**
```
S = UΣVᵀ
R* = U·Vᵀ  (with det correction)
t* = x̄ - R*·ȳ
```

**Iterate** until convergence (typically 5-10 iterations)

---

### 4. Final Similarity Score

Primary losses (no pose):
```
L_mmd = KXX + KYY - 2·KXY
L_cos = 1 - KXY / sqrt((KXX+ε)(KYY+ε))
```
Both are zero iff X and Y are identical multisets (with same weights/features). Choose based on scale sensitivity and task.

---

## Differentiability

### Forward Pass
```
Nodes/weights/features → (optional center) → accumulate KXX, KYY, KXY → L_mmd or L_cos
```

### Backward Pass (Gaussian coord kernel)
```
∂K/∂x_i = - (x_i - y_j)/σ² · K

∂KXX/∂x_i = 2 Σ_k q_i q_k ∂K((x_i,f_i),(x_k,f_k))/∂x_i
∂KXY/∂x_i =   Σ_j q_i p_j ∂K((x_i,f_i),(y_j,g_j))/∂x_i
∂MMD²/∂x_i = ∂KXX/∂x_i - 2 ∂KXY/∂x_i
```
Feature gradients follow from the chosen κ_f. Gradients through centering propagate via the mean-subtraction.

---

## Architecture: Hybrid PyTorch-CUDA

### Why Hybrid?

| Component | Implementation | Reason |
|-----------|---------------|---------|
| Kernel parameters (σ, τ, M) | **PyTorch** | Easy autograd, optimizer integration |
| Node features | **PyTorch** | Neural network processing |
| KDE/MMD accumulators | **CUDA** | O(N²) dense, near-linear with cutoffs |
| Gradient computation | **CUDA** | Avoid memory overhead |

### Design Pattern

```python
class KernelSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))

    def forward(self, X, Y, q=None, p=None, center=True):
        sigma = torch.exp(self.log_sigma)
        Kxx, Kyy, Kxy = cuda_kde_similarity(X, Y, q, p, sigma, center)
        mmd2 = Kxx + Kyy - 2 * Kxy
        return mmd2
```

---

## Training Pipeline

### Phase 1: Generate Synthetic Training Data

```python
def generate_training_pairs():
    """
    Create graph pairs with known similarity labels
    """
    
    # Case 1: Same graph, different pose (label=1.0)
    G_base = random_graph()
    G_q = rotate(G_base, random_R())
    G_t = G_base + noise(0.05)
    label = 1.0
    
    # Case 2: Different graphs (label=0.0)
    G_q = random_graph()
    G_t = random_graph()
    label = 0.0
    
    # Case 3: Partial overlap (label=0.7)
    G_q = G_base
    G_t = subsample(G_base, 0.7) + noise(0.1)
    label = 0.7
    
    return (G_q, G_t, label)
```

### Phase 2: Train Kernel Parameters

```python
model = KernelSimilarity()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for G_q, G_t, label in dataloader:
        # Forward
        pred_similarity = model(G_q.pos, G_t.pos, G_q.w, G_t.w)
        
        # Loss: match ground truth
        loss = (pred_similarity - label) ** 2
        
        # Backprop
        loss.backward()
        optimizer.step()
```

### Phase 3: Freeze and Use as Loss

```python
# Freeze learned parameters
for param in model.parameters():
    param.requires_grad = False

# Use as loss in downstream task
def task_loss(generated_graph, target_graph):
    similarity = model(generated_graph, target_graph)
    return -similarity  # Maximize similarity
```

---

## Implementation Strategy

### 1. Point Cloud Version (Baseline)

```
Input: P_q ∈ ℝ^(N_q × 3), P_t ∈ ℝ^(N_t × 3)
Output: similarity ∈ ℝ
```

**Pure 3D positions, no features**

### 2. Graph Version (Single Pair)

```
Input: 
  - G_q = (pos_q, features_q, edge_index_q)
  - G_t = (pos_t, features_t, edge_index_t)
  
Output: similarity ∈ ℝ
```

**Use features to compute node weights:**
```python
w_q = MLP(features_q)  # Learn importance weights
w_t = MLP(features_t)
```

### 3. Batched Graph Version

```
Input:
  - Batch_q = [G₁^q, G₂^q, ..., G_B^q]
  - Batch_t = [G₁^t, G₂^t, ..., G_B^t]
  
Output: similarities ∈ ℝ^B
```

**Process pairs in parallel on GPU**

---

## Graph-Specific Extensions

### Node Weights from Features

Instead of uniform weights, learn from features:

```python
class NodeWeightNet(nn.Module):
    def forward(self, node_features):
        # features: [N, d]
        weights = self.mlp(node_features)  # [N, 1]
        weights = torch.softmax(weights, dim=0)  # Normalize
        return weights
```

### Edge-Aware Kernels

Incorporate edge information:

```python
# Diffuse node features along edges
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(d_in, 64)
        self.conv2 = GCNConv(64, d_out)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Use enriched features for weights
enriched_features = graph_encoder(features, edge_index)
weights = weight_net(enriched_features)
```

---

## Performance Considerations

### Memory Complexity

- **Dense weights matrix**: `O(N_q × N_t)`
- **Sparse with 3σ cutoff**: `O(N_q × k)` where `k ≈ 27` (constant!)

### Time Complexity

| Operation        | Complexity       |
|------------------|------------------|
| Centering        | O(N)             |
| Kernel sums      | O(N_q × N_t)     |
| Kernel sums (cut)| O(N_q × k)       |
| Total (no EM)    | O(N²) or O(Nk)   |

### GPU Utilization

- **Batched processing**: Process multiple pairs simultaneously
- **Warp-level reductions**: Use CUDA intrinsics
- **Shared memory**: Cache frequently accessed data
- **Spatial hashing**: Sparse neighborhood computation

---

## Optional: Registration via EM (KC)
When rigid pose is required, an EM/MM procedure over (R, t) (Habeck, Tsin & Kanade) can be used to maximize kernel correlation. This is not needed for a pure similarity loss and adds iteration latency, but remains useful when pose is part of the task.

## Code Pointers
- Pure similarity (this repo): `cuda/standalone/kde_similarity.cu` — forward sums for KXX/KYY/KXY and MMD/cosine losses.
- Registration reference (this repo): `cuda/standalone/se3_kernel_loss.cu` — EM/MM with SVD for pose.

---

## Advantages Over Alternatives

### vs. ICP (Iterative Closest Point)
- ✅ No hard correspondences needed
- ✅ Robust to partial overlap
- ✅ Differentiable
- ❌ Slower (O(N²) vs O(N log N))

### vs. Deep Learning Matching
- ✅ Explicit geometric reasoning
- ✅ Fewer trainable parameters
- ✅ Better sample efficiency
- ❌ Not purely learned end-to-end

### vs. Point Cloud Networks (PointNet, DGCNN)
- ✅ Rotation-equivariant by design
- ✅ No data augmentation needed
- ✅ Interpretable similarity score
- ❌ Requires optimization loop (slower)

### vs. Spectral Methods
- ✅ Works with partial overlap
- ✅ Differentiable
- ✅ No eigendecomposition needed
- ❌ Local optima possible

---

## Applications

### 1. Molecular Docking
```python
# Score protein-ligand binding
similarity = model(protein_graph, ligand_graph)
binding_affinity = similarity_to_affinity(similarity)
```

### 2. Point Cloud Registration
```python
# Align scans from different sensors
similarity = model(lidar_scan, rgbd_scan)
quality_score = similarity
```

### 3. Shape Retrieval
```python
# Find similar 3D models
for shape in database:
    score = model(query_shape, shape)
    ranking.append((shape, score))
```

### 4. Graph Matching
```python
# Match molecular structures
similarity = model(molecule1_graph, molecule2_graph)
is_isomorphic = (similarity > threshold)
```

---

## Hyperparameters

### Critical Parameters

| Parameter | Symbol | Range | Effect |
|-----------|--------|-------|--------|
| Bandwidth | σ | 0.1-2.0 | Tolerance to misalignment |
| Temperature | τ | 0.1-10.0 | Sharpness of discrimination |
| EM iterations | I | 1-20 | Accuracy vs speed tradeoff |
| Learning rate | α | 1e-4 - 1e-2 | Training stability |

### Tuning Strategy

1. **Start large σ**: Begin with σ=1.0 (coarse alignment)
2. **Anneal**: Gradually decrease to σ=0.2 (fine details)
3. **Multi-scale**: Use σ schedule: [1.0, 0.5, 0.3, 0.2]
4. **Temperature**: Start τ=1.0, adjust based on task

---

## Failure Cases and Mitigations

### Problem 1: Symmetric Shapes

**Issue**: Multiple optima with same κ value

**Solution**: 
- Add symmetry-breaking term
- Use multiple random initializations
- Learn asymmetric features

### Problem 2: Very Different Node Counts

**Issue**: Normalization by N_q × N_t may not be appropriate

**Solution**:
```python
# Normalize by min or max instead
similarity = kappa / min(N_q, N_t)
# Or use geometric mean
similarity = kappa / sqrt(N_q * N_t)
```

### Problem 3: Local Minima

**Issue**: EM can get stuck in local optima

**Solution**:
- Multi-scale σ schedule (coarse-to-fine)
- Multiple random initializations
- Global optimization (random search + local refinement)

### Problem 4: Partial Overlap

**Issue**: κ is small when clouds only partially overlap

**Solution**:
- Use normalized metric: κ / max_possible_κ
- Add outlier detection
- Use robust kernel (e.g., Student-t instead of Gaussian)

---

## Future Extensions

### 1. Learnable Pose Initialization
```python
# Predict initial R, t from global features
R_init, t_init = InitNet(global_features_q, global_features_t)
# Then refine with EM
```

### 2. Attention-Based Weights
```python
# Cross-attention between graphs
attn_weights = CrossAttention(features_q, features_t)
# Use in kernel correlation
```

### 3. Deformable Alignment
```python
# Allow local non-rigid deformations
deformation = DeformNet(local_features)
transformed_pos = pos + deformation
```

### 4. Multimodal Kernels
```python
# Combine geometric and semantic kernels
k_geom = exp(-||p_i - p_j||²/σ²)
k_sem = exp(-||f_i - f_j||²/σ_f²)
k_total = α·k_geom + (1-α)·k_sem
```

---

## References

### Core Papers
1. **Habeck (2024)**: "Matching biomolecular structures by registration of point clouds"
   - Kernel correlation for structure alignment
   - Majorization-minimization algorithm

2. **Zhang et al. (2024)**: "Correspondence-Free SE(3) Point Cloud Registration in RKHS"
   - Equivariant learning for registration
   - Unsupervised training framework

3. **Tsin & Kanade (2004)**: "A correlation-based approach to robust point set registration"
   - Original kernel correlation idea

### Related Work
- Continuous Visual Odometry (CVO)
- Gaussian Mixture Model registration
- Deep Closest Point (DCP)
- PointNet / PointNet++
- E(n)-equivariant networks

---

## Summary

**What we're building:**

A learnable, differentiable, rotation-equivariant similarity metric for graphs/point clouds that:

1. ✅ Requires no correspondences
2. ✅ Handles different node counts
3. ✅ Is fast (CUDA implementation)
4. ✅ Is trainable (PyTorch parameters)
5. ✅ Can be frozen and used as a loss function

**Core innovation:**

Combining classical geometric optimization (EM algorithm) with modern deep learning (learnable kernel parameters) to get the best of both worlds: geometric interpretability + data-driven adaptation.

**Next steps:**

1. Implement CUDA kernels for fast alignment
2. Create PyTorch wrapper with learnable parameters
3. Generate synthetic training data
4. Train kernel parameters
5. Freeze and deploy as similarity loss
