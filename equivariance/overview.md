# Understanding SO(3)/SE(3) Equivariant Neural Networks

## Core Concepts

### What is Equivariance?

**Equivariance** is the mathematical property that describes how a function's output transforms when its input is transformed. For a neural network to be equivariant means:

```
f(T(x)) = T'(f(x))
```

Where:
- `T` is a transformation applied to the input
- `T'` is the corresponding transformation applied to the output
- If the input rotates, the output rotates in a predictable way

### The Groups

#### SO(3) - Special Orthogonal Group in 3D
- The group of **rotations** in 3D space
- Preserves lengths and angles
- No reflections, only rotations
- Example: Rotating a molecule shouldn't change its properties

#### SE(3) - Special Euclidean Group in 3D
- The group of **rotations AND translations** in 3D space
- SE(3) = SO(3) + Translations
- Example: Moving and rotating a protein shouldn't change its binding affinity

#### E(3) - Euclidean Group in 3D
- Includes rotations, translations, AND reflections
- E(3) = O(3) + Translations
- Most general symmetry for 3D Euclidean space

### Why Do We Care?

Physical systems have inherent symmetries:
1. **Data Efficiency**: Models learn faster with less data
2. **Generalization**: Better performance on unseen orientations
3. **Physical Correctness**: Predictions respect natural laws
4. **Interpretability**: Outputs have geometric meaning

## Irreducible Representations (Irreps)

### What Are Irreps?

Irreps are the "building blocks" of group representations. For SO(3):

- **l=0**: Scalars (1 dimension) - invariant to rotation
  - Examples: temperature, pressure, mass
  
- **l=1**: Vectors (3 dimensions) - rotate like position vectors
  - Examples: velocity, force, dipole moment
  
- **l=2**: Rank-2 tensors (5 dimensions) - quadrupole moments
  - Examples: stress tensor, quadrupole moment

- **l=3, 4, ...**: Higher-order tensors

### Spherical Harmonics

SO(3) irreps are represented using **spherical harmonics** Y_l^m:
- `l` is the degree (0, 1, 2, ...)
- `m` ranges from -l to +l (giving 2l+1 components)
- They form an orthonormal basis on the sphere

## cuEquivariance vs e3nn

### cuEquivariance (NVIDIA)
- **Focus**: High-performance CUDA-accelerated operations
- **Architecture**: Segmented Tensor Products (STP)
- **Flexibility**: Can define custom irrep bases
- **Performance**: Optimized for GPU, 10-100x faster
- **Use Case**: Production models (DiffDock, MACE, Allegro)

### e3nn
- **Focus**: Educational and research-friendly
- **Architecture**: More explicit tensor product operations
- **Flexibility**: Rich tutorials and documentation
- **Use Case**: Research, learning, prototyping

## How to Use SO(3)/SE(3) Representations

### 1. Representing Data

```python
import cuequivariance as cue

# Define irreps: 32 scalars + 16 vectors + 8 rank-2 tensors
irreps = cue.Irreps("SO3", "32x0 + 16x1 + 8x2")

# This means:
# - 32 channels of scalars (rotation invariant)
# - 16 channels of vectors (3D each, rotate with input)
# - 8 channels of rank-2 tensors (5D each, transform accordingly)
```

### 2. Node Features in Graphs

For a molecular graph:
```python
# Each atom can have:
# - Scalar features: charge, mass, electronegativity
# - Vector features: velocity, force
# - Higher-order: polarizability tensor

node_features = cue.Irreps("SO3", "10x0 + 3x1 + 1x2")
# 10 scalar channels + 3 vector channels + 1 rank-2 tensor channel
```

### 3. Tensor Products

The key operation in equivariant networks:

```python
# Create a fully connected tensor product
input1 = cue.Irreps("SO3", "4x0 + 1x1")  # 4 scalars + 1 vector
input2 = cue.Irreps("SO3", "4x0 + 2x1")  # 4 scalars + 2 vectors
output = cue.Irreps("SO3", "4x0 + 3x1")  # 4 scalars + 3 vectors

etp = cue.descriptors.fully_connected_tensor_product(
    input1, input2, output
)
```

### 4. Message Passing Example

```python
# Typical GNN layer with equivariance:
# 1. Compute edge features (vectors between nodes)
# 2. Apply tensor product to combine node + edge features
# 3. Aggregate messages
# 4. Update node features maintaining equivariance
```

## Visualizing Representations

### What to Visualize?

1. **Scalar Fields (l=0)**: Heat maps, contour plots
2. **Vector Fields (l=1)**: Arrow plots showing direction and magnitude
3. **Spherical Functions (l≥0)**: Plot on sphere surface
4. **Graph Structures**: Nodes with embedded feature vectors

### Visualization Strategy

For a graph with SO(3) equivariant features:
- Show node positions in 3D
- Overlay vector features as arrows
- Color nodes by scalar features
- Show edges between connected nodes
- Optionally: rotate entire system to show equivariance

## Key Applications

1. **Molecular Property Prediction**: 
   - Predict energy, forces on atoms
   - Quantum chemistry calculations

2. **Protein-Ligand Binding**:
   - DiffDock: Predict how drugs bind to proteins
   - Orientation matters but shouldn't affect binding energy

3. **Material Science**:
   - Crystal structure prediction
   - Material property prediction (conductivity, etc.)

4. **Physics Simulations**:
   - Particle dynamics
   - Fluid flow

## Mathematical Details

### Clebsch-Gordan Coefficients

When combining two irreps, we use CG coefficients:
```
l1 ⊗ l2 = ⊕ C^{l1,l2}_l × l
```

Example: Vector ⊗ Vector:
```
1 ⊗ 1 = 0 + 1 + 2
```
(scalar + vector + rank-2 tensor)

### Wigner D-matrices

Describe how irreps transform under rotations:
```
D^l(R) |l,m⟩ = Σ D^l_{m',m}(R) |l,m'⟩
```

## Practical Tips

1. **Start Simple**: Begin with `l=0` and `l=1` (scalars and vectors)
2. **Use Pre-built Layers**: cuEquivariance/e3nn provide ready-to-use layers
3. **Check Equivariance**: Test by rotating inputs and checking outputs rotate correctly
4. **Profile Performance**: Higher-order irreps are more expensive
5. **Understand Your Data**: What geometric information is actually needed?

## Common Pitfalls

1. **Mixing Frames**: Ensure all vectors are in the same coordinate system
2. **Forgetting Normalization**: Spherical harmonics need proper normalization
3. **Over-parametrization**: Don't use high-order irreps unless needed
4. **Ignoring Parity**: For O(3), consider even/odd parity

## Resources

- cuEquivariance Docs: https://docs.nvidia.com/cuda/cuequivariance/
- e3nn Tutorial: https://blondegeek.github.io/e3nn_tutorial/
- Paper: "E(3)-Equivariant Graph Neural Networks" (Satorras et al.)
- Paper: "NequIP" (Batzner et al.)