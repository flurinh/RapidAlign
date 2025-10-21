# SE(3)-Invariant Kernel Alignment Loss

## Motivation
We need a loss that judges whether two embedded graphs share the same 3D structure while being insensitive to global translation and rotation. The loss must be fast enough for GPU training loops, differentiable end-to-end, and stable when graphs only partially overlap. We build on the kernel correlation idea to obtain a closed-form pose update and combine it with sparse CUDA accumulation.

## Setup
Let a graph be encoded as a weighted point set \((X, q)\) with node positions \(x_i \in \mathbb{R}^3\) and non-negative weights \(q_i\) (e.g. node degree, mass, or learned confidence). Likewise \((Y, p)\) describes the comparison graph. Pre-centering removes translation:
\[
\tilde{x}_i = x_i - \mu_X, \quad \tilde{y}_j = y_j - \mu_Y, \quad \mu_X = \sum_i q_i x_i \Big/ \sum_i q_i.
\]
We use a Gaussian kernel with bandwidth \(\sigma\) to soft-match nodes inside a 3\(\sigma\) neighborhood. Sparse neighbor lists (uniform grid or hash) bound the double sum on GPU.

## Soft Correspondence Weights
For a current pose \((R, t)\) we define the unnormalized pair weights
\[
\hat{w}_{ij} = q_i p_j \exp\!\left(-\frac{\lVert \tilde{x}_i - R \tilde{y}_j - t \rVert^2}{2\sigma^2}\right)
\]
restricted to neighbors. Normalized weights are \(w_{ij} = \hat{w}_{ij} / \kappa(R, t)\) with
\[
\kappa(R, t) = \sum_{i,j} \hat{w}_{ij}.
\]
We view \(\kappa\) as a kernel correlation score; larger is better.

## Closed-Form Pose Update
Given the weights \(w_{ij}\) we obtain the pose that maximizes \(\kappa\) via a single majorization minimization step:
- Weighted centroids: \(\bar{x} = \sum_{i,j} w_{ij} \tilde{x}_i\) and \(\bar{y} = \sum_{i,j} w_{ij} \tilde{y}_j\).
- Cross-covariance: \(S = \sum_{i,j} w_{ij} (\tilde{x}_i - \bar{x})(\tilde{y}_j - \bar{y})^\top\).
- SVD: \(S = U \Sigma V^\top\). Set \(R^{\star} = U \operatorname{diag}(1,1,\det(UV^\top)) V^\top\).
- Translation: \(t^{\star} = \bar{x} - R^{\star} \bar{y}\).

One EM/MM iteration (weights \(\rightarrow\) pose \(\rightarrow\) weights) is sufficient in practice for the loss. More iterations improve accuracy at extra cost; we keep it to one for a lightweight loss.

## Loss Definition
We plug \((R^{\star}, t^{\star})\) back into the correlation and use a negative log-likelihood style loss with a temperature term \(\tau\):
\[
\mathcal{L}(X, Y) = -\log \left(\kappa(R^{\star}, t^{\star}) + \varepsilon\right) / \tau.
\]
Small \(\tau\) sharpens discrimination; \(\varepsilon\) avoids \(\log 0\). The loss is zero when the structures perfectly overlap.

## Backpropagation
Gradients flow through three components:
1. **Correlation term.** Direct derivative with respect to centered coordinates (with pose detached) is
   \[
   \frac{\partial \kappa}{\partial \tilde{x}_i} = \sum_{j} \hat{w}_{ij} \frac{R^{\star} \tilde{y}_j + t^{\star} - \tilde{x}_i}{\sigma^2}.
   \]
   The loss gradient is \(\partial \mathcal{L} / \partial \tilde{x}_i = -\tau^{-1}\, \kappa^{-1}\, \partial \kappa / \partial \tilde{x}_i\). The expression for \(\tilde{y}_j\) mirrors this with \(-R^{\star\top}\).
2. **Pose dependence.** For higher fidelity differentiate through the SVD used in \(R^{\star}\). Modern autodiff libraries already support this (batched SVD in PyTorch, JAX). Guard against near-singular \(S\) by adding a small diagonal \(\lambda I\) before SVD.
3. **Weight normalization.** The normalized weights contribute an additional term: \(w_{ij} = \hat{w}_{ij} / \kappa\) implies
   \[
   \frac{\partial w_{ij}}{\partial \tilde{x}_k} = \frac{1}{\kappa} \frac{\partial \hat{w}_{ij}}{\partial \tilde{x}_k} - \frac{\hat{w}_{ij}}{\kappa^2} \frac{\partial \kappa}{\partial \tilde{x}_k}.
   \]
   Implementations often stop gradients through \(w_{ij}\) (detach pose) for extra stability; empirically this still improves geometry.

## CUDA Implementation Sketch
1. **Neighbor search:** Hash points into a uniform grid with cell size \(\approx 2\sigma\). Build compact neighbor lists per point via shared memory buckets.
2. **Pair accumulation kernel:** Each thread block processes one source cell. Iterate over neighbor cells in target; accumulate \(\hat{w}_{ij}\), \(\hat{w}_{ij} \tilde{x}_i\), \(\hat{w}_{ij} \tilde{y}_j\), and the outer product contributions to \(S\) using warp-level reductions.
3. **Pose update:** Launch a batched SVD kernel (3×3 matrices) or use cuSOLVER’s `gesvdjBatched` to recover \(R^{\star}\) and \(t^{\star}\).
4. **Loss / gradient:** Re-run the pair kernel with the aligned pose (or reuse cached deltas) to compute \(\kappa\) and the per-point gradients. Expose a custom `autograd::Function` that stores \(w_{ij}\) and deltas for the backward pass.

## Practical Tips
- Bandwidth scheduling: start with a large \(\sigma\) (global structure) and anneal to finer values during training.
- Use weight clamping to avoid domination by a single node; normalize \(q_i\) and \(p_j\) to sum to one per graph.
- When symmetry causes ambiguous rotations, blend consecutive iterations or regularize toward the identity pose.
- Gradients near \(\kappa \approx 0\) can explode; cap the minimum correlation and optionally switch the loss to \(1 - \kappa\) if working with very sparse overlap.

This formulation yields a translation- and rotation-equivariant loss with a single EM-style update, minimalist hyperparameters, and GPU-friendly kernels.
