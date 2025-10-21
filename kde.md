# KDE/MMD Loss Notes

RapidAlign’s loss is based on evaluating a Gaussian kernel between point sets
\( X = \{x_i\}_{i=1}^N \) and \( Y = \{y_j\}_{j=1}^M \):

\[
    K(x_i, y_j) = \exp\left(-\frac{\|x_i - y_j\|^2}{2\sigma^2}\right).
\]

We accumulate the self-terms and cross-terms (weighted if needed) to obtain
\(K_{XX}\), \(K_{YY}\), \(K_{XY}\) and derive MMD² or a cosine similarity.

## Tunable Parameters

1. **Bandwidth(s) \(\sigma\)**
   - Small \(\sigma\) → sensitive to fine detail/noise.
   - Large \(\sigma\) → emphasises coarse structure, smooths over local detail.
   - Multi-band: learn several \(\sigma_k\) plus mixture weights for a
     multi-scale response.

2. **Mixture weights** (multi-band)
   - Softmaxed parameters ensuring positive weights that sum to one, letting the
     loss focus on informative scales.

3. **Node weights**
   - Per-node scalars (uniform or learned) set how much each point contributes.

4. **Optional temperature / scaling**
   - Global factor to rescale the loss or adjust sensitivity.

## Why Fit the Kernel?

A hand-picked \(\sigma\) can misbehave: too small and the loss spikes for tiny
differences; too large and different shapes look identical. By learning
\(\sigma\) (and mixtures) on synthetic noise/masking examples, we calibrate the
loss so it correlates with corruption level:

- Identical clouds → near-zero loss.
- Mild noise/masking → small loss.
- Heavy noise or structural mismatch → large loss.

## Fitting Workflow

1. Generate synthetic pairs \((X, Y_\lambda)\) using noise/masking level
   \(\lambda\).
2. Evaluate the current loss \(L(\sigma; X, Y_\lambda)\).
3. Regress \(L\) against \(\lambda\) (e.g. minimise MSE) to encourage a smooth
   monotonic relationship.
4. Update \(\sigma\) and mixture weights via gradient descent.
5. Save the tuned parameters and use them in downstream models.

Once tuned, the kernel acts as a reliable, differentiable shape similarity
measure that responds smoothly to degradations, making it ideal for denoisers,
autoencoders, and diffusion/flow decoders.
