"""
Lloyd-Max optimal scalar quantizer for N(0,1), computed via classic Lloyd's
algorithm (k-means in 1D) on a large Monte-Carlo sample, with importance-
sampling-free direct large-N convergence + a deterministic RNG seed so
results are reproducible.

Why N(0,1): TurboQuant's whole trick is applying a random orthogonal
rotation (Walsh-Hadamard here) to each K/V vector before quantizing.
By the Central Limit Theorem / concentration of measure, the rotated
coordinates of a high-dimensional vector are approximately i.i.d. Gaussian
regardless of the original per-coordinate distribution. So a single
Gaussian-optimal codebook works for (almost) any input distribution -
that's *why* the rotation step exists at all instead of just doing
per-channel quantization directly.
"""
import numpy as np

def lloyd_max_gaussian(bits, n_samples=20_000_000, iters=300, seed=42):
    n_levels = 1 << bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples).astype(np.float64)
    samples.sort()

    # init centroids at quantiles of N(0,1) (better than uniform init)
    qs = (np.arange(n_levels) + 0.5) / n_levels
    from scipy.special import ndtri  # fallback below if scipy missing
    centroids = ndtri(qs)

    for it in range(iters):
        # assign: nearest centroid boundaries (centroids sorted, so this is
        # searchsorted on midpoints)
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(midpoints, samples)
        new_centroids = np.zeros(n_levels)
        for k in range(n_levels):
            mask = idx == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
            else:
                new_centroids[k] = centroids[k]
        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < 1e-7:
            break

    # final MSE
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(midpoints, samples)
    recon = centroids[idx]
    mse = np.mean((samples - recon) ** 2)
    return centroids, mse, it + 1

if __name__ == "__main__":
    for bits in (3, 4):
        c, mse, iters = lloyd_max_gaussian(bits)
        print(f"// {bits}-bit ({1<<bits} levels), converged in {iters} iters, MSE={mse:.6f}")
        print(f"static const float TQ{bits}_CODEBOOK[{1<<bits}] = {{")
        print("    " + ", ".join(f"{v:.8f}f" for v in c))
        print("};")
        print()
