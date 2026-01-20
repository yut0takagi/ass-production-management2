import numpy as np

def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    pairwise squared Euclidean distances between rows of X and Y
    X: (n, d), Y: (m, d)
    returns: (n, m)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X2 = np.sum(X**2, axis=1, keepdims=True)      # (n, 1)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T    # (1, m)
    return X2 + Y2 - 2.0 * (X @ Y.T)

def _rbf_kernel_from_sqdist(D2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-D2 / (2.0 * sigma**2))

def _median_heuristic_sigma(X: np.ndarray, Y: np.ndarray, max_points: int = 1000, seed: int = 0) -> float:
    """
    Median heuristic for RBF bandwidth.
    """
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    n = Z.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Z = Z[idx]

    D2 = _sq_dists(Z, Z)
    # 対角0を除いた上三角から距離を取る
    iu = np.triu_indices(D2.shape[0], k=1)
    vals = D2[iu]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    med = np.median(vals)
    return float(np.sqrt(0.5 * med)) if med > 0 else 1.0  # よく使われる変形

def mmd_rbf_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Unbiased estimator of MMD^2 with RBF kernel.
    X: (n, d), Y: (m, d)
    returns: MMD (not squared)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Unbiased MMD needs at least 2 samples in each set.")

    if sigma is None:
        sigma = _median_heuristic_sigma(X, Y)

    Kxx = _rbf_kernel_from_sqdist(_sq_dists(X, X), sigma)
    Kyy = _rbf_kernel_from_sqdist(_sq_dists(Y, Y), sigma)
    Kxy = _rbf_kernel_from_sqdist(_sq_dists(X, Y), sigma)

    # 対角を除いた平均（unbiased）
    sum_Kxx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    sum_Kyy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    sum_Kxy = np.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2.0 * sum_Kxy
    # 数値誤差で負になるのを抑制
    return float(np.sqrt(max(mmd2, 0.0)))