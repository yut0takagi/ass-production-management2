import numpy as np
from scipy.stats import wasserstein_distance


def sliced_wasserstein_distance(X, Y, n_projections=200, seed=0):
    """
    Sliced Wasserstein Distance (1-Wasserstein) for multivariate samples.
    X: (n, d), Y: (m, d)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    d = X.shape[1]
    if Y.shape[1] != d:
        raise ValueError("X and Y must have the same feature dimension.")

    rng = np.random.default_rng(seed)

    # ランダム方向（単位ベクトル）を生成
    thetas = rng.normal(size=(n_projections, d))
    thetas /= (np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12)

    # 射影して 1次元へ
    X_proj = X @ thetas.T   # (n, n_projections)
    Y_proj = Y @ thetas.T   # (m, n_projections)

    # 各射影で 1D Wasserstein を計算して平均
    dists = [wasserstein_distance(X_proj[:, k], Y_proj[:, k]) for k in range(n_projections)]
    return float(np.mean(dists))

