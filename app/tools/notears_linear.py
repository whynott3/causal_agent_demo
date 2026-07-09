"""轻量 NOTEARS 线性实现（基于 xunzheng/notears 思路改写）。

参考实现：
https://github.com/xunzheng/notears

本模块仅保留项目所需最小能力：
- 输入数值矩阵 X (n, d)
- 输出估计权重矩阵 W (d, d)，其中 W[i, j] 表示 i -> j 的边权
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


def _loss_l2(W: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    """L2 loss and gradient."""
    n = X.shape[0]
    R = X - X @ W
    loss = 0.5 / n * np.sum(R * R)
    G_loss = -1.0 / n * X.T @ R
    return float(loss), G_loss


def _acyclicity(W: np.ndarray) -> tuple[float, np.ndarray]:
    """Acyclicity constraint h(W) and gradient."""
    d = W.shape[0]
    E = slin.expm(W * W)
    h = float(np.trace(E) - d)
    G_h = E.T * W * 2.0
    return h, G_h


def _adj(w: np.ndarray) -> np.ndarray:
    d = int(math.sqrt(w.shape[0]))
    return w.reshape([d, d])


def estimate_notears_adjacency(
    X: np.ndarray,
    *,
    lambda1: float = 0.01,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
) -> np.ndarray:
    """估计 NOTEARS 线性邻接矩阵。

    Args:
        X: shape (n, d) 的数值矩阵。
        lambda1: L1 正则系数。
        max_iter: 增广拉格朗日外层迭代次数。
        h_tol: DAG 约束容忍阈值。
        rho_max: 罚项上限。

    Returns:
        W_est: shape (d, d)，W[i, j] 表示 i -> j 的权重。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵。")
    n, d = X.shape
    if n < 5 or d < 2:
        raise ValueError("NOTEARS 至少需要 n>=5 且 d>=2。")

    # 用变量拆分实现 L1：W = W_pos - W_neg, 且 W_pos/W_neg >= 0
    w_est = np.zeros(2 * d * d, dtype=float)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = []
    for i in range(d):
        for j in range(d):
            if i == j:
                bnds += [(0.0, 0.0), (0.0, 0.0)]
            else:
                bnds += [(0.0, None), (0.0, None)]

    def _func(w: np.ndarray) -> tuple[float, np.ndarray]:
        w_pos = w[: d * d]
        w_neg = w[d * d :]
        W = _adj(w_pos - w_neg)
        loss, G_loss = _loss_l2(W, X)
        h_val, G_h = _acyclicity(W)
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * np.sum(w)
        G_smooth = G_loss + (rho * h_val + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return float(obj), g_obj

    for _ in range(max_iter):
        w_new: np.ndarray | None = None
        h_new = 0.0
        while rho < rho_max:
            sol = sopt.minimize(
                fun=_func,
                x0=w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds,
                options={"maxiter": 1500},
            )
            w_new = np.asarray(sol.x, dtype=float)
            W_new = _adj(w_new[: d * d] - w_new[d * d :])
            h_new, _ = _acyclicity(W_new)
            if h_new <= 0.25 * h:
                break
            rho *= 10.0
        if w_new is None:
            break
        w_est = w_new
        h = h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est[: d * d] - w_est[d * d :])
    np.fill_diagonal(W_est, 0.0)
    return W_est

