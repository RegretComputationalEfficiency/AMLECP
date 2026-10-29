"""Estimators for the R2 rerun of Section 5.1.

All batch estimators minimize the same regularized least-squares objective
    D_k(v) = 0.5 * ||y - A v||_2^2 + eta_k * r(v),   r(v) = 0.5 * ||v||_p^2,  p = 4/3,
where the rows of A are a_t = (x_t, -p_t) and y_t is the observed demand, i.e. the
(unit-variance) Gaussian negative log-likelihood of the linear demand model.  The data
term is NOT averaged over |I_k|, matching D_k in the paper.

  * AMLE-PGD  : proximal gradient (Algorithm 2) with the certificate stop
                ||G^{j+1}||_2 <= eps_k, which verifies Condition 1.
  * AMLE-MOSEK: the same problem solved to optimality by MOSEK (eps_k = 0).
  * ILQX      : Lasso, 0.5 ||y - A v||^2 + lambda_t ||v||_1 (Ban and Keskin 2021).
  * AdaGrad   : recursive AdaGrad on the per-period squared loss (port of R1 AdaGrad.py).
  * SPO+      : SPO+ with doubly-robust labels; the one-hot price choice is solved by an
                exact argmax instead of a Gurobi MILP (identical solution).
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Proximal operator of lam * 0.5 * ||u||_p^2  (Lemma C.2)
# ---------------------------------------------------------------------------


def _scalar_roots(a: np.ndarray, c: float, p: float) -> np.ndarray:
    """Unique t >= 0 with t + c * t^(p-1) = a, elementwise (a >= 0, c >= 0)."""
    if c <= 0.0:
        return a.copy()
    if abs(p - 2.0) < 1e-12:
        return a / (1.0 + c)
    if abs(p - 4.0 / 3.0) < 1e-12:
        # w = t^(1/3) solves w^3 + c w - a = 0; stable real root via the sinh form.
        s = math.sqrt(c / 3.0)
        w = 2.0 * s * np.sinh(np.arcsinh(1.5 * a / (c * s)) / 3.0)
        # one Newton polish step on w^3 + c w - a
        w = w - (w ** 3 + c * w - a) / (3.0 * w * w + c)
        w = np.maximum(w, 0.0)
        return w ** 3
    # generic p in (1, 2): safeguarded Newton on h(t) = t + c t^(p-1) - a
    t = np.minimum(a, (a / c) ** (1.0 / (p - 1.0)))
    for _ in range(60):
        tp = np.maximum(t, 1e-300)
        h = tp + c * tp ** (p - 1.0) - a
        dh = 1.0 + c * (p - 1.0) * tp ** (p - 2.0)
        t_new = np.clip(tp - h / dh, 0.0, a)
        if np.max(np.abs(t_new - t)) <= 1e-15 * (1.0 + np.max(a)):
            t = t_new
            break
        t = t_new
    return t


def prox_half_sq_pnorm(v: np.ndarray, lam: float, p: float) -> np.ndarray:
    """argmin_u lam * 0.5 * ||u||_p^2 + 0.5 * ||u - v||_2^2  (Lemma C.2)."""
    if lam <= 0.0:
        return v.copy()
    a = np.abs(v)
    vp = float(np.sum(a ** p) ** (1.0 / p))
    if vp == 0.0:
        return np.zeros_like(v)
    if abs(p - 2.0) < 1e-12:
        return v / (1.0 + lam)

    def gap(s: float) -> float:
        t = _scalar_roots(a, lam * s ** (2.0 - p), p)
        return s - float(np.sum(t ** p) ** (1.0 / p))

    s_star = brentq(gap, 0.0, vp, xtol=1e-14 * vp, rtol=4 * np.finfo(float).eps, maxiter=200)
    return np.sign(v) * _scalar_roots(a, lam * s_star ** (2.0 - p), p)


def grad_half_sq_pnorm(v: np.ndarray, p: float) -> np.ndarray:
    """Gradient of 0.5 * ||v||_p^2."""
    a = np.abs(v)
    vp = float(np.sum(a ** p) ** (1.0 / p))
    if vp == 0.0:
        return np.zeros_like(v)
    return vp ** (2.0 - p) * a ** (p - 1.0) * np.sign(v)


def objective(A: np.ndarray, y: np.ndarray, v: np.ndarray, eta: float, p: float) -> float:
    r = y - A @ v
    return 0.5 * float(r @ r) + eta * 0.5 * float(np.sum(np.abs(v) ** p) ** (2.0 / p))


def foc_residual(A: np.ndarray, y: np.ndarray, v: np.ndarray, eta: float, p: float) -> float:
    """||grad D_k(v)||_2 (Condition 1 when C_k is inactive)."""
    g = -(A.T @ (y - A @ v)) + eta * grad_half_sq_pnorm(v, p)
    return float(np.linalg.norm(g))


# ---------------------------------------------------------------------------
# AMLE-PGD: proximal gradient with certificate stop (Algorithm 2)
# ---------------------------------------------------------------------------


def power_iteration(A: np.ndarray, u0: np.ndarray | None, iters: int = 50, tol: float = 1e-10) -> tuple[float, np.ndarray]:
    """Largest eigenvalue of A^T A (warm-started)."""
    u = np.ones(A.shape[1]) if u0 is None else u0.copy()
    u /= np.linalg.norm(u)
    lam = 0.0
    for _ in range(iters):
        w = A.T @ (A @ u)
        lam_new = float(np.linalg.norm(w))
        u = w / lam_new
        if abs(lam_new - lam) <= tol * lam_new:
            lam = lam_new
            break
        lam = lam_new
    return lam, u


def amle_pgd(A: np.ndarray, y: np.ndarray, eta: float, eps: float, v0: np.ndarray, L: float,
             p: float, max_iter: int) -> dict:
    """Proximal gradient on 0.5||y - Av||^2 + eta * 0.5||v||_p^2 with step 1/L.

    Stops at the first j with ||G^{j+1}||_2 <= eps, where
    G^{j+1} = (v^j - v^{j+1})/alpha + grad f(v^{j+1}) - grad f(v^j)
    belongs to grad f(v^{j+1}) + eta grad r(v^{j+1}) (+ N_C), i.e. Condition 1 holds.
    """
    alpha = 1.0 / L
    v = v0.copy()
    g = -(A.T @ (y - A @ v))
    G_norm = math.inf
    for j in range(1, max_iter + 1):
        v_new = prox_half_sq_pnorm(v - alpha * g, alpha * eta, p)
        g_new = -(A.T @ (y - A @ v_new))
        G_norm = float(np.linalg.norm((v - v_new) / alpha + g_new - g))
        v, g = v_new, g_new
        if G_norm <= eps:
            return {"v": v, "iters": j, "stop": "cert", "G": G_norm}
    return {"v": v, "iters": max_iter, "stop": "maxiter", "G": G_norm}


# ---------------------------------------------------------------------------
# AMLE-MOSEK and ILQX (cvxpy + MOSEK), same objective scale as AMLE-PGD
# ---------------------------------------------------------------------------


def _solve_with_fallback(build) -> tuple[np.ndarray, str]:
    """Solve with MOSEK; if MOSEK stalls (status UNKNOWN), retry on the objective scaled by
    1/N (same minimizer), then with CLARABEL -- the same safeguard as R1's cvxMLEpnorm."""
    import cvxpy as cp

    for label, scale, solver in (("mosek", False, cp.MOSEK), ("mosek_scaled", True, cp.MOSEK),
                                 ("clarabel", True, cp.CLARABEL)):
        v, prob = build(scale)
        try:
            prob.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            continue
        if v.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(v.value, dtype=np.float64), label
    raise RuntimeError("all solvers failed")


def amle_mosek(A: np.ndarray, y: np.ndarray, eta: float, p: float, W: float) -> tuple[np.ndarray, str]:
    import cvxpy as cp

    def build(scale: bool):
        v = cp.Variable(A.shape[1])
        obj = 0.5 * cp.sum_squares(y - A @ v) + 0.5 * eta * cp.square(cp.pnorm(v, p))
        if scale:
            obj = obj / A.shape[0]
        return v, cp.Problem(cp.Minimize(obj), [cp.norm1(v) <= W])

    return _solve_with_fallback(build)


def ilqx_lasso(A: np.ndarray, y: np.ndarray, lam: float, W: float) -> tuple[np.ndarray, str]:
    import cvxpy as cp

    def build(scale: bool):
        v = cp.Variable(A.shape[1])
        obj = 0.5 * cp.sum_squares(y - A @ v) + lam * cp.norm1(v)
        if scale:
            obj = obj / A.shape[0]
        return v, cp.Problem(cp.Minimize(obj), [cp.norm1(v) <= W])

    return _solve_with_fallback(build)


# ---------------------------------------------------------------------------
# AdaGrad (port of R1 Linear_code/AdaGrad.py; same initialization and projection)
# ---------------------------------------------------------------------------


def project_l1(v: np.ndarray, W: float) -> np.ndarray:
    a = np.abs(v)
    if a.sum() <= W:
        return v
    s = np.sort(a)[::-1]
    cs = np.cumsum(s)
    rho = np.nonzero(s > (cs - W) / np.arange(1, s.size + 1))[0][-1]
    tau = (cs[rho] - W) / (rho + 1.0)
    return np.sign(v) * np.maximum(a - tau, 0.0)


class AdaGradOnline:
    def __init__(self, dim: int, W: float, lr: float, init: float = 0.5, epsilon: float = 1e-8):
        self.v = np.full(dim, init, dtype=np.float64)
        self.G = np.zeros(dim, dtype=np.float64)
        self.W, self.lr, self.epsilon = float(W), float(lr), float(epsilon)

    def update(self, a: np.ndarray, d: float) -> float:
        t0 = time.perf_counter()
        resid = d - float(a @ self.v)
        grad = -resid * a
        self.G += grad * grad
        self.v = project_l1(self.v - self.lr * grad / (np.sqrt(self.G) + self.epsilon), self.W)
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# SPO+ (Appendix B), exact argmax over the one-hot price choice
# ---------------------------------------------------------------------------


def fit_revenue_quadratic_ls(X: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Least squares R ~ p * beta^T x + beta_{d+1} p^2 on the exploration history (as in R1)."""
    Phi = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float64)
    Phi[:, :-1] = X * p[:, None]
    Phi[:, -1] = p * p
    beta, *_ = np.linalg.lstsq(Phi, p * d, rcond=None)
    return beta


def dr_labels(X: np.ndarray, p: np.ndarray, d: np.ndarray, price_grid: np.ndarray,
              explore_grid: np.ndarray) -> np.ndarray:
    """c_ij = R_hat_ij + 1[p_i = p_j] (R_i - R_hat_ij) / pi_j, pi_j = 1/M_exp on the exploration grid."""
    beta = fit_revenue_quadratic_ls(X, p, d)
    lin = X @ beta[:-1]
    c = price_grid[None, :] * lin[:, None] + beta[-1] * (price_grid ** 2)[None, :]
    j_obs = np.array([int(np.argmin(np.abs(price_grid - pi))) for pi in p])
    on_grid = np.isclose(price_grid[j_obs], p, rtol=1e-6, atol=1e-8)
    in_explore = np.any(np.isclose(price_grid[j_obs][:, None], explore_grid[None, :], rtol=1e-6, atol=1e-8), axis=1)
    ok = on_grid & in_explore
    rows = np.nonzero(ok)[0]
    r_obs = p * d
    pi = 1.0 / explore_grid.size
    c[rows, j_obs[rows]] = c[rows, j_obs[rows]] + (r_obs[rows] - c[rows, j_obs[rows]]) / pi
    return c.astype(np.float32)


def train_spoplus(X: np.ndarray, c: np.ndarray, lr: float, epochs: int, l2: float, hidden: int,
                  seed: int):
    """Train MLP x -> c_hat with the SPO+ loss for the maximization problem max_{w one-hot} c^T w:
        l(c_hat, c) = max_j (2 c_hat_j - c_j) - 2 c_hat_{j*(c)} + max_j c_j
    (the MAXIMIZE branch of PyEPO's SPOPlus; subgradient 2 (w(2 c_hat - c) - w*(c)))."""
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    n_price = c.shape[1]
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_price))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X.astype(np.float32))
    Ct = torch.from_numpy(c)
    j_true = torch.argmax(Ct, dim=1)
    z_true = Ct.max(dim=1).values
    n = Xt.shape[0]
    bs = min(64, max(1, n))
    gen = torch.Generator().manual_seed(int(seed) + 1)
    net.train()
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            ch = net(Xt[idx])
            cb = Ct[idx]
            loss = ((2.0 * ch - cb).max(dim=1).values
                    - 2.0 * ch.gather(1, j_true[idx].unsqueeze(1)).squeeze(1)
                    + z_true[idx]).mean()
            if l2 > 0:
                loss = loss + l2 * sum((w ** 2).sum() for w in net.parameters())
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net
