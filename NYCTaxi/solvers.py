"""Solvers for the update step.

amle_pgd    : Algorithm 2 (proximal gradient with the stopping test of Condition 1) for
              min_theta  L(theta) + eta * 0.5 * ||theta||_p^2,  L = logistic negative log-likelihood.
              The step size 1/L is chosen by backtracking and never falls below 1/L_max,
              where L_max = lambda_max(Phi^T Phi) / 4 bounds the Hessian of L.
amle_mosek  : the same problem solved to optimality by MOSEK.
lasso_mosek : min_theta L(theta) + lam * ||theta||_1 solved by MOSEK (ILQX).
"""
import time
import numpy as np
from numba import njit


# ---------------------------------------------------------------- proximal step
@njit(cache=True)
def _v_of_s(g, lam, qm1, w):
    """Solve w^(q-1) + lam * w = g coordinatewise (w = v^(p-1)); return sum v^p."""
    tot = 0.0
    for i in range(g.shape[0]):
        gi = g[i]
        if gi == 0.0:
            w[i] = 0.0
            continue
        wi = gi ** (1.0 / qm1)            # root for lam = 0, an upper bound
        if lam > 0.0 and gi / lam < wi:
            wi = gi / lam
        for _ in range(60):               # Newton from the right on a convex increasing function
            h = wi ** qm1 + lam * wi - gi
            d = qm1 * wi ** (qm1 - 1.0) + lam
            wn = wi - h / d
            if wn <= 0.0:
                wn = 0.5 * wi
            if abs(wn - wi) <= 1e-15 * wi:
                wi = wn
                break
            wi = wn
        w[i] = wi
        tot += wi ** (qm1 + 1.0)          # v^p = w^(p/(p-1)) = w^q
    return tot


@njit(cache=True)
def prox_half_sq_pnorm(gamma, c, q):
    """argmin_u  c/2 ||u||_p^2 + 1/2 ||u - gamma||_2^2,  p = q/(q-1), q >= 2."""
    k = gamma.shape[0]
    out = np.zeros(k)
    g = np.abs(gamma)
    if c <= 0.0 or g.max() == 0.0:
        return gamma.copy()
    qm1 = q - 1.0
    p = q / qm1
    e = (2.0 - p) / p                     # ||v||_p^(2-p) = (sum v^p)^e
    w = np.zeros(k)
    if e == 0.0:                          # p = 2: u = gamma / (1 + c)
        return gamma / (1.0 + c)
    # root of F(s) = (sum v(s)^p)^e - s, decreasing in s, on [0, hi]
    lo = 0.0
    hi = _v_of_s(g, 0.0, qm1, w) ** e
    flo = hi
    fhi = _v_of_s(g, c * hi, qm1, w) ** e - hi
    s = hi
    side = 0
    for _ in range(200):
        if fhi == 0.0:
            s = hi
            break
        s = (lo * fhi - hi * flo) / (fhi - flo)      # regula falsi (Illinois)
        fs = _v_of_s(g, c * s, qm1, w) ** e - s
        if abs(fs) <= 1e-14 * (abs(s) + 1e-300) or hi - lo <= 1e-15 * hi:
            break
        if fs > 0.0:
            lo = s; flo = fs
            if side == 1:
                fhi *= 0.5
            side = 1
        else:
            hi = s; fhi = fs
            if side == -1:
                flo *= 0.5
            side = -1
    _v_of_s(g, c * s, qm1, w)
    for i in range(k):
        out[i] = np.sign(gamma[i]) * w[i] ** qm1
    return out


# ---------------------------------------------------------------- AMLE-PGD
def _loss_from_index(u, y):
    return float(np.sum(np.logaddexp(0.0, u)) - y @ u)


def amle_pgd(Phi, y, eta, eps, theta0, L_max, L_init, q, max_iter=100000):
    """Returns theta, info(dict). Stops when ||G||_2 <= eps (certifies Condition 1)."""
    t0 = time.perf_counter()
    th = theta0.copy()
    u = Phi @ th
    f = _loss_from_index(u, y)
    g = Phi.T @ (1.0 / (1.0 + np.exp(-u)) - y)
    L = min(max(L_init, 1e-12), L_max)
    n_grad = 1; n_trial = 0; Gn = np.inf
    for j in range(1, max_iter + 1):
        L = max(L * 0.5, 1e-12)
        while True:
            thn = prox_half_sq_pnorm(th - g / L, eta / L, q)
            d = thn - th
            un = Phi @ thn
            fn = _loss_from_index(un, y)
            n_trial += 1
            gn = None
            if L >= L_max:
                break
            dd = d @ d
            if abs(g @ d) + 0.5 * L * dd > 1e-8 * abs(f):
                # sufficient decrease of the quadratic upper model
                if fn <= f + g @ d + 0.5 * L * dd:
                    break
            else:
                # decrease below rounding of f: use the gradient form, which implies it by convexity
                gn = Phi.T @ (1.0 / (1.0 + np.exp(-un)) - y)
                if (gn - g) @ d <= 0.5 * L * dd:
                    break
            L = min(2.0 * L, L_max)
        if gn is None:
            gn = Phi.T @ (1.0 / (1.0 + np.exp(-un)) - y)
        n_grad += 1
        G = -L * d + gn - g
        th, u, f, g = thn, un, fn, gn
        Gn = float(np.sqrt(G @ G))
        if Gn <= eps:
            break
    return th, dict(time=time.perf_counter() - t0, iters=j, trials=n_trial, L=L, G=Gn)


def amle_apgd(Phi, y, eta, eps, theta0, L_max, L_init, q, max_iter=100000):
    """Accelerated variant of amle_pgd: extrapolation with adaptive restart.

    Every iterate is a proximal gradient step from the extrapolated point z, so the same
    test ||G||_2 <= eps with G = L (z - theta+) + grad L(theta+) - grad L(z) certifies Condition 1.
    """
    t0 = time.perf_counter()
    th = theta0.copy(); u = Phi @ th
    z, uz = th, u
    L = min(max(0.5 * L_init, 1e-12), L_max)
    tk = 1.0; n_trial = 0; Gn = np.inf
    for j in range(1, max_iter + 1):
        fz = _loss_from_index(uz, y)
        gz = Phi.T @ (1.0 / (1.0 + np.exp(-uz)) - y)
        while True:
            thn = prox_half_sq_pnorm(z - gz / L, eta / L, q)
            d = thn - z
            un = Phi @ thn
            n_trial += 1
            gn = Phi.T @ (1.0 / (1.0 + np.exp(-un)) - y)
            dd = d @ d
            if L >= L_max:
                break
            if abs(gz @ d) + 0.5 * L * dd > 1e-8 * abs(fz):
                if _loss_from_index(un, y) <= fz + gz @ d + 0.5 * L * dd:
                    break
            elif (gn - gz) @ d <= 0.5 * L * dd:
                break
            L = min(2.0 * L, L_max)
        G = -L * d + gn - gz
        Gn = float(np.sqrt(G @ G))
        if Gn <= eps:
            th = thn
            break
        if (z - thn) @ (thn - th) > 0.0:          # restart
            tk = 1.0; z, uz = thn, un
        else:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)); b = (tk - 1.0) / tn
            z = thn + b * (thn - th); uz = un + b * (un - u); tk = tn
        th, u = thn, un
    return th, dict(time=time.perf_counter() - t0, iters=j, trials=n_trial, L=L, G=Gn)


# ---------------------------------------------------------------- MOSEK
def _logistic_model(Phi, y, name):
    from mosek.fusion import Model, Domain, Expr, Matrix
    n, k = Phi.shape
    M = Model(name)
    M.setSolverParam("numThreads", 1)
    M.setSolverParam("intpntCoTolRelGap", 1e-9)
    M.setSolverParam("intpntCoTolPfeas", 1e-9)
    M.setSolverParam("intpntCoTolDfeas", 1e-9)
    th = M.variable("theta", k, Domain.unbounded())
    t = M.variable(n)
    # softplus(z_i) <= t_i with z = -(2y-1) * Phi theta
    S = Matrix.dense(-(2.0 * y - 1.0)[:, None] * Phi)
    z = Expr.mul(S, th)
    u = M.variable(n, Domain.unbounded()); v = M.variable(n, Domain.unbounded())
    M.constraint(Expr.add(u, v), Domain.lessThan(1.0))
    one = Expr.constTerm(n, 1.0)
    M.constraint(Expr.hstack(u, one, Expr.sub(z, t)), Domain.inPExpCone())
    M.constraint(Expr.hstack(v, one, Expr.neg(t)), Domain.inPExpCone())
    return M, th, Expr.sum(t)


def _solve(M, th, t_build):
    """Returns (theta, info). theta is None when MOSEK returns no usable solution."""
    from mosek.fusion import AccSolutionStatus, SolutionStatus
    M.acceptedSolutionStatus(AccSolutionStatus.Anything)
    t0 = time.perf_counter()
    M.solve()
    wall = time.perf_counter() - t0
    status = M.getPrimalSolutionStatus()
    ok = status in (SolutionStatus.Optimal, SolutionStatus.Feasible, SolutionStatus.Unknown)
    try:
        sol = np.array(th.level()) if ok else None
    except Exception:
        sol = None
    if sol is not None and not np.all(np.isfinite(sol)):
        sol = None
    info = dict(time=M.getSolverDoubleInfo("optimizerTime"), wall=wall, build=t_build,
                iters=M.getSolverIntInfo("intpntIter"), optimal=status == SolutionStatus.Optimal)
    M.dispose()
    return sol, info


def amle_mosek(Phi, y, eta, q):
    from mosek.fusion import Domain, Expr, ObjectiveSense
    t0 = time.perf_counter()
    M, th, loss = _logistic_model(Phi, y, "amle")
    k = Phi.shape[1]
    p = q / (q - 1.0)
    # ||theta||_p <= w :  |theta_i| <= r_i^(1/p) w^(1-1/p),  sum r_i = w
    w = M.variable(Domain.greaterThan(0.0)); r = M.variable(k, Domain.greaterThan(0.0))
    M.constraint(Expr.sub(Expr.sum(r), w), Domain.equalsTo(0.0))
    M.constraint(Expr.hstack(r, Expr.repeat(w, k, 0), th), Domain.inPPowerCone(1.0 / p))
    # w^2 <= 2 * s * (1/2)  ->  s >= w^2
    s = M.variable(Domain.greaterThan(0.0))
    M.constraint(Expr.vstack(s, Expr.constTerm(0.5), w), Domain.inRotatedQCone())
    M.objective(ObjectiveSense.Minimize, Expr.add(loss, Expr.mul(0.5 * eta, s)))
    return _solve(M, th, time.perf_counter() - t0)


def lasso_mosek(Phi, y, lam):
    from mosek.fusion import Domain, Expr, ObjectiveSense
    t0 = time.perf_counter()
    M, th, loss = _logistic_model(Phi, y, "lasso")
    k = Phi.shape[1]
    a = M.variable(k, Domain.greaterThan(0.0))
    M.constraint(Expr.sub(a, th), Domain.greaterThan(0.0))
    M.constraint(Expr.add(a, th), Domain.greaterThan(0.0))
    M.objective(ObjectiveSense.Minimize, Expr.add(loss, Expr.mul(lam, Expr.sum(a))))
    return _solve(M, th, time.perf_counter() - t0)
