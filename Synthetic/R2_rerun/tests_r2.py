"""Checks for the R2 rerun: environment = R1 environment, solver correctness."""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["R2_DESIGN"] = "base"   # the checks against R1 refer to the base design
import env_r2 as env          # noqa: E402
import solvers_r2 as S        # noqa: E402

R1_RESULTS = os.path.join(env.LINEAR_DIR, "results")


def test_environment_matches_r1(n_t: int = 300) -> None:
    sp = env.signal_power()
    with open(os.path.join(R1_RESULTS, "Low_SNR0", "snr_calibration.csv")) as f:
        cal = next(csv.DictReader(f))
    assert abs(sp - float(cal["signal_power"])) < 1e-8 * sp, (sp, cal["signal_power"])
    for snr, folder in (("low", "Low_SNR0"), ("mid", "Mid_SNR0"), ("high", "High_SNR0")):
        with open(os.path.join(R1_RESULTS, folder, "snr_calibration.csv")) as f:
            cal = next(csv.DictReader(f))
        assert abs(env.noise_scale(snr) - float(cal["noise_scale"])) < 1e-9 * env.noise_scale(snr)
    ref = {}
    with open(os.path.join(R1_RESULTS, "Low_SNR0", "toy_oracle_revenue_by_k_t.csv")) as f:
        for row in csv.DictReader(f):
            if int(row["t"]) < n_t:
                ref[(int(row["k"]), int(row["t"]))] = float(row["oracle_revenue"])
    sigma = env.noise_scale("low")
    worst = 0.0
    for k_inst in range(1, 6):
        X = env.feature_trajectory(k_inst, n_t)
        for t in range(n_t):
            o = env.oracle_revenue(float(X[t] @ env.THETA0), sigma)
            worst = max(worst, abs(o - ref[(k_inst - 1, t)]) / ref[(k_inst - 1, t)])
    assert worst < 1e-10, worst
    print("environment == R1: signal_power, noise scales and oracle revenues match (max rel diff {:.1e})".format(worst))


def test_schedule() -> None:
    is_exp, refit = env.schedule()
    print("schedule: m = {}, exploration periods = {}, refits = {}".format(
        env.M_EXPLORE, int(is_exp.sum()), int(refit.sum())))
    assert env.M_EXPLORE == 27
    assert np.all(is_exp[np.nonzero(refit)[0]])


P_TESTED = (4 / 3, 2.0, env.p_of_q(1.0 + math.log(env.KAPPA)))   # q = 4, 2, 1 + log kappa


def test_prox_against_cvxpy() -> None:
    import cvxpy as cp
    rng = np.random.default_rng(0)
    for p in P_TESTED:
        worst_kkt, worst_gap = 0.0, -np.inf
        for lam in (1e-3, 0.1, 1.0, 10.0, 1e3):
            v = rng.normal(size=40) * rng.choice([0.01, 1.0, 100.0], size=40)
            u = cp.Variable(40)
            cp.Problem(cp.Minimize(lam * 0.5 * cp.square(cp.pnorm(u, p)) + 0.5 * cp.sum_squares(u - v))).solve(
                solver=cp.MOSEK)
            mine = S.prox_half_sq_pnorm(v, lam, p)
            # optimality condition u + lam * grad r(u) = v  (Lemma C.2)
            kkt = np.max(np.abs(mine + lam * S.grad_half_sq_pnorm(mine, p) - v)) / (1 + np.max(np.abs(v)))

            def obj(w):
                return lam * 0.5 * np.sum(np.abs(w) ** p) ** (2.0 / p) + 0.5 * np.sum((w - v) ** 2)

            worst_kkt = max(worst_kkt, kkt)
            worst_gap = max(worst_gap, obj(mine) - obj(u.value))
        assert worst_kkt < 1e-9 and worst_gap <= 1e-8, (p, worst_kkt, worst_gap)
        print("prox of lam*0.5||.||_p^2, p={:.4f}: optimality residual {:.1e}; objective never above MOSEK's".format(
            p, worst_kkt))


def test_pgd_against_mosek() -> None:
    rng = np.random.default_rng(1)
    N, dim = 300, 60
    A = np.hstack([rng.uniform(0.5, 5.0, size=(N, dim - 1)), -rng.choice(env.EXPLORE_GRID, size=(N, 1))])
    v_true = np.concatenate([rng.uniform(0, 1, dim - 1), [5.0]])
    y = A @ v_true + rng.normal(scale=5.0, size=N)
    eta = 50.0
    lam, _ = S.power_iteration(A, None)
    for p in P_TESTED:
        v_ms, _ = S.amle_mosek(A, y, eta, p, 1e4)
        res = S.amle_pgd(A, y, eta, 1e-6, np.full(dim, 0.1), 1.02 * lam, p, 2_000_000)
        foc = S.foc_residual(A, y, res["v"], eta, p)
        gap = (S.objective(A, y, res["v"], eta, p) - S.objective(A, y, v_ms, eta, p))
        print("PGD vs MOSEK, p={:.4f}: iters={}, stop={}, ||G||={:.2e}, ||grad D||={:.2e}, obj gap={:.2e}".format(
            p, res["iters"], res["stop"], res["G"], foc, gap))
        assert res["stop"] == "cert" and abs(gap) < 1e-5 * abs(S.objective(A, y, v_ms, eta, p))
        # certificate is a valid upper bound on the first-order residual when C_k is inactive
        assert foc <= res["G"] * (1 + 1e-6) + 1e-9


def test_spoplus_loss() -> None:
    import torch
    rng = np.random.default_rng(2)
    c = torch.tensor(rng.normal(size=(5, 7)), dtype=torch.float64)
    ch = torch.tensor(rng.normal(size=(5, 7)), dtype=torch.float64, requires_grad=True)
    j = torch.argmax(c, 1)
    loss = (2 * ch - c).max(1).values - 2 * ch.gather(1, j[:, None]).squeeze(1) + c.max(1).values
    assert torch.all(loss >= -1e-12)
    loss0 = (2 * c - c).max(1).values - 2 * c.gather(1, j[:, None]).squeeze(1) + c.max(1).values
    assert torch.allclose(loss0, torch.zeros(5, dtype=torch.float64))
    loss.sum().backward()
    wq = torch.nn.functional.one_hot(torch.argmax(2 * ch.detach() - c, 1), 7)
    ws = torch.nn.functional.one_hot(j, 7)
    assert torch.allclose(ch.grad, 2.0 * (wq - ws).double())
    print("SPO+ (maximization) loss: nonnegative, zero at c_hat = c, gradient = 2(w(2c_hat-c) - w*(c))")


if __name__ == "__main__":
    test_schedule()
    test_environment_matches_r1()
    test_prox_against_cvxpy()
    test_pgd_against_mosek()
    test_spoplus_loss()
    print("all tests passed")
