"""Simulation environment for the R2 rerun of Section 5.1 (Case Study I).

The data-generating process is identical to the R1 experiment
(``Linear_code/Toy_main.py`` / ``opensource_Gaussian/run_Gaussian_comparison.py``):
same dimension, same true parameters (seed 4242), same feature sampler, same price
grids and the same oracle.  ``tests_r2.py`` checks that the oracle revenues coincide
with ``Linear_code/results/*_SNR0/toy_oracle_revenue_by_k_t.csv``.

What changes relative to R1 (to match the paper's Algorithm 1):
  * every exploration period enters the estimation history I_k;
  * the exploration blocks are those of Algorithm 1 with m_0 = m_k = floor(kappa^(1-2/q)) = 27:
    M_0 = [1, m_0] and M_k = [k^2, k^2 + m_k - 1] for k >= 1 (periods numbered from 1);
    the exploration periods are the union of the blocks;
  * the estimate is refit at the end of each block M_k and used until the next refit;
  * noise and exploration prices are drawn from an independent stream per trial k
    (R1 used np.random.seed(k + t), which makes trials shifted copies of each other)
    and are common to all methods (common random numbers).

Designs (environment variable R2_DESIGN, set by ``run_r2.py --design``):
  * base  : the R1 parameters (approximately sparse: 13% of the entries of theta0 in [1, 2], the rest
            in [0.001, 0.015]);
  * dense : theta0 has i.i.d. Gaussian entries with the same mean and standard deviation as the R1
            vector, so the mean intercept, the spread of theta0^T x and ||theta0||_2 (hence sigma at
            each SNR) are unchanged and only the sparsity structure differs.  Features, prices and
            gamma0 are the same as in base.
  * heavy : Case II of the q-sensitivity study (heavy-tailed features, sparse theta0; see below).
  * heavy100 : heavy with prices measured in units of 100 in the estimation problem (PRICE_UNIT), which
            reduces the condition number of A^T A from about 1.8e7 to about 7e4; demand, revenue, the
            price grids and the oracle are unchanged.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR_DIR = os.path.dirname(HERE)
if LINEAR_DIR not in sys.path:
    sys.path.insert(0, LINEAR_DIR)

from truncate_normal_sample_time_varying import l4_heavy_tail_sample, truncate_normal_sample_time_varying  # noqa: E402

# ---------------------------------------------------------------------------
# Problem constants (identical to R1)
# ---------------------------------------------------------------------------
N_PROD = 1
D_FEAT = 3 ** 6                      # 729 contextual features
KAPPA = D_FEAT + 1                   # parameter dimension: (theta0, gamma)
Q_NORM = 4.0                         # q in Assumption 2(ii)
P_REG = Q_NORM / (Q_NORM - 1.0)      # regularizer exponent p = 4/3
T_HORIZON = 10_000
K_TRIALS = 10
DATA_SEED = 4242                     # true parameters (as in R1)
NOISE_SEED = 20261017                # per-trial noise / exploration stream (new)

FEATURE_MU, FEATURE_SIGMA = 0.0, 15.0
FEATURE_A, FEATURE_B = 0.5, 5.0
FEATURE_C = 100.0                    # l_q budget of the sampler (never active)
W_L1 = 1.0e4                         # C_k = R^kappa; l1 bound kept as a numerical safeguard (never active)

PRICE_GRID = np.linspace(30.0, 50.0, 401)
EXPLORE_GRID = np.linspace(30.0, 50.0, 6)
TARGET_SNR = {"low": 0.1, "mid": 1.0, "high": 10.0}
N_CALIB = 5000

M_EXPLORE = int(math.floor(KAPPA ** (1.0 - 2.0 / Q_NORM)))   # = 27
M0_EXPLORE = M_EXPLORE

# True parameters, generated exactly as in R1.
_rng = np.random.default_rng(DATA_SEED)
THETA0 = _rng.uniform(0.001, 0.015, size=(N_PROD, N_PROD * D_FEAT))
_hi = _rng.random(size=(N_PROD, N_PROD * D_FEAT)) < 0.15
THETA0[_hi] = _rng.uniform(1.0, 2.0, size=int(np.sum(_hi)))
THETA0 = THETA0.ravel().astype(np.float64)

DESIGNS = ("base", "dense", "heavy", "heavy100")
DESIGN = os.environ.get("R2_DESIGN", "base")
if DESIGN not in DESIGNS:
    raise ValueError("unknown R2_DESIGN {!r}; choose from {}".format(DESIGN, DESIGNS))
DENSE_SEED = 777
L4_MEAN_SHIFT = 0.3                  # heavy design only (R1 used 0.05)
GAMMA0 = 5.0
if DESIGN == "dense":
    _z = np.random.default_rng(DENSE_SEED).standard_normal(D_FEAT)
    _z = (_z - _z.mean()) / _z.std()
    THETA0 = THETA0.mean() + THETA0.std() * _z
elif DESIGN in ("heavy", "heavy100"):
    # Case II of the q-sensitivity study (Appendix B): heavy-tailed features (R1 sampler l4_heavy_tail_sample:
    # N(0, 0.1^2) coordinates shifted by L4_MEAN_SHIFT, a collinear pair whose second coordinate receives jumps
    # of random sign and size U(5, 10) with probability 0.05), the sparse parameter vector of R1, theta_1 = 0.1,
    # horizon 5,000, and prices on [0, E[theta0^T x] / theta_1], the price at which the expected demand of an
    # average customer vanishes (501 decision prices, 11 exploration prices).
    THETA0 = np.zeros(D_FEAT)
    THETA0[:5] = (50.0, 20.0, 0.3, 0.2, 0.1)
    GAMMA0 = 0.1
    T_HORIZON = 5_000
    _P_HI = L4_MEAN_SHIFT * float(THETA0.sum()) / GAMMA0      # every coordinate of x has mean L4_MEAN_SHIFT
    PRICE_GRID = np.linspace(0.0, _P_HI, 501)
    EXPLORE_GRID = np.linspace(0.0, _P_HI, 11)
# Price unit of the estimation problem: the regression row is a = (x, -p / PRICE_UNIT), so the fitted price
# coefficient is gamma * PRICE_UNIT.  Demand, revenue and the price grids stay in the original unit.
PRICE_UNIT = 100.0 if DESIGN == "heavy100" else 1.0
V_TRUE = np.concatenate([THETA0, [GAMMA0 * PRICE_UNIT]])   # v = (theta0, gamma * unit); demand mean = a^T v


# ---------------------------------------------------------------------------
# Features, noise, exploration draws
# ---------------------------------------------------------------------------
def sample_feature(k_inst: int, t: int) -> np.ndarray:
    if DESIGN in ("heavy", "heavy100"):
        return l4_heavy_tail_sample(D_FEAT, int(k_inst), int(t), 0, mean_shift=L4_MEAN_SHIFT).astype(np.float64)
    return truncate_normal_sample_time_varying(
        4, FEATURE_C, FEATURE_A, FEATURE_B, D_FEAT,
        int(k_inst), int(t), 0, mu=FEATURE_MU, sigma=FEATURE_SIGMA,
    ).astype(np.float64)


@lru_cache(maxsize=2)
def feature_trajectory(k_inst: int, T: int = T_HORIZON) -> np.ndarray:
    X = np.empty((int(T), D_FEAT), dtype=np.float64)
    for t in range(int(T)):
        X[t] = sample_feature(k_inst, t)
    X.setflags(write=False)
    return X


@lru_cache(maxsize=4)
def trial_draws(k_inst: int, T: int = T_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal noise and exploration-price indices for trial k (common to all methods)."""
    rng = np.random.default_rng([NOISE_SEED, int(k_inst)])
    z = rng.standard_normal(int(T))
    idx = rng.integers(0, EXPLORE_GRID.size, size=int(T))
    z.setflags(write=False)
    idx.setflags(write=False)
    return z, idx


# ---------------------------------------------------------------------------
# SNR calibration (same definition and sample as R1)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def signal_power() -> float:
    """||theta0||^2 * lambda_min(E[x x^T]), estimated as in R1 (k = 0, N_CALIB draws)."""
    Xc = np.empty((N_CALIB, D_FEAT), dtype=np.float64)
    for i in range(N_CALIB):
        Xc[i] = sample_feature(0, i)
    lam_min = max(float(np.linalg.eigvalsh(Xc.T @ Xc / N_CALIB)[0]), 0.0)
    return float(THETA0 @ THETA0) * lam_min


def noise_scale(snr: str) -> float:
    return float(math.sqrt(signal_power() / TARGET_SNR[snr]))


# ---------------------------------------------------------------------------
# Revenue
# ---------------------------------------------------------------------------
def expected_positive_part(mu: np.ndarray | float, sigma: float) -> np.ndarray:
    """E[max(mu + eps, 0)] for eps ~ N(0, sigma^2)."""
    mu = np.asarray(mu, dtype=np.float64)
    z = mu / sigma
    return mu * 0.5 * (1.0 + erf(z / math.sqrt(2.0))) + sigma * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_revenue(p: float, mu_x: float, sigma: float) -> float:
    """p * E[max(theta0^T x - gamma0 p + eps, 0)] (same revenue notion as the R1 oracle)."""
    return float(p * expected_positive_part(mu_x - GAMMA0 * p, sigma))


def oracle_revenue(mu_x: float, sigma: float) -> float:
    rev = PRICE_GRID * expected_positive_part(mu_x - GAMMA0 * PRICE_GRID, sigma)
    return float(rev.max())


def exploit_price(v_hat: np.ndarray, x: np.ndarray) -> float:
    """Plug-in price on PRICE_GRID: argmax_p p * max(theta_hat^T x - gamma_hat p, 0) (as in R1)."""
    d_hat = float(v_hat[:D_FEAT] @ x) - float(v_hat[D_FEAT]) / PRICE_UNIT * PRICE_GRID
    return float(PRICE_GRID[int(np.argmax(PRICE_GRID * np.maximum(d_hat, 0.0)))])


# ---------------------------------------------------------------------------
# Exploration schedule (Algorithm 1)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def schedule(T: int = T_HORIZON, m0: int = M0_EXPLORE, m: int = M_EXPLORE) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_explore[t], refit_at[t]) for t = 0, ..., T-1; index t is period t + 1 of Algorithm 1.

    Algorithm 1 with m_k = m: M_0 = [1, m0] and M_k = [k^2, k^2 + m - 1] for k >= 1, truncated at T.
    The blocks with k^2 + m - 1 >= (k+1)^2 overlap (here k <= 12); the exploration periods are the union of
    the blocks and all other periods are exploitation periods.  Epoch k consists of periods k^2, ..., (k+1)^2 - 1.
    The estimate is refit at the last period of each block (blocks ending at the same period give one refit),
    except at period T, after which no customer arrives.  With T = 10,000 and m0 = m = 27: periods 1-195 are
    all exploration, 2,518 exploration periods in total, 99 refits.
    """
    T, m0, m = int(T), int(m0), int(m)
    is_explore = np.zeros(T, dtype=bool)
    refit_at = np.zeros(T, dtype=bool)
    is_explore[:m0] = True
    ends = {m0}
    k = 1
    while k * k <= T:
        start, end = k * k, min(k * k + m - 1, T)   # periods, inclusive
        is_explore[start - 1:end] = True
        ends.add(end)
        k += 1
    for end in ends:
        if end < T:
            refit_at[end - 1] = True
    is_explore.setflags(write=False)
    refit_at.setflags(write=False)
    return is_explore, refit_at


def eta_k(eta_coef: float, n_hist: int, q: float = Q_NORM) -> float:
    """eta_k = eta0 * kappa^(1/q - 1/2) * sqrt(|I_k|) (corrected Proposition 1)."""
    return float(eta_coef) * KAPPA ** (1.0 / q - 0.5) * math.sqrt(n_hist)


def eps_k(eps_coef: float, n_hist: int, q: float = Q_NORM) -> float:
    """eps_k = eps0 * kappa^(1/2 - 1/q) * sqrt(|I_k|) (Theorem 1, eq. (7))."""
    return float(eps_coef) * KAPPA ** (0.5 - 1.0 / q) * math.sqrt(n_hist)


def p_of_q(q: float) -> float:
    """Regularizer exponent p = q / (q - 1) (conjugate of the feature norm q)."""
    return float(q) / (float(q) - 1.0)


def lasso_lambda(lam_coef: float, period: int) -> float:
    """ILQX penalty lambda_t = lambda0 t^(1/4) sqrt(log kappa + log t), t = current period (Ban and Keskin 2021)."""
    t = max(int(period), 1)
    return float(lam_coef) * t ** 0.25 * math.sqrt(math.log(D_FEAT + 1) + math.log(t + 1))
