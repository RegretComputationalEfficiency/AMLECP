from __future__ import annotations

import os

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import json
import time
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Suppress Warnings and Set Up ---
warnings.filterwarnings('ignore')
tqdm.pandas(desc="Pandas Apply Progress")

# --- Config: paths, simulator scenarios, and hyperparameters ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SIMULATOR_DIR = PROJECT_DIR / "Simulator"
# User-built December 2014 test arrivals CSV (see 2014_data/README.md).
# Override with env TEST_DATA_PATH, e.g. after combine/filter/(optional spread) pipeline.
DATA_PATH = Path(
    os.environ.get(
        "TEST_DATA_PATH",
        str(BASE_DIR / "data" / "2014_december_test_arrivals.csv"),
    )
)
TIME_COLUMN = "pickup_time"

# Optional calendar slice. Set DATA_DAY = None to use the full sorted CSV (capped by TOTAL_ARRIVALS_TO_SIMULATE).
#   None                         → no day filter
#   "2014-12-11"                  → single day [HOUR_START:00, HOUR_END:59] (requires HOUR_END >= HOUR_START)
#   ("2014-12-11", "2014-12-12")  → two continuous days: first day from HOUR_START:00 through
#                                   second day through HOUR_END:59 (HOUR_END may be < HOUR_START, e.g. 19→5)
DATA_DAY = "2014-12-06" # ("2014-12-25", "2014-12-26") # "2014-12-26" 
DATA_HOUR_START = 7   # first day: inclusive from :00   # 5-8
DATA_HOUR_END =   9  # second day: inclusive through :59 (may be earlier hour than START)   # 4

# Match Simulator/beta_simulator.py PAPER4_EXPERIMENTS.
BETA_SIMULATOR_SCENARIOS = [
    "pair_1p5__noise_05",
    "pair_1p5_2p0__noise_05",
    "pair_1p5_2p0__noise_10",
    "pair_1p5_2p0__noise_20",
]
BETA_SENSITIVITY_OUTPUT_ROOT = SIMULATOR_DIR / "beta_sensitivity_outputs_2013_4"

# Hyperparameters to test.
HYPERPARAMETER_PAIRS = [
    (5e-2, 5e-2), # (eta_coeff, lambda_factor)
    (1e-1, 1e-1),
    (5e-1, 5e-1),  
]

# Simulation parameters.
TOTAL_ARRIVALS_TO_SIMULATE = int(1e5)  
M0_EXPLORATION_COUNT = 30 
DEFAULT_MK =  3 
M_K_VALUES = [12]*50 + [3]*150  

RANDOM_PRICE_MIN = 10.0
RANDOM_PRICE_MAX = 30.0 
RANDOM_PRICE_GRID_SIZE = 5
W_L1_RADIUS_AGENT = 50.0
PGD_WARM = 50 


PRICE_GRID = np.linspace(2.5, 70, 5000)
EXPLORATION_PRICES = np.linspace(RANDOM_PRICE_MIN, RANDOM_PRICE_MAX, RANDOM_PRICE_GRID_SIZE)

# Penalty schedules kept here for quick tuning.
ETA_K_POWER_VALUES = [0.0, 0.25, 0.5]  # PGD only; ILQX runs once per (eta, lam, stop_eps)
P_PENALTY_NORM_VAL = 4 / 3
LAMBDA_N_POWER = 0.25

# Adaptive PGD learning-rate controls.
LR_DIM_SCALE_EXP = 0.5 # 0.5
LR_COEFF_BY = 2e0 #1e0
STOP_EPS_VALUES = [1e0, 1.5e0]
MOMENTUM_BETA = 0.9
PLATEAU_PATIENCE = 800
LR_PLATEAU_DECAY = 0.97
MIN_LR = 1e-9
LOSS_IMPROVE_TOL = 1e-5
REL_LOSS_IMPROVE_FRAC = 1e-6
MAX_GRAD_NORM = 1e7
CLIP_BACKOFF_SCALE_THRESHOLD = 0.03
CLIP_BACKOFF_PATIENCE = 2000
CLIP_BACKOFF_DECAY = 0.985
LOSS_EVAL_EVERY = 5  # PGD: evaluate loss / plateau bookkeeping every K iters (#7)

# Repeat & output controls.
NUM_REPEATS = 20
OUTPUT_ROOT = BASE_DIR / "TestData" / "sim_output"
# Stream rows to disk during simulation (avoids holding ~300k-step DataFrames in RAM).
STREAM_FLUSH_EVERY = 5000
STREAM_CSV_CHUNKSIZE = 50000
# End-of-script FacetGrid/bar plots reload every CSV; use plot_cumulative_regret.py instead.
GENERATE_INLINE_PLOTS = False

# --- Set Random Seed for Reproducibility ---
MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)
print(f"--- Using single generator (Master Seed: {MASTER_SEED}) ---")

BETA_SIMULATOR = None
SCALER_SIMULATOR = None
SCALED_FEATURES_SIMULATOR = None
SIMULATOR_CONTEXTUAL_FEATURES = []
AGENT_CONTEXTUAL_FEATURES = []
interaction_feature_names = []
AGENT_MODEL_FEATURES = []


def eta_k_value(cumulative_exploration_samples, eta_k_power):
    return cumulative_exploration_samples ** eta_k_power


def lambda_t_value(lambda_t_factor, n_obs, d_coeffs):
    log_term = np.log(d_coeffs + 1e-9) + np.log(n_obs + 1e-9)
    return lambda_t_factor * (n_obs ** LAMBDA_N_POWER) * np.sqrt(log_term)


def pgd_penalty_gradient(coeffs, eta_coeff, cumulative_exploration_samples, eta_k_power):
    return eta_coeff * eta_k_value(cumulative_exploration_samples, eta_k_power) * coeffs

# --- CVXPY and MOSEK Imports ---
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    if cp.MOSEK in cp.installed_solvers():
        MOSEK_SOLVER_AVAILABLE_IN_CVXPY = True
        print("CVXPY and MOSEK solver imported successfully.")
    else:
        MOSEK_SOLVER_AVAILABLE_IN_CVXPY = False
        print("CVXPY imported, but MOSEK is not available as a CVXPY solver.")
except ImportError:
    CVXPY_AVAILABLE = False
    MOSEK_SOLVER_AVAILABLE_IN_CVXPY = False
    print("Warning: CVXPY not found. The CVX MLE strategy will not be available.")

# --- L1 Projection Function ---
def project_l1(vector, W):
    vector = np.array(vector).flatten()
    if np.sum(np.abs(vector)) <= W:
        return vector.copy()
    sorted_indices = np.argsort(np.abs(vector))[::-1]
    sorted_abs = np.abs(vector)[sorted_indices]
    cumulative_sum = np.cumsum(sorted_abs)
    arange_rho = np.arange(1, len(sorted_abs) + 1)
    rho_candidates = np.where(sorted_abs > (cumulative_sum - W) / arange_rho)[0]
    rho = rho_candidates[-1] if len(rho_candidates) > 0 else len(sorted_abs) -1
    theta = (cumulative_sum[rho] - W) / (rho + 1.0)
    return np.sign(vector) * np.maximum(np.abs(vector) - theta, 0)

def p_norm(x, p):
    return np.sum(np.abs(x) ** p) ** (1 / p)

# --- 0. Load Simulator Components & Processed Data ---
def load_beta_simulator_components(scenario_name):
    scenario_dir = BETA_SENSITIVITY_OUTPUT_ROOT / scenario_name
    try:
        df_betas = pd.read_csv(scenario_dir / 'beta_simulator_parameters.csv')
        intercept = df_betas.loc[df_betas['feature'] == 'intercept', 'coefficient'].iloc[0]
        coeffs_df = df_betas[df_betas['feature'] != 'intercept']
        coefficients = pd.Series(coeffs_df['coefficient'].values, index=coeffs_df['feature'].values)
        beta_simulator = {'intercept': intercept, 'coefficients': coefficients}
        scaler_simulator = joblib.load(scenario_dir / "beta_simulator_scaler.joblib")
        with open(scenario_dir / "beta_simulator_scaled_features_list.json", 'r') as f:
            scaled_feature_names_for_simulator = json.load(f)
        print(f"Beta_simulator components loaded successfully: {scenario_name}")
        return beta_simulator, scaler_simulator, scaled_feature_names_for_simulator
    except Exception as e:
        print(f"Error loading Beta_simulator components for {scenario_name}: {e}")
        print(f"Please ensure simulator files exist under: {scenario_dir}")
        return None, None, None


def parse_data_day_window(data_day, hour_start, hour_end):
    """Return (start_ts, end_ts, label) or (None, None, None) when data_day is None."""
    if data_day is None:
        return None, None, None

    h0, h1 = int(hour_start), int(hour_end)

    if isinstance(data_day, tuple):
        if len(data_day) != 2 or not all(isinstance(x, str) for x in data_day):
            raise ValueError("DATA_DAY tuple must be (first_day, second_day) with two YYYY-MM-DD strings.")
        day1 = pd.Timestamp(data_day[0])
        day2 = pd.Timestamp(data_day[1])
        if day2 < day1:
            raise ValueError(f"Invalid day range: {data_day[0]} .. {data_day[1]}")
        start_ts = day1.normalize() + pd.Timedelta(hours=h0)
        end_ts = day2.normalize() + pd.Timedelta(hours=h1, minutes=59, seconds=59)
        overnight = h1 < h0
        label = (
            f"continuous [{start_ts}, {end_ts}] "
            f"({data_day[0]} {h0:02d}:00 → {data_day[1]} {h1:02d}:59"
            f"{', overnight' if overnight else ''})"
        )
    elif isinstance(data_day, str):
        day = pd.Timestamp(data_day)
        if h1 < h0:
            raise ValueError(
                f"Single-day window requires DATA_HOUR_END ({h1}) >= DATA_HOUR_START ({h0}). "
                "Use DATA_DAY = (first_day, second_day) for overnight spans."
            )
        start_ts = day.normalize() + pd.Timedelta(hours=h0)
        end_ts = day.normalize() + pd.Timedelta(hours=h1, minutes=59, seconds=59)
        label = f"single day [{start_ts}, {end_ts}]"
    else:
        raise TypeError("DATA_DAY must be None, a str, or a (first_day, second_day) tuple of str.")

    if end_ts < start_ts:
        raise ValueError(f"Invalid time window: end {end_ts} is before start {start_ts}.")
    return start_ts, end_ts, label


try:
    processed_df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded processed data. Shape: {processed_df.shape}")
    # If duplicate pickup_time values were spread (see 2014_data/README.md), some
    # rows carry fractional seconds while unique timestamps stay second-level.
    _ts = processed_df[TIME_COLUMN]
    try:
        processed_df[TIME_COLUMN] = pd.to_datetime(_ts, format='mixed')
    except (TypeError, ValueError):
        processed_df[TIME_COLUMN] = pd.to_datetime(_ts.astype(str).str.strip(), errors='coerce')
    processed_df.sort_values(by=TIME_COLUMN, inplace=True)
    processed_df.reset_index(drop=True, inplace=True)
    print(f"Processed data sorted by {TIME_COLUMN}.")

    _window_start, _window_end, _window_label = parse_data_day_window(DATA_DAY, DATA_HOUR_START, DATA_HOUR_END)
    if _window_start is not None:
        _before = len(processed_df)
        processed_df = processed_df[
            (processed_df[TIME_COLUMN] >= _window_start) & (processed_df[TIME_COLUMN] <= _window_end)
        ].reset_index(drop=True)
        print(f"Time window filter: {_window_label}\n  {_before:,} → {len(processed_df):,} rows.")
        if processed_df.empty:
            raise ValueError(f"No rows in time window: {_window_label}")

    # When a calendar slice is set, simulate all arrivals in that window; otherwise cap at TOTAL_ARRIVALS_TO_SIMULATE.
    if DATA_DAY is not None:
        EFFECTIVE_TOTAL_ARRIVALS = len(processed_df)
    else:
        EFFECTIVE_TOTAL_ARRIVALS = min(TOTAL_ARRIVALS_TO_SIMULATE, len(processed_df))
    print(f"Arrivals to simulate: {EFFECTIVE_TOTAL_ARRIVALS:,}")
except Exception as e:
    print(f"Error loading processed data: {e}")
    print(f"Please ensure the path to the processed data CSV is correct: {DATA_PATH}")
    processed_df = pd.DataFrame()

if processed_df.empty:
    print("Processed data is empty. Exiting.")
    exit()

if 'EFFECTIVE_TOTAL_ARRIVALS' not in globals():
    EFFECTIVE_TOTAL_ARRIVALS = min(TOTAL_ARRIVALS_TO_SIMULATE, len(processed_df))


def configure_feature_sets(beta_simulator):
    global SIMULATOR_CONTEXTUAL_FEATURES
    global AGENT_CONTEXTUAL_FEATURES
    global interaction_feature_names
    global AGENT_MODEL_FEATURES

    SIMULATOR_CONTEXTUAL_FEATURES = [f for f in beta_simulator['coefficients'].index if not f.startswith('price_x_')]
    AGENT_CONTEXTUAL_FEATURES = SIMULATOR_CONTEXTUAL_FEATURES
    interaction_feature_names = [f"price_x_{feature}" for feature in AGENT_CONTEXTUAL_FEATURES]
    AGENT_MODEL_FEATURES = AGENT_CONTEXTUAL_FEATURES + interaction_feature_names

# --- 1. World Simulator (vectorized via Z = A + B*price decomposition) ---
PRICE_GRID = np.linspace(2.5, 70, 500)
EXPLORATION_PRICES = np.linspace(RANDOM_PRICE_MIN, RANDOM_PRICE_MAX, RANDOM_PRICE_GRID_SIZE)

_oracle_cache = {}

def _build_oracle_cache():
    """Pre-compute numpy arrays for vectorized A+B*price decomposition (called once per scenario)."""
    _oracle_cache.clear()
    coef_index = BETA_SIMULATOR['coefficients'].index.tolist()
    coef_values = BETA_SIMULATOR['coefficients'].values.astype(np.float64)
    sc_mean = SCALER_SIMULATOR.mean_ if SCALER_SIMULATOR is not None else np.zeros(len(SCALED_FEATURES_SIMULATOR))
    sc_scale = SCALER_SIMULATOR.scale_ if SCALER_SIMULATOR is not None else np.ones(len(SCALED_FEATURES_SIMULATOR))
    sc_lookup = {f: i for i, f in enumerate(SCALED_FEATURES_SIMULATOR)}
    ctx_idx_map = {f: i for i, f in enumerate(AGENT_CONTEXTUAL_FEATURES)}

    a_const = float(BETA_SIMULATOR['intercept'])
    ctx_indices, ctx_scales = [], []
    int_indices, int_scales = [], []

    for i, fname in enumerate(coef_index):
        c = coef_values[i]
        if c == 0.0:
            continue
        if fname.startswith('price_x_'):
            bi = ctx_idx_map.get(fname[8:])
            if bi is None:
                continue
            int_indices.append(bi)
            if fname in sc_lookup:
                j = sc_lookup[fname]
                int_scales.append(c / sc_scale[j])
                a_const -= c * sc_mean[j] / sc_scale[j]
            else:
                int_scales.append(c)
        else:
            ci = ctx_idx_map.get(fname)
            if ci is not None:
                ctx_indices.append(ci)
                if fname in sc_lookup:
                    j = sc_lookup[fname]
                    ctx_scales.append(c / sc_scale[j])
                    a_const -= c * sc_mean[j] / sc_scale[j]
                else:
                    ctx_scales.append(c)
            elif fname in sc_lookup:
                j = sc_lookup[fname]
                a_const += c * (0.0 - sc_mean[j]) / sc_scale[j]

    _oracle_cache['a_const'] = a_const
    _oracle_cache['ctx_idx'] = np.array(ctx_indices, dtype=int)
    _oracle_cache['ctx_sc'] = np.array(ctx_scales, dtype=np.float64)
    _oracle_cache['int_idx'] = np.array(int_indices, dtype=int)
    _oracle_cache['int_sc'] = np.array(int_scales, dtype=np.float64)


def _oracle_AB(features_arr):
    """Vectorized: Z = A + B*price from numpy feature array (indexed by AGENT_CONTEXTUAL_FEATURES order)."""
    A = _oracle_cache['a_const'] + np.dot(_oracle_cache['ctx_sc'], features_arr[_oracle_cache['ctx_idx']])
    B = np.dot(_oracle_cache['int_sc'], features_arr[_oracle_cache['int_idx']])
    return float(A), float(B)


def _agent_AB(features_arr, beta_package):
    """Vectorized: Z = A + B*price from numpy feature array for agent model."""
    coeffs = beta_package['_coeffs_arr']
    sc_m = beta_package['_sc_mean']
    sc_s = beta_package['_sc_scale']
    n_ctx = beta_package['_n_ctx']
    A = float(beta_package['intercept']) + float(np.dot(coeffs[:n_ctx], (features_arr - sc_m[:n_ctx]) / sc_s[:n_ctx]))
    int_c = coeffs[n_ctx:]
    B = float(np.dot(int_c, features_arr / sc_s[n_ctx:]))
    A -= float(np.dot(int_c, sc_m[n_ctx:] / sc_s[n_ctx:]))
    return A, B


def simulate_passenger_acceptance_true(price_offered, features_arr):
    A, B = _oracle_AB(features_arr)
    Z = A + B * float(price_offered)
    p_accept = 1.0 / (1.0 + np.exp(-np.clip(Z, -700, 700)))
    outcome = 1 if rng.random() <= p_accept else 0
    return outcome, float(outcome - p_accept)

# --- 2. Agent Learning Components ---

def cvxpy_logistic_lasso_estimator(X_train_scaled_df, y_train_series, feature_names, previous_beta_params,
                                   cumulative_exploration_samples, W_l1_radius, lambda_t_factor, **kwargs):
    if not CVXPY_AVAILABLE or not MOSEK_SOLVER_AVAILABLE_IN_CVXPY: return None, 0.05

    n_obs, d_coeffs = X_train_scaled_df.shape
    X_matrix = X_train_scaled_df[feature_names].values.astype(np.float64)
    y_pm1 = (2 * y_train_series.values.astype(np.float64) - 1)
    lambda_val = lambda_t_value(lambda_t_factor, n_obs, d_coeffs)

    start_time = time.time()
    beta0_cp = cp.Variable()
    beta_coeffs_cp = cp.Variable(d_coeffs)

    if previous_beta_params:
        beta0_cp.value = previous_beta_params['intercept']
        beta_coeffs_cp.value = previous_beta_params['coefficients'].reindex(feature_names).fillna(0).values

    z_cp = X_matrix @ beta_coeffs_cp + beta0_cp
    obj = cp.Minimize(cp.sum(cp.logistic(cp.multiply(-y_pm1, z_cp))) + lambda_val * cp.norm1(beta_coeffs_cp))
    prob = cp.Problem(obj, [cp.norm1(beta_coeffs_cp) <= W_l1_radius])
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False)

    elapsed = time.time() - start_time
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimal_coeffs = beta_coeffs_cp.value if beta_coeffs_cp.value is not None else np.zeros(d_coeffs)
        beta_out = {
            'intercept': beta0_cp.value if beta0_cp.value is not None else 0.0,
            'coefficients': pd.Series(optimal_coeffs, index=feature_names)
        }
        return beta_out, elapsed
    else:
        return None, elapsed

def classify_snr_bucket(y_values):
    positive_rate = float(np.mean(y_values)) if len(y_values) else 0.5
    balance = min(positive_rate, 1.0 - positive_rate)
    if balance < 0.10:
        return 'low'
    if balance < 0.25:
        return 'mid'
    return 'high'


def logistic_regularized_loss(beta0, coeffs, X, Y, penalty_weight):
    Z = beta0 + X @ coeffs
    logistic_loss = np.mean(np.logaddexp(0.0, Z) - Y * Z)
    ridge_penalty = 0.5 * penalty_weight * np.dot(coeffs, coeffs)
    return float(logistic_loss + ridge_penalty)


def PGD(X_train_scaled_df, y_train_series, feature_names, previous_beta_params,
        m_k_exploration_batch_size, cumulative_exploration_samples, W_l1_radius,
        p_penalty_norm_val, num_iters_max, eta_coeff, eta_k_power, stop_eps=0.5, **kwargs):
    d, n_obs = len(feature_names), len(X_train_scaled_df)
    X_matrix, Y_vector = X_train_scaled_df[feature_names].values.astype(np.float64), y_train_series.values.astype(np.float64)

    beta0, coeffs = (previous_beta_params['intercept'], previous_beta_params['coefficients'].reindex(feature_names).fillna(0).values) if previous_beta_params else (0.0, np.zeros(d))
    penalty_weight = eta_coeff * eta_k_value(cumulative_exploration_samples, eta_k_power)

    # [#6] Compute Z = b0 + X @ cfs once per (b0, cfs) and reuse for both gradient and loss.
    def compute_Z(b0, cfs):
        return b0 + X_matrix @ cfs

    def grads_from_Z(Z, cfs):
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z, -700.0, 700.0)))
        error = prob - Y_vector
        grad_b0 = float(np.sum(error))
        grad_cfs = X_matrix.T @ error + pgd_penalty_gradient(cfs, eta_coeff, cumulative_exploration_samples, eta_k_power)
        return grad_b0, grad_cfs

    def loss_from_Z(Z, cfs):
        logistic_loss = np.mean(np.logaddexp(0.0, Z) - Y_vector * Z)
        ridge_penalty = 0.5 * penalty_weight * np.dot(cfs, cfs)
        return float(logistic_loss + ridge_penalty)

    start_time = time.time()
    lr = LR_COEFF_BY / max(1.0, d ** LR_DIM_SCALE_EXP)
    # lr = base_lr * max(eta_coeff, 1e-12) if eta_coeff > 0 else base_lr
    velocity_b0 = 0.0
    velocity_coeffs = np.zeros(d)
    Z_curr = compute_Z(beta0, coeffs)
    best_loss = loss_from_Z(Z_curr, coeffs)
    dim_scale = d ** (1.0 / p_penalty_norm_val - 0.5) if d > 0 else 1.0
    grad_stop_threshold = stop_eps * (max(1, cumulative_exploration_samples) ** 0.25) * dim_scale
    plateau_count = 0
    clip_backoff_count = 0

    for i in range(num_iters_max):
        grad_b0, grad_cfs = grads_from_Z(Z_curr, coeffs)
        # [#8] avoid np.append + np.linalg.norm allocation
        raw_grad_norm = float(np.sqrt(grad_b0 * grad_b0 + float(grad_cfs @ grad_cfs)))
        if raw_grad_norm < grad_stop_threshold and i > 0:
            break

        clip_scale = 1.0
        if raw_grad_norm > MAX_GRAD_NORM:
            clip_scale = MAX_GRAD_NORM / raw_grad_norm
            grad_b0 *= clip_scale
            grad_cfs *= clip_scale

        velocity_b0 = MOMENTUM_BETA * velocity_b0 + (1 - MOMENTUM_BETA) * grad_b0
        velocity_coeffs = MOMENTUM_BETA * velocity_coeffs + (1 - MOMENTUM_BETA) * grad_cfs

        step = lr / max(1, n_obs ** 0.5)
        next_beta0 = beta0 - step * velocity_b0
        next_coeffs = project_l1(coeffs - step * velocity_coeffs, W_l1_radius)
        Z_next = compute_Z(next_beta0, next_coeffs)

        # [#7] only evaluate loss / run plateau bookkeeping every K iters (and at last iter)
        eval_loss_now = (i % LOSS_EVAL_EVERY == 0) or (i == num_iters_max - 1)
        if eval_loss_now:
            next_loss = loss_from_Z(Z_next, next_coeffs)
            absolute_improvement = best_loss - next_loss
            relative_improvement = absolute_improvement / max(abs(best_loss), 1.0)
            improved = (
                absolute_improvement > LOSS_IMPROVE_TOL
                or relative_improvement > REL_LOSS_IMPROVE_FRAC
            )

            if improved:
                beta0, coeffs = next_beta0, next_coeffs
                Z_curr = Z_next
                best_loss = next_loss
                plateau_count = 0
                clip_backoff_count = 0
            else:
                plateau_count += 1
                if next_loss <= best_loss:
                    beta0, coeffs = next_beta0, next_coeffs
                    Z_curr = Z_next

            if clip_scale < CLIP_BACKOFF_SCALE_THRESHOLD and not improved:
                clip_backoff_count += 1
            else:
                clip_backoff_count = 0

            if plateau_count >= PLATEAU_PATIENCE:
                lr = max(MIN_LR, lr * LR_PLATEAU_DECAY)
                plateau_count = 0
            if clip_backoff_count >= CLIP_BACKOFF_PATIENCE:
                lr = max(MIN_LR, lr * CLIP_BACKOFF_DECAY)
                clip_backoff_count = 0
            if lr <= MIN_LR and not improved:
                break
        else:
            # skip loss eval: accept proposed step (matches the "next_loss <= best_loss" branch
            # semantics in expectation, since momentum direction is descent-aligned),
            # without touching plateau / clip-backoff counters.
            beta0, coeffs = next_beta0, next_coeffs
            Z_curr = Z_next
    
    elapsed = time.time() - start_time
    beta_out = {'intercept': beta0, 'coefficients': pd.Series(coeffs, index=feature_names)}
    return beta_out, elapsed

def train_beta_agent_mle(exploration_data_df, current_arrival_idx, previous_beta_package, W_l1_radius_param, estimator_type, estimator_params, current_epoch_k=None):
    if len(exploration_data_df) < 20 or exploration_data_df['accepted'].nunique() < 2: return None, 0.01

    df_train = exploration_data_df.copy()
    for feat in AGENT_CONTEXTUAL_FEATURES:
        df_train[f"price_x_{feat}"] = df_train['price_offered'] * df_train[feat]

    X_raw, y = df_train[AGENT_MODEL_FEATURES], df_train['accepted']
    scaler = StandardScaler().fit(X_raw)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=AGENT_MODEL_FEATURES)

    # Warm-start: use exact MOSEK for the first PGD_WARM epochs, then switch to gradient PGD.
    use_mosek_warm = (
        estimator_type == 'pgd'
        and current_epoch_k is not None
        and current_epoch_k <= PGD_WARM
        and CVXPY_AVAILABLE
        and MOSEK_SOLVER_AVAILABLE_IN_CVXPY
    )
    if use_mosek_warm:
        # Reuse cvxpy_logistic_lasso_estimator; borrow lambda_t_factor from pgd params if available,
        # otherwise fall back to a default proportional to eta_coeff.
        eta_c = estimator_params.get('eta_coeff', 0.05)
        lam_factor = estimator_params.get('lambda_t_factor', eta_c)
        cvx_params = {'lambda_t_factor': lam_factor}
        estimator_func = cvxpy_logistic_lasso_estimator
        call_params = cvx_params
        # print(f"  [PGD warm-MOSEK] epoch {current_epoch_k} ≤ {PGD_WARM}: using exact MOSEK solve.")
    else:
        estimator_func = {'cvx': cvxpy_logistic_lasso_estimator, 'pgd': PGD}[estimator_type]
        call_params = estimator_params

    beta_content, time_taken = estimator_func(
        X_train_scaled_df=X_scaled, y_train_series=y, feature_names=AGENT_MODEL_FEATURES,
        previous_beta_params=previous_beta_package,
        cumulative_exploration_samples=len(exploration_data_df),
        W_l1_radius=W_l1_radius_param, **call_params)

    if beta_content:
      full_package = {
          'intercept': beta_content['intercept'],
          'coefficients': beta_content['coefficients'],
          'scaler': scaler,
          '_coeffs_arr': beta_content['coefficients'].reindex(AGENT_MODEL_FEATURES).fillna(0.0).values.astype(np.float64),
          '_sc_mean': scaler.mean_.astype(np.float64),
          '_sc_scale': scaler.scale_.astype(np.float64),
          '_n_ctx': len(AGENT_CONTEXTUAL_FEATURES),
          'scaled_features': scaler.get_feature_names_out().tolist(),
          'feature_names': AGENT_MODEL_FEATURES
      }
      return full_package, time_taken
    else:
        return None, time_taken

def p_accept_agent_model(price, features, beta_package):
    if not beta_package or 'scaler' not in beta_package: return 0.5
    fa = np.array([float(features.get(f, 0.0)) for f in AGENT_CONTEXTUAL_FEATURES]) if isinstance(features, dict) else features
    A, B = _agent_AB(fa, beta_package)
    Z = A + B * float(price)
    return float(1.0 / (1.0 + np.exp(-np.clip(Z, -700, 700))))

def p_accept_oracle_model(price, features):
    fa = np.array([float(features.get(f, 0.0)) for f in AGENT_CONTEXTUAL_FEATURES]) if isinstance(features, dict) else features
    A, B = _oracle_AB(fa)
    Z = A + B * float(price)
    return float(1.0 / (1.0 + np.exp(-np.clip(Z, -700, 700))))

def find_optimal_price_generic(features_arr, beta_package, for_oracle=False):
    fallback_price = rng.uniform(10.0, 20.0)
    if not beta_package and not for_oracle: return fallback_price, 0.0

    A, B = _oracle_AB(features_arr) if for_oracle else _agent_AB(features_arr, beta_package)
    Z = A + B * PRICE_GRID
    prob = 1.0 / (1.0 + np.exp(-np.clip(Z, -700, 700)))
    rev = PRICE_GRID * prob
    idx = int(np.argmax(rev))
    return float(PRICE_GRID[idx]), float(rev[idx])


class SimulationResultsWriter:
    """Append simulation rows to CSV in batches to limit memory use."""

    def __init__(self, path: Path, flush_every: int = STREAM_FLUSH_EVERY):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[dict] = []
        self._header: list[str] | None = None
        self._file = None
        self._writer: csv.DictWriter | None = None

    def write_row(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._writer is None:
            self._header = list(self._buffer[0].keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._header, extrasaction="ignore")
            self._writer.writeheader()
        for row in self._buffer:
            self._writer.writerow({k: row.get(k) for k in self._header})
        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def _revenue_regret_columns(active_agents: list[str]) -> list[str]:
    rev_reg_cols = ['arrival_idx', 'arrival_time', 'k_sq_epoch', 'phase', 'oracle_exp_revenue']
    for ag in active_agents:
        rev_reg_cols.extend([
            f'reward_{ag}', f'regret_{ag}', f'stale_regret_{ag}',
            f'fresh_params_{ag}', f'price_{ag}',
        ])
    if 'cvx' in active_agents:
        rev_reg_cols.extend([
            'price_cvx_instant',
            'regret_cvx_instant',
            'regret_cvx_expected_real',
            'timeliness_loss_cvx',
        ])
    rev_reg_cols.extend(['reward_original', 'regret_original', 'price_original'])
    return rev_reg_cols


def finalize_revenue_regret_csv(
    pgd_csv: Path,
    out_csv: Path,
    active_agents: list[str],
    cvx_csv: Path | None = None,
    chunksize: int = STREAM_CSV_CHUNKSIZE,
) -> None:
    """Write revenue_regret.csv in chunks (optionally merge ILQX cvx columns)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wanted = _revenue_regret_columns(active_agents)
    first = True
    pgd_reader = pd.read_csv(pgd_csv, chunksize=chunksize)
    cvx_reader = pd.read_csv(cvx_csv, chunksize=chunksize) if cvx_csv is not None else None
    for pgd_chunk in pgd_reader:
        if cvx_reader is not None:
            cvx_chunk = next(cvx_reader)
            for col in cvx_chunk.columns:
                if 'cvx' in col:
                    pgd_chunk[col] = cvx_chunk[col].values
        out_cols = [c for c in wanted if c in pgd_chunk.columns]
        pgd_chunk[out_cols].to_csv(out_csv, index=False, mode='w' if first else 'a', header=first)
        first = False

  
def save_oracle_revenue_from_csv(source_csv: Path, out_dir: Path, chunksize: int = STREAM_CSV_CHUNKSIZE) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = out_dir / "oracle_revenue.csv"
    first = True
    for chunk in pd.read_csv(
        source_csv,
        usecols=['arrival_idx', 'oracle_price', 'oracle_exp_revenue'],
        chunksize=chunksize,
    ):
        chunk.to_csv(oracle_path, index=False, mode='w' if first else 'a', header=first)
        first = False


def load_revenue_regret_subset(csv_path: Path, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Load only columns needed for post-run text analysis."""
    sample = pd.read_csv(csv_path, nrows=0)
    base = {'arrival_idx', 'arrival_time', 'k_sq_epoch', 'phase', 'oracle_exp_revenue',
            'reward_original', 'regret_original', 'price_original'}
    agents = ['pgd', 'cvx']
    for ag in agents:
        base.update({
            f'reward_{ag}', f'regret_{ag}', f'stale_regret_{ag}',
            f'fresh_params_{ag}', f'price_{ag}', f'epsilon_{ag}',
        })
    base.update({
        'price_cvx_instant',
        'regret_cvx_instant',
        'regret_cvx_expected_real',
        'timeliness_loss_cvx',
    })
    if extra_cols:
        base.update(extra_cols)
    usecols = [c for c in sample.columns if c in base]
    return pd.read_csv(csv_path, usecols=usecols)


# --- 3. Main Simulation Function ---
def run_simulation(eta_coeff_param, lambda_t_factor_param, stop_eps_param, eta_k_power_param,
                   total_arrivals, data_df, progress_desc, active_agents=None, run_seed=None,
                   stream_csv_path: Path | None = None):
    """
    Runs a full simulation for a given set of hyperparameters.
    active_agents: e.g. ['pgd'], ['cvx'], or ['pgd', 'cvx']. eta_k_power only affects PGD.
    run_seed: if set, resets global rng so ILQX/PGD splits stay comparable.
    """
    global rng
    if run_seed is not None:
        rng = np.random.default_rng(run_seed)

    if active_agents is None:
        active_agents = ['pgd']
        if CVXPY_AVAILABLE and MOSEK_SOLVER_AVAILABLE_IN_CVXPY:
            active_agents.append('cvx')

    # --- Simulation State Initialization ---
    state = {
        agent: {
            'beta_package': None, 'beta_history': {}, 'mle_job': False, 'mle_job_key': None,
            'mle_complete_time': np.datetime64('NaT'), 'mle_package_to_apply': None,
            'exploration_data': [], 'mle_trigger_events': [], 'mle_completion_events': [], 'solving_times': [],
            # Epoch exploit lock: if MLE for k is pending at exploit start, use last completed β for all exploit steps in epoch k.
            'exploit_lock_epoch': None, 'exploit_lock_beta': None, 'exploit_lock_fresh': None,
        }
        for agent in active_agents
    }
    simulation_results = []
    stream_writer = (
        SimulationResultsWriter(stream_csv_path, flush_every=STREAM_FLUSH_EVERY)
        if stream_csv_path is not None else None
    )
    current_k_sq_epoch_val, current_k_sq_epoch_expl_done_count = 0, 0

    # --- Pre-extract data into numpy arrays (avoid per-step pandas iloc) ---
    _n = min(total_arrivals, len(data_df))
    _timestamps = data_df[TIME_COLUMN].values[:_n]
    _feat_names = list(AGENT_CONTEXTUAL_FEATURES)
    _features_mat = data_df[_feat_names].values[:_n].astype(np.float64)
    _fares = data_df['fare_amount'].values[:_n].astype(np.float64)

    # --- Pre-compute epoch index for each step (replace per-step while loop) ---
    _arrival_1idx = np.arange(1, _n + 1, dtype=np.int64)
    _epoch_arr = np.zeros(_n, dtype=int)
    for _k in range(1, int(np.sqrt(_n)) + 2):
        _lo, _hi = _k * _k, (_k + 1) * (_k + 1)
        _mask = (_arrival_1idx >= _lo) & (_arrival_1idx < _hi)
        _epoch_arr[_mask] = _k

    # --- Main Simulation Loop ---
    for t in tqdm(range(total_arrivals), desc=progress_desc):
        current_arrival_timestamp = _timestamps[t]
        features_arr = _features_mat[t]

        oracle_price, oracle_exp_reward = find_optimal_price_generic(features_arr, None, for_oracle=True)
        original_fare = float(_fares[t])
        accepted_original, epsilon_original = simulate_passenger_acceptance_true(original_fare, features_arr)
        reward_original = original_fare * accepted_original

        for agent_type in active_agents:
            if state[agent_type]['mle_job'] and current_arrival_timestamp >= state[agent_type]['mle_complete_time']:
                if (trained_package := state[agent_type]['mle_package_to_apply']):
                    state[agent_type]['beta_history'][state[agent_type]['mle_job_key']] = trained_package
                    state[agent_type]['beta_package'] = trained_package
                    state[agent_type]['mle_completion_events'].append({'key': state[agent_type]['mle_job_key'], 'completion_idx': t})
                state[agent_type]['mle_job'] = False
                state[agent_type]['mle_package_to_apply'] = None

        arrival_num_1_indexed = t + 1
        new_k_sq_epoch_val = int(_epoch_arr[t])

        if new_k_sq_epoch_val != current_k_sq_epoch_val and new_k_sq_epoch_val > 0:
            current_k_sq_epoch_val = new_k_sq_epoch_val
            current_k_sq_epoch_expl_done_count = 0
            for agent_type in active_agents:
                state[agent_type]['exploit_lock_epoch'] = None
                state[agent_type]['exploit_lock_beta'] = None
                state[agent_type]['exploit_lock_fresh'] = None

        exploration_samples_for_this_epoch = 0
        is_current_epoch_full_exploration = False
        if current_k_sq_epoch_val > 0:
            current_epoch_total_length = (2 * current_k_sq_epoch_val) + 1
            m_k_intended_for_epoch = M_K_VALUES[current_k_sq_epoch_val - 1] if 0 < current_k_sq_epoch_val <= len(M_K_VALUES) else DEFAULT_MK
            is_current_epoch_full_exploration = (m_k_intended_for_epoch >= current_epoch_total_length)
            exploration_samples_for_this_epoch = current_epoch_total_length if is_current_epoch_full_exploration else m_k_intended_for_epoch

        phase = ""
        if t < M0_EXPLORATION_COUNT: phase = 'm0_explore'
        elif current_k_sq_epoch_val > 0:
            phase = f'k_sq_epoch_{current_k_sq_epoch_val}_explore' if is_current_epoch_full_exploration or current_k_sq_epoch_expl_done_count < exploration_samples_for_this_epoch else f'k_sq_epoch_{current_k_sq_epoch_val}_exploit'
        else: phase = 'm0_buffer'

        step_results = {
            'arrival_idx': t,
            'arrival_time': str(current_arrival_timestamp),
            'k_sq_epoch': current_k_sq_epoch_val,
            'phase': phase,
            'oracle_price': oracle_price,
            'oracle_exp_revenue': oracle_exp_reward,
            'price_original': original_fare,
        }
        for agent_name in ['pgd', 'cvx']:
            step_results[f'epsilon_{agent_name}'] = None
            step_results[f'fresh_params_{agent_name}'] = None
            step_results[f'stale_regret_{agent_name}'] = 0.0
        step_results['price_cvx_instant'] = None
        step_results['regret_cvx_instant'] = None
        step_results['regret_cvx_expected_real'] = None
        step_results['timeliness_loss_cvx'] = None

        if 'explore' in phase or 'buffer' in phase:
            price_offered = float(rng.choice(EXPLORATION_PRICES))
            accepted, epsilon = simulate_passenger_acceptance_true(price_offered, features_arr)
            reward = price_offered * accepted
            log_entry = {'price_offered': price_offered, 'accepted': accepted}
            for _i, _fn in enumerate(_feat_names):
                log_entry[_fn] = float(features_arr[_i])
            for agent_type in active_agents:
                state[agent_type]['exploration_data'].append(log_entry)
                step_results[f'reward_{agent_type}'] = reward
                step_results[f'regret_{agent_type}'] = oracle_exp_reward - reward
                step_results[f'price_{agent_type}'] = price_offered
                step_results[f'epsilon_{agent_type}'] = epsilon
            if phase == f'k_sq_epoch_{current_k_sq_epoch_val}_explore' and not is_current_epoch_full_exploration:
                current_k_sq_epoch_expl_done_count += 1
        else:
            for agent_type in active_agents:
                k_epoch = current_k_sq_epoch_val
                if phase.endswith('_exploit') and k_epoch > 0:
                    if state[agent_type]['exploit_lock_epoch'] != k_epoch:
                        pending_k = (
                            state[agent_type]['mle_job']
                            and state[agent_type]['mle_job_key'] == k_epoch
                        )
                        if pending_k:
                            beta_to_use = state[agent_type]['beta_history'].get(k_epoch - 1)
                            mle_params_were_fresh = False
                        else:
                            beta_to_use = state[agent_type]['beta_package']
                            mle_params_were_fresh = True
                        state[agent_type]['exploit_lock_epoch'] = k_epoch
                        state[agent_type]['exploit_lock_beta'] = beta_to_use
                        state[agent_type]['exploit_lock_fresh'] = mle_params_were_fresh
                    else:
                        beta_to_use = state[agent_type]['exploit_lock_beta']
                        mle_params_were_fresh = state[agent_type]['exploit_lock_fresh']
                else:
                    mle_params_were_fresh = not state[agent_type]['mle_job']
                    beta_to_use = (
                        state[agent_type]['beta_package']
                        if mle_params_were_fresh
                        else state[agent_type]['beta_history'].get(state[agent_type]['mle_job_key'] - 1)
                    )

                step_results[f'fresh_params_{agent_type}'] = mle_params_were_fresh

                price_offered, _ = find_optimal_price_generic(features_arr, beta_to_use)
                accepted, epsilon = simulate_passenger_acceptance_true(price_offered, features_arr)
                reward = price_offered * accepted
                regret = oracle_exp_reward - reward
                step_results[f'reward_{agent_type}'] = reward
                step_results[f'regret_{agent_type}'] = regret
                step_results[f'stale_regret_{agent_type}'] = 0.0 if mle_params_were_fresh else regret
                step_results[f'price_{agent_type}'] = price_offered
                step_results[f'epsilon_{agent_type}'] = epsilon

                # Counterfactual timeliness analysis (ILQX only):
                # compare real asynchronous deployment vs instant deployment of beta_k at exploit start.
                if agent_type == 'cvx' and phase.endswith('_exploit') and k_epoch > 0:
                    beta_instant = beta_to_use
                    pending_k_for_cvx = (
                        state['cvx']['mle_job']
                        and state['cvx']['mle_job_key'] == k_epoch
                        and state['cvx']['mle_package_to_apply'] is not None
                    )
                    if pending_k_for_cvx:
                        beta_instant = state['cvx']['mle_package_to_apply']

                    if beta_instant is not None:
                        price_instant, _ = find_optimal_price_generic(features_arr, beta_instant)
                        p_real = p_accept_oracle_model(price_offered, features_arr)
                        p_inst = p_accept_oracle_model(price_instant, features_arr)
                        regret_real_expected = oracle_exp_reward - (price_offered * p_real)
                        regret_instant = oracle_exp_reward - (price_instant * p_inst)
                        step_results['price_cvx_instant'] = price_instant
                        step_results['regret_cvx_instant'] = regret_instant
                        step_results['regret_cvx_expected_real'] = regret_real_expected
                        step_results['timeliness_loss_cvx'] = regret_real_expected - regret_instant

        trigger_mle_now = False
        if phase.startswith('k_sq_epoch_') and phase.endswith('_explore'):
            trigger_mle_now = (is_current_epoch_full_exploration and arrival_num_1_indexed == ((current_k_sq_epoch_val + 1)**2) - 1) or \
                              (not is_current_epoch_full_exploration and current_k_sq_epoch_expl_done_count == exploration_samples_for_this_epoch)

        if trigger_mle_now:
            for agent_type in active_agents:
                if not state[agent_type]['mle_job']:
                    state[agent_type]['mle_trigger_events'].append({'key': current_k_sq_epoch_val, 'trigger_idx': t})
                    data_for_mle = pd.DataFrame(state[agent_type]['exploration_data'])
                    prev_beta = state[agent_type]['beta_history'].get(current_k_sq_epoch_val - 1)

                    params_map = {
                        'pgd': {'m_k_exploration_batch_size': exploration_samples_for_this_epoch, 'p_penalty_norm_val': P_PENALTY_NORM_VAL, 'num_iters_max': 2000, 'eta_coeff': eta_coeff_param, 'eta_k_power': eta_k_power_param, 'stop_eps': stop_eps_param},
                        'cvx': {'lambda_t_factor': lambda_t_factor_param}
                    }

                    package, time_taken = train_beta_agent_mle(data_for_mle, t, prev_beta, W_L1_RADIUS_AGENT, estimator_type=agent_type, estimator_params=params_map[agent_type], current_epoch_k=current_k_sq_epoch_val)

                    state[agent_type]['solving_times'].append({
                        'epoch_k': current_k_sq_epoch_val,
                        'trigger_idx': t,
                        'n_samples': len(data_for_mle),
                        'time_taken_sec': time_taken
                    })
                    state[agent_type]['mle_job'] = True
                    state[agent_type]['mle_job_key'] = current_k_sq_epoch_val
                    state[agent_type]['mle_package_to_apply'] = package
                    state[agent_type]['mle_complete_time'] = current_arrival_timestamp + np.timedelta64(int(time_taken * 1e9), 'ns')

        step_results['reward_original'] = reward_original
        step_results['regret_original'] = oracle_exp_reward - reward_original
        if stream_writer is not None:
            stream_writer.write_row(step_results)
        else:
            simulation_results.append(step_results)

    if stream_writer is not None:
        stream_writer.close()
        return None, state
    return pd.DataFrame(simulation_results), state

# --- 4. NEW: Timeliness Analysis Function ---
def analyze_update_timeliness(results_df, state, agent_type, experiment_key):
    """
    Analyzes parameter update timeliness. Denominators use ALL epochs with exploit phases
    (identical for both agents), so skipped triggers count as "not timely".
    """
    agent_name_map = {'pgd': 'AMLE-PGD', 'cvx': 'ILQX'}
    agent_name = agent_name_map.get(agent_type, agent_type.upper())
    scenario_name, eta_coeff, lambda_factor, stop_eps, eta_k_power = experiment_key
    print(f"\n--- Timeliness Analysis for {agent_name} [{scenario_name}] (c_k={eta_coeff}, lam={lambda_factor}, stop_eps={stop_eps}, eta_k_pow={eta_k_power}) ---")

    trigger_events = state[agent_type]['mle_trigger_events']
    completion_events = state[agent_type]['mle_completion_events']
    completion_map = {event['key']: event['completion_idx'] for event in completion_events}

    exploit_mask = results_df['phase'].str.contains('_exploit', na=False)
    exploit_rows = results_df[exploit_mask]
    if exploit_rows.empty:
        print("No exploitation phases found.")
        return

    all_exploit_epochs = sorted(exploit_rows['k_sq_epoch'].unique())
    exploit_start = {}
    for k in all_exploit_epochs:
        exploit_start[k] = int(exploit_rows.loc[exploit_rows['k_sq_epoch'] == k, 'arrival_idx'].iloc[0])

    total_epochs = len(all_exploit_epochs)
    print(f"Total epochs with exploit phase: {total_epochs}  |  MLE triggered: {len(trigger_events)}  |  MLE completed: {len(completion_events)}")

    # --- 1. Intra-Epoch: MLE for epoch k completes before exploit phase of epoch k ---
    intra_timely = 0
    for k in all_exploit_epochs:
        if k in completion_map and completion_map[k] <= exploit_start[k]:
            intra_timely += 1
    print(f"1. Intra-Epoch (update k ready before exploit k):  {intra_timely} / {total_epochs} ({intra_timely/total_epochs:.2%})")

    # --- 2. Freshness Rate: exploit steps using params fresh at epoch-k exploit start (epoch-locked) ---
    fresh_col = f'fresh_params_{agent_type}'
    fresh_decisions = int(exploit_rows[fresh_col].sum())
    total_decisions = len(exploit_rows)
    print(f"2. Freshness Rate (exploit epochs fresh at exploit start):  {fresh_decisions} / {total_decisions} ({fresh_decisions/max(1,total_decisions):.2%})")

    # --- 3. Inter-Epoch: MLE for epoch k completes before exploit phase of epoch k+1 ---
    inter_timely, inter_total = 0, 0
    for k in all_exploit_epochs:
        if (k + 1) not in exploit_start:
            continue
        inter_total += 1
        if k in completion_map and completion_map[k] <= exploit_start[k + 1]:
            inter_timely += 1
    if inter_total > 0:
        print(f"3. Inter-Epoch (update k ready before exploit k+1):  {inter_timely} / {inter_total} ({inter_timely/inter_total:.2%})")

    stale_col = f'stale_regret_{agent_type}'
    regret_col = f'regret_{agent_type}'
    if stale_col in exploit_rows.columns:
        stale_regret_sum = float(exploit_rows[stale_col].sum())
    elif fresh_col in exploit_rows.columns and regret_col in exploit_rows.columns:
        stale_mask = ~exploit_rows[fresh_col].astype(bool)
        stale_regret_sum = float(exploit_rows.loc[stale_mask, regret_col].sum())
    else:
        stale_regret_sum = float('nan')
    total_exploit_regret = float(exploit_rows[regret_col].sum()) if regret_col in exploit_rows.columns else float('nan')
    if pd.notna(stale_regret_sum) and pd.notna(total_exploit_regret) and total_exploit_regret != 0:
        print(
            f"4. Stale Regret (exploit, stale params):  {stale_regret_sum:.2f} / {total_exploit_regret:.2f} "
            f"({stale_regret_sum / total_exploit_regret:.2%} of exploit regret)"
        )
    elif pd.notna(stale_regret_sum):
        print(f"4. Stale Regret (exploit, stale params):  {stale_regret_sum:.2f}")


def compute_cvx_timeliness_counterfactual_row(results_df, experiment_key) -> dict | None:
    """Return one flat metrics row for ILQX-real vs ILQX-instant counterfactual (or None if unavailable)."""
    scenario_name, eta_coeff, lambda_factor, stop_eps, eta_k_power = experiment_key
    needed = {'regret_cvx_instant', 'regret_cvx_expected_real', 'timeliness_loss_cvx'}
    if not needed.issubset(results_df.columns):
        return None

    exploit_rows = results_df[results_df['phase'].str.contains('_exploit', na=False)].copy()
    exploit_rows = exploit_rows.dropna(subset=['regret_cvx_instant', 'regret_cvx_expected_real', 'timeliness_loss_cvx'])
    if exploit_rows.empty:
        return None

    real_reg = float(exploit_rows['regret_cvx_expected_real'].sum())
    instant_reg = float(exploit_rows['regret_cvx_instant'].sum())
    loss = float(exploit_rows['timeliness_loss_cvx'].sum())
    rel = (loss / instant_reg) if instant_reg != 0 else float('nan')

    row = {
        'scenario': scenario_name,
        'eta_coeff': eta_coeff,
        'lambda_factor': lambda_factor,
        'stop_eps': stop_eps,
        'eta_k_pow': eta_k_power,
        'n_exploit_steps': len(exploit_rows),
        'exploit_regret_ilqx_real': real_reg,
        'exploit_regret_ilqx_instant': instant_reg,
        'timeliness_loss_abs': loss,
        'timeliness_loss_pct_of_instant': rel,
    }

    if 'fresh_params_cvx' in exploit_rows.columns:
        row['ilqx_fresh_rate_exploit'] = float(exploit_rows['fresh_params_cvx'].astype(bool).mean())

    max_arrival = int(exploit_rows['arrival_idx'].max())
    for pct in (0.05, 0.10, 0.20):
        pct_label = int(pct * 100)
        early_cutoff = int(max_arrival * pct)
        early = exploit_rows[exploit_rows['arrival_idx'] <= early_cutoff]
        if early.empty:
            row[f'early_{pct_label}pct_loss_abs'] = float('nan')
            row[f'early_{pct_label}pct_loss_pct_of_instant'] = float('nan')
            continue
        early_loss = float(early['timeliness_loss_cvx'].sum())
        early_instant = float(early['regret_cvx_instant'].sum())
        row[f'early_{pct_label}pct_loss_abs'] = early_loss
        row[f'early_{pct_label}pct_loss_pct_of_instant'] = (
            early_loss / early_instant if early_instant != 0 else float('nan')
        )

    stale_mask = (
        ~exploit_rows['fresh_params_cvx'].astype(bool)
        if 'fresh_params_cvx' in exploit_rows.columns
        else pd.Series(False, index=exploit_rows.index)
    )
    stale_loss = float(exploit_rows.loc[stale_mask, 'timeliness_loss_cvx'].sum()) if stale_mask.any() else 0.0
    stale_instant_reg = (
        float(exploit_rows.loc[stale_mask, 'regret_cvx_instant'].sum()) if stale_mask.any() else float('nan')
    )
    row['stale_only_loss_abs'] = stale_loss
    row['stale_only_loss_pct_of_stale_instant'] = (
        stale_loss / stale_instant_reg
        if (pd.notna(stale_instant_reg) and stale_instant_reg != 0)
        else float('nan')
    )

    row['worst_density_bucket_loss_abs'] = float('nan')
    row['worst_density_bucket_loss_pct_of_instant'] = float('nan')
    row['worst_density_bucket_n'] = np.nan
    row['worst_density_bucket_density_min'] = float('nan')
    row['worst_density_bucket_density_max'] = float('nan')

    if 'arrival_time' in exploit_rows.columns and exploit_rows['arrival_time'].notna().any():
        tmp = exploit_rows[['arrival_time', 'timeliness_loss_cvx', 'regret_cvx_instant']].copy()
        tmp['arrival_time'] = pd.to_datetime(tmp['arrival_time'], errors='coerce')
        tmp = tmp.dropna(subset=['arrival_time'])
        if len(tmp) >= 20:
            dt = tmp['arrival_time'].diff().dt.total_seconds().abs()
            median_dt = float(dt.dropna().median()) if dt.notna().any() else 1.0
            if not np.isfinite(median_dt) or median_dt <= 0:
                median_dt = 1.0
            dt = dt.fillna(median_dt).clip(lower=1e-6)
            tmp['density_proxy'] = 1.0 / dt
            try:
                tmp['density_bin'] = pd.qcut(tmp['density_proxy'], q=5, duplicates='drop')
                g = tmp.groupby('density_bin', observed=True).agg(
                    loss=('timeliness_loss_cvx', 'sum'),
                    instant=('regret_cvx_instant', 'sum'),
                    n=('density_proxy', 'size'),
                    dens_min=('density_proxy', 'min'),
                    dens_max=('density_proxy', 'max'),
                ).reset_index()
                g['rel'] = np.where(g['instant'] != 0, g['loss'] / g['instant'], np.nan)
                if not g.empty and g['rel'].notna().any():
                    worst = g.loc[g['rel'].idxmax()]
                    row['worst_density_bucket_loss_abs'] = float(worst['loss'])
                    row['worst_density_bucket_loss_pct_of_instant'] = float(worst['rel'])
                    row['worst_density_bucket_n'] = int(worst['n'])
                    row['worst_density_bucket_density_min'] = float(worst['dens_min'])
                    row['worst_density_bucket_density_max'] = float(worst['dens_max'])
            except ValueError:
                pass

    return row


def analyze_cvx_timeliness_counterfactual(results_df, experiment_key) -> dict | None:
    """Counterfactual: ILQX-real (async) vs ILQX-instant (same beta_k, no deployment lag)."""
    scenario_name, eta_coeff, lambda_factor, stop_eps, eta_k_power = experiment_key
    row = compute_cvx_timeliness_counterfactual_row(results_df, experiment_key)

    print(
        f"\n--- ILQX Timeliness Counterfactual [{scenario_name}] "
        f"(c_k={eta_coeff}, lam={lambda_factor}, stop_eps={stop_eps}, eta_k_pow={eta_k_power}) ---"
    )
    if row is None:
        print("Counterfactual columns are missing or no usable exploit rows (older CSV or empty run).")
        return None

    print(f"Exploit expected regret (ILQX-real async):    {row['exploit_regret_ilqx_real']:.2f}")
    print(f"Exploit expected regret (ILQX-instant cf):    {row['exploit_regret_ilqx_instant']:.2f}")
    print(
        f"Timeliness loss (real - instant):             {row['timeliness_loss_abs']:.2f} "
        f"({row['timeliness_loss_pct_of_instant']:.2%} of instant)"
    )
    for pct_label in (5, 10, 20):
        e_loss = row.get(f'early_{pct_label}pct_loss_abs', float('nan'))
        e_rel = row.get(f'early_{pct_label}pct_loss_pct_of_instant', float('nan'))
        if pd.notna(e_loss):
            print(f"Early-{pct_label}% timeliness loss:                 {e_loss:.2f} ({e_rel:.2%} of instant)")
    print(f"Stale-only timeliness loss contribution:      {row['stale_only_loss_abs']:.2f}")
    stale_rel = row.get('stale_only_loss_pct_of_stale_instant', float('nan'))
    if pd.notna(stale_rel):
        print(f"Stale-only normalized loss:                   {stale_rel:.2%} of stale instant regret")
    worst_rel = row.get('worst_density_bucket_loss_pct_of_instant', float('nan'))
    if pd.notna(worst_rel):
        print(
            "Worst density-bucket normalized loss:         "
            f"{row['worst_density_bucket_loss_abs']:.2f} ({worst_rel:.2%}, n={int(row['worst_density_bucket_n'])}, "
            f"density~[{row['worst_density_bucket_density_min']:.4g}, {row['worst_density_bucket_density_max']:.4g}] 1/sec)"
        )
    return row


def save_timeliness_counterfactual_csv(rows: list[dict], out_path: Path | None = None) -> Path | None:
    """Write aggregated counterfactual metrics for all runs."""
    if not rows:
        return None
    out_path = out_path or (OUTPUT_ROOT / "timeliness_counterfactual.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ['scenario', 'eta_coeff', 'lambda_factor', 'stop_eps', 'eta_k_pow'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return out_path


def _run_inline_plots(repeat_result_paths: dict[tuple, list[Path]]) -> None:
    """Optional end-of-script figures (heavy); prefer plot_cumulative_regret.py."""
    from matplotlib.colors import to_hex
    import matplotlib.cm as _cm

    mean_results = {}
    for key, repeat_paths in repeat_result_paths.items():
        mean_cols = None
        header = None
        aligned = []
        for repeat_idx, repeat_path in enumerate(repeat_paths):
            if mean_cols is None:
                header = pd.read_csv(repeat_path, nrows=0).columns
                mean_cols = [
                    c for c in header
                    if c.startswith(('reward_', 'regret_', 'stale_regret_', 'price_', 'epsilon_'))
                    or c in ['oracle_price', 'oracle_exp_revenue']
                ]
            base_cols = ['arrival_idx', 'k_sq_epoch', 'phase']
            keep_cols = [c for c in base_cols + mean_cols if c in header]
            tmp = pd.read_csv(repeat_path, usecols=keep_cols).copy()
            tmp['_repeat_idx'] = repeat_idx
            aligned.append(tmp)

        stacked = pd.concat(aligned, ignore_index=True)
        mean_df = (
            stacked
            .groupby('arrival_idx', as_index=False)
            .agg({
                **({'k_sq_epoch': 'first'} if 'k_sq_epoch' in stacked.columns else {}),
                **({'phase': 'first'} if 'phase' in stacked.columns else {}),
                **{c: 'mean' for c in mean_cols if c in stacked.columns},
            })
        )
        for _ag in ['pgd', 'cvx', 'original']:
            if f'regret_{_ag}' in mean_df.columns:
                mean_df[f'cumulative_regret_{_ag}'] = mean_df[f'regret_{_ag}'].cumsum()
        mean_results[key] = mean_df

    if not mean_results:
        return

    _scenario_order = list(dict.fromkeys(k[0] for k in mean_results.keys()))
    _marker_pool = ['o', '^', 's', 'D', 'v', 'P']
    _scenario_markers = {sn: _marker_pool[i % len(_marker_pool)] for i, sn in enumerate(_scenario_order)}
    _unique_etas = sorted(set(k[1] for k in mean_results.keys()))
    _unique_lams = sorted(set(k[2] for k in mean_results.keys()))
    _RED_SHADES = ['#B71C1C', '#D32F2F', '#E53935', '#EF5350', '#FF8A80']
    _BLUE_SHADES = ['#0D47A1', '#1565C0', '#1E88E5', '#42A5F5', '#64B5F6']
    _greens = _cm.get_cmap('Greens')
    _ilqx_marker_pool = ['^', 'v', 'D', 'P', 'X', 'h']
    _scenario_markers_ilqx = {sn: _ilqx_marker_pool[i % len(_ilqx_marker_pool)] for i, sn in enumerate(_scenario_order)}

    def _param_color(unique_vals, val, palette):
        vals = sorted(unique_vals)
        if len(vals) <= 1:
            return palette[2]
        idx = vals.index(val)
        t = idx / max(1, len(vals) - 1)
        pal_idx = int(round(t * (len(palette) - 1)))
        return palette[pal_idx]

    _green_shades = {}
    for _i, _sn in enumerate(_scenario_order):
        _t = 0.65 if len(_scenario_order) <= 1 else 0.40 + 0.50 * _i / (len(_scenario_order) - 1)
        _green_shades[_sn] = to_hex(_greens(_t))

    _marker_stride = max(1, int(EFFECTIVE_TOTAL_ARRIVALS / 20))
    _markevery_pgd = (0, _marker_stride)
    _markevery_ilqx = (_marker_stride // 2, _marker_stride)

    print("\nGenerating Combined Cumulative Regret plot...")
    fig, ax = plt.subplots(figsize=(14, 9))
    _tsp_plotted, _ilqx_plotted = set(), set()
    for key, df in mean_results.items():
        scenario_name, eta, lam, seps, eta_pow = key
        mk = _scenario_markers[scenario_name]
        short_scn = scenario_name.replace('__', ' ').replace('_', ' ')
        ax.plot(df['arrival_idx'], df['cumulative_regret_pgd'],
                color=_param_color(_unique_etas, eta, _RED_SHADES), marker=mk,
                markevery=_markevery_pgd, markersize=6, linewidth=1.2,
                label=f"PGD  [{short_scn}] η={eta}, λ={lam}, η^k={eta_pow}")
        if 'cumulative_regret_cvx' in df.columns:
            ilqx_key = (scenario_name, lam)
            if ilqx_key not in _ilqx_plotted:
                _ilqx_plotted.add(ilqx_key)
                ax.plot(df['arrival_idx'], df['cumulative_regret_cvx'],
                        color=_param_color(_unique_lams, lam, _BLUE_SHADES),
                        marker=_scenario_markers_ilqx[scenario_name],
                        markevery=_markevery_ilqx, markersize=6, linewidth=1.2,
                        label=f"ILQX [{short_scn}] λ={lam}")
        if scenario_name not in _tsp_plotted:
            _tsp_plotted.add(scenario_name)
            ax.plot(df['arrival_idx'], df['cumulative_regret_original'],
                    color=_green_shades[scenario_name], marker=mk, markevery=_markevery_pgd,
                    markersize=5, linewidth=1.2, linestyle='--', label=f"TSP  [{short_scn}]")
    ax.set_title('Cumulative Regret Comparison', fontsize=16)
    ax.set_xlabel('Arrival Index (t)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    _regret_pdf = OUTPUT_ROOT / "cumulative_regret.pdf"
    fig.savefig(_regret_pdf, format='pdf', bbox_inches='tight')
    print(f"Regret plot saved to: {_regret_pdf}")
    plt.show()

    print("\nGenerating Combined Summary Bar Charts...")
    summary_data = []
    for key, df in mean_results.items():
        scenario_name, eta, lam, seps, eta_pow = key
        experiment_label = f"{scenario_name}; $\\tilde{{c}}_k$={eta}, $\\lambda_t$={lam}, stop_eps={seps}, $\\eta_k$^{eta_pow}"
        summary_data.append({
            'Experiment': experiment_label, 'Strategy': 'AMLE-PGD',
            'Total Reward': df['reward_pgd'].sum(), 'Total Regret': df['regret_pgd'].sum(),
        })
        if 'reward_cvx' in df.columns:
            summary_data.append({
                'Experiment': experiment_label, 'Strategy': 'ILQX',
                'Total Reward': df['reward_cvx'].sum(), 'Total Regret': df['regret_cvx'].sum(),
            })
    summary_df = pd.DataFrame(summary_data)
    tsp_rows, seen_tsp = [], set()
    for key, df in mean_results.items():
        scenario_name = key[0]
        if scenario_name in seen_tsp:
            continue
        seen_tsp.add(scenario_name)
        tsp_rows.append({
            'Experiment': scenario_name, 'Strategy': 'TSP',
            'Total Reward': df['reward_original'].sum(),
            'Total Regret': df['regret_original'].sum(),
        })
    summary_df = pd.concat([summary_df, pd.DataFrame(tsp_rows)], ignore_index=True)
    _strategy_palette = {'AMLE-PGD': '#D32F2F', 'ILQX': '#1E88E5', 'TSP': 'green'}
    for y_col, title in [('Total Reward', 'Total Cumulative Reward'), ('Total Regret', 'Total Cumulative Regret')]:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=summary_df, x='Experiment', y=y_col, hue='Strategy', palette=_strategy_palette, ax=ax)
        ax.set_title(title, fontsize=16)
        plt.xticks(rotation=15, ha='right')
        fig.tight_layout()
        plt.show()


# --- 5. Experiment Setup and Execution ---
all_result_paths: dict[tuple, Path] = {}
all_states = {}
all_repeat_result_paths: dict[tuple, list[Path]] = {}


def print_feature_volatility_analysis(scenario_name):
    print("\n" + "="*20 + f" Feature Volatility Analysis [{scenario_name}] " + "="*20)
    sim_features_df = processed_df.loc[:EFFECTIVE_TOTAL_ARRIVALS - 1, AGENT_CONTEXTUAL_FEATURES]
    print(f"Analysis based on the first {EFFECTIVE_TOTAL_ARRIVALS} arrivals.")

    binary_features = [col for col in sim_features_df.columns if sim_features_df[col].dropna().isin([0, 1]).all()]
    continuous_features = [col for col in sim_features_df.columns if col not in binary_features]

    print("\n--- Continuous Features ---")
    if continuous_features:
        stats_continuous = sim_features_df[continuous_features].agg(['mean', 'var', 'min', 'max']).transpose()
        stats_continuous['range'] = stats_continuous['max'] - stats_continuous['min']
        print(stats_continuous[['mean', 'var', 'range']])
    else:
        print("No continuous features found.")

    print("\n--- Binary Features ---")
    if binary_features:
        stats_binary = sim_features_df[binary_features].mean().reset_index()
        stats_binary.columns = ['feature', 'ratio_of_ones']
        print(stats_binary)
    else:
        print("No binary features found.")
    print("="*55)


def save_experiment_outputs(results_df, final_state, out_dir, active_agents):
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_df is not None:
        rev_reg_cols = _revenue_regret_columns(active_agents)
        existing = [c for c in rev_reg_cols if c in results_df.columns]
        results_df[existing].to_csv(out_dir / "revenue_regret.csv", index=False)

    solve_records = []
    for ag in active_agents:
        for rec in final_state[ag]['solving_times']:
            solve_records.append({'agent': ag, **rec})
    if solve_records:
        pd.DataFrame(solve_records).to_csv(out_dir / "solving_time.csv", index=False)


def save_oracle_revenue(results_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = out_dir / "oracle_revenue.csv"
    results_df[['arrival_idx', 'oracle_price', 'oracle_exp_revenue']].to_csv(oracle_path, index=False)


# --- Main Experiment Loop (with repeats) ---
for repeat_idx in range(NUM_REPEATS):
    run_seed_global = MASTER_SEED + 1000 * repeat_idx
    rng = np.random.default_rng(run_seed_global)
    print(f"\n{'='*20} REPEAT {repeat_idx + 1}/{NUM_REPEATS} (seed={run_seed_global}) {'='*20}")

    for scenario_name in BETA_SIMULATOR_SCENARIOS:
        BETA_SIMULATOR, SCALER_SIMULATOR, SCALED_FEATURES_SIMULATOR = load_beta_simulator_components(scenario_name)
        if BETA_SIMULATOR is None:
            print(f"Skipping scenario because simulator components are missing: {scenario_name}")
            continue

        configure_feature_sets(BETA_SIMULATOR)
        _build_oracle_cache()
        missing_features = [col for col in AGENT_CONTEXTUAL_FEATURES if col not in processed_df.columns]
        if missing_features:
            raise ValueError(f"Processed data is missing required contextual features for {scenario_name}: {missing_features}")

        if repeat_idx == 0:
            print_feature_volatility_analysis(scenario_name)

        oracle_saved_this_scenario = False
        ilqx_cache_paths: dict[tuple, Path] = {}
        ilqx_cache_states: dict[tuple, dict] = {}
        run_seed = run_seed_global

        for eta_coeff, lambda_factor in HYPERPARAMETER_PAIRS:
            ilqx_key = (scenario_name, lambda_factor)

            if CVXPY_AVAILABLE and MOSEK_SOLVER_AVAILABLE_IN_CVXPY and ilqx_key not in ilqx_cache_paths:
                ilqx_cache_dir = OUTPUT_ROOT / scenario_name / f"repeat_{repeat_idx}" / "_ilqx_cache"
                ilqx_stream_path = ilqx_cache_dir / f"lam{lambda_factor}_stream.csv"
                if ilqx_stream_path.exists() and ilqx_stream_path.stat().st_size > 0:
                    print(f"\n--- ILQX cache hit (λ={lambda_factor}): reusing {ilqx_stream_path.name} ---")
                    cvx_state = {'pgd': {'mle_trigger_events': [], 'mle_completion_events': [], 'solving_times': []},
                                 'cvx': {'mle_trigger_events': [], 'mle_completion_events': [], 'solving_times': []}}
                else:
                    print(f"\n--- ILQX-only (once per scenario+λ): {ilqx_key}, repeat={repeat_idx} ---")
                    _, cvx_state = run_simulation(
                        eta_coeff_param=eta_coeff,
                        lambda_t_factor_param=lambda_factor,
                        stop_eps_param=STOP_EPS_VALUES[0],
                        eta_k_power_param=ETA_K_POWER_VALUES[0],
                        total_arrivals=EFFECTIVE_TOTAL_ARRIVALS,
                        data_df=processed_df,
                        progress_desc=f"[R{repeat_idx}] {scenario_name} ILQX-only (λ={lambda_factor})",
                        active_agents=['cvx'],
                        run_seed=run_seed,
                        stream_csv_path=ilqx_stream_path,
                    )
                ilqx_cache_paths[ilqx_key] = ilqx_stream_path
                ilqx_cache_states[ilqx_key] = cvx_state

            for stop_eps in STOP_EPS_VALUES:
                base_key = (scenario_name, eta_coeff, lambda_factor, stop_eps)

                for eta_k_power in ETA_K_POWER_VALUES:
                    key = (*base_key, eta_k_power)
                    progress_description = f"[R{repeat_idx}] {scenario_name} PGD (c_k={eta_coeff}, eps={stop_eps}, eta_pow={eta_k_power})"
                    print(f"\n--- Starting: {key}, repeat={repeat_idx} ---")

                    hp_dir = OUTPUT_ROOT / scenario_name / f"repeat_{repeat_idx}" / f"eta{eta_coeff}_lam{lambda_factor}_stopeps{stop_eps}_etapow{eta_k_power}"
                    revenue_path_check = hp_dir / "revenue_regret.csv"
                    if revenue_path_check.exists() and revenue_path_check.stat().st_size > 0:
                        print(f"  [SKIP] revenue_regret.csv already exists, loading cached result.")
                        all_result_paths[key] = revenue_path_check
                        # Reconstruct minimal state for timeliness analysis (no in-memory state available)
                        cached_state = {ag: {'mle_trigger_events': [], 'mle_completion_events': [], 'solving_times': []} for ag in ['pgd', 'cvx']}
                        all_states[key] = cached_state
                        all_repeat_result_paths.setdefault(key, []).append(revenue_path_check)
                        if not oracle_saved_this_scenario:
                            oracle_saved_this_scenario = True  # assume oracle already saved
                        continue

                    pgd_stream_path = hp_dir / "_stream_full.csv"
                    _, pgd_state = run_simulation(
                        eta_coeff_param=eta_coeff,
                        lambda_t_factor_param=lambda_factor,
                        stop_eps_param=stop_eps,
                        eta_k_power_param=eta_k_power,
                        total_arrivals=EFFECTIVE_TOTAL_ARRIVALS,
                        data_df=processed_df,
                        progress_desc=progress_description,
                        active_agents=['pgd'],
                        run_seed=run_seed,
                        stream_csv_path=pgd_stream_path,
                    )

                    merge_agents = ['pgd']
                    final_state = pgd_state
                    cvx_stream = ilqx_cache_paths.get(ilqx_key)
                    if cvx_stream is not None:
                        merge_agents.append('cvx')
                        final_state = {**pgd_state, **ilqx_cache_states[ilqx_key]}

                    if not oracle_saved_this_scenario:
                        oracle_dir = OUTPUT_ROOT / scenario_name / f"repeat_{repeat_idx}"
                        save_oracle_revenue_from_csv(pgd_stream_path, oracle_dir)
                        oracle_saved_this_scenario = True

                    revenue_path = hp_dir / "revenue_regret.csv"
                    finalize_revenue_regret_csv(
                        pgd_stream_path,
                        revenue_path,
                        merge_agents,
                        cvx_csv=cvx_stream,
                    )
                    pgd_stream_path.unlink(missing_ok=True)

                    save_experiment_outputs(None, final_state, hp_dir, list(final_state.keys()))
                    all_result_paths[key] = revenue_path
                    all_states[key] = final_state
                    all_repeat_result_paths.setdefault(key, []).append(revenue_path)

print("\n" + "="*20 + " ALL SIMULATIONS COMPLETE " + "="*20)
print(f"Output files saved under: {OUTPUT_ROOT}")


# --- 6. Noise Chasing Analysis ---
print("\n" + "="*20 + " Noise Chasing Analysis " + "="*20)

for key, csv_path in all_result_paths.items():
    scenario_name, eta_coeff, lambda_factor, stop_eps, eta_k_power = key
    print(f"\n--- Analysis for Experiment: [{scenario_name}] ($\\tilde{{c}}_k$={eta_coeff}, $\\lambda_t$={lambda_factor}, stop_eps={stop_eps}, eta_k_pow={eta_k_power}) ---")

    results_df = load_revenue_regret_subset(csv_path)
    results_df['epsilon_pgd_lagged'] = results_df['epsilon_pgd'].shift(1) if 'epsilon_pgd' in results_df.columns else np.nan
    
    if 'price_pgd' in results_df.columns and 'epsilon_pgd_lagged' in results_df.columns:
        corr_pgd = results_df[['price_pgd', 'epsilon_pgd_lagged']].corr().iloc[0, 1]
        print(f"Correlation between AMLE-PGD Price(t) and Noise(t-1): {corr_pgd:.4f}")

    if 'epsilon_cvx' in results_df.columns and not results_df['epsilon_cvx'].isnull().all():
        results_df['epsilon_cvx_lagged'] = results_df['epsilon_cvx'].shift(1)
        corr_cvx = results_df[['price_cvx', 'epsilon_cvx_lagged']].corr().iloc[0, 1]
        print(f"Correlation between ILQX Price(t) and Noise(t-1):    {corr_cvx:.4f}")

# --- 7. NEW: Parameter Update Timeliness Analysis ---
print("\n" + "="*20 + " Parameter Update Timeliness Analysis " + "="*20)
timeliness_counterfactual_rows: list[dict] = []
for key, csv_path in all_result_paths.items():
    final_state = all_states[key]
    results_df = load_revenue_regret_subset(csv_path)
    active_agents = list(final_state.keys())
    for agent in active_agents:
        analyze_update_timeliness(results_df, final_state, agent, key)
    if 'cvx' in active_agents:
        cf_row = analyze_cvx_timeliness_counterfactual(results_df, key)
        if cf_row is not None:
            timeliness_counterfactual_rows.append(cf_row)

_timeliness_csv = save_timeliness_counterfactual_csv(timeliness_counterfactual_rows)
if _timeliness_csv is not None:
    print(f"\nSaved timeliness counterfactual summary: {_timeliness_csv} ({len(timeliness_counterfactual_rows)} row(s))")
else:
    print("\nNo timeliness counterfactual rows to save (re-run simulations with updated PGD_ILQX.py).")


# --- 8. Optional inline plots ---
if GENERATE_INLINE_PLOTS:
    _run_inline_plots(all_repeat_result_paths)
else:
    print("\nSkipping inline plots (GENERATE_INLINE_PLOTS=False). Use plot_cumulative_regret.py for figures.")
