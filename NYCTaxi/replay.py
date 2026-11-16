"""FIFO deployment with paired instantaneous and delayed expected revenues.

Rows have coefficients [intercept, Z main effects, Z * price / price_unit].
Exploration observations are identical in both replays. Delay changes the deployed
estimate, not the estimates learned from that fixed exploration data.
"""
import numpy as np
from scipy.special import expit, lambertw


def index_AB(Z, theta):
    d = Z.shape[1]
    return theta[0] + Z @ theta[1:1 + d], Z @ theta[1 + d:]


def prob(A, B, price, price_unit=10.0):
    return expit(A + B * np.asarray(price) / price_unit)


def best_price(A, B, price_unit=10.0, price_lo=2.5, price_hi=70.0):
    A, B = np.asarray(A), np.asarray(B)
    w = np.real(lambertw(np.exp(np.minimum(A - 1.0, 700.0))))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(B < 0, (1.0 + w) / (-B), np.inf) * price_unit
    return np.clip(p, price_lo, price_hi)


def schedule(T, m0=410, mk=28):
    """0-based exploration indices and 1-based completed-block triggers.

    Initial block [1,m0], later blocks [k^2,k^2+mk-1]; overlaps count once.
    A partially observed final block does not trigger an update.
    """
    if T < 0 or m0 < 1 or mk < 1:
        raise ValueError("T must be nonnegative; exploration lengths must be positive")
    explore = np.zeros(T, bool)
    explore[:m0] = True
    trig = [m0] if m0 <= T else []
    k = 1
    while k * k <= T:
        end = k * k + mk - 1
        explore[k * k - 1:min(end, T)] = True
        if m0 < end <= T:
            trig.append(end)
        k += 1
    return np.flatnonzero(explore), np.asarray(trig, dtype=np.int64)


def fifo_finish(start, secs):
    """One worker solves every snapshot in trigger order, without dropping jobs."""
    finish = np.empty(len(start))
    previous = -np.inf
    for j, (t, seconds) in enumerate(zip(start, secs)):
        previous = max(t, previous) + seconds
        finish[j] = previous
    return finish


def _policy_prices(Z, thetas, est, price_unit, price_lo, price_hi):
    p = np.full(len(est), np.clip(20.0, price_lo, price_hi))
    # Contiguous groups avoid materializing an arrivals-by-coefficients matrix.
    edges = np.r_[0, np.flatnonzero(np.diff(est)) + 1, len(est)]
    for a, b in zip(edges[:-1], edges[1:]):
        if a < b and est[a] >= 0:
            p[a:b] = best_price(*index_AB(Z[a:b], thetas[est[a]]),
                                price_unit, price_lo, price_hi)
    return p


def evaluate(Z, clock, fare, true_theta, explore_idx, explore_price, trig,
             thetas, secs, price_unit=10.0, price_lo=2.5, price_hi=70.0):
    """Return summary rows, cumulative curves, and update-level timing details.

    clock and secs are seconds. trig is 1-based, explore_idx is 0-based.
    summary has windows first5/last20/all (fractions of all arrivals). Revenue
    includes common exploration revenue. delay_loss = instant minus delayed,
    so a negative result means delay improved realized expected revenue.
    theta_gap_l2 excludes fallback customers; fallback_share reports their share.
    All share and mean gap metrics use exploitation customers only.
    updates.eligible includes triggers having exploitation before the next trigger;
    its first exploitation arrival is that update's deadline.
    """
    Z = np.asarray(Z)
    T = len(Z)
    clock, fare = np.asarray(clock, float), np.asarray(fare, float)
    explore_idx, trig = np.asarray(explore_idx, np.int64), np.asarray(trig, np.int64)
    explore_price, secs = np.asarray(explore_price, float), np.asarray(secs, float)
    thetas, true_theta = np.asarray(thetas, float), np.asarray(true_theta, float)
    if T < 1 or Z.ndim != 2 or clock.shape != (T,) or fare.shape != (T,):
        raise ValueError("A nonempty, equally sized feature/clock/fare stream is required")
    if not np.all(np.isfinite(clock)) or np.any(np.diff(clock) < 0):
        raise ValueError("clock must be finite and ordered")
    if len(trig) != len(secs) or thetas.shape != (len(trig), 1 + 2 * Z.shape[1]):
        raise ValueError("Every trigger needs one coefficient vector and duration")
    if true_theta.shape != (1 + 2 * Z.shape[1],):
        raise ValueError("true_theta has the wrong dimension")
    if (np.any(trig < 1) or np.any(trig > T) or np.any(np.diff(trig) <= 0)
            or np.any(secs < 0) or not np.all(np.isfinite(secs))):
        raise ValueError("Triggers must increase within the horizon; durations must be finite and nonnegative")
    if (len(explore_idx) != len(explore_price) or np.any(explore_idx < 0)
            or np.any(explore_idx >= T) or np.any(np.diff(explore_idx) <= 0)):
        raise ValueError("Exploration indices and prices must be paired, sorted and unique")
    if not (0 < price_lo < price_hi and price_unit > 0):
        raise ValueError("Invalid price interval or unit")
    is_explore = np.zeros(T, bool)
    is_explore[explore_idx] = True
    ex = np.flatnonzero(~is_explore)
    target = np.searchsorted(trig, ex + 1, side="right") - 1
    finish = fifo_finish(clock[trig - 1], secs)
    used = np.searchsorted(finish, clock[ex], side="right") - 1
    # When timestamps tie, a later row has not arrived at the current row yet.
    used = np.minimum(used, target)
    Zex = Z[ex]
    instant_p = _policy_prices(Zex, thetas, target, price_unit, price_lo, price_hi)
    delayed_p = _policy_prices(Zex, thetas, used, price_unit, price_lo, price_hi)
    A, B = index_AB(Z, true_theta)
    oracle_p = best_price(A, B, price_unit, price_lo, price_hi)
    revenues = {"oracle": oracle_p * prob(A, B, oracle_p, price_unit),
                "tsp": fare * prob(A, B, fare, price_unit)}
    explore_rev = explore_price * prob(A[explore_idx], B[explore_idx], explore_price, price_unit)
    for name, p in (("instant", instant_p), ("delayed", delayed_p)):
        r = np.empty(T)
        r[explore_idx] = explore_rev
        r[ex] = p * prob(A[ex], B[ex], p, price_unit)
        revenues[name] = r
    theta_gap = np.full(len(ex), np.nan)
    edges = np.r_[0, np.flatnonzero((np.diff(used) != 0) | (np.diff(target) != 0)) + 1, len(ex)]
    for a, b in zip(edges[:-1], edges[1:]):
        if a < b and used[a] >= 0 and target[a] >= 0:
            theta_gap[a:b] = np.linalg.norm(thetas[target[a]] - thetas[used[a]])
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    rows = []
    for window, lo, hi in (("first5", 0, max(1, int(np.ceil(.05 * T)))),
                           ("last20", int(np.floor(.8 * T)), T), ("all", 0, T)):
        mask = (ex >= lo) & (ex < hi)
        row = dict(window=window, start=lo + 1, end=hi, n_arrivals=hi - lo,
                   n_exploit=int(mask.sum()))
        row.update({"revenue_" + k: float(v[lo:hi].sum()) for k, v in revenues.items()})
        row["delay_loss"] = row["revenue_instant"] - row["revenue_delayed"]
        row["delay_loss_pct"] = 100 * row["delay_loss"] / row["revenue_instant"] if row["revenue_instant"] else 0.0
        for k in ("instant", "delayed", "tsp"):
            row["regret_" + k] = row["revenue_oracle"] - row["revenue_" + k]
        row.update(theta_gap_l2=mean(theta_gap[mask & np.isfinite(theta_gap)]),
                   mean_abs_price_gap=mean(np.abs(instant_p[mask] - delayed_p[mask])),
                   stale_share=mean((used != target)[mask]), fallback_share=mean((used < 0)[mask]))
        rows.append(row)
    # An update wholly superseded during exploration has no exploitation deadline.
    pos = np.searchsorted(ex, trig)
    first = np.full(len(trig), T, dtype=np.int64)
    valid = pos < len(ex)
    first[valid] = ex[pos[valid]]
    next_trigger = np.r_[trig[1:] - 1, T] if len(trig) else np.empty(0, dtype=np.int64)
    eligible = valid & (first < next_trigger)
    deadline = np.full(len(trig), np.nan)
    deadline[eligible] = clock[first[eligible]]
    timely = eligible & (finish <= deadline)
    updates = dict(finish=finish, deadline=deadline, eligible=eligible, timely=timely,
                   timely_rate=mean(timely[eligible]), n_updates=len(trig),
                   n_eligible=int(eligible.sum()), solve_total=float(secs.sum()),
                   solve_median=float(np.median(secs)) if len(secs) else 0.0,
                   solve_p95=float(np.percentile(secs, 95)) if len(secs) else 0.0)
    grid = np.unique(np.linspace(1, T, min(3000, T)).astype(np.int64))
    curve = {"t": grid}
    curve.update({"revenue_" + k: np.cumsum(v)[grid - 1] for k, v in revenues.items()})
    for k in ("instant", "delayed", "tsp"):
        curve["regret_" + k] = curve["revenue_oracle"] - curve["revenue_" + k]
    curve["delay_loss"] = curve["revenue_instant"] - curve["revenue_delayed"]
    return dict(summary=rows, curve=curve, updates=updates)


def self_check():
    assert schedule(0)[0].size == schedule(0)[1].size == 0
    e, tr = schedule(3)
    assert np.array_equal(e, np.arange(3)) and not len(tr)
    assert np.array_equal(fifo_finish([1, 2, 8], [3, 2, 1]), [4, 6, 9])
    Z = np.ones((10, 1)); clock = np.arange(10, dtype=float)
    th = np.array([[2., 0., -2.], [4., 0., -1.]])
    args = (Z, clock, np.full(10, 10.), th[1], [0, 3], [10., 10.], [1, 4], th)
    r = evaluate(*args, [0., 0.])
    assert all(x["delay_loss"] == 0 for x in r["summary"])
    assert r["updates"]["timely_rate"] == 1
    r = evaluate(*args, [3., 2.])
    assert np.array_equal(r["updates"]["finish"], [3., 5.])
    assert np.array_equal(r["updates"]["deadline"], [1., 4.])
    assert r["summary"][-1]["fallback_share"] == .25
    # Neither future update may affect the first five arrivals before it finishes.
    th_changed = th.copy(); th_changed[1] = [100., 10., -20.]
    r2 = evaluate(*args[:-1], th_changed, [3., 2.])
    assert np.array_equal(r["curve"]["revenue_delayed"][:5], r2["curve"]["revenue_delayed"][:5])
    e, tr = schedule(3)
    r = evaluate(np.ones((3, 1)), np.arange(3), np.ones(3) * 10, th[0], e,
                 np.ones(3) * 10, tr, np.empty((0, 3)), [])
    assert r["summary"][-1]["n_exploit"] == 0 and r["updates"]["n_eligible"] == 0
    print("replay checks passed")


if __name__ == "__main__":
    self_check()
