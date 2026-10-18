"""Moment-diagnostic sanity check, not a neural-network mechanism classifier.

Run with Python 3.12 + NumPy. Outputs simulations only; no GPU is used.
The noise drift budget is an assumed bound, not estimated from these responses.
Intervals are pointwise percentile bootstrap intervals, not a theorem or power study.
"""

from __future__ import annotations


import numpy as np


DELTAS = np.array([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
CAL = (np.abs(DELTAS) > 0) & (np.abs(DELTAS) <= 0.5)
HOLD = np.abs(DELTAS) >= 0.75
ZERO = int(np.flatnonzero(DELTAS == 0)[0])
CASES = (
    "fixed_pool", "random_pool", "selection", "near_one_hot_pool",
    "heteroscedastic_pool", "routing_drift", "nonlinear_readout",
    "context_contaminated_selection",
)


def simulate(case: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Columns share nuisance seeds; rows are independent scene families."""
    p, sigma = 0.35, 0.12
    # Include both paired nuisance and condition-specific additive noise so
    # uncertainty does not disappear through perfect cancellation across delta.
    noise = (0.5 * rng.normal(size=(n, 1))
             + np.sqrt(0.75) * rng.normal(size=(n, len(DELTAS))))
    weight = np.full((n, 1), p)
    if case == "selection":
        weight = (rng.random((n, 1)) < p).astype(float)
    elif case in ("random_pool", "near_one_hot_pool"):
        concentration = 4.0 if case == "random_pool" else 0.01
        weight = rng.beta(p * concentration, (1-p) * concentration, (n, 1))
    elif case == "routing_drift":
        weight = np.broadcast_to(p + 0.2 * DELTAS**2, (n, len(DELTAS)))

    signal = weight * DELTAS
    if case == "nonlinear_readout":
        signal = np.tanh(2 * signal) / 2
    if case == "heteroscedastic_pool":
        # Same theoretical mean and variance as hard selection at every delta.
        sd = np.sqrt(sigma**2 + p*(1-p)*DELTAS**2)
    else:
        sd = sigma
    # For context_contaminated_selection, both selected object slots contain
    # the same contextual average p*delta. The output is exactly fixed_pool.
    return signal + noise * sd


def statistics(y: np.ndarray) -> tuple[float, float, float]:
    means = y.mean(axis=0)
    p_hat = float(DELTAS[CAL] @ means[CAL] / (DELTAS[CAL] @ DELTAS[CAL]))
    variances = y.var(axis=0, ddof=1)
    gap = float(np.mean(
        p_hat*(1-p_hat)*DELTAS[HOLD]**2
        - (variances[HOLD] - variances[ZERO])
    ))
    # Signed mean drift along delta, evaluated on held-out response means.
    drift = float(np.mean((means[HOLD] - p_hat*DELTAS[HOLD]) / DELTAS[HOLD]))
    return p_hat, gap, drift


def diagnose(y: np.ndarray, rng: np.random.Generator, boots: int,
             noise_budget: float, mean_tolerance: float) -> dict:
    point = statistics(y)
    samples = np.array([
        statistics(y[rng.integers(0, len(y), size=len(y))])
        for _ in range(boots)
    ])
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    if low[2] > mean_tolerance or high[2] < -mean_tolerance:
        decision = "shared_mean_model_fails"
    elif not 0 <= point[0] <= 1:
        decision = "shared_mean_model_fails"
    elif low[2] < -mean_tolerance or high[2] > mean_tolerance:
        # Failure to detect a departure is not evidence of equivalence.
        decision = "shared_mean_model_unverified"
    elif low[1] > noise_budget:
        decision = "reject_restricted_selection_if_assumptions_hold"
    else:
        decision = "inconclusive"
    return {
        "p_hat": point[0], "gap": point[1],
        "gap_ci_low": float(low[1]), "gap_ci_high": float(high[1]),
        "heldout_mean_drift": point[2],
        "drift_ci_low": float(low[2]), "drift_ci_high": float(high[2]),
        "decision": decision,
    }

