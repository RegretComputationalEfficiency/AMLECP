"""NYC taxi price-acceptance simulator (``beta_simulator``).

This package builds and serves a logistic *passenger-acceptance* model that acts
as the environment/simulator for the online-pricing experiments in this repo.

Public API (re-exported here via relative imports):

- :func:`load_beta_simulator` -- load trained intercept/coefficients.
- :func:`load_scaler_and_features` -- load the fitted scaler + scaled-feature list.
- :func:`simulate_passenger_acceptance` -- sample an accept/reject decision.

Typical usage::

    from Simulator import (
        load_beta_simulator,
        load_scaler_and_features,
        simulate_passenger_acceptance,
    )

    beta = load_beta_simulator(scenario_name="pair_1p5_2p0__noise_05",
                               base_dir="Simulator/beta_sensitivity_outputs")
    scaler, scaled = load_scaler_and_features(
        scenario_name="pair_1p5_2p0__noise_05",
        base_dir="Simulator/beta_sensitivity_outputs",
    )
    accepted = simulate_passenger_acceptance(20.0, trip_features, beta, scaler, scaled)
"""

from .simulator_response import (
    load_beta_simulator,
    load_scaler_and_features,
    simulate_passenger_acceptance,
)

__all__ = [
    "load_beta_simulator",
    "load_scaler_and_features",
    "simulate_passenger_acceptance",
]
