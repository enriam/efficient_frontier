from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=False)
class Asset:
    ret: float  # mean
    volat: float  # stdev


def calculate_markowitz_weights_three_assets(
    assets: list[Asset],
    correlations: list[float],
    max_portfolio_volat: float,
) -> tuple[list[float], float]:
    """
    Calculate the optimal long-only Markowitz portfolio for three assets.

    The function maximizes expected portfolio return subject to:
    - weights summing to 1
    - weights between 0 and 1
    - portfolio volatility <= max_portfolio_volat

    Parameters
    ----------
    assets:
        List with 3 objects Asset: [asset1, asset2, asset3].
    correlations:
        List with correlations: [corr12, corr13, corr23].
    max_portfolio_volat:
        Maximum portfolio volatility accepted by the investor.

    Returns
    -------
    tuple[list[float], float]
        Optimal weights [w1, w2, w3] and expected portfolio return.
    """

    num_assets = 3
    if len(assets) != num_assets:
        msg = "assets must contain exactly 3 Asset objects."
        raise ValueError(msg)

    if len(correlations) != num_assets:
        msg = "correlations must contain exactly 3 values."
        raise ValueError(msg)

    if max_portfolio_volat < 0:
        msg = "max_portfolio_volat must be non-negative."
        raise ValueError(msg)

    returns = np.array([asset.ret for asset in assets], dtype=float)
    volats = np.array([asset.volat for asset in assets], dtype=float)

    if np.any(volats < 0):
        msg = "Asset volatilities must be non-negative."
        raise ValueError(msg)

    if any(corr < -1 or corr > 1 for corr in correlations):
        msg = "Correlations must be between -1 and 1."
        raise ValueError(msg)

    corr12, corr13, corr23 = correlations
    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    epsilon = -1e-10
    if np.any(np.linalg.eigvalsh(correlation_matrix) < epsilon):
        msg = "The correlation matrix is not valid."
        raise ValueError(msg)

    covariance_matrix = np.outer(volats, volats) * correlation_matrix

    def portfolio_return(weights: np.ndarray) -> float:
        return float(weights @ returns)

    def portfolio_volat(weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ covariance_matrix @ weights))

    def negative_portfolio_return(weights: np.ndarray) -> float:
        return -portfolio_return(weights)

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                max_portfolio_volat - portfolio_volat(weights)
            ),
        },
    ]

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    initial_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    result = minimize(
        negative_portfolio_return,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        msg = f"Optimization failed: {result.message}"
        raise ValueError(msg)

    optimal_weights = result.x
    optimal_return = portfolio_return(optimal_weights)

    return optimal_weights.tolist(), optimal_return
