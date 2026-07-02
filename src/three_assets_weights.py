# ruff: noqa: E501

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from src.asset import Asset


def weights_3a_with_max_volatility(
    assets: Sequence[Asset],
    correlations: Sequence[float],
    max_volatility: float,
) -> dict[str, float | np.ndarray]:
    """
    Calculate the optimal long-only weights for a three-asset portfolio subject
    to a maximum portfolio volatility constraint.

    The function maximizes the expected portfolio return under these constraints:
    - The portfolio has exactly three assets.
    - The weights must sum to 1.
    - Each weight must be between 0 and 1.
    - The portfolio volatility must be less than or equal to max_volatility.

    Parameters
    ----------
    assets:
        Sequence of three Asset objects. Each asset must have:
        - name: asset name.
        - avg: expected return or historical mean return.
        - std: standard deviation.

    corr12:
        Correlation between asset 1 and asset 2.

    corr13:
        Correlation between asset 1 and asset 3.

    corr23:
        Correlation between asset 2 and asset 3.

    max_volatility:
        Maximum allowed portfolio volatility.

    Returns
    -------
    dict[str, float | np.ndarray]
        Dictionary containing:
        - weights: optimal portfolio weights.
        - portfolio_return: expected return of the optimal portfolio.
        - portfolio_volatility: volatility of the optimal portfolio.
    """

    if len(assets) != 3:  # noqa: PLR2004
        msg = "Exactly three assets are required."
        raise ValueError(msg)

    if max_volatility < 0:
        msg = "Maximum volatility must be non-negative."
        raise ValueError(msg)

    corr12 = correlations[0]
    corr13 = correlations[1]
    corr23 = correlations[2]

    if not -1 <= corr12 <= 1:
        msg = "corr12 must be between -1 and 1."
        raise ValueError(msg)

    if not -1 <= corr13 <= 1:
        msg = "corr13 must be between -1 and 1."
        raise ValueError(msg)

    if not -1 <= corr23 <= 1:
        msg = "corr23 must be between -1 and 1."
        raise ValueError(msg)

    means = np.array([asset.avg for asset in assets])
    stds = np.array([asset.std for asset in assets])

    if np.any(stds < 0):
        msg = "Asset standard deviations must be non-negative."
        raise ValueError(msg)

    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    epsilon = 1e-10
    if np.any(eigenvalues < -epsilon):
        msg = "The correlations do not define a valid correlation matrix."
        raise ValueError(msg)

    covariance_matrix = np.outer(stds, stds) * correlation_matrix

    eigenvalues = np.linalg.eigvalsh(covariance_matrix)

    if np.any(eigenvalues < -epsilon):
        msg = "The covariance matrix is not positive semidefinite."
        raise ValueError(msg)

    def portfolio_return(weights):
        return weights @ means

    def portfolio_variance(weights):
        return weights @ covariance_matrix @ weights

    def portfolio_volatility(weights):
        return np.sqrt(max(portfolio_variance(weights), 0))

    def negative_portfolio_return(weights):
        return -portfolio_return(weights)

    bounds = [(0, 1), (0, 1), (0, 1)]

    sum_constraint = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    volatility_constraint = {
        "type": "ineq",
        "fun": lambda weights: max_volatility - portfolio_volatility(weights),
    }

    initial_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    min_volatility_result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
    )

    if not min_volatility_result.success:
        msg = "Minimum-volatility portfolio optimization failed."
        raise RuntimeError(msg)

    min_volatility = portfolio_volatility(min_volatility_result.x)

    if max_volatility < min_volatility - 1e-10:
        msg = "No feasible portfolio exists for the given maximum volatility."
        raise ValueError(msg)

    result = minimize(
        negative_portfolio_return,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint, volatility_constraint],
    )

    if not result.success:
        msg = "Optimal portfolio optimization failed."
        raise RuntimeError(msg)

    weights = result.x
    portfolio_return_value = portfolio_return(weights)
    portfolio_volatility_value = portfolio_volatility(weights)

    return {
        "weights": weights,
        "portfolio_return": portfolio_return_value,
        "portfolio_volatility": portfolio_volatility_value,
    }


def global_min_volatility_3a(
    stds: Sequence[float],
    correlations: Sequence[float],
) -> dict[str, float | np.ndarray]:
    """
    Calculate the minimum possible volatility of a long-only three-asset portfolio.

    The function minimizes portfolio variance subject to:
    - The portfolio has exactly three assets.
    - The weights must sum to 1.
    - Each weight must be between 0 and 1.

    Parameters
    ----------
    stds:
        Sequence with the standard deviations of the three assets.

    corr12:
        Correlation between asset 1 and asset 2.

    corr13:
        Correlation between asset 1 and asset 3.

    corr23:
        Correlation between asset 2 and asset 3.

    Returns
    -------
    dict[str, float | np.ndarray]
        Dictionary containing:
        - weights: weights of the minimum-volatility portfolio.
        - volatility: minimum portfolio volatility.
        - variance: minimum portfolio variance.
        - covariance_matrix: covariance matrix used in the calculation.
    """

    num_assets = 3
    if len(stds) != num_assets:
        msg = "Exactly three standard deviations are required."
        raise ValueError(msg)

    stds = np.array(stds, dtype=float)

    if np.any(stds < 0):
        msg = "Standard deviations must be non-negative."
        raise ValueError(msg)

    corr12 = correlations[0]
    corr13 = correlations[1]
    corr23 = correlations[2]

    if not -1 <= corr12 <= 1:
        msg = "corr12 must be between -1 and 1."
        raise ValueError(msg)

    if not -1 <= corr13 <= 1:
        msg = "corr13 must be between -1 and 1."
        raise ValueError(msg)

    if not -1 <= corr23 <= 1:
        msg = "corr23 must be between -1 and 1."
        raise ValueError(msg)

    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    eigenvalues = np.linalg.eigvalsh(correlation_matrix)

    epsilon = 1e-10
    if np.any(eigenvalues < -epsilon):
        msg = "The correlations do not define a valid correlation matrix."
        raise ValueError(msg)

    covariance_matrix = np.outer(stds, stds) * correlation_matrix

    def portfolio_variance(weights):
        return weights @ covariance_matrix @ weights

    bounds = [(0, 1), (0, 1), (0, 1)]

    sum_constraint = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    initial_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
    )

    if not result.success:
        msg = "Minimum-volatility portfolio optimization failed."
        raise RuntimeError(msg)

    weights = result.x
    variance = portfolio_variance(weights)
    # volatility = np.sqrt(max(variance, 0))

    return np.sqrt(max(variance, 0))
