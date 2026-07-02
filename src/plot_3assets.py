# ruff: noqa: E501 TRY003 EM101 FBT003

from collections.abc import Sequence
from typing import Any

# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import minimize

from src.asset import Asset


def plot_three_asset_frontier(
    assets: Sequence[Asset],
    corr12: float,
    corr13: float,
    corr23: float,
    ax: Axes,
    **kwargs: Any,
) -> tuple[Axes, dict[str, Any]]:
    """
    Plot the mean-variance frontier for a long-only portfolio with three financial assets.

    The function draws:
    - The efficient part of the frontier.
    - The inefficient part of the minimum-variance frontier, using a different color and style.
    - A cloud of random feasible portfolios, if requested.
    - The minimum-risk portfolio.
    - The three portfolios with 100% allocation to each asset, if requested.

    Parameters
    ----------
    assets:
        Sequence of three Asset objects. Each object must have:
        - name: asset name.
        - avg: expected return or historical mean return.
        - std: standard deviation.

    corr12:
        Correlation between asset 1 and asset 2.

    corr13:
        Correlation between asset 1 and asset 3.

    corr23:
        Correlation between asset 2 and asset 3.

    **kwargs:
        Optional plotting and calculation parameters.

        Calculation options:
        - n_portfolios: number of random portfolios to generate. Default is 5000.
        - n_frontier_points: number of target-return points used for the frontier. Default is 150.
        - random_seed: random seed for reproducibility. Default is None.

        Visibility options:
        - show_random_portfolios: whether to show the random portfolio cloud. Default is True.
        - show_asset_points: whether to show the 100% asset portfolios. Default is True.

        Figure options:
        - figsize: matplotlib figure size. Default is (10, 6).
        - title: chart title. Default is "Three-Asset Mean-Variance Frontier".
        - x_label: x-axis label. Default is "Risk (standard deviation)".
        - y_label: y-axis label. Default is "Expected return".

        Labels:
        - efficient_label: label for the efficient frontier.
        - inefficient_label: label for the inefficient frontier.
        - random_label: label for the random portfolio cloud.
        - min_risk_label: label for the minimum-risk portfolio.
        - asset_label_prefix: prefix for the 100% asset portfolio labels.

        Colors and styles:
        - efficient_color
        - inefficient_color
        - random_color
        - min_risk_color
        - asset_color
        - efficient_linestyle
        - inefficient_linestyle
        - efficient_linewidth
        - inefficient_linewidth
        - random_alpha
        - random_size
        - point_size

    Returns
    -------
    tuple[Figure, Axes, dict[str, Any]]
        A tuple containing:
        - fig: the matplotlib Figure object.
        - ax: the matplotlib Axes object.
        - data: dictionary with calculated portfolio data, including:
            - covariance_matrix
            - random_weights
            - random_returns
            - random_risks
            - frontier_weights
            - frontier_returns
            - frontier_risks
            - min_risk_weights
            - min_risk_return
            - min_risk_risk
    """

    if len(assets) != 3:  # noqa: PLR2004
        raise ValueError("Exactly three assets are required.")

    for asset in assets:
        if asset.std < 0:
            raise ValueError("Asset standard deviations must be non-negative.")

    if not -1 <= corr12 <= 1:
        raise ValueError("corr12 must be between -1 and 1.")

    if not -1 <= corr13 <= 1:
        raise ValueError("corr13 must be between -1 and 1.")

    if not -1 <= corr23 <= 1:
        raise ValueError("corr23 must be between -1 and 1.")

    n_portfolios = kwargs.get("n_portfolios", 5000)
    n_frontier_points = kwargs.get("n_frontier_points", 150)
    random_seed = kwargs.get("random_seed")

    show_random_portfolios = kwargs.get("show_random_portfolios", True)
    show_asset_points = kwargs.get("show_asset_points", True)

    # figsize = kwargs.get("figsize", (10, 6))
    title = kwargs.get("title", "Three-Asset Mean-Variance Frontier")
    x_label = kwargs.get("x_label", "Risk (standard deviation)")
    y_label = kwargs.get("y_label", "Expected return")

    efficient_label = kwargs.get("efficient_label", "Efficient frontier")
    inefficient_label = kwargs.get("inefficient_label", "Inefficient frontier")
    random_label = kwargs.get("random_label", "Random portfolios")
    min_risk_label = kwargs.get("min_risk_label", "Minimum-risk portfolio")
    asset_label_prefix = kwargs.get("asset_label_prefix", "100%")

    efficient_color = kwargs.get("efficient_color", "black")
    inefficient_color = kwargs.get("inefficient_color", "gray")
    random_color = kwargs.get("random_color", "lightgray")
    min_risk_color = kwargs.get("min_risk_color", "red")
    asset_color = kwargs.get("asset_color", "blue")

    efficient_linestyle = kwargs.get("efficient_linestyle", "-")
    inefficient_linestyle = kwargs.get("inefficient_linestyle", "--")
    efficient_linewidth = kwargs.get("efficient_linewidth", 2.5)
    inefficient_linewidth = kwargs.get("inefficient_linewidth", 2.0)

    random_alpha = kwargs.get("random_alpha", 0.35)
    random_size = kwargs.get("random_size", 12)
    point_size = kwargs.get("point_size", 70)

    means = np.array([asset.avg for asset in assets])
    stds = np.array([asset.std for asset in assets])

    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    covariance_matrix = np.outer(stds, stds) * correlation_matrix

    eigenvalues = np.linalg.eigvalsh(covariance_matrix)

    epsilon = -1e-10
    if np.any(eigenvalues < epsilon):
        raise ValueError("The covariance matrix is not positive semidefinite.")

    rng = np.random.default_rng(random_seed)

    random_weights = rng.dirichlet(alpha=np.ones(3), size=n_portfolios)
    random_returns = random_weights @ means
    random_variances = np.einsum(
        "ij,jk,ik->i", random_weights, covariance_matrix, random_weights
    )
    random_risks = np.sqrt(np.maximum(random_variances, 0))

    def portfolio_return(weights):
        return weights @ means

    def portfolio_variance(weights):
        return weights @ covariance_matrix @ weights

    def portfolio_risk(weights):
        return np.sqrt(max(portfolio_variance(weights), 0))

    bounds = [(0, 1), (0, 1), (0, 1)]
    initial_weights = np.array([1 / 3, 1 / 3, 1 / 3])

    sum_constraint = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    min_risk_result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
    )

    if not min_risk_result.success:
        raise RuntimeError("Minimum-risk portfolio optimization failed.")

    min_risk_weights = min_risk_result.x
    min_risk_return = portfolio_return(min_risk_weights)
    min_risk_risk = portfolio_risk(min_risk_weights)

    target_returns = np.linspace(
        np.min(means), np.max(means), n_frontier_points
    )

    frontier_weights = []
    frontier_returns = []
    frontier_risks = []

    for target_return in target_returns:
        return_constraint = {
            "type": "eq",
            "fun": lambda weights, target_return=target_return: (
                portfolio_return(weights) - target_return
            ),
        }

        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_constraint, return_constraint],
        )

        if result.success:
            weights = result.x
            frontier_weights.append(weights)
            frontier_returns.append(portfolio_return(weights))
            frontier_risks.append(portfolio_risk(weights))

    frontier_weights = np.array(frontier_weights)
    frontier_returns = np.array(frontier_returns)
    frontier_risks = np.array(frontier_risks)

    efficient_mask = frontier_returns >= min_risk_return
    inefficient_mask = frontier_returns < min_risk_return

    # fig, ax = plt.subplots(figsize=figsize)

    if show_random_portfolios:
        ax.scatter(
            random_risks,
            random_returns,
            s=random_size,
            alpha=random_alpha,
            color=random_color,
            label=random_label,
        )

    ax.plot(
        frontier_risks[inefficient_mask],
        frontier_returns[inefficient_mask],
        color=inefficient_color,
        linestyle=inefficient_linestyle,
        linewidth=inefficient_linewidth,
        label=inefficient_label,
    )

    ax.plot(
        frontier_risks[efficient_mask],
        frontier_returns[efficient_mask],
        color=efficient_color,
        linestyle=efficient_linestyle,
        linewidth=efficient_linewidth,
        label=efficient_label,
    )

    ax.scatter(
        min_risk_risk,
        min_risk_return,
        s=point_size,
        color=min_risk_color,
        label=min_risk_label,
        zorder=5,
    )

    ax.annotate(
        min_risk_label,
        xy=(min_risk_risk, min_risk_return),
        xytext=(8, 8),
        textcoords="offset points",
    )

    if show_asset_points:
        for asset_index, asset in enumerate(assets):
            asset_weights = np.zeros(3)
            asset_weights[asset_index] = 1

            asset_return = portfolio_return(asset_weights)
            asset_risk = portfolio_risk(asset_weights)
            asset_label = f"{asset_label_prefix} {asset.name}"

            ax.scatter(
                asset_risk,
                asset_return,
                s=point_size,
                color=asset_color,
                zorder=5,
            )

            ax.annotate(
                asset_label,
                xy=(asset_risk, asset_return),
                xytext=(8, 8),
                textcoords="offset points",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # data = {
    #     "covariance_matrix": covariance_matrix,
    #     "random_weights": random_weights,
    #     "random_returns": random_returns,
    #     "random_risks": random_risks,
    #     "frontier_weights": frontier_weights,
    #     "frontier_returns": frontier_returns,
    #     "frontier_risks": frontier_risks,
    #     "min_risk_weights": min_risk_weights,
    #     "min_risk_return": min_risk_return,
    #     "min_risk_risk": min_risk_risk,
    # }

    return ax


def calc_pf3a_metrics(
    assets: Sequence[Asset],
    weights: Sequence[float],
    correlations: Sequence[float],
) -> tuple[float, float]:
    """
    Calculate the expected return and volatility of a three-asset portfolio.

    Parameters
    ----------
    assets:
        Sequence of three Asset objects. Each asset must have:
        - name: asset name.
        - avg: expected return or historical mean return.
        - std: standard deviation.

    weights:
        Sequence of three portfolio weights. The weights must sum to 1.

    corr12:
        Correlation between asset 1 and asset 2.

    corr13:
        Correlation between asset 1 and asset 3.

    corr23:
        Correlation between asset 2 and asset 3.

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - portfolio_return: weighted average return of the portfolio.
        - portfolio_volatility: standard deviation of the portfolio.
    """
    num_assets = 3
    if len(assets) != num_assets:
        raise ValueError("Exactly three assets are required.")

    if len(weights) != num_assets:
        raise ValueError("Exactly three weights are required.")

    if not np.isclose(sum(weights), 1):
        raise ValueError("Portfolio weights must sum to 1.")

    corr12 = correlations[0]
    corr13 = correlations[1]
    corr23 = correlations[2]

    if not -1 <= corr12 <= 1:
        raise ValueError("corr12 must be between -1 and 1.")

    if not -1 <= corr13 <= 1:
        raise ValueError("corr13 must be between -1 and 1.")

    if not -1 <= corr23 <= 1:
        raise ValueError("corr23 must be between -1 and 1.")

    means = np.array([asset.avg for asset in assets])
    stds = np.array([asset.std for asset in assets])
    weights = np.array(weights)

    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    covariance_matrix = np.outer(stds, stds) * correlation_matrix

    portfolio_return = weights @ means
    portfolio_variance = weights @ covariance_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_return, portfolio_volatility


def validate_three_asset_correlations(
    corr12: float,
    corr13: float,
    corr23: float,
) -> None:
    """
    Validate whether three pairwise correlations define a valid
    three-asset correlation matrix.

    Parameters
    ----------
    corr12:
        Correlation between asset 1 and asset 2.

    corr13:
        Correlation between asset 1 and asset 3.

    corr23:
        Correlation between asset 2 and asset 3.

    Returns
    -------
    None
        Raises ValueError if the correlations are not valid.
    """

    correlations = [corr12, corr13, corr23]

    if any(
        correlation < -1 or correlation > 1 for correlation in correlations
    ):
        raise ValueError("Each correlation must be between -1 and 1.")

    determinant = (
        1 + 2 * corr12 * corr13 * corr23 - corr12**2 - corr13**2 - corr23**2
    )
    epsilon = -1e-10
    if determinant < epsilon:
        raise ValueError(
            "The three correlations do not define a valid correlation matrix."
        )


def validate_correlation_matrix(
    correlation_matrix: Sequence[Sequence[float]],
    tolerance: float = 1e-10,
) -> None:
    """
    Validate whether a correlation matrix is valid for any number of assets.

    A valid correlation matrix must be:
    - Square.
    - Symmetric.
    - Equal to 1 on the diagonal.
    - Between -1 and 1 in all entries.
    - Positive semidefinite.

    Parameters
    ----------
    correlation_matrix:
        Square correlation matrix. Its shape must be (n_assets, n_assets).

    tolerance:
        Numerical tolerance used when checking symmetry, diagonal values,
        and positive semidefiniteness. Default is 1e-10.

    Returns
    -------
    None
        Raises ValueError if the matrix is not a valid correlation matrix.
    """

    matrix = np.array(correlation_matrix, dtype=float)

    min_dim = 2
    if matrix.ndim != min_dim:
        raise ValueError("The correlation matrix must be two-dimensional.")

    n_rows, n_columns = matrix.shape

    if n_rows != n_columns:
        raise ValueError("The correlation matrix must be square.")

    if not np.allclose(matrix, matrix.T, atol=tolerance):
        raise ValueError("The correlation matrix must be symmetric.")

    if not np.allclose(np.diag(matrix), np.ones(n_rows), atol=tolerance):
        raise ValueError(
            "The diagonal values of the correlation matrix must be 1."
        )

    if np.any(matrix < -1 - tolerance) or np.any(matrix > 1 + tolerance):
        raise ValueError("All correlations must be between -1 and 1.")

    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            "The correlation matrix must be positive semidefinite."
        )
