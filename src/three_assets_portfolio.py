from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize

from src.asset import Asset


def plot_3a_frontier(  # noqa: PLR0913
    asset_1: Asset,
    asset_2: Asset,
    asset_3: Asset,
    correlation_12: float,
    correlation_13: float,
    correlation_23: float,
    ax: Axes | None = None,
    num_portfolios: int = 10_000,
    **kwargs: Any,
) -> Axes:
    # asset1_label = kwargs.get("asset1_label", "Asset 1")
    # asset2_label = kwargs.get("asset2_label", "Asset 2")
    # asset3_label = kwargs.get("asset3_label", "Asset 3")
    efficient_label = kwargs.get("efficient_label", "Efficient frontier")
    inefficient_label = kwargs.get("inefficient_label", "Efficient frontier")
    min_variance_label = kwargs.get(
        "min_variance_label",
        "Minimum variance portfolio",
    )

    x_axis_label = kwargs.get("x_axis_label", "Risk, standard deviation")
    y_axis_label = kwargs.get("y_axis_label", "Expected return")
    plot_title = kwargs.get(
        "plot_title",
        "Three-Asset Portfolio Efficient Frontier",
    )

    # asset1_color = kwargs.get("asset1_color", "red")
    # asset2_color = kwargs.get("asset2_color", "orange")
    # asset3_color = kwargs.get("asset3_color", "green")
    frontier_color = kwargs.get("frontier_color", "steelblue")
    min_variance_color = kwargs.get("min_variance_color", "blue")
    portfolio_cloud_color = kwargs.get("portfolio_cloud_color", "lightgray")

    random_seed = kwargs.get("random_seed", 42)
    show_portfolio_cloud = kwargs.get("show_portfolio_cloud", True)
    format_as_percent = kwargs.get("format_as_percent", True)

    correlations = [correlation_12, correlation_13, correlation_23]

    if any(
        correlation < -1 or correlation > 1 for correlation in correlations
    ):
        msg = "Correlations must be between -1 and 1."
        raise ValueError(msg)

    if asset_1.volat < 0 or asset_2.volat < 0 or asset_3.volat < 0:
        msg_0 = "Asset standard deviations must be non-negative."
        raise ValueError(msg_0)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    means = np.array(
        [
            asset_1.ret,
            asset_2.ret,
            asset_3.ret,
        ]
    )

    stds = np.array(
        [
            asset_1.volat,
            asset_2.volat,
            asset_3.volat,
        ]
    )

    correlation_matrix = np.array(
        [
            [1.0, correlation_12, correlation_13],
            [correlation_12, 1.0, correlation_23],
            [correlation_13, correlation_23, 1.0],
        ]
    )

    epsilon = -1e-10
    if np.any(np.linalg.eigvalsh(correlation_matrix) < epsilon):
        msg_1 = "The correlation matrix is not valid."
        raise ValueError(msg_1)

    covariance_matrix = np.outer(stds, stds) * correlation_matrix

    rng = np.random.default_rng(random_seed)
    weights = rng.dirichlet(np.ones(3), size=num_portfolios)

    portfolio_returns = weights @ means
    portfolio_variances = np.einsum(
        "ij,jk,ik->i",
        weights,
        covariance_matrix,
        weights,
    )
    portfolio_risks = np.sqrt(portfolio_variances)

    sorted_indices = np.argsort(portfolio_risks)
    sorted_risks = portfolio_risks[sorted_indices]
    sorted_returns = portfolio_returns[sorted_indices]

    cumulative_max_returns = np.maximum.accumulate(sorted_returns)
    efficient_mask = sorted_returns >= cumulative_max_returns

    efficient_risks = sorted_risks[efficient_mask]
    efficient_returns = sorted_returns[efficient_mask]

    min_variance_index = np.argmin(portfolio_risks)

    if show_portfolio_cloud:
        ax.scatter(
            portfolio_risks,
            portfolio_returns,
            color=portfolio_cloud_color,
            alpha=0.25,
            s=10,
            label=inefficient_label,
        )

    ax.plot(
        efficient_risks,
        efficient_returns,
        color=frontier_color,
        linewidth=2,
        label=efficient_label,
    )

    # ax.scatter(
    #     asset_1.volat,
    #     asset_1.ret,
    #     color=asset1_color,
    #     label=asset1_label,
    #     zorder=3,
    # )

    # ax.scatter(
    #     asset_2.volat,
    #     asset_2.ret,
    #     color=asset2_color,
    #     label=asset2_label,
    #     zorder=3,
    # )

    # ax.scatter(
    #     asset_3.volat,
    #     asset_3.ret,
    #     color=asset3_color,
    #     label=asset3_label,
    #     zorder=3,
    # )

    ax.scatter(
        portfolio_risks[min_variance_index],
        portfolio_returns[min_variance_index],
        color=min_variance_color,
        label=min_variance_label,
        zorder=4,
    )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(plot_title)

    if format_as_percent:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    return ax


def weights_3a_pf_long_only(
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

    num_correlations = 3
    if len(correlations) != num_correlations:
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


# Ejemplo uso
# assets = [
#     Asset(ret=0.08, volat=0.15),
#     Asset(ret=0.12, volat=0.22),
#     Asset(ret=0.06, volat=0.10),
# ]

# correlations = [0.25, 0.10, 0.40]

# weights, expected_return = calculate_markowitz_weights_three_assets(
#     assets=assets,
#     correlations=correlations,
#     max_portfolio_volat=0.16,
# )
# ---


def weights_3a_pf_bounds(
    assets: list[Asset],
    correlations: list[float],
    max_portfolio_volat: float,
    **kwargs: Any,
) -> tuple[list[float], float]:
    """
    Calculate the optimal long-only Markowitz portfolio for three assets.

    Optional kwargs
    ---------------
    min_weights:
        Minimum allowed weights [min_w1, min_w2, min_w3].
        Defaults to [0.0, 0.0, 0.0].
    """

    num_assets = 3
    num_correlations = 3

    min_weights = kwargs.get("min_weights", [0.0, 0.0, 0.0])

    if len(assets) != num_assets:
        msg = "assets must contain exactly 3 Asset objects."
        raise ValueError(msg)

    if len(correlations) != num_correlations:
        msg = "correlations must contain exactly 3 values."
        raise ValueError(msg)

    if len(min_weights) != num_assets:
        msg = "min_weights must contain exactly 3 values."
        raise ValueError(msg)

    if max_portfolio_volat < 0:
        msg = "max_portfolio_volat must be non-negative."
        raise ValueError(msg)

    if any(min_weight < 0 for min_weight in min_weights):
        msg = "Minimum weights must be non-negative."
        raise ValueError(msg)

    if sum(min_weights) > 1.0:
        msg = "The sum of minimum weights cannot be greater than 1."
        raise ValueError(msg)

    returns = np.array([asset.ret for asset in assets], dtype=float)
    volats = np.array([asset.volat for asset in assets], dtype=float)
    min_weights_array = np.array(min_weights, dtype=float)

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

    epsilon = 1e-10
    if np.any(np.linalg.eigvalsh(correlation_matrix) < -epsilon):
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

    bounds = [
        (float(min_weights_array[0]), 1.0),
        (float(min_weights_array[1]), 1.0),
        (float(min_weights_array[2]), 1.0),
    ]

    initial_weights = min_weights_array + (
        (1.0 - np.sum(min_weights_array)) / num_assets
    )

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


def risk_return_3a_pf(
    assets: list[Asset],
    weights: list[float],
    correlations: list[float],
) -> tuple[float, float]:
    """
    Calculate portfolio risk and return for a three-asset portfolio.

    Parameters
    ----------
    assets:
        List with 3 Asset objects: [asset_1, asset_2, asset_3].
    weights:
        List with portfolio weights: [w1, w2, w3].
    correlations:
        List with correlations: [corr12, corr13, corr23].

    Returns
    -------
    tuple[float, float]
        portfolio_risk, portfolio_return
    """

    num_assets = 3
    num_correlations = 3

    if len(assets) != num_assets:
        msg = "assets must contain exactly 3 Asset objects."
        raise ValueError(msg)

    if len(weights) != num_assets:
        msg = "weights must contain exactly 3 values."
        raise ValueError(msg)

    if len(correlations) != num_correlations:
        msg = "correlations must contain exactly 3 values."
        raise ValueError(msg)

    if any(asset.volat < 0 for asset in assets):
        msg = "Asset standard deviations must be non-negative."
        raise ValueError(msg)

    if any(corr < -1 or corr > 1 for corr in correlations):
        msg = "Correlations must be between -1 and 1."
        raise ValueError(msg)

    if not np.isclose(sum(weights), 1.0):
        msg = "Portfolio weights must sum to 1."
        raise ValueError(msg)

    corr12, corr13, corr23 = correlations

    returns = np.array([asset.ret for asset in assets], dtype=float)
    volats = np.array([asset.volat for asset in assets], dtype=float)
    weights_array = np.array(weights, dtype=float)

    correlation_matrix = np.array(
        [
            [1.0, corr12, corr13],
            [corr12, 1.0, corr23],
            [corr13, corr23, 1.0],
        ]
    )

    epsilon = 1e-10
    if np.any(np.linalg.eigvalsh(correlation_matrix) < -epsilon):
        msg = "The correlation matrix is not valid."
        raise ValueError(msg)

    covariance_matrix = np.outer(volats, volats) * correlation_matrix

    pf_return = float(weights_array @ returns)
    pf_variance = float(weights_array @ covariance_matrix @ weights_array)
    pf_risk = float(np.sqrt(pf_variance))

    return pf_risk, pf_return


def scatter_3a_portfolio(
    pf_risk: float,
    pf_return: float,
    ax: Axes,
    **kwargs: Any,
) -> Axes:
    """Plot the risk-return point in the efficient frontier"""

    # Get kwargs
    point_label = kwargs.get("point_label", "Selected portfolio")
    point_color = kwargs.get("point_color", "darkorange")
    point_marker = kwargs.get("point_marker", "D")
    line_color = kwargs.get("line_color", point_color)

    # Check input values
    if pf_risk < 0.0:
        msg = "Portfolio risk must be non-negative"
        raise ValueError(msg)

    ax.scatter(
        pf_risk,
        pf_return,
        color=point_color,
        marker=point_marker,
        label=point_label,
        zorder=5,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.vlines(
        x=pf_risk,
        ymin=y_min,
        ymax=pf_return,
        color=line_color,
        linestyle=":",
        linewidth=1,
        alpha=1.0,
    )

    ax.hlines(
        y=pf_return,
        xmin=x_min,
        xmax=pf_risk,
        color=line_color,
        linestyle=":",
        linewidth=1,
        alpha=1.0,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()

    return
