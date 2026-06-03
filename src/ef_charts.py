from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=False)
class Asset:
    ret: float
    volat: float  # Variance


def plot_two_asset_frontier(
    asset_1: Asset,
    asset_2: Asset,
    correlation: float,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """
    Plot the risk-return curve for a two-asset portfolio.

    Parameters
    ----------
    asset_1 : Asset
        First asset, with mean as expected return and var as variance.
    asset_2 : Asset
        Second asset, with mean as expected return and var as variance.
    correlation : float
        Correlation coefficient between the two assets.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object where the chart will be drawn.
    **kwargs : Any
        Optional plot customization parameters.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    # Get kwargs
    num_portfolios = kwargs.get("num_portfolios", 500)
    efficient_label = kwargs.get("efficient_label", "Efficient Portfolios")
    inefficient_label = kwargs.get(
        "inefficient_label", "Inefficient Portfolios"
    )
    min_variance_label = kwargs.get(
        "min_variance_label",
        "Minimum variance portfolio",
    )
    x_axis_label = kwargs.get("x_axis_label", "Risk, standard deviation")
    y_axis_label = kwargs.get("y_axis_label", "Expected return")
    plot_title = kwargs.get(
        "plot_title",
        "Two-Asset Portfolio Efficient Frontier",
    )

    efficient_color = kwargs.get("efficient_color", "steelblue")
    inefficient_color = kwargs.get("inefficient_color", "grey")
    min_variance_color = kwargs.get("min_variance_color", "blue")

    # Check input values
    if not -1 <= correlation <= 1:
        msg = "Correlation must be between -1 and 1."
        raise ValueError(msg)

    if asset_1.volat < 0 or asset_2.volat < 0:
        msg_0 = "Asset std must be non-negative."
        raise ValueError(msg_0)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    std_1 = asset_1.volat
    std_2 = asset_2.volat
    covariance = correlation * std_1 * std_2

    weight_1 = np.linspace(0, 1, num_portfolios)
    weight_2 = 1 - weight_1

    portfolio_returns = weight_1 * asset_1.ret + weight_2 * asset_2.ret

    portfolio_variances = (
        weight_1**2 * std_1**2
        + weight_2**2 * std_2**2
        + 2 * weight_1 * weight_2 * covariance
    )

    portfolio_risks = np.sqrt(portfolio_variances)
    min_variance_index = int(np.argmin(portfolio_risks))

    left_slice = slice(0, min_variance_index + 1)
    right_slice = slice(min_variance_index, num_portfolios)

    left_mean_return = np.mean(portfolio_returns[left_slice])
    right_mean_return = np.mean(portfolio_returns[right_slice])

    if left_mean_return < right_mean_return:
        inefficient_slice = left_slice
        efficient_slice = right_slice
    else:
        inefficient_slice = right_slice
        efficient_slice = left_slice

    ax.plot(
        portfolio_risks[inefficient_slice],
        portfolio_returns[inefficient_slice],
        linestyle="--",
        color=inefficient_color,
        label=inefficient_label,
    )

    ax.plot(
        portfolio_risks[efficient_slice],
        portfolio_returns[efficient_slice],
        linestyle="-",
        color=efficient_color,
        label=efficient_label,
    )

    ax.scatter(
        portfolio_risks[min_variance_index],
        portfolio_returns[min_variance_index],
        color=min_variance_color,
        label=min_variance_label,
        zorder=4,
    )

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    return ax


def plot_two_asset_portfolio(  # noqa: PLR0913
    asset_1: Asset,
    asset_2: Asset,
    weight_1: float,
    weight_2: float,
    correlation: float,
    ax: Axes,
    **kwargs: Any,
) -> tuple[float, float]:
    """
    Calculate and plot the risk-return point for a two-asset portfolio.

    Parameters
    ----------
    asset_1 : Asset
        First asset, with mean as expected return and var as variance.
    asset_2 : Asset
        Second asset, with mean as expected return and var as variance.
    weight_1 : float
        Portfolio weight assigned to asset 1.
    weight_2 : float
        Portfolio weight assigned to asset 2.
    correlation : float
        Correlation coefficient between the two assets.
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the point will be drawn.
    **kwargs : Any
        Optional plot customization parameters.

    Returns
    -------
    tuple[float, float]
        Portfolio risk and portfolio return.
    """

    # Get kwargs
    point_label = kwargs.get("point_label", "Selected portfolio")
    point_color = kwargs.get("point_color", "darkorange")
    point_marker = kwargs.get("point_marker", "D")
    line_color = kwargs.get("line_color", point_color)

    # Check input values
    if not -1 <= correlation <= 1:
        msg = "Correlation must be between -1 and 1."
        raise ValueError(msg)

    if asset_1.volat < 0 or asset_2.volat < 0:
        msg_0 = "Asset variances must be non-negative."
        raise ValueError(msg_0)

    if not np.isclose(weight_1 + weight_2, 1.0):
        msg_1 = "Portfolio weights must sum to 1."
        raise ValueError(msg_1)

    std_1 = asset_1.volat
    std_2 = asset_2.volat
    covariance = correlation * std_1 * std_2

    portfolio_return = weight_1 * asset_1.ret + weight_2 * asset_2.ret

    portfolio_variance = (
        weight_1**2 * std_1**2
        + weight_2**2 * std_2**2
        + 2 * weight_1 * weight_2 * covariance
    )

    portfolio_risk = np.sqrt(portfolio_variance)

    ax.scatter(
        portfolio_risk,
        portfolio_return,
        color=point_color,
        marker=point_marker,
        label=point_label,
        zorder=5,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.vlines(
        x=portfolio_risk,
        ymin=y_min,
        ymax=portfolio_return,
        color=line_color,
        linestyle=":",
        linewidth=1,
        alpha=1.0,
    )

    ax.hlines(
        y=portfolio_return,
        xmin=x_min,
        xmax=portfolio_risk,
        color=line_color,
        linestyle=":",
        linewidth=1,
        alpha=1.0,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()

    return portfolio_risk, portfolio_return


def calc_two_asset_pf_metrics(
    asset_1: Asset,
    asset_2: Asset,
    weight_1: float,
    weight_2: float,
    correlation: float,
) -> tuple[float, float]:
    """
    Calculate portfolio risk and return for a two-asset portfolio.

    Returns
    -------
    tuple[float, float]
        portfolio_risk, portfolio_return
    """

    if asset_1.volat < 0 or asset_2.volat < 0:
        msg = "Asset standard deviations must be non-negative."
        raise ValueError(msg)

    if not -1 <= correlation <= 1:
        msg_0 = "Correlation must be between -1 and 1."
        raise ValueError(msg_0)

    pf_return = weight_1 * asset_1.ret + weight_2 * asset_2.ret

    covariance = correlation * asset_1.volat * asset_2.volat

    pf_variance = (
        weight_1**2 * asset_1.volat**2
        + weight_2**2 * asset_2.volat**2
        + 2 * weight_1 * weight_2 * covariance
    )

    pf_risk = np.sqrt(pf_variance)

    return pf_risk, pf_return
