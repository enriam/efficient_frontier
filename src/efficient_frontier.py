import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_two_asset_efficient_frontier(
    mean_return_1,
    mean_return_2,
    variance_1,
    variance_2,
    correlation,
    ax=None,
    num_portfolios=500,
    **kwargs,
):
    """
    Plot the risk-return curve for a two-asset portfolio.

    Parameters
    ----------
    mean_return_1 : float
        Expected return of asset 1.
    mean_return_2 : float
        Expected return of asset 2.
    variance_1 : float
        Variance of asset 1.
    variance_2 : float
        Variance of asset 2.
    correlation : float
        Correlation coefficient between the two assets.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object where the chart will be drawn.
        If None, a new figure and axes are created.
    num_portfolios : int
        Number of portfolio weights to evaluate.
    **kwargs
        Optional plot customization parameters:
        - asset1_label
        - asset2_label
        - portfolio_label
        - x_axis_label
        - y_axis_label
        - plot_title
        - asset1_color
        - asset2_color
        - min_variance_color

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    asset1_label = kwargs.get("asset1_label", "Asset 1")
    asset2_label = kwargs.get("asset2_label", "Asset 2")
    portfolio_label = kwargs.get("portfolio_label", "Portfolio frontier")

    x_axis_label = kwargs.get("x_axis_label", "Risk, standard deviation")
    y_axis_label = kwargs.get("y_axis_label", "Expected return")
    plot_title = kwargs.get("plot_title", "Two-Asset Portfolio Efficient Frontier")

    asset1_color = kwargs.get("asset1_color", "blue")
    asset2_color = kwargs.get("asset2_color", "orange")
    min_variance_color = kwargs.get("min_variance_color", "red")

    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1.")

    if variance_1 < 0 or variance_2 < 0:
        raise ValueError("Variances must be non-negative.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    std_1 = np.sqrt(variance_1)
    std_2 = np.sqrt(variance_2)
    covariance = correlation * std_1 * std_2

    weights_1 = np.linspace(0, 1, num_portfolios)
    weights_2 = 1 - weights_1

    portfolio_returns = weights_1 * mean_return_1 + weights_2 * mean_return_2

    portfolio_variances = (
        weights_1**2 * variance_1
        + weights_2**2 * variance_2
        + 2 * weights_1 * weights_2 * covariance
    )

    portfolio_risks = np.sqrt(portfolio_variances)
    min_variance_index = np.argmin(portfolio_risks)

    ax.plot(
        portfolio_risks,
        portfolio_returns,
        label=portfolio_label,
    )

    ax.scatter(
        std_1,
        mean_return_1,
        color=asset1_color,
        label=asset1_label,
        zorder=3,
    )

    ax.scatter(
        std_2,
        mean_return_2,
        color=asset2_color,
        label=asset2_label,
        zorder=3,
    )

    ax.scatter(
        portfolio_risks[min_variance_index],
        portfolio_returns[min_variance_index],
        color=min_variance_color,
        label=kwargs.get("min_variance_label", "Minimum variance portfolio"),
        zorder=4,
    )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_two_portfolios(pf1, pf2):
    plot_two_asset_efficient_frontier(*pf1)
    plot_two_asset_efficient_frontier(*pf2)

    