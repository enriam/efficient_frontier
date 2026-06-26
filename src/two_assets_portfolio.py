import math

# from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from src.asset import Asset

# @dataclass(frozen=False)
# class Asset:
#     ret: float
#     volat: float  # Variance


def plot_2a_frontier(
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

    efficient_style = kwargs.get("efficient_style", "-")
    inefficient_style = kwargs.get("inefficient_style", "--")

    # Check input values
    if not -1 <= correlation <= 1:
        msg = "Correlation must be between -1 and 1."
        raise ValueError(msg)

    if asset_1.std < 0 or asset_2.std < 0:
        msg_0 = "Asset std must be non-negative."
        raise ValueError(msg_0)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    std_1 = asset_1.std
    std_2 = asset_2.std
    covariance = correlation * std_1 * std_2

    weight_1 = np.linspace(0, 1, num_portfolios)
    weight_2 = 1 - weight_1

    portfolio_returns = weight_1 * asset_1.avg + weight_2 * asset_2.avg

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
        linestyle=inefficient_style,
        color=inefficient_color,
        label=inefficient_label,
    )

    ax.plot(
        portfolio_risks[efficient_slice],
        portfolio_returns[efficient_slice],
        linestyle=efficient_style,
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
    # ax.legend()
    ax.grid(visible=True, alpha=0.3)

    return ax


def scatter_2a_portfolio(  # noqa: PLR0913
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

    if asset_1.std < 0 or asset_2.std < 0:
        msg_0 = "Asset variances must be non-negative."
        raise ValueError(msg_0)

    if not np.isclose(weight_1 + weight_2, 1.0):
        msg_1 = "Portfolio weights must sum to 1."
        raise ValueError(msg_1)

    std_1 = asset_1.std
    std_2 = asset_2.std
    covariance = correlation * std_1 * std_2

    portfolio_return = weight_1 * asset_1.avg + weight_2 * asset_2.avg

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


def risk_return_2a_pf(
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

    if asset_1.std < 0 or asset_2.std < 0:
        msg = "Asset standard deviations must be non-negative."
        raise ValueError(msg)

    if not -1 <= correlation <= 1:
        msg_0 = "Correlation must be between -1 and 1."
        raise ValueError(msg_0)

    if not np.isclose(weight_1 + weight_2, 1.0):
        msg = "Portfolio weights must sum to 1."
        raise ValueError(msg)

    if weight_1 < 0 or weight_2 < 0:
        msg = "Portfolio weights must be non-negative."
        raise ValueError(msg)

    pf_return = weight_1 * asset_1.avg + weight_2 * asset_2.avg

    covariance = correlation * asset_1.std * asset_2.std

    pf_variance = (
        weight_1**2 * asset_1.std**2
        + weight_2**2 * asset_2.std**2
        + 2 * weight_1 * weight_2 * covariance
    )

    pf_risk = np.sqrt(pf_variance)

    return pf_risk, pf_return


def weights_2a_pf_long_only(
    asset1: Asset,
    asset2: Asset,
    corr: float,
    target_vol: float,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """
    Calcula los pesos óptimos long-only de una cartera de dos activos.

    Objetivo:
        maximizar rentabilidad esperada

    Restricciones:
        w1 + w2 = 1
        0 <= w1 <= 1
        0 <= w2 <= 1
        volatilidad cartera <= target_vol

    Cada activo debe tener:
        asset.rent   -> rentabilidad esperada
        asset.volat  -> volatilidad

    Returns
    -------
    tuple[float, float]
        Pesos óptimos (w1, w2).
    """

    mu1 = asset1.avg
    mu2 = asset2.avg

    sigma1 = asset1.std
    sigma2 = asset2.std

    rho = corr
    sigma_t = target_vol

    if sigma1 < 0 or sigma2 < 0 or sigma_t < 0:
        msg = "Las volatilidades deben ser no negativas."
        raise ValueError(msg)

    if not -1 <= rho <= 1:
        msg_0 = "La correlación debe estar entre -1 y 1."
        raise ValueError(msg_0)

    # Varianza de la cartera:
    # sigma_p^2(w1) = a*w1^2 + b*w1 + c
    a = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
    b = 2 * rho * sigma1 * sigma2 - 2 * sigma2**2
    c = sigma2**2

    candidates = []

    # Extremos long-only: 100% activo 1 y 100% activo 2
    if sigma1 <= sigma_t + tol:
        candidates.append((1.0, 0.0))

    if sigma2 <= sigma_t + tol:
        candidates.append((0.0, 1.0))

    # Soluciones interiores que cumplen exactamente sigma_p = sigma_t
    # a*w1^2 + b*w1 + c = sigma_t^2
    if abs(a) > tol:
        discriminant = b**2 - 4 * a * (c - sigma_t**2)

        if discriminant >= -tol:
            discriminant = max(discriminant, 0.0)
            sqrt_disc = math.sqrt(discriminant)

            w1_a = (-b - sqrt_disc) / (2 * a)
            w1_b = (-b + sqrt_disc) / (2 * a)

            for w1 in (w1_a, w1_b):
                w2 = 1 - w1

                if -tol <= w1 <= 1 + tol and -tol <= w2 <= 1 + tol:
                    w1 = min(max(w1, 0.0), 1.0)  # noqa: PLW2901
                    w2 = 1 - w1

                    pf_volat = volatility_2a_pf(w1, sigma1, sigma2, rho)
                    if pf_volat <= sigma_t + tol:
                        candidates.append((w1, w2))

    # Si los dos activos juntos permiten una cartera de menor volatilidad,
    # puede haber una solución factible aunque ningún activo por separado lo
    # sea.
    if not candidates:
        msg = "No existe cartera long-only que cumpla la volatilidad target."
        raise ValueError(msg)

    # Elegimos la cartera factible con mayor rentabilidad esperada
    w1_opt, w2_opt = max(candidates, key=lambda w: w[0] * mu1 + w[1] * mu2)

    return w1_opt, w2_opt


def volatility_2a_pf(w1, sigma1, sigma2, rho):
    """
    Volatilidad de una cartera de dos activos usando correlación.
    """

    w2 = 1 - w1

    var = (
        w1**2 * sigma1**2
        + w2**2 * sigma2**2
        + 2 * w1 * w2 * rho * sigma1 * sigma2
    )

    return math.sqrt(max(var, 0.0))


def min_vol_2a_pf(vol1, vol2, corr, tol=1e-12):
    """
    Calcula la mínima volatilidad posible de una cartera long-only
    de dos activos, en función de sus volatilidades y correlación.

    Restricciones:
        w1 + w2 = 1
        0 <= w1 <= 1
        0 <= w2 <= 1

    Parameters
    ----------
    vol1 : float
        Volatilidad del activo 1.
    vol2 : float
        Volatilidad del activo 2.
    corr : float
        Correlación entre ambos activos.

    Returns
    -------
    tuple[float, float, float]
        mínima volatilidad, peso activo 1, peso activo 2
    """

    sigma1 = vol1
    sigma2 = vol2
    rho = corr

    if sigma1 < 0 or sigma2 < 0:
        msg = "Las volatilidades deben ser no negativas."
        raise ValueError(msg)

    if not -1 <= rho <= 1:
        msg_0 = "La correlación debe estar entre -1 y 1."
        raise ValueError(msg_0)

    # Caso degenerado: ambos activos sin volatilidad
    if sigma1 < tol and sigma2 < tol:
        return 0.0, 0.5, 0.5

    # Varianza:
    # sigma_p^2(w1) = a*w1^2 + b*w1 + c
    a = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
    b = 2 * rho * sigma1 * sigma2 - 2 * sigma2**2
    # c = sigma2**2

    # Si a es cero o casi cero, la función es plana o casi plana.
    # Evaluamos extremos long-only.
    if abs(a) < tol:
        candidates = [0.0, 1.0]
    else:
        # Peso que minimiza la varianza sin restricción long-only
        w1_gmv = -b / (2 * a)

        # Restricción long-only
        w1_gmv = min(max(w1_gmv, 0.0), 1.0)

        candidates = [w1_gmv, 0.0, 1.0]

    best_w1 = min(
        candidates,
        key=lambda w1: variance_2a_pf(w1, sigma1, sigma2, rho),
    )

    best_w2 = 1 - best_w1

    min_var = variance_2a_pf(best_w1, sigma1, sigma2, rho)
    min_vol = math.sqrt(max(min_var, 0.0))

    return min_vol, best_w1, best_w2


def variance_2a_pf(w1, vol1, vol2, corr):
    """
    Varianza de una cartera de dos activos usando correlación.
    """

    w2 = 1 - w1

    return w1**2 * vol1**2 + w2**2 * vol2**2 + 2 * w1 * w2 * corr * vol1 * vol2


# ========================================================================
# ========================================================================


def optimal_two_asset_weights_long_only(
    asset1, asset2, corr, target_vol, tol=1e-12
):
    """
    Calcula los pesos óptimos long-only de una cartera de dos activos
    con una volatilidad máxima target_vol.

    Objetivo:
        maximizar rentabilidad esperada

    Restricciones:
        w1 + w2 = 1
        0 <= w1 <= 1
        0 <= w2 <= 1
        sigma_p <= target_vol

    Cada activo debe tener:
        asset.rent   -> rentabilidad esperada
        asset.volat  -> volatilidad

    Returns
    -------
    tuple[float, float]
        Pesos óptimos (w1, w2).
    """

    mu1 = asset1.avg
    mu2 = asset2.avg

    sigma1 = asset1.std
    sigma2 = asset2.std

    rho = corr
    sigma_t = target_vol

    if sigma1 < 0 or sigma2 < 0 or sigma_t < 0:
        msg = "Las volatilidades deben ser no negativas."
        raise ValueError(msg)

    if not -1 <= rho <= 1:
        msg_0 = "La correlación debe estar entre -1 y 1."
        raise ValueError(msg_0)

    min_vol, w1_gmv, w2_gmv = min_vol_two_assets(sigma1, sigma2, rho)

    if sigma_t < min_vol - tol:
        msg_1 = (
            f"No existe cartera long-only con volatilidad <= {sigma_t:.6f}. "
            f"La volatilidad mínima posible es {min_vol:.6f}."
        )
        raise ValueError(msg_1)

    if abs(sigma_t - min_vol) <= tol:
        return w1_gmv, w2_gmv

    max_vol = max(sigma1, sigma2)

    # Si el target permite invertir 100% en cualquier activo,
    # elegimos directamente el de mayor rentabilidad esperada.
    if sigma_t >= max_vol - tol:
        if mu1 >= mu2:
            return 1.0, 0.0
        return 0.0, 1.0

    # En este punto:
    # min_vol < sigma_t < max(sigma1, sigma2)
    #
    # Resolvemos la frontera:
    # sigma_p^2(w1) = sigma_t^2
    #
    # La solución óptima estará en el borde de la restricción,
    # salvo que el activo con mayor rentabilidad ya sea factible,
    # cosa que ya hemos capturado con el caso anterior.

    a = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
    b = 2 * rho * sigma1 * sigma2 - 2 * sigma2**2
    c = sigma2**2 - sigma_t**2

    discriminant = b**2 - 4 * a * c

    if discriminant < -tol:
        msg_2 = "No existe solución real para esa volatilidad target."
        raise ValueError(msg_2)

    discriminant = max(discriminant, 0.0)
    sqrt_disc = math.sqrt(discriminant)

    w1_a = (-b - sqrt_disc) / (2 * a)
    w1_b = (-b + sqrt_disc) / (2 * a)

    candidates = []

    for w1 in (w1_a, w1_b):
        w2 = 1 - w1

        if -tol <= w1 <= 1 + tol and -tol <= w2 <= 1 + tol:
            w1 = min(max(w1, 0.0), 1.0)  # noqa: PLW2901
            w2 = 1 - w1

            vol = portfolio_volatility_two_assets(w1, sigma1, sigma2, rho)

            if vol <= sigma_t + tol:
                candidates.append((w1, w2))

    if not candidates:
        msg_3 = "No existe cartera long-only que cumpla la volatilidad target."
        raise ValueError(msg_3)

    # Elegimos la cartera factible de mayor rentabilidad esperada.
    w1_opt, w2_opt = max(candidates, key=lambda w: w[0] * mu1 + w[1] * mu2)

    return w1_opt, w2_opt


def min_vol_two_assets(vol1, vol2, corr, tol=1e-12):
    """
    Calcula la mínima volatilidad long-only de una cartera de dos activos.

    Returns
    -------
    tuple[float, float, float]
        mínima volatilidad, peso activo 1, peso activo 2
    """

    sigma1 = vol1
    sigma2 = vol2
    rho = corr

    denominator = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2

    if abs(denominator) < tol:
        candidates = [0.0, 1.0]
    else:
        w1_gmv = (sigma2**2 - rho * sigma1 * sigma2) / denominator

        w1_gmv = min(max(w1_gmv, 0.0), 1.0)
        candidates = [w1_gmv, 0.0, 1.0]

    best_w1 = min(
        candidates,
        key=lambda w1: portfolio_variance_two_assets(w1, sigma1, sigma2, rho),
    )

    best_w2 = 1 - best_w1

    min_var = portfolio_variance_two_assets(best_w1, sigma1, sigma2, rho)
    min_vol = math.sqrt(max(min_var, 0.0))

    return min_vol, best_w1, best_w2


def portfolio_variance_two_assets(w1, vol1, vol2, corr):
    """
    Varianza de una cartera de dos activos usando correlación.
    """

    w2 = 1 - w1

    return w1**2 * vol1**2 + w2**2 * vol2**2 + 2 * w1 * w2 * corr * vol1 * vol2


def portfolio_volatility_two_assets(w1, vol1, vol2, corr):
    """
    Volatilidad de una cartera de dos activos usando correlación.
    """

    var = portfolio_variance_two_assets(w1, vol1, vol2, corr)
    return math.sqrt(max(var, 0.0))
