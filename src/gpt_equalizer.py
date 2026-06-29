"""Utilities for plotting annual investment returns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        msg = "returns must be a pandas.DataFrame"
        raise TypeError(msg)
    if returns.empty or returns.shape[1] == 0:
        msg = "returns must contain at least one asset and one observation"
        raise ValueError(msg)
    if not isinstance(returns.index, pd.DatetimeIndex):
        msg = "returns must have a pandas.DatetimeIndex"
        raise TypeError(msg)
    if not returns.index.is_monotonic_increasing:
        msg = "returns index must be sorted in ascending order"
        raise ValueError(msg)
    if not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes
    ):
        msg = "all returns columns must be numeric"
        raise TypeError(msg)


def _resolve_weights(
    number_of_assets: int, weights: Sequence[float] | None
) -> np.ndarray:
    if weights is None:
        return np.full(number_of_assets, 1 / number_of_assets)

    resolved = np.asarray(weights, dtype=float)
    if resolved.ndim != 1 or len(resolved) != number_of_assets:
        msg = "weights must contain one value for each asset"
        raise ValueError(msg)
    if not np.isfinite(resolved).all():
        msg = "weights must contain only finite values"
        raise ValueError(msg)
    if not np.isclose(resolved.sum(), 1.0):
        msg = "weights must add up to 1"
        raise ValueError(msg)
    return resolved


def _compound_by_year(
    returns: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Compound periodic returns into calendar-year returns."""
    return (1 + returns).groupby(returns.index.year).prod(min_count=1) - 1


def plot_annual_returns(
    returns: pd.DataFrame,
    weights: Sequence[float] | None = None,
    ax: Axes | None = None,
    **kwargs: object,
) -> Axes:
    """Plot annual asset and periodically rebalanced portfolio returns.

    Parameters
    ----------
    returns:
        Periodic decimal returns for each asset, indexed by date.
    weights:
        Portfolio weights in the same order as ``returns.columns``. Equal
        weights are used when omitted.
    ax:
        Matplotlib axes on which to draw. A new one is created when omitted.
    **kwargs:
        Supports ``title``; its default value is ``"Portfolio Correlations"``.
    """
    _validate_returns(returns)
    resolved_weights = _resolve_weights(returns.shape[1], weights)

    unknown_kwargs = set(kwargs) - {"title", "pf_label"}
    if unknown_kwargs:
        names = ", ".join(sorted(unknown_kwargs))
        msg = f"unsupported keyword argument(s): {names}"
        raise TypeError(msg)

    title = str(kwargs.get("title", "Portfolio Correlations"))
    pf_label = str(kwargs.get("pf_label", "Portfolio"))

    annual_assets = _compound_by_year(returns)
    periodic_portfolio = returns.mul(resolved_weights, axis="columns").sum(
        axis="columns", min_count=returns.shape[1]
    )
    annual_portfolio = _compound_by_year(periodic_portfolio)

    if annual_portfolio.dropna().empty:
        msg_0 = "returns do not contain a complete period for the portfolio"
        raise ValueError(msg_0)

    if ax is None:
        _, ax = plt.subplots()

    for asset in annual_assets.columns:
        ax.plot(
            annual_assets.index,
            annual_assets[asset],
            linewidth=1,
            alpha=0.4,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=str(asset),
        )
    ax.plot(
        annual_portfolio.index,
        annual_portfolio,
        linewidth=3,
        solid_capstyle="round",
        solid_joinstyle="round",
        color="black",
        label=pf_label,
    )

    # maximum = annual_portfolio.max()
    # minimum = annual_portfolio.min()
    # ax.axhline(maximum, color="teal", linewidth=1)
    # ax.axhline(minimum, color="darkred", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)

    # label_style = {
    #     "transform": ax.get_yaxis_transform(),
    #     "ha": "left",
    #     "va": "center",
    #     "color": "white",
    #     "fontweight": "bold",
    #     "fontsize": 8,
    #     "clip_on": True,
    # }
    # ax.text(
    #     0,
    #     maximum,
    #     "MAX",
    #     bbox={"facecolor": "teal", "edgecolor": "teal"},
    #     **label_style,
    # )
    # ax.text(
    #     0,
    #     minimum,
    #     "MIN",
    #     bbox={"facecolor": "darkred", "edgecolor": "darkred"},
    #     **label_style,
    # )

    first_year = int(annual_assets.index.min())
    last_year = int(annual_assets.index.max())
    first_tick = (first_year // 5) * 5
    last_tick = ((last_year + 4) // 5) * 5
    ax.set_xticks(np.arange(first_tick, last_tick + 1, 5))
    ax.set_xlim(
        first_tick, last_tick if first_tick != last_tick else last_tick + 1
    )
    ax.set_ylim(-0.50, 0.80)
    ax.set_yticks(np.arange(-0.5, 0.81, 0.1))

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    ax.tick_params(axis="both", color="lightgrey", labelcolor="grey")

    ax.spines["bottom"].set_position(("data", -0.5))
    ax.spines[["top", "bottom", "right", "left"]].set(visible=False)

    ax.grid(visible=True, which="major", axis="y", c="lightgrey", ls=":")

    ax.set_title(title)

    ax.legend(loc="upper right")

    return ax
