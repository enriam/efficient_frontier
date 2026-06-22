from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

# base colors to create colormaps for positive and negative returns
POS_CLR = "teal"
NEG_CLR = "darkred"


def equalizer(
    pf_assets,
    ax: Axes = None,
    total_colname: str | None = None,
    weights: list | None = None,
    **kwargs: Any,
):
    """
    Plots the historical portfolio returns and the returns of the individual
    memebers of the portfolio to gives a visual idea of how well the different
    members combine to produce a balanced portfolio.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    pf_assets: pandas DataFrame
        The returns for each member of the portfolio (and optionally the total
        return for the portfolio)

    ax: Axes
        The axes to draw the chart

    title : str
        Plot title (default: 'Equalizer')

    total_colname : str
        The name of the column with the total returns for the portfolio. If
        is left empty, a new column with the total return will be calculated.

    weights: list
        The weights of each member of the portfolio, in the same order as the
        columns in 'pf_assets'. If left empty, all portfolio members will be
        assigned the same weight. This parameter takes effect only if
        'total_colname' is left empty.

    legend: Boolean
        If set to True, the chart will plot with different colors the lines
        corresponding the members of the portfolio and will show a legend to
        identify them (default: False).

    exclude: int
        Number of outliers to exclude from both negative and positive returns.

    Returns
    -------
    ax, d, df['total']: list

        ax, the axes with the chart
        d, dictionary with statistics {'min', 'max'}
        df['total'], pandas.Series with total returns for portfolio
    """
    # --- Get kwargs
    title = kwargs.get("title", "Equalizer")
    num_outliers_off = kwargs.get("num_outliers_off", 0)
    show_legend = kwargs.get("show_legend", False)

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check total column
    if total_colname is None:  # df has no pf total column
        total_col = "Total"
        if weights is None:  # means assets must have equal weights
            assets = len(pf_assets.columns)
            pfweights = [1 / assets] * assets
        else:
            pfweights = weights
        # calulate portfolio Total
        df = pf_assets.copy()
        df[total_col] = (df * pfweights).sum(axis=1)
    else:
        total_col = total_colname  # df contains pf total col
        df = pf_assets

    # check exclude outliers
    if abs(num_outliers_off) > len(pf_assets.index) / 2 - 1:
        print('Too many points for arg "exclude", has been reset to 0')
        outliers = 0
    else:
        outliers = int(abs(num_outliers_off))

    # --- CREATE PLOTS
    # draw lines for all portfolio members except pf total
    members = df.drop(columns=total_col)
    if show_legend:
        ax.plot(
            members,
            linewidth=2,
            alpha=0.3,
            label=members.columns,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.legend()

    else:
        ax.plot(
            members,
            linewidth=2.5,
            color="lightgrey",
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # draw line for portfolio total
    ax.plot(
        df[total_col],
        color="steelblue",
        linewidth=5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    # title and axis labels
    try:
        year_min = str(df.index.min().year)  # first year with data
        year_max = str(df.index.max().year)  # last year with data
        xlabel = year_min + " to " + year_max
    except Exception:
        year_min = str(df.index.min())  # first index label
        year_max = str(df.index.max())  # last index label
        xlabel = "Periods " + year_min + " to " + year_max

    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel(xlabel, size=12, fontweight="bold")
    ax.set_ylabel("Return", size=12, fontweight="bold")

    # spines: hide right and top
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    ax.tick_params(labelbottom=False)

    # x axis
    ax.set_xlim(df.index.min(), df.index.max())

    # y axis
    yticks = [n / 100 for n in range(-50, 60, 10)]
    ax.set_ylim(-0.50, 0.50)
    ax.set_yticks(yticks)

    # y axis labels as percent
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # horizontal lines for max and min returns
    bull_clr = POS_CLR  # color for positive returns
    bear_clr = NEG_CLR  # color for negative returns
    # remove outliers
    ordered_rets = sorted(df[total_col].to_list())
    if outliers > 0:
        ordered_rets = ordered_rets[outliers:-outliers]
    pf_min = min(ordered_rets)
    pf_max = max(ordered_rets)

    ax.axhline(
        y=pf_max,
        xmin=0,
        xmax=1,
        color=bull_clr,
        linestyle="dotted",
        linewidth=3,
    )
    ax.text(
        df.index.min(),
        pf_max,
        "MAX",
        color="white",
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox={"facecolor": bull_clr, "edgecolor": bull_clr, "alpha": 1},
    )
    ax.axhline(
        y=pf_min,
        xmin=0,
        xmax=1,
        color=bear_clr,
        linestyle="dotted",
        linewidth=3,
    )
    ax.text(
        df.index.min(),
        pf_min,
        "MIN",
        color="white",
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox={"facecolor": bear_clr, "edgecolor": bear_clr, "alpha": 1},
    )

    # horizontal line for y=0
    ax.axhline(
        y=0, xmin=0, xmax=1, color="black", linestyle="solid", linewidth=2
    )

    # # text box to inform of outliers excluded
    # if outliers > 0:
    #     outliers_txt = f'(Top & Bottom Outliers Excluded: {outliers})'
    #     ax.text(0.5, 1, outliers_txt, color='black', fontsize=9,
    #             transform=ax.transAxes, ha='center', va='top',
    #             bbox=dict(facecolor='white', edgecolor='white', alpha=1))

    # create dictionary with statistics
    d = {"min": pf_min, "max": pf_max}
    # d = {'balance': balance, 'intensity': intensity}
    return ax, d, df[total_col]
