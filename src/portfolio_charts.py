# --- PLOTTING LIBRARY
# This is a collection of functions that will generate charts almost identical
# to the charts you can find in the web site portfoliocharts.com
# For detailed information about the charts visit portfoliocharts.com

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# --- IMPORTS
# import itertools
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    PercentFormatter,
)

# from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.stats import hmean


# --- DEFINITIONS
# class for customized exceptions
class Exception(Exception):
    pass


# base colors to create colormaps for positive and negative returns
POS_CLR = "teal"
NEG_CLR = "darkred"


##############################################################################
# SUPPORT FUNCTIONS
##############################################################################
def ulcer_index(data, type="prices"):
    """
    This function calculates the Ulcer Index (UI) of an investment. The UI is
    a measure of the depth and duration of drawdowns in prices from earlier
    highs. For a detailed description of how this indicator is calculated
    please visit:
    http://www.tangotools.com/ui/ui.htm

    Parameters
    ----------
    data: pandas.Series
        Time Series with prices or returns for a specific investment.

    type: str
        Two values are accepted, 'prices' or 'returns', to indicate the type of
        information that is passed with 'data'

    Returns
    -------
    ui: scalar
        The UI for the investment
    """

    # UI is calculated with price data. If data type is 'returns', the
    # values have to be converted to prices.

    # --- DATA TYPE = prices
    if type == "prices":
        prices = data.copy()
    # --- DATA TYPE = returns -> convert returns tu prices
    elif type == "returns":
        # calculate cummulative returns
        cum_rets = (1 + data).cumprod()  # = to prices when initial value is 1
        # build a start row, insert it in first place and reset index
        start_row = pd.Series([1], index=[data.index[0]])
        # prices = start_row.append(cum_rets).reset_index(drop=True)
        prices = pd.concat([start_row, cum_rets], ignore_index=True)

    # --- CALCULATE ULCER INDEX
    peaks = prices.cummax()
    drawdowns = (prices / peaks - 1) * 100
    sum_sqr_dd = (drawdowns**2).sum()  # sum of square drawdowns
    ui = (sum_sqr_dd / len(prices)) ** 0.5
    return ui


# -----------------------------------------------------------------------
def sequential_cmap(base_clr):
    """
    Creates a sequential colormap from 'base_clr' to 'white', ranging from 0 to
    256  in the RGB spectrum.

    Parameters
    ----------
    base_clr: color
        Base color to create the colormap

    Returns
    -------
    ListedColormap
    """

    # get RGB code for base_clr
    base_rgb = mpl.colors.to_rgb(base_clr)
    # create and populate np array [r, g, b, alpha]
    N = 256  # RGB spectrum
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(base_rgb[0], 1, N)
    vals[:, 1] = np.linspace(base_rgb[1], 1, N)
    vals[:, 2] = np.linspace(base_rgb[2], 1, N)
    # Note: vals[:, 3] is the alpha channel and remains 1
    # create and return colormap
    return colors.ListedColormap(vals)


# -----------------------------------------------------------------------
def rolling_CAGR(data, periods=10):
    """
    Calculates the CAGR for all rolling windows of 'periods'-length within the
    data timeframe.

    Parameters
    ----------
    data: pandas.Series
        Series with the simple return for each period of time.

    periods: int
        Length of the window (default: 10)

    Returns
    -------
    pandas.Series:
        A Series with the rolling CAGRs
    """

    # check validity of periods
    if periods < 1 or periods > len(data) + 1:
        raise Exception('Argument "periods" must be in [1, len(data)]')
    # required rolling windows
    nwindows = len(data) + 1 - periods
    # calculate cummulative returns into numpy array
    cum_rets = np.array(
        [(1 + data[i : i + periods]).cumprod()[-1] for i in range(0, nwindows)]
    )
    # calculate CAGR = (Vi/Vf)^(1/n) - 1
    cagr = np.power(cum_rets, 1 / periods) - 1
    # convert to Series and return
    return pd.Series(cagr, index=data.index[0:nwindows]).to_period("Y")


# -----------------------------------------------------------------------
def rolling_cumrets(data, periods=10):
    """
    Calculates the cummulative return for all rolling windows of
    'periods'-length within the data timeframe.

    Parameters
    ----------
    data: pandas.Series
        Series with the simple return for each period of time.

    periods: int
        Length of the window (default: 10)

    Returns
    -------
    pandas.Series:
        A Series with the rolling cummulative returns
    """

    # check validity of periods
    if periods < 1 or periods > len(data):
        raise Exception('Argument "periods" must be in [1, len(data)]')
    # required rolling windows
    windows = len(data) - periods + 1
    # define index for pandas Series
    idx = data.index[0:windows]
    # calculate cummulative returns for each window (use np.array() to make it
    # work also with np arrays and lists)
    cum_rets = np.array(
        [
            (1 + data.iloc[i : i + periods]).cumprod().iloc[-1]
            for i in range(0, windows)
        ]
    )

    return pd.Series(cum_rets, index=idx).to_period("Y")


##############################################################################
# CHART FUNCTIONS
##############################################################################
def annual_returns(data, ax=None, title="Annual Returns"):
    """
    The function creates a chart with the distribution of all annual returns
    for a given investment.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data : pandas Time Series
        A sequence of returns for a given portfolio

    ax : Axes
        The axes to draw the chart

    title : str
       Plot title (default: 'Annual Returns')

    Returns
    -------
    ax, d : list

        ax, the axes with the chart
        d, dictionary with statistics {'min', 'max', 'avg', 'stdev'}
    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # --- DEFINITIONS
    # colormaps
    cm_pos = sequential_cmap(POS_CLR).reversed()  # cmap for positive returns
    pos_clrs = cm_pos(np.linspace(0.2, 1, 6))  # colors for positive returns
    cm_neg = sequential_cmap(NEG_CLR)  # cmap for negative returns
    neg_clrs = cm_neg(np.linspace(0, 0.8, 6))  # colors for negative returns
    # concatenate colors
    newcolors = np.vstack((pos_clrs, neg_clrs))

    # --- CREATE CHART
    hist_bins = [
        -0.50,
        -0.40,
        -0.30,
        -0.20,
        -0.10,
        0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]
    n, bins, patches = ax.hist(
        data, bins=hist_bins, weights=np.ones(len(data)) / len(data)
    )

    # set bars face color based on their position = return range
    for patch in patches:
        xpos = patch.get_xy()[0]  # x coordinate
        patch.set_facecolor(newcolors[round(xpos * 10)])
        patch.zorder = 3  # draw above grid lines

    # title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Return", size=12, fontweight="bold")
    ax.set_ylabel("Frequency", size=12, fontweight="bold")

    # grid -> horizontal with major and minor lines
    ax.yaxis.grid(
        which="major", color="dimgrey", linestyle="solid", linewidth=0.7
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.grid(
        which="minor", color="darkgrey", linestyle="solid", linewidth=0.3
    )

    # x axis
    ax.set_xticks(hist_bins)
    ax.set_xlim(-0.60, 0.60)

    # x axis labels as percent
    xlabels = [(str(round(item * 100)) + "%") for item in hist_bins]
    xlabels[0] = "< " + xlabels[0]  # customize first label
    xlabels[-1] = xlabels[-1] + " >"  # customize last label
    ax.set_xticklabels(xlabels)

    # y axis labels as percent
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # --- ADD BASIC STATISTICS INFO
    # statistics
    rmin = data.min()  # min annual return
    rmax = data.max()  # max annual return
    ravg = data.mean()  # average return
    rstd = data.std()  # standard deviation
    rneg = np.sum(np.array(data < 0)) / len(data)  # % of neg returns

    # marker parameters
    mk_size = 70  # marker size
    mk_ypos = ax.get_ylim()[1] / 40  # markers y pos = 1/40 of y axis length
    z = 6  # z-order

    # plot statistics
    ax.scatter(
        rmin,
        mk_ypos,
        color="red",
        edgecolor="black",
        marker="v",
        label=f"Min: {round(rmin * 100):,.1f}%",
        s=mk_size,
        zorder=z,
    )
    ax.scatter(
        rmax,
        mk_ypos,
        color="limegreen",
        edgecolor="black",
        marker="v",
        label=f"Max: {round(rmax * 100):,.1f}%",
        s=mk_size,
        zorder=z,
    )
    ax.scatter(
        ravg,
        mk_ypos,
        color="darkorange",
        edgecolor="black",
        marker="o",
        label=f"Avg: {round(ravg * 100):,.1f}%",
        s=mk_size,
        zorder=z,
    )
    ax.scatter(
        rstd,
        mk_ypos,
        color="skyblue",
        edgecolor="black",
        marker="d",
        label=f"Dev: {round(rstd * 100):,.1f}%",
        s=mk_size,
        zorder=z,
    )

    # legend
    ax.legend(scatteryoffsets=[0.5], framealpha=1, loc="upper right")

    # vertical line at x=0
    rneg_txt = f" << Loss freq: {str(round(rneg * 100, 1))}%"
    ax.axvline(
        x=0, ymin=0, ymax=1, color="black", linestyle="dashed", linewidth=1
    )

    # text informing loss frequency
    ax.text(
        0.493,
        0.94,
        rneg_txt,
        color="white",
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="tab:red", edgecolor="darkred", alpha=1),
    )

    # create dictionary with statistics
    d = {"min": rmin, "max": rmax, "avg": ravg, "stdev": rstd}
    return ax, d


# -----------------------------------------------------------------------
def equalizer(
    pf_assets,
    ax=None,
    title="Equalizer",
    total_colname=None,
    weights=None,
    legend=False,
    exclude=0,
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
    if abs(exclude) > len(pf_assets.index) / 2 - 1:
        print('Too many points for arg "exclude", has been reset to 0')
        outliers = 0
    else:
        outliers = int(abs(exclude))

    # --- CREATE PLOTS
    # draw lines for all portfolio members except pf total
    members = df.drop(columns=total_col)
    if legend:
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
        bbox=dict(facecolor=bull_clr, edgecolor=bull_clr, alpha=1),
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
        bbox=dict(facecolor=bear_clr, edgecolor=bear_clr, alpha=1),
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


# -----------------------------------------------------------------------
def rolling_returns(
    data, ax=None, periods=10, title="Rolling Returns", isrolling=False
):
    """
    Draws a bar chart with the CAGR return for each of the rolling windows of
    'periods'-length within the data timeframe.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data: pandas.Series
        The data to plot. Can be returns or already calculated rolling returns.
        In this last case, 'isrolling' should be set to True.

    ax: Axe
        The axes to plot the data

    periods: int
        The number of periods for the rolling window calculation (default: 10)
        This parameter will be ignored if 'isrolling' is True.

    title: str
        The title desired for the chart (default: 'Rolling Returns')

    isrolling: Boolean
        If False, the function will calculate the rolling returns in windows
        of 'periods' length. If True, no calculation will be carried out and
        'data' will be plot directly.

    Returns
    -------
    ax, pf_data: list

        ax, the axes with the chart
        pf_data, pandas series with the rolling returns
    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check input parameters
    if isrolling:
        pf_data = data.copy()  # data is rolling returns
    else:
        if periods > len(data):  # periods is too long
            raise Exception('argument "periods" cannot be longer than data.')
        elif periods < 1:  # periods is invalid
            raise Exception('argument "periods" must be greater that 0.')
        # convert data to rolling CAGR
        else:
            pf_data = rolling_CAGR(data, periods)

    # --- CREATE CHART
    container = ax.bar(pf_data.index.year, pf_data.values, color=POS_CLR)

    # customize bar colors
    for patch in container:
        # place patch above grid lines
        patch.zorder = 3
        # change face color based on height
        if patch.get_height() < 0:
            patch.set_fc(NEG_CLR)

    # title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Start Year", size=12, fontweight="bold")
    ax.set_ylabel(
        "Rolling " + str(periods) + " Years Real CAGR",
        size=12,
        fontweight="bold",
    )

    # y axis labels to percent format
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # grid: horizontal with major and minor lines
    ax.yaxis.grid(
        which="major", color="lightgrey", linestyle="solid", linewidth=0.7
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.grid(
        which="minor", color="lightgrey", linestyle="solid", linewidth=0.3
    )
    # return and end function
    return ax, pf_data


# -----------------------------------------------------------------------
def target_accuracy(
    data, ax=None, capital=10_000, periods=10, title="Target Accuracy"
):
    """
    This function creates a chart that measures the spread of real-world growth
    of a portfolio over every possible start year compared to the simple
    long-term average, in rolling windows of periods length.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data : pandas Time Series
        A sequence of returns for a given portfolio

    ax : Axes
        The axes to draw the chart

    capital: scalar
        Initial portfolio capital

    periods: int
        The number of periods for the rolling window calculation (default=10)

    title : str
       Plot title (default: 'Target Accuracy')

    Returns
    -------
    ax, d, df_cumrets: list

        ax, the axis with the chart
        d, dictionary with statistics {'max', 'pc85', 'avg', 'pc15', 'min'}
        df_cumrets, pandas dataframe with cummulative returns for all windows
    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check initial capital
    if capital < 1:
        raise Exception("arg 'capital' must be greater than 1")

    # check number of periods
    if periods < 1:
        raise Exception("arg 'periods' must be greater than 1")
    elif periods > 15:
        nperiods = 15  # periods for return calculations
    else:
        nperiods = periods

    # --- CALCULATIONS
    # Statistics: max, percentil 85, average, percentil 15, min
    chart_periods = 15  # number of periods to plot
    df = pd.DataFrame(
        index=range(0, chart_periods + 1),
        columns=["min", "pc15", "avg", "pc85", "max"],
    )
    avg = data.mean()  # average return
    df.iloc[0] = [1, 1, 1, 1, 1]  # portfolio capital starts at 1
    # iterate all investment periods
    for n in range(1, chart_periods + 1):
        # calc all rolling cummulative returns for that investment period
        df_cumrets = rolling_cumrets(data, periods=n)
        # populate df with the required statistics
        df.loc[n, "min"] = df_cumrets.min()
        df.loc[n, "pc15"] = df_cumrets.quantile(0.15)
        df.loc[n, "avg"] = np.power(1 + avg, n)  # Cf = Ci (1 + r) ^ n
        df.loc[n, "pc85"] = df_cumrets.quantile(0.85)
        df.loc[n, "max"] = df_cumrets.max()

    # --- CREATE CHART
    # plot lines for each statistic
    ax.plot(df.index, df["max"], color="teal", linewidth=2)
    ax.plot(df.index, df["pc85"], color="teal", linewidth=0.5)
    ax.plot(df.index, df["avg"], color="grey", linewidth=2)
    ax.plot(df.index, df["pc15"], color="darkred", linewidth=0.5)
    ax.plot(df.index, df["min"], color="darkred", linewidth=2)

    # fill areas between plots
    x = list(df.index)
    y1, y2 = list(df["max"]), list(df["pc85"])
    y3 = list(df["avg"])
    y4, y5 = list(df["pc15"]), list(df["min"])

    ax.fill_between(x, y1, y2, facecolor="teal", alpha=0.50, zorder=5)
    ax.fill_between(x, y2, y3, facecolor="teal", alpha=0.25, zorder=5)
    ax.fill_between(x, y3, y4, facecolor="darkred", alpha=0.25, zorder=5)
    ax.fill_between(x, y4, y5, facecolor="darkred", alpha=0.50, zorder=5)

    # title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Years Invested", size=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value", size=12, fontweight="bold")

    # x axis
    ax.set_xticks(range(0, 16))
    ax.set_xlim(0, 15)

    # y axis
    ax.set_yscale("log")  # we want log axis to account for percent growth
    ax.set_ylim(0.1, 10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%0.1f"))

    # grid: vertical major lines, horizontal major and minor lines
    ax.xaxis.grid(which="major", color="lightgrey", linestyle="solid", lw=0.7)
    ax.yaxis.grid(which="both", color="lightgrey", linestyle="solid", lw=0.7)

    # --- ADD STATISTICS
    # marker parameters
    mk = "D"  # marker type
    ms = 60  # markers size
    mew = 0.8  # markers edge size
    # z = 6  # z-order
    mk_xpos = nperiods  # x-position

    # markers y-positions
    mk_max = df["max"][nperiods]
    mk_pc85 = df["pc85"][nperiods]
    mk_avg = df["avg"][nperiods]
    mk_pc15 = df["pc15"][nperiods]
    mk_min = df["min"][nperiods]

    # add vertical line at n periods invested
    ax.axvline(x=nperiods, ymin=0, ymax=1, c="dimgray", ls="solid", lw=2)

    # draw markers for statistics
    ax.scatter(
        mk_xpos,
        mk_max,
        marker=mk,
        s=ms,
        color="teal",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Max: {round(mk_max * capital):,}",
    )
    ax.scatter(
        mk_xpos,
        mk_pc85,
        marker=mk,
        s=ms,
        color="paleturquoise",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Strecht: {round(mk_pc85 * capital):,}",
    )
    ax.scatter(
        mk_xpos,
        mk_avg,
        marker=mk,
        s=ms,
        color="gray",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Average: {round(mk_avg * capital):,}",
    )
    ax.scatter(
        mk_xpos,
        mk_pc15,
        marker=mk,
        s=ms,
        color="darksalmon",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Baseline: {round(mk_pc15 * capital):,}",
    )
    ax.scatter(
        mk_xpos,
        mk_min,
        marker=mk,
        s=ms,
        color="darkred",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Min: {round(mk_min * capital):,}",
    )

    # legend
    num_cols = 5  # default columns
    fontsize = 10  # default font size

    # change defaults depending on fig width
    fig_width = plt.gcf().get_size_inches()[0]  # get figure width
    if fig_width < 7:
        fontsize = 8
        num_cols = 2
    elif fig_width < 8:
        fontsize = 8

    ax.legend(
        ncol=num_cols,
        loc="lower center",
        shadow=True,
        fontsize=fontsize,
        scatteryoffsets=[0.5],
        handlelength=1,
        bbox_to_anchor=(0.5, 0.05),
    )

    # create dictionary and return
    d = {
        "max": mk_max,
        "pc85": mk_pc85,
        "avg": mk_avg,
        "pc15": mk_pc15,
        "min": mk_min,
    }
    return ax, d, df_cumrets


# -----------------------------------------------------------------------
def portfolio_growth(
    data,
    ax=None,
    initial=10_000,
    annual_in=10_000,
    goal=1_000_000,
    title="Portfolio Growth",
):
    """
    This function creates a chart that shows the performance of the portfolio
    for every historical start date. The darker the color, the older de start
    date. The chart also shows the minimum and maximum years that were needed
    in the portfolio history to achieve the goal value.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data: pandas Time Series
        A sequence of returns for a given portfolio

    ax: Axes
        The axes to draw the chart

    start_val: scalar
        Initial portfolio capital

    annual_contrib: scalar
        Annual contribution to portfolio capital

    goal: scalar
        Target capital value

    title: str
       Plot title (default: 'Portfolio Growth')

    Returns
    -------
    ax, d, pf_growth: list

        ax, the axes with the chart
        d, dictionary with statistics {'min years', 'max years'}
        pf_growth, pandas.DataFrame with all historical portfolio growths

    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check money quantities
    if initial <= 0:
        Exception('argument "start_val" must be positive')
    if annual_in <= 0:
        Exception('argument "annual_contrib" must be positive')
    if goal <= 0:
        Exception('argument "goal" must be positive')

    # --- DEFINITIONS
    chart_periods = 30  # number of periods to show in chart
    nperiods = len(data.index)  # number of periods to calculate
    years_min, years_max = nperiods, 0  # min, max years to achieve goal
    ini_val = round(initial / 1_000)  # initial value in thousands
    contrib = round(annual_in / 1_000)  # annual contribution in thousands
    pf_goal = round(goal / 1_000)  # goal in thousands

    # create colormap to use in chart
    cmap = sequential_cmap(POS_CLR)
    # from new colormap create a list of colors for the cycler
    cycler_clrs = cmap(np.linspace(0, 0.6, nperiods))
    plt.gca().set_prop_cycle(plt.cycler(color=cycler_clrs))

    # --- CALCULATIONS
    # create NaN dataframe to hold portfolio growth values
    #   - columns will be the equivalent to the investment starting year
    #   - rows will be the number of invested years
    pf_growth = pd.DataFrame(
        np.nan,
        index=range(0, nperiods + 1),
        columns=list(map(str, list(range(0, nperiods + 1)))),
    )

    # calculate portfolio values starting at each year and till the end of data
    # eg: column 0 will be equivalent to 1970 investment start; then each row
    # will hold the porfolio value for an investment period (row 10 will have
    # portfolio value after 10 years invested); then column 2 will hold values
    # for portfolio started in 1971, and so on...

    # iterate columns -> each column means an investment start year
    for m in range(0, nperiods + 1):
        # create NaN values Series to hold column values
        # (NaN instead of 0 because 0 will be plotted by matplotlib functions)
        ser_val = pd.Series(data=np.nan, index=range(0, nperiods + 1))
        ser_val.iloc[0] = ini_val  # year 0 starts with initial portfolio value
        # iterate rows -> each row means an investment period
        for n in range(1, nperiods + 1 - m):
            # value(n) = (1 + return(n-1)) * value(n-1) + contribution
            # assume contribution is made at the end of each period
            ser_val.iloc[n] = (1 + data[m:].iloc[n - 1]) * ser_val.iloc[
                n - 1
            ] + contrib
        # each column is assigned the resulting Series, which will have one
        # element less in each iteration
        pf_growth[str(m)] = ser_val

    # find min, max years to achieve goal
    years = []
    for m in range(0, nperiods + 1):
        for n in range(0, nperiods + 1):
            if pf_growth[str(m)][n] >= pf_goal:
                years.append(n)
                break
    years_min = min(years)
    years_max = max(years)

    # --- CREATE CHART
    # plot only the periods shown in the chart (chart_periods) plus one
    ax.plot(
        pf_growth.index[: chart_periods + 1],
        pf_growth[: chart_periods + 1],
        lw=2,
    )
    # plot goal line and text
    ax.axhline(
        y=pf_goal, xmin=0, xmax=1, color="grey", linestyle="solid", lw=2
    )

    # -- Text Labels
    # label with min years will be limited to 30; if higher -> add '+'
    suffix = ""
    if years_min > chart_periods:
        years_min = chart_periods
        suffix = "+"

    ax.text(
        years_min,
        pf_goal,
        str(years_min) + suffix,
        fontsize=10,
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="darkorange",
            edgecolor="darkorange",
            alpha=1,
        ),
    )

    # label with max years will be limited to 30; if higher -> add '+'
    suffix = ""
    if years_max > chart_periods:
        years_max = chart_periods
        suffix = "+"

    ax.text(
        years_max,
        pf_goal,
        str(years_max) + suffix,
        fontsize=10,
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="darkorange",
            edgecolor="darkorange",
            alpha=1,
        ),
    )

    # label for goal horizontal line
    ax.text(
        0.2,
        pf_goal,
        f"Goal: {pf_goal:,.0f}",
        fontsize=10,
        color="white",
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="grey",
            edgecolor="grey",
            alpha=1,
        ),
    )

    # set title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Years Invested", size=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value", size=12, fontweight="bold")

    # x axis
    ax.set_xlim(0, chart_periods)
    ax.set_xticks(range(0, chart_periods + 1))

    # y axis
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter("{x:,.0f} K")

    # grid: vertical major lines, horizontal major lines
    ax.xaxis.grid(which="major", color="lightgrey", linestyle="solid", lw=0.7)
    ax.yaxis.grid(which="major", color="lightgrey", linestyle="solid", lw=0.7)

    # create dictionary and return
    d = {"min years": years_min, "max years": years_max}
    return ax, d, pf_growth


# -----------------------------------------------------------------------
def drawdowns(data, ax=None, title="Drawdowns"):
    """
    The Drawdowns chart maps every single portfolio loss from any high point
    along the way. The deepest drawdown is the largest compound loss for the
    portfolio regardless of start date. The longest drawdown is the longest
    amount of time that a particular portfolio fell below its initial value.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data: pandas.Series
        The anual returns of a portfolio

    ax: matplotlib.Axes
        The axes where the chart will be drawn

    title: str
        Title for the chart (default: 'Drawdowns')

    Returns
    -------
    ax, d, df_cumrets: list

        ax, the axes with the chart
        d, dictionary with statistics {'deepest_dd', 'longest_dd', 'ui'}
        df_cumrets, pandas dataframe with cummulative returns for all windows
    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # --- DEFINITIONS
    nperiods = len(data.index)  # number of periods in data
    chart_periods = 15  # number of periods to show in chart

    # colormap
    cmap = sequential_cmap(NEG_CLR)  # new cmap based on clr for neg values
    cycler_clrs = cmap(
        np.linspace(0, 0.6, nperiods)
    )  # list of colors from cmap
    plt.gca().set_prop_cycle(plt.cycler(color=cycler_clrs))  # apply to cycler

    # --- CALCULATIONS
    # -- Cummulative returns
    pf_rets = data.copy()
    pf_rets.reset_index(drop=True, inplace=True)
    pf_cumrets = pd.DataFrame()  # will hold cummulative returns
    # fill columns -> each one a consecutive investment start year
    for m in range(0, nperiods):
        col_data = pf_rets[m:]
        col_data.reset_index(drop=True, inplace=True)
        pf_cumrets = pd.concat(
            [pf_cumrets, (1 + col_data).cumprod()], axis=1, ignore_index=True
        )
    pf_cumrets = pf_cumrets - 1
    # add a first row with 0s
    zeros_row_df = pd.DataFrame(
        np.zeros((1, nperiods)), index=[0], columns=list(range(len(pf_rets)))
    )
    pf_cumrets = pd.concat([zeros_row_df, pf_cumrets], ignore_index=True)

    # -- Longest drawdown
    negs = pf_cumrets[pf_cumrets < 0]  # df with neg values
    negs.dropna(axis=1, how="all", inplace=True)  # drop cols with no negs
    # duration = idx_last - idx_first; store in pairs of (duration, column)
    durations = [
        (negs[n].last_valid_index() - negs[n].first_valid_index(), n)
        for n in negs.columns
    ]
    # Note: the for loop above works only if col names are ints
    max_dur = max(durations, key=lambda x: x[0])  # Note: returns 1st ocurrence
    longest_dd = max_dur[0] + 2  # +1: substraction idx reduces dd one year
    # +1: another year is needed to go positive

    # -- Deepest drawdown
    deepest_dd = pf_cumrets.min().min()

    # -- Ulcer Index
    ui = round(ulcer_index(data, type="returns"), 1)

    # --- CREATE CHART
    ax.plot(pf_cumrets, lw=2)

    # fill between plots and x axis
    ax.set_facecolor("gainsboro")  # color outside of fill area
    for n in range(0, len(pf_cumrets.columns)):
        if (pf_cumrets[n] < 0).sum() > 0:  # only columns with negative values
            ax.fill_between(
                pf_cumrets[n].index,
                pf_cumrets[n],
                fc="white",
                interpolate=False,
            )

    # title and axis labels
    #   - title will work as x axis label
    #   - x axis label will work as title
    ax.set_title("Years to Recover", size=12, fontweight="bold")
    ax.set_xlabel(title, size=16, fontweight="bold")
    ax.set_ylabel("Drowdown Percentage", size=12, fontweight="bold")

    # x axis
    ax.set_xlim(0, chart_periods)
    ax.set_xticks(range(0, chart_periods + 1))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False)

    # y axis
    ax.set_ylim(-1.0, 0.0)
    ax.set_yticks(np.linspace(-1.0, 0, 11))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # y labels to positive percentages
    ylabels = list(map(str, sorted((range(0, 110, 10)), reverse=True)))
    ylabels = [label + "%" for label in ylabels]
    ax.set_yticklabels(ylabels)

    # grid: vertical major lines, horizontal major and minor lines
    ax.xaxis.grid(which="major", color="silver", linestyle="solid", lw=0.7)
    ax.yaxis.grid(which="both", color="silver", linestyle="solid", lw=0.7)

    # deepest drawdown
    ax.axhline(
        y=deepest_dd, xmin=0, xmax=1, color="grey", lw=2, linestyle="solid"
    )
    ax.text(
        0.1,
        deepest_dd - 0.01,
        "Deepest: " + str(round(deepest_dd * -100)) + "%",
        fontsize=9,
        color="white",
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="grey",
            edgecolor="grey",
            alpha=1,
        ),
    )

    # longest drawdown
    ax.axvline(
        x=longest_dd, ymin=0, ymax=1, color="grey", lw=2, linestyle="solid"
    )
    ax.text(
        longest_dd + 0.05,
        -0.01,
        "Longest: " + str(longest_dd),
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="grey",
            edgecolor="grey",
            alpha=1,
        ),
    )

    # ulcer index
    ax.text(
        1,
        -0.9,
        "Ulcer Index: " + str(ui),
        fontsize=11,
        color="white",
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="Square, pad=0.2",
            facecolor="darkred",
            edgecolor="darkred",
            alpha=1,
        ),
    )

    # create dictionary with statistics and return
    d = {"deepest": deepest_dd, "longest": longest_dd, "ulcer index": ui}
    return ax, d, pf_cumrets


# -----------------------------------------------------------------------
def heatmap(data, ax=None, title="Heatmap"):
    """
    Shows the CAGR for all investing timeframes based on the year the
    investment started and how long the investment lasted. At the bottom of
    the chart there is a guide to interpret the color of the squares. Also,
    mouse hover on the chart will show the exact return for each individual
    square.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    ax: matplotlib.Axes (default=None)
        The axes where the chart will be drawn. If not provided, new axis will
        be created.

    data: pandas.Series
        Time Series with the anual returns of a portfolio, be them nominal,
        inflation adjusted, cost adjusted, etc. The labels for the y axis will
        be the year of each entry in the index.

    title: str
        Title for the chart (default: 'Drawdowns')

    Returns
    -------
    ax, pf_cagr: list

        ax, the axes with the chart
        pf_cagr, dataframe with the CAGRs calculated for the input data

    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(7, 10)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # --- DEFINITIONS
    nperiods = len(data)  # number of periods in data
    chart_cols = 35  # number of periods to show in x axis

    # --- colormap
    bounds = np.array([-3, 0, 3, 6, 9])  # 66% POS_CLR, 34% NEG_CLR
    # base cmaps for positive and negative returns
    cm_pos = sequential_cmap(POS_CLR).reversed()  # top POS_CLR, down WHITE
    cm_neg = sequential_cmap(NEG_CLR)  # top WHITE, down NEG_CLR
    # generate list of colors for new cmap (take into account 'bounds')
    pos_clrs = cm_pos(np.linspace(0.02, 0.8, 170))  # avoid full white
    neg_clrs = cm_neg(np.linspace(0.2, 1, 86))  # avoid full white
    # create new cmap and norm
    clr_range = np.vstack((neg_clrs, pos_clrs))  # unify color list
    cmap = colors.ListedColormap(clr_range, name="RedTeal")  # create colormap
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

    # --- CALCULATIONS
    # calculate CAGR matrix
    pf_cagr = pd.DataFrame()
    for m in range(1, nperiods + 1):
        # pf_cagr[m] = rolling_CAGR(data, periods=m)
        pf_cagr = pd.concat([pf_cagr, rolling_CAGR(data, periods=m)], axis=1)
    # convert to % format
    pf_cagr = round(pf_cagr * 100, 2)

    # --- CREATE CHART
    heatmap = ax.pcolor(
        pf_cagr.to_numpy(), cmap=cmap, norm=norm, ec="black", lw=0.5, snap=True
    )

    # -- Insert colorbar within its own axis (guide to heatmap colors)
    axins = inset_axes(
        ax,
        width="25%",
        height="1.8%",
        loc="lower right",
        bbox_to_anchor=(-0.1, 0.10, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.gcf().colorbar(
        heatmap,
        cax=axins,
        label="CAGR",
        extend="both",
        extendrect=True,
        drawedges=True,
        extendfrac="auto",
        orientation="horizontal",
        format=PercentFormatter(xmax=100, decimals=0),
    )
    # colorbar label
    cbar.ax.xaxis.set_label_position("top")
    # colorbar tick labels
    for label in cbar.ax.xaxis.get_ticklabels():
        label.set_fontsize(8)

    # titles and axis labels
    ax.set_title(title, size=16, fontweight="bold", pad=15)
    ax.set_xlabel("Years Held", size=10, labelpad=7, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Start Year", size=10, fontweight="bold", loc="center")

    # x axis
    max_cols = min(chart_cols, pf_cagr.shape[1])
    ax.xaxis.tick_top()
    # rearrange ticks so that tick labels appear centered
    ax.set_xticks(np.arange(max_cols) + 0.5, minor=False)
    ax.set_xlim(0, max_cols)
    # labels = num years invested, one out of every five
    col_names = []
    for n in range(1, max_cols + 1):
        if n % 5 == 0:
            col_names.append(str(n))
        else:
            col_names.append("")
    ax.tick_params(axis="x", top=False)
    ax.set_xticklabels(col_names, fontdict={"fontsize": 8})

    # y axis
    ax.invert_yaxis()
    # rearrange ticks so that tick labels appear centered
    ax.set_yticks(np.arange(pf_cagr.shape[0]) + 0.5, minor=False)
    # make all labels (=years) visible
    ax.tick_params(axis="y", left=False)
    ax.set_yticklabels(pf_cagr.index, fontdict={"fontsize": 8})

    # --- EVENT HANDLING
    # create anotation empty box and set invisible
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(1, -1),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="whitesmoke"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)
    row, col = -1, -1  # will keep last mouse position

    # function to handle events
    def on_move(event):
        """
        Shows/hides annotation labels with the CAGR value corresponding to
        the mouse position in the heatmap.
        """

        nonlocal row, col  # used to store last mouse position

        # check if event is inside axes but not inside colorbar axes
        if event.inaxes != ax:
            # reset position to a value of (x, y) outside the axis
            row, col = -1, -1
            # and set any visible annotation to invisible
            if annot.get_visible():
                annot.set_visible(False)
                plt.gcf().canvas.draw_idle()

            return

        # get position (shift 0.5% because axis ticks were shifted)
        new_row = abs(int(event.ydata - 0.005))
        new_col = abs(int(event.xdata - 0.005))

        # proceed only if mouse moved to a new row or col
        if new_row == row and new_col == col:
            return

        # set last annotation invisible
        if annot.get_visible():
            annot.set_visible(False)
            plt.gcf().canvas.draw_idle()

        # get value for new position and check for NaN
        cagr = pf_cagr.iloc[new_row, new_col]
        if np.isnan(cagr):  # NaN values = white area in the axis
            return
        # update annotation
        annot.xy = (new_col, new_row)
        annot.set_text(str(cagr))
        annot.set_visible(True)
        plt.gcf().canvas.draw_idle()

        # update last position
        row, col = new_row, new_col

        # end function
        return

    # connect function with event
    binding_id = plt.connect("motion_notify_event", on_move)  # noqa: F841

    # end chart: return heatmap and CAGR data frame
    return ax, pf_cagr


def long_term_returns(data, ax=None, periods=10, title="Long Term Returns"):
    """
    Draws a chart showing the full range of inflation-adjusted CAGRs of a
    portfolio based on how long you invested.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data: pandas.Series
        The data to plot. Can be returns or already calculated rolling returns.
        In this last case, 'isrolling' should be set to True.

    ax: Axe
        The axes to plot the data

    periods: int
        The number of periods for the rolling window calculation (default: 10)
        This parameter will be ignored if 'isrolling' is True.

    title: str
        The title desired for the chart (default: 'Rolling Returns')

    Returns
    -------
    ax, d, df_cagr: list

        ax, the axes with the chart
        d, dictionary with statistics {'max', 'pc85', 'med', 'pc15', 'min'}
        df_cagr, pandas series with the rolling CAGRs
    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check number of periods
    if periods < 1:
        raise Exception("arg 'periods' must be greater than 1")
    elif periods > 15:
        nperiods = 15  # periods for return calculations
    else:
        nperiods = periods

    # --- CALCULATIONS
    # Statistics: max, percentil 85, average, percentil 15, min
    chart_periods = 15  # number of periods to plot
    df = pd.DataFrame(
        index=range(0, chart_periods + 1),
        columns=["min", "pc15", "med", "pc85", "max"],
    )
    # iterate all investment periods
    for n in range(1, chart_periods + 1):
        # calculate all rolling CAGRs for that investment period
        df_cagr = rolling_CAGR(data, n)
        # populate df with the statistics
        df["min"][n] = df_cagr.min()
        df["pc15"][n] = df_cagr.quantile(0.15)
        df["med"][n] = df_cagr.median()
        df["pc85"][n] = df_cagr.quantile(0.85)
        df["max"][n] = df_cagr.max()

    # --- CREATE CHART
    # plot lines for each statistic
    ax.plot(df.index, df["max"], color="teal", linewidth=2)
    ax.plot(df.index, df["pc85"], color="teal", linewidth=0.5)
    ax.plot(df.index, df["med"], color="grey", linewidth=2)
    ax.plot(df.index, df["pc15"], color="darkred", linewidth=0.5)
    ax.plot(df.index, df["min"], color="darkred", linewidth=2)

    # fill areas between plots
    x = list(df.index)
    y1, y2 = list(df["max"]), list(df["pc85"])
    y3 = list(df["med"])
    y4, y5 = list(df["pc15"]), list(df["min"])

    ax.fill_between(x, y1, y2, facecolor="teal", alpha=0.50, zorder=5)
    ax.fill_between(x, y2, y3, facecolor="grey", alpha=0.25, zorder=5)
    ax.fill_between(x, y3, y4, facecolor="grey", alpha=0.25, zorder=5)
    ax.fill_between(x, y4, y5, facecolor="darkred", alpha=0.50, zorder=5)

    # title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Years Invested", size=12, fontweight="bold")
    ax.set_ylabel("CAGR", size=12, fontweight="bold")

    # x axis
    ax.set_xticks(range(0, 16))
    ax.set_xlim(1, 15)

    # y axis
    yticks = [n / 100 for n in range(-50, 60, 10)]
    ax.set_ylim(-0.50, 0.50)
    ax.set_yticks(yticks)

    # y axis labels as percent
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # grid: vertical major lines, horizontal major and minor lines
    ax.xaxis.grid(which="major", color="lightgrey", linestyle="solid", lw=0.7)
    ax.yaxis.grid(which="major", color="lightgrey", linestyle="solid", lw=0.7)

    # --- ADD STATISTICS
    # marker parameters
    mk = "D"  # marker type
    ms = 60  # markers size
    mew = 0.8  # markers edge size
    # z = 6  # z-order
    mk_xpos = nperiods  # x-position

    # markers y-positions
    mk_max = df["max"][nperiods]
    mk_pc85 = df["pc85"][nperiods]
    mk_med = df["med"][nperiods]
    mk_pc15 = df["pc15"][nperiods]
    mk_min = df["min"][nperiods]

    # add vertical line at n periods invested
    ax.axvline(x=nperiods, ymin=0, ymax=1, c="dimgray", ls="solid", lw=2)
    # ax.text(nperiods, 0.49, str(nperiods) + ' years', color='white',
    #         fontsize=9, ha='left', va='top',
    #         bbox=dict(facecolor='grey', edgecolor='grey', alpha=1))

    # draw markers for statistics
    ax.scatter(
        mk_xpos,
        mk_min,
        marker=mk,
        s=ms,
        color="darkred",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Min: {round(mk_min * 100, 1)}%",
    )
    ax.scatter(
        mk_xpos,
        mk_pc15,
        marker=mk,
        s=ms,
        color="darksalmon",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Baseline: {round(mk_pc15 * 100, 1)}%",
    )
    ax.scatter(
        mk_xpos,
        mk_med,
        marker=mk,
        s=ms,
        color="gray",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Median: {round(mk_med * 100, 1)}%",
    )
    ax.scatter(
        mk_xpos,
        mk_pc85,
        marker=mk,
        s=ms,
        color="paleturquoise",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Strecht: {round(mk_pc85 * 100, 1)}%",
    )
    ax.scatter(
        mk_xpos,
        mk_max,
        marker=mk,
        s=ms,
        color="teal",
        edgecolors="black",
        linewidths=mew,
        zorder=6,
        label=f"Max: {round(mk_max * 100, 1)}%",
    )

    # legend
    num_cols = 5  # default columns
    fontsize = 10  # default font size

    # change defaults depending on fig width
    fig_width = plt.gcf().get_size_inches()[0]  # get figure width
    if fig_width < 7:
        fontsize = 8
        num_cols = 2
    elif fig_width < 8:
        fontsize = 8

    ax.legend(
        ncol=num_cols,
        loc="lower center",
        shadow=True,
        fontsize=fontsize,
        scatteryoffsets=[0.5],
        handlelength=1,
        bbox_to_anchor=(0.5, 0.05),
    )

    # create dictionary and return
    d = {
        "max": mk_max,
        "pc85": mk_pc85,
        "med": mk_med,
        "pc15": mk_pc15,
        "min": mk_min,
    }
    return ax, d, df_cagr


def safe_withdrawal_rates(
    data, ax=None, years=30, preserve=0, title="Safe Withdrawal Rates"
):
    """This function creates a chart that shows the safe withdrawal rate for
    any asset allocation over a variety of retirement durations based on
    real-life sequence of returns. Those looking to retire early or leave
    money to heirs can also see the perpetual withdrawal rate that protected
    the original inflation-adjusted principal.

    For a detailed explanation visit https://portfoliocharts.com

    Parameters
    ----------
    data: pandas Time Series
        A sequence of returns for a given portfolio

    ax: Axes
        The axes to draw the chart

    years: positive integer between 10 and 40
        Life expectancy after retirement

    preserve: scalar
        Percentage of initial portfolio to preserve after the end of the life
        expectancy period. Values can be between 0 and 100.

    title: str
       Plot title (default: 'Portfolio Growth')

    Returns
    -------
    ax, swr: list

        ax, the axes with the chart
        swr, safe withdrawal rate

    """

    # --- CHECK ARGUMENTS
    # if a figure does not exist, create fig and axis
    if not plt.get_fignums():
        plt.gcf()
        plt.gcf().set_size_inches(10, 6)
        ax = plt.gca()
    # if fig exists but axis not passed, create one
    elif not ax:
        ax = plt.gca()

    # check years
    retirement = 30  # retirement length (default = 30)
    if years < 1:
        retirement = 1
    elif years > 40:
        retirement = 40
    else:
        retirement = years

    # check % preservation
    goal = 0  # % of initial portfolio remaining at the end (default = 0)
    if 0 <= preserve <= 100:
        goal = preserve / 100
    elif preserve > 100:
        goal = 1

    # --- DEFINITIONS
    years_min, years_max = 10, 40  # max/min years to show in chart
    wr_min, wr_max = 0.0, 0.1  # min/max swr to show in chart
    nperiods = len(data.index)  # number of periods to calculate

    # create colormap to use in chart
    cmap = sequential_cmap(POS_CLR)
    # from new colormap create a list of colors for the cycler
    cycler_clrs = cmap(np.linspace(0, 0.6, nperiods))
    plt.gca().set_prop_cycle(plt.cycler(color=cycler_clrs))

    # functions
    def safe_rates(data, end_pct):
        """Calculates safe withdrawal rates for all possible retirement periods
        in a Series of returns."""

        # add row on the top with return value = 0
        pf_returns = pd.concat([pd.Series([0]), data], ignore_index=True)
        # compute cumulative growth
        pf_cumrets = (1 + pf_returns).cumprod()
        # create NaN dataframe for withdrawal rates
        pf_wr = pd.Series(np.nan, index=pf_returns.index)
        # row 0 -> 0 years retirement length, makes no sense
        # for the rest of rows, process until first NaN value
        for n in range(1, pf_cumrets.count()):
            subset = pf_cumrets.iloc[:n]
            N = len(subset)
            pf_wr[n] = hmean(subset) / N * (1 - end_pct / pf_cumrets.iloc[n])

        # remove first row (NaN value)
        pf_wr = pf_wr[1:].reset_index(drop=True)
        return pf_wr

    # def init_param_guess(data):
    #     """ """
    #     n = data.count()
    #     x = data.index[:n]
    #     y = data[:n]
    #     p = np.polyfit(x, np.log(y), 1, w=y**2)
    #     # p = np.polyfit(x, np.log(y), 1)
    #     a = np.exp(p[1])
    #     b = -p[0]
    #     print(f'b = {b}, a = {a}')

    #     return a, b

    def exp_decay(x, a, b, c):
        return a * np.e ** (-b * x) + c

    # --- CALCULATIONS
    # build DataFrame with series of portfolio returns for all periods
    pf_returns = pd.DataFrame()
    for n in range(len(data.index)):
        pf_returns = pd.concat(
            [pf_returns, data.shift(-n)], axis=1, ignore_index=True
        )
    # change index labels to starting year in string format
    pf_returns.index = list(map(str, list(data.index.year)))
    # transpose to have starting year as column name
    pf_returns = pf_returns.transpose()
    # drop columns with less than 'years_min' periods of data
    pf_returns = pf_returns.iloc[:, :-years_min]

    # calculate SWRs & PWRs for each starting year
    pf_swr = pd.DataFrame()
    pf_pwr = pd.DataFrame()
    for col in pf_returns.columns:
        pf_swr = pd.concat([pf_swr, safe_rates(pf_returns[col], goal)], axis=1)
        pf_pwr = pd.concat([pf_pwr, safe_rates(pf_returns[col], 1)], axis=1)
    # rename columns
    pf_swr = pf_swr.set_axis(pf_returns.columns, axis=1)
    pf_pwr = pf_pwr.set_axis(pf_returns.columns, axis=1)

    # --- CURVE FITTING
    fit_curve = True
    if fit_curve:
        study_year = "2000"
        test_year = pf_swr.loc[10:, study_year]
        test_year2 = pf_pwr.loc[10:, study_year]
        points = test_year.count()
        print(f"Points: {points}")
        test_data = test_year[:points]
        test_data2 = test_year2[:points]
        X = test_data.index[:points]
        Y = test_data[:points]
        Z = test_data2[:points]

        # curve_fit() with initial guess
        a, b, c = 0.3, 0.15, Z.iloc[-1]
        a = 1 - c
        # a_low, a_high = a - 0.00000001, a + 0.00000001
        # c_low, c_high = c - 0.00000001, c + 0.00000001
        popt, pcov = curve_fit(
            exp_decay, X, Y, [a, b, c], bounds=([0, 0, 0], [1, 1, 0.1])
        )
        a_fitted, b_fitted, c_fitted = popt
        print(f"Guess: a = {a}          b = {b}          c = {c}")
        print(f"Fit:   a = {a_fitted}   b = {b_fitted}   c = {c_fitted}")

        # plot fitted curves
        x_fitted = np.linspace(Z.index[-1], 50, 50 - Z.index[-1])
        y_fitted = exp_decay(x_fitted, a_fitted, b_fitted, c_fitted)
        ax.scatter(
            pf_swr[study_year].index, pf_swr[study_year], label="Raw data"
        )
        ax.scatter(x_fitted, y_fitted, label="Fitted", c="r")
        ax.axhline(c, xmin=0, xmax=50, c="k", ls="dotted", lw=2)
        ax.set_ylim(0.0, 0.1)
        ax.set_xlim(0, 50)

        ax.legend()

        return ax, 0, 1, 2
    # --- CREATE CHART
    # get values for swr & pwr
    swr = pf_swr.iloc[retirement].min()
    col = pf_swr.iloc[retirement].idxmin()
    pwr = pf_pwr.loc[retirement, col]

    # plot all periods
    ax.plot(pf_swr.index, pf_swr, lw=2)
    # highlight portfolio with min wr (= redraw in a diff color)
    ax.plot(pf_swr[col].index, pf_swr[col], lw=2, color="darkorange")
    # plot pwr for portfolio with min wr
    ax.plot(pf_pwr[col].index, pf_pwr[col], lw=2, color="olive")

    # fill areas between plots
    x = list(pf_pwr[col].index)
    y1, y2 = pf_swr[col].to_list(), pf_pwr[col].to_list()
    y3 = 0

    ax.fill_between(x, y1, y2, facecolor="olive", alpha=0.15, zorder=5)
    ax.fill_between(x, y2, y3, facecolor="olive", alpha=0.35, zorder=5)

    # draw vertical line at retirement length
    if years_min <= years <= years_max:
        ax.axvline(retirement, ymin=0, ymax=1, c="dimgray", ls="solid", lw=2)

        # draw markers for swr
        if wr_min <= swr <= wr_max:
            ax.scatter(
                retirement,
                swr + 0.0003,
                marker="D",
                s=50,
                color="orange",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
            ax.text(
                retirement,
                swr + 0.005,
                f"{round(swr * 100, 1)}% SWR",
                fontsize=8,
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="square",
                    facecolor="darkorange",
                    edgecolor="darkorange",
                    alpha=1,
                ),
            )
        # draw markers for pwr
        if wr_min <= pwr <= wr_max:
            ax.scatter(
                retirement,
                pwr + 0.0003,
                marker="D",
                s=50,
                color="olive",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
            ax.text(
                retirement,
                pwr - 0.005,
                f"{round(pwr * 100, 1)}% PWR",
                fontsize=8,
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="square",
                    facecolor="olive",
                    edgecolor="olive",
                    alpha=1,
                ),
            )

    # title and axis labels
    ax.set_title(title, size=14, fontweight="bold")
    ax.set_xlabel("Retirement Length", size=12, fontweight="bold")
    ax.set_ylabel("Withdrawal Rate", size=12, fontweight="bold")

    # axis limits
    ax.set_xlim(years_min, years_max)
    ax.set_ylim(wr_min, wr_max)
    ax.set_yticks(np.arange(wr_min, wr_max + 0.01, 0.01))

    # y axis format as percentages
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # grid: vertical major lines, horizontal major and minor lines
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.grid(which="both", color="silver", linestyle="solid", lw=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(which="both", color="silver", linestyle="solid", lw=0.6)

    # return data & end safe_withdrawal_rates()
    return ax, swr, pwr, col
