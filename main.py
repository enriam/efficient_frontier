import matplotlib.pyplot as plt
from src.efficient_frontier import plot_two_asset_efficient_frontier, plot_two_portfolios

fig, ax = plt.subplots(figsize=(9, 6))

plot_two_asset_efficient_frontier(
    mean_return_1=0.08,
    mean_return_2=0.12,
    variance_1=0.04,
    variance_2=0.09,
    correlation=0.25,
    ax=ax,
    asset1_label="Bond ETF",
    asset2_label="Equity ETF",
    portfolio_label="Possible portfolios",
    x_axis_label="Volatility",
    y_axis_label="Expected annual return",
    plot_title="Diversification Effect by Correlation",
    asset1_color="navy",
    asset2_color="darkorange",
    min_variance_color="crimson",
)

plt.show()

# pf1 = [0.08, 0.12, 0.04, 0.09, 0.25, ax]
# pf2 = [0.80, 0.12, 0.06, 0.1, -0.25, ax]

# plot_two_portfolios(pf1, pf2)
# plt.show()