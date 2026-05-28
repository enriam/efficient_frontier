import matplotlib.pyplot as plt
from src.efficient_frontier import plot_two_asset_efficient_frontier

fig, ax = plt.subplots(figsize=(9, 6))

plot_two_asset_efficient_frontier(
    mean_return_1=0.08,
    mean_return_2=0.12,
    variance_1=0.04,
    variance_2=0.09,
    correlation=0.25,
    ax=ax,
)

plt.show()