## Two-Asset Efficient Frontier Explorer

This Streamlit app demonstrates how the efficient frontier of a two-asset portfolio changes as the correlation between the assets varies.

Users can adjust the expected return, variance, and correlation of two assets and immediately see the resulting risk-return curve plotted with Matplotlib. The visualization highlights the individual assets, the full set of possible portfolio combinations, and the minimum-variance portfolio.

The goal of the app is to make portfolio diversification intuitive: when asset correlation decreases, the frontier bends further to the left, showing how combining imperfectly correlated assets can reduce portfolio risk. When correlation increases, the diversification benefit becomes smaller, and the portfolio frontier approaches a straight line between the two assets.

This app is intended as an educational tool for learning basic portfolio theory, risk-return tradeoffs, and the role of correlation in diversification.