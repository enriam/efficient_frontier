from pathlib import Path

import numpy as np
import pandas as pd


def read_price_series(
    file_path: str,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.Series:
    """
    Read a CSV file and return a price series indexed by date.
    """
    path = Path(file_path)

    if not path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    data = pd.read_csv(path)

    if date_column not in data.columns:
        msg_0 = f"Missing '{date_column}' column in file: {file_path}"
        raise ValueError(msg_0)

    if price_column not in data.columns:
        msg_1 = f"Missing '{price_column}' column in file: {file_path}"
        raise ValueError(msg_1)

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column)
    data = data.set_index(date_column)

    return data[price_column]


def calculate_returns(
    price_series: pd.Series, method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from a price series.
    """
    if method == "simple":
        return price_series.pct_change()

    if method == "log":
        return np.log(price_series / price_series.shift(1))

    msg = "Method must be either 'simple' or 'log'"
    raise ValueError(msg)


def load_asset_returns(
    asset_files: dict[str, str],
    date_column: str = "Date",
    price_column: str = "Close",
    return_method: str = "simple",
    *,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Read asset price files, calculate returns, and combine them into one
    DataFrame.
    """
    returns_data = {}

    for asset_name, file_path in asset_files.items():
        prices = read_price_series(
            file_path=file_path,
            date_column=date_column,
            price_column=price_column,
        )

        returns = calculate_returns(prices, return_method)
        returns_data[asset_name] = returns

    returns_df = pd.DataFrame(returns_data)

    if drop_missing:
        returns_df = returns_df.dropna()

    return returns_df
