import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --- STREAMLIT UTILS -----------------------------------------
def check_maintenance_status():
    if st.secrets.under_maintenance:
        maintenance_img = html_img_with_href(
            "images/under_maintenance.jpg",
            "",
            width="100%",
            height="100%",
        )
        st.write(maintenance_img, unsafe_allow_html=True)
        st.stop()


def footnote():
    st.write("&nbsp;")
    st.write("&copy; 2026 Enrique Amat [enriam.codes@gmail.com]")


def html_img_with_href(img_file, url, width, height):
    if url == "":
        url = None

    try:
        with Path(img_file).open("rb") as f:
            data = f.read()
            img = base64.b64encode(data).decode()
    except FileNotFoundError:
        html_code = "" if url is None else f'<a href="{url}">{url}</a>'

    else:
        if url is None:
            html_code = f"""<img src="data:image/png;base64,{img}"
                style="width:{width}; height:{height}"/>"""
        else:
            html_code = f"""
            <a href="{url}">
                <img src="data:image/png;base64,{img}"
                style="width:{width}; height:{height}"/>
            </a>"""

    return html_code


# --- FINANCE UTILS -----------------------------------------
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


def calculate_asset_returns(
    asset_files: dict[str, str],
    date_column: str = "date",
    price_column: str = "close",
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


def load_asset_prices(
    asset_files: dict[str, str],
    date_column: str = "date",
    price_column: str = "close",
    *,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Read asset price files and combine them into one DataFrame.
    """
    prices = {}

    for asset_name, file_path in asset_files.items():
        data = read_price_series(
            file_path=file_path,
            date_column=date_column,
            price_column=price_column,
        )
        prices[asset_name] = data

    prices_df = pd.DataFrame(prices)
    if drop_missing:
        prices_df = prices_df.dropna()

    return prices_df
