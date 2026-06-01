# ruff: noqa: E501

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import src.ef_charts as ef
from src import utils

# from src.ef_chart import Asset, plot_two_asset_ef

# --- Global Config (this must be the first streamlit command in the page)
st.set_page_config(
    page_title="Frontera Eficiente",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"About": "&copy; 2026 Enrique Amat (enriam.codes@gmail.com)"},
)

# --- Maintenance check
utils.check_maintenance_status()

# --- Page Deployment
st.title("Deversificación y correlación")

text01 = """
El concepto de "diversificación" siempre ha estado presente en los inversores de forma intuitiva, incluso forma parte de la cultura popular: "nunca pongas todos los huevos en la misma cesta". Lo que Markowitz y la *Modern Portfolio Theory (MPT)* aportaron fue formalizar matemáticamente cómo hacerlo y demostrar que el riesgo de una cartera de inversión no depende solo de la volatilidad de cada activo, sino también de cómo se mueven unos en relación a los otros. Esto último se conoce formalmente como **correlación** entre activos. 

Antes de ver cómo afecta la correlación entre activos a una cartera de inversión es muy útil entender el concepto de "frontera eficiente".

## La frontera eficiente

Supongamos que tenemos dos activos de inversión A y B, el primero representando a una acción y el segundo a un bono. Cada uno tiene su rentabilidad y su riesgo, por ejemplo los de la tabla siguiente: 

"""

text02 = """
Si variamos los pesos de cada activo podemos crear distintas carteras: 60%-40%, 70%-30%, 100%-0%. Cada cartera nos dará una rentabilidad distinta y nos expondrá a un riesgo distinto. Esos valores de rentabilidad y riesgo se calculan con las ecuaciones de Markowitz. Y si representamos todas esas combinaciones de pesos en una gráfica en la que el eje X represente la rentabilidad y el eje Y el riesgo, obtendremos la frontera eficiente.

Puedes seleccionar los pesos de los activos y verás cómo cambian la rentabilidad y el riesgo de tu cartera en función de ellos. 
"""

# --- Portfolio 1 -------------------------------------------
bond = ef.Asset(mean=0.05, var=0.012)
stock = ef.Asset(mean=0.15, var=0.1)
corr1 = 0.0

# Portfolio numbers
bond_ret = round(bond.mean * 100, 1)
bond_std = round(np.sqrt(bond.var) * 100, 1)
stock_ret = round(stock.mean * 100, 1)
stock_std = round(np.sqrt(stock.var) * 100, 1)

table = f"""
|             |Rentabilidad |Riesgo     |
|-------------|-------------|-----------|
|**Bono**     |{bond_ret}   |{bond_std} |
|**Acción**   |{stock_ret}  |{stock_std}|
"""
st.markdown(text01)
st.markdown(table)
st.markdown(text02)
st.write("")

# Chart 1: changing weights
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    st.write("###### Pesos Bono - Acción")
    weight = st.slider(
        label="Pesos",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        format="%d%%",
        label_visibility="collapsed",
        # width="stretch",
    )


with col_a2:
    st.metric("**Bono**", f"{weight} %")
with col_a3:
    st.metric("**Acción**", f"{100 - weight} %")


b_weight = weight / 100
s_weight = 1 - b_weight

b_std = np.sqrt(bond.var)
s_std = np.sqrt(stock.var)
covariance = corr1 * b_std * s_std

pf_mean = b_weight * bond.mean + s_weight * stock.mean
pf_var = (
    b_weight**2 * bond.var
    + s_weight**2 * stock.var
    + 2 * b_weight * s_weight * covariance
)

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr1,
    ax=ax1,
    asset1_label="Bono 100%",
    asset2_label="Acción 100%",
    portfolio_label="Carteras eficientes",
    portfolio_color="darkorange",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="Cartera de riesgo mínimo",
    inefficient_label="Carteras ineficientes",
    plot_title="Frontera Eficiente",
    asset1_color="green",  # bond
    asset2_color="crimson",  # stock
    min_variance_color="navy",
)

ax1 = ef.plot_two_asset_portfolio(
    asset_1=bond,
    asset_2=stock,
    weight_1=b_weight,
    weight_2=s_weight,
    correlation=corr1,
    ax=ax1,
    point_label="Cartera seleccionada",
    point_color="darkorange",
    point_marker="D",
)


st.pyplot(fig1, use_container_width=True)

# --- Portfolio 2 -------------------------------------------
st.write("###### Pesos Bono - Acción")
corr2 = st.slider(
    label="Corr",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    format="%0.1f",
    label_visibility="collapsed",
)

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=0.0,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label="Cartera $\\rho$ = 0.0",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
    efficient_color="darkgrey",
    inefficient_color="darkgrey",
    min_variance_color="navy",
)

ax2 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr2,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {corr2}",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="Cartera de mínima volatilidad",
    efficient_color="darkorange",
    inefficient_color="darkgrey",
    min_variance_color="navy",
)

# ax2.set_xlim(-0.02, 0.32)

st.pyplot(fig2, use_container_width=True)


# --- COPYRIGHT & DISCLAIMER
utils.footnote()
