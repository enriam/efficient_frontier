# ruff: noqa: E501

import matplotlib.pyplot as plt
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

Supongamos que tenemos dos activos de inversión A y B, que representar una **acción** y un **bono**, y que tienen las siguientes rentabilidades y riesgos:
"""

text02 = """
Si variamos los pesos de cada activo podemos crear distintas carteras: 60%-40%, 70%-30%, 100%-0%, cada una dando una rentabilidad distinta y exponiéndonos a un riesgo distinto. 

La representación gráfica de todas las carteras posibles es la frontera eficiente. Se representa en una gráfica donde el eje X es el riesgo y el eje Y la rentabilidad. 

Si cambiamos los pesos de los activos, cambian la rentabilidad y el riesgo de la cartera. 
"""

# --- CHART 1 -------------------------------------------
# Portfolio numbers
bond = ef.Asset(ret=0.06, volat=0.07)
stock = ef.Asset(ret=0.14, volat=0.16)
corr1 = -0.50

bond_ret = round(bond.ret * 100, 1)
bond_std = round(bond.volat * 100, 1)
stock_ret = round(stock.ret * 100, 1)
stock_std = round(stock.volat * 100, 1)

table = f"""
|             |Rentabilidad |Riesgo       |Correlación|
|-------------|-------------|-------------|-----------|
|**Bono**     |{bond_ret} % |{bond_std} % |           |
|**Acción**   |{stock_ret} %|{stock_std} %|{corr1}    |
"""

st.markdown(text01)
st.write(" ")
st.markdown(table)
st.write(" ")
st.markdown(text02)
st.write(" ")

# Chart 1: changing weights
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("###### Pesos Bono - Acción")
    st.write(" ")
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

# b_std = np.sqrt(bond.volat)
# s_std = np.sqrt(stock.volat)
# covariance = corr1 * b_std * s_std

# pf_mean = b_weight * bond.ret + s_weight * stock.ret
# pf_var = (
#     b_weight**2 * bond.volat
#     + s_weight**2 * stock.volat
#     + 2 * b_weight * s_weight * covariance
# )

fig1, ax1 = plt.subplots(figsize=(9, 6))

ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr1,
    ax=ax1,
    plot_title="Frontera Eficiente",
    efficient_label="Carteras eficientes",
    inefficient_label="Carteras ineficientes",
    min_variance_label="Cartera de riesgo mínimo",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
)

ef.plot_two_asset_portfolio(
    asset_1=bond,
    asset_2=stock,
    weight_1=b_weight,
    weight_2=s_weight,
    correlation=corr1,
    ax=ax1,
)
st.write(
    f"Yeeeea {ef.calc_two_asset_pf_metrics(bond, stock, b_weight, s_weight, corr1)}"
)
# ax1.set_xlim(-0.02, stock.volat + 0.02)

st.pyplot(fig1, use_container_width=True)

text03 = """
La línea gris discontinua representa las carteras ineficientes porque cualquiera de ellas tiene una cartera equivalente en la línea azul que, con el mismo riesgo, da más rentabilidad.

Un aspecto interesante es que la cartera con el mínimo riesgo no es la que está formada por bonos al 100%.

Pero lo que en realidad nos interesa no es decidir los pesos de la cartera. Como inversores debemos decidir el riesgo que estamos dispuestos a asumir, y que las ecuaciones de Markowitz nos digan qué pesos debemos asignar a los activos y qué rentabilidad podemos esperar.
"""
st.write(text03)
st.write(" ")

# --- CHART 2 -------------------------------------------
# st.write("###### Riesgo")
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    risk = st.slider(
        label="**Riesgo**",
        # min_value=min_volat * 100,
        min_value=0.0,
        max_value=stock.volat * 100,
        value=(stock.volat * 100) / 2,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    risk /= 100
    # weights, ret = ef.calc_weights_for_risk(bond, stock, corr1, risk / 100)
with col_a2:
    st.metric("**Riesgo**", f"{round(risk * 100, 1)} %")
with col_a3:
    # st.metric("**Rentabilidad**", f"{round(ret * 100, 1)} %")
    st.metric("**Rentabilidad**", "pepe")

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr1,
    ax=ax2,
    plot_title="Pesos en función del riesgo",
    efficient_label="",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
    efficient_color="darkgrey",
    inefficient_color="darkgrey",
    min_variance_color="navy",
)

# ax2.set_xlim(-0.02, stock.volat + 0.02)

# ax2 = ef.plot_two_asset_frontier(
#     asset_1=bond,
#     asset_2=stock,
#     correlation=corr1,
#     ax=ax2,
#     plot_title="Pesos en función del riesgo",
#     efficient_label="",
#     inefficient_label="",
#     x_axis_label="Riesgo",
#     y_axis_label="Rentabilidad",
#     min_variance_label="Cartera de mínima volatilidad",
#     efficient_color="darkorange",
#     inefficient_color="darkgrey",
#     min_variance_color="navy",
# )

# ax2.set_xlim(-0.02, 0.32)

st.pyplot(fig2, use_container_width=True)


# --- CHART 3 -------------------------------------------
st.write("###### Correlación")
corr2 = st.slider(
    label="Corr",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    format="%0.1f",
    label_visibility="collapsed",
)

fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=0.0,
    ax=ax3,
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

ax3 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr2,
    ax=ax3,
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

# ax3.set_xlim(-0.02, 0.32)

st.pyplot(fig3, use_container_width=True)


# --- COPYRIGHT & DISCLAIMER
utils.footnote()
