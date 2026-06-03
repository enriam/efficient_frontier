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


# --- Function definitions
def get_md_file(file_path: str):
    with open(file_path, encoding="utf-8") as f:  # noqa: PTH123
        return f.read()


# --- Page Deployment
st.title("Diversificación y correlación")

text01 = """
El concepto de diversificación siempre ha estado presente en la mente del inversor de manera intuitiva, pero fue Henry Markowitz quien, en 1952, formalizó el concepto matemáticamente y demostró que el riesgo de una cartera de inversión no depende solo del riesgo de cada activo, sino también de cómo se mueven unos activos en relación a los otros. Esto último se conoce como **correlación**. Así nació la Teoría Moderna de Carteras.  
"""

text02 = """
Antes de ver cómo la correlación afecta a una cartera de inversión es útil entender lo que es la "frontera eficiente" y cómo se representa.

## La frontera eficiente

Supongamos que tenemos dos activos de inversión A y B, que representan una **acción** y un **bono** con las siguientes características:
"""

text03 = """
Variando los pesos de cada activo podemos crear distintas carteras: 60%-40%, 70%-30%, 100%-0%, cada una con rentabilidad y riesgo distintos. 

La representación gráfica de todas las carteras posibles con estos dos activos, es decir, todas las combinaciones de pesos, es una linea en la gráfica donde el eje X es el riesgo (volatilidad) y el eje Y la rentabilidad. 
"""

# --- Two Asset Portfolio
bond = ef.Asset(ret=0.06, volat=0.07)
stock = ef.Asset(ret=0.14, volat=0.16)
corr = -0.50

table = f"""
|             |Rentabilidad                 |Riesgo                         |
|-------------|-----------------------------|-------------------------------|
|**Bono**     |{round(bond.ret * 100, 1)} % |{round(bond.volat * 100, 1)} % |
|**Acción**   |{round(stock.ret * 100, 1)} %|{round(stock.volat * 100, 1)} %|

**Correlación** $\\rho$ = {corr}
"""

st.markdown(text01)
st.write(" ")
msg = ":blue[¿Quieres conocer las ecuaciones de Markowitz?]"
with st.expander(msg):
    st.markdown(get_md_file("md\\markowitz_1.md"))
st.markdown(text02)
st.markdown(table)
st.write(" ")
st.write(text03)
st.write(" ")

# --- CHART 1 ------modifying weights -------------------------------------
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

fig1, ax1 = plt.subplots(figsize=(9, 6))

ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr,
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
    correlation=corr,
    ax=ax1,
)
# ax1.set_xlim(-0.02, stock.volat + 0.02)

st.pyplot(fig1, use_container_width=True)

text03 = """
La línea azul es la **frontera eficiente** y representa las carteras óptimas. La línea gris discontinua representa las carteras ineficientes porque cualquiera de ellas tiene una cartera equivalente en la línea azul que, con el mismo riesgo, da más rentabilidad.

**Observación 1**: un aspecto interesante que podemos ver en la gráfica es que la cartera con el mínimo riesgo no es la que está formada por bonos al 100%. Esto sería cierto si consideráramos que el bono tiene riesgo 0.

Normalmente no nos va a interesar decidir los pesos de la cartera. Como inversores debemos decidir el riesgo que estamos dispuestos a asumir. A partir de ahí, resolviendo las ecuaciones de Markowitz encontraremos los pesos óptimos para obtener la máxima rentabilidad posible.

## Cartera de riesgo limitado

Consideremos el riesgo que estamos dispuestos a asumir en nuestra inversión, y queremos saber la cartera óptima que podemos formar con el bono y la acción anteriores. Necesitamos saber los pesos de cada activo para no superar el riesgo tolerable y rentabilidad que podemos esperar de esa cartera.
"""
st.write(text03)
st.write(" ")

# --- CHART 2 -------------------------------------------
# Global Minimum Volatility
gvm, *_ = ef.min_vol_two_assets(bond.volat, stock.volat, corr)

# st.write("###### Riesgo")
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    pf_risk = st.slider(
        label="**Riesgo**",
        min_value=gvm * 100,
        max_value=stock.volat * 100,
        value=(stock.volat * 100) / 2,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    pf_risk /= 100

w1, w2 = ef.optimal_two_asset_weights_long_only(bond, stock, corr, pf_risk)
pf_ret = w1 * bond.ret + w2 * stock.ret

with col_a2:
    st.metric("**Riesgo**", f"{round(pf_risk * 100, 1)} %")
with col_a3:
    # st.metric("**Rentabilidad**", f"{round(ret * 100, 1)} %")
    st.metric("**Rentabilidad**", f"{round(pf_ret * 100, 1)} %")

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr,
    ax=ax2,
    plot_title="Pesos en función del riesgo",
    efficient_label="",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
)

weights_text = f"""
Bono = {round(w1 * 100, 1)} %

Acción = {round(w2 * 100, 1)} %
"""
ax2.scatter(
    pf_risk,
    pf_ret,
    c="darkorange",
    marker="D",
    zorder=6,
    label=f"Cartera riesgo {round(pf_risk * 100, 1)} %",
)
ax2.annotate(
    weights_text,
    xy=(pf_risk, pf_ret),
    xycoords="data",
    xytext=(0.7, 0.1),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "ec": "grey"},
    bbox={"boxstyle": "square", "fc": "white", "ec": "darkorange", "pad": 0.3},
)
# ax2.set_xlim(-0.02, stock.volat + 0.02)
ax2.legend()
st.pyplot(fig2, use_container_width=True)


# --- CHART 3 -------------------------------------------
text04 = """
## Correlación

Hemos visto cómo la frontera eficiente contiene todas las carteras óptimas que podemos formar y cómo cambiando los pesos variamos la rentabilidad y el riesgo de nuestra cartera. También hemos visto que si fijamos el riesgo máximo que queremos asumir, eso fija los pesos de los activos y la rentabilidad que podemos esperar.

Ahora vamos a ver el efecto que la correlación de los activos tiene en una  cartera de inversión. Lo vamos a hacer pintando las fronteras eficientes de dos carteras, ambas con activos que tienen la misma rentabilidad y el mismo riesgo, pero con coeficientes de correlación distintos.
"""

st.markdown(text04)
# st.write("###### Correlación")
col1, _, col2 = st.columns([3, 1, 3])
with col1:
    corr1 = st.slider(
        label="**Correlación Cartera 1**",
        min_value=-1.0,
        max_value=1.0,
        value=-0.5,
        step=0.1,
        format="%0.1f",
        label_visibility="visible",
    )
with col2:
    corr2 = st.slider(
        label="**Correlación Cartera 2**",
        min_value=-1.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        format="%0.1f",
        label_visibility="visible",
    )

fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3 = ef.plot_two_asset_frontier(
    asset_1=bond,
    asset_2=stock,
    correlation=corr1,
    ax=ax3,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {corr1}",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
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
)
ax3.legend()
# ax3.set_xlim(-0.02, 0.32)

st.pyplot(fig3, use_container_width=True)


# --- COPYRIGHT & DISCLAIMER
utils.footnote()
