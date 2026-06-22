# ruff: noqa: E501

import matplotlib.pyplot as plt
import streamlit as st

import src.three_assets_portfolio as ef3
import src.two_assets_portfolio as ef2
from src import utils
from src.gpt_equalizer import plot_annual_returns

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

text = """
El concepto de diversificación siempre ha estado presente en la mente del inversor de manera intuitiva, pues es obvio que invertir todo el dinero en un único activo tiene más riesgo que invertir en diferentes activos. Pero fue Henry Markowitz quien, en 1952, formalizó el concepto matemáticamente, dando lugar al nacimiento de la Teoría Moderna de Carteras. Vamos a explorar la TMC de forma intuitiva, utilizando las menos ecuaciones posibles.

Supongamos que tenemos un activo con un histórico de rentabilidades, digamos que anuales. Estadísticamente, la rentabilidad esperada de ese activo sería la media de esas rentabilidades anuales. Se representa por la letra griega mu: $\\mu$.

Por otro lado, las rentabilidades anuales no coinciden con la media, algunas son mayores y otras menores. Esa dispersión se mide estadísticamente con una variable llamada **varianza**. No hace falta entrar en el detalle del cálculo  de la varianza, lo importante es que Markowitz identificó el riesgo de un activo con su varianza, dando así una medida matemática del riesgo. La varianza ser representa por la letra griega sigma al cuadrado: $\\sigma^2$.

Aquí merece la pena introducir otro concepto: la **volatilidad**, que es muy utilizado porque mide lo mismo que la varianza pero en las mismas unidades que la rentabilidad, es decir, en porcentaje. La volatilidad, que en estadística se llama desviación estándar, no es más que la raíz cuadrada de la varianza: $\\sigma$.

Así pues tenemos:

- **$\\mu$**, rentabilidad esperada de un activo: la media de las rentabilidades históricas
- **$\\sigma^2$** ó **$\\sigma$**, varianza ó volatilidad: representa el riesgo de un activo, cuantifica la incertidumbre de que no se cumpla la rentabilidad esperada  

### Cartera de inversión
Supongamos ahora una cartera muy simple, formada por solo dos activos, cada uno con un peso o porcentaje en la cartera: $w_1$ y $w_2$ (obviamente $w_1 + w_2$ = 1). Estos activos, según lo explicado, tendrán su rentabilidad y su riesgo: ($\\mu_1$, $\\sigma^2_1$) y ($\\mu_2$, $\\sigma^2_2$). Lo que nos dice el modelo de Markowitz es lo siguiente:

**Rentabilidad**
$$
\\mu_p = w_1 \\mu_1 + w_2 \\mu_2
$$

La rentabilidad esperada de la cartera, $r_p$, depende de las rentabilidades de los activos en proporción a sus pesos. 

**Riesgo**
$$
\\sigma_p^2 = w_1^2 \\sigma_1^2 + w_2^2 \\sigma_2^2 
+ 2 w_1 w_2 \\rho_{12} \\sigma_1 \\sigma_2
$$

El riesgo (varianza) de la cartera depende de los riesgo de los activos, ya sean en forma de varianza, $\\sigma^2$, o de volatilidad $\\sigma$ (eso no debe confundirnos, es irrelevante), pero depende también de una nueva variable, $\\rho$ (letra griega rho), que representa la **correlación** entre activos. Y este nuevo factor es muy importante.

**Correlación**

La correlación es uno de los conceptos más importantes en la teoría de carteras de Markowitz. Es una variable estadística que indica en qué medida los precios de dos activos se mueven juntos, tanto en dirección como en magnitud relativa. Su valor está dentro del intervalo [-1, 1] y se interpreta de la siguiente manera:

- $\\rho = +1$: movimiento perfectamente sincronizado. Si el activo 1 sube un 2%, el activo 2 sube proporcionalmente siempre. No hay ningún beneficio de combinarlos.
- $\\rho = 0$: los movimientos son independientes. Saber qué hizo el activo 1 no te dice nada sobre qué hará el activo 2.
- $\\rho = -1$: movimiento perfectamente opuesto. Cuando el activo 1 sube, el activo 2 baja en la misma proporción. Combinados en la proporción correcta, el riesgo se anula por completo.

Por tanto, lo que nos dice la fórmula del riesgo es que, si combinamos en la carterar activos correlacionados positivamente, aumentamos el riesgo, y si combinamos activos correlacionados negativamente, reducimos el riesgo. Por tanto, el concepto de diversificar adquiere una nueva dimensión a través de la correlación.

*Nota: en la práctica, los activos financieros rara vez tienen correlaciones extremas. Dos acciones del mismo sector suelen tener $\\rho$ entre 0.5 y 0.8. Activos de clases distintas (renta variable vs. bonos soberanos, por ejemplo) suelen tener correlaciones más bajas o incluso negativas en ciertos regímenes de mercado.*

Veamos de forma práctica estos conceptos.

## La frontera eficiente

Supongamos que tenemos dos activos de inversión A y B, que representan una **acción** y un **bono** con las siguientes características:
"""
st.markdown(text)

# --- Two Asset Portfolio
bond = ef2.Asset(name="Bono", ret=0.06, volat=0.07)
stock = ef2.Asset(name="Acción", ret=0.14, volat=0.16)
corr = -0.50

table1 = f"""
|             |Rentabilidad                 |Riesgo                         |
|-------------|-----------------------------|-------------------------------|
|**Bono**     |{round(bond.ret * 100, 1)} % |{round(bond.volat * 100, 1)} % |
|**Acción**   |{round(stock.ret * 100, 1)} %|{round(stock.volat * 100, 1)} %|

Supongamos también (aunque ahora no es relevante) que tienen una **correlación** $\\rho = -${-corr}
"""
st.markdown(table1)


text = """
Variando los pesos de cada activo podemos crear distintas carteras: 60%-40%, 70%-30%, 100%-0%, cada una con rentabilidad y riesgo distintos. 

Vamos a representar todas las carteras posibles con estos dos activos, es decir, todas las combinaciones de pesos, en una gráfica donde el eje X es el riesgo (volatilidad) y el eje Y la rentabilidad.

Puedes cambiar los pesos de los activos con el deslizador para ver cómo cambia la rentabilidad y el riesgo de la cartera. 
"""
st.markdown(text)
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

ef2.plot_2a_frontier(
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

ef2.scatter_2a_portfolio(
    asset_1=bond,
    asset_2=stock,
    weight_1=b_weight,
    weight_2=s_weight,
    correlation=corr,
    ax=ax1,
)
# ax1.set_xlim(-0.02, stock.volat + 0.02)

st.pyplot(fig1, use_container_width=True)

text = """
La línea azul es la **frontera eficiente** y representa las carteras óptimas. La línea gris discontinua representa las carteras ineficientes porque cualquiera de ellas tiene una cartera equivalente en la línea azul que, con el mismo riesgo, da más rentabilidad.

Aunque lo analizaremos con más detalle un poco más adelante, un aspecto interesante que podemos ver en la gráfica es que la cartera con el mínimo riesgo no es la que está formada por bonos al 100%. Esto es así porque la acción y el bono que hemos elegido tienen correlación negativa, es decir, los riesgos se compensan parcialmente y, por tanto, tener un poco de la acción en la cartera reduce el riesgo más que tener todo al 100% en bonos.

El concepto "frontera" puede parecer un poco innecesario dado que estamos hablando de una línea. Pero vamos a complicar un poco la cartera introduciendo un tercer activo para que se comprenda mejor el concepto.
"""

st.markdown(text)
st.write(" ")

# --- Three Asset Portfolio
bond = ef2.Asset(name="Bono1", ret=0.06, volat=0.07)
stock = ef2.Asset(name="Acción", ret=0.14, volat=0.16)
bond2 = ef2.Asset(name="Bono2", ret=0.08, volat=0.10)
corr_12 = -0.50
corr_13 = 0.25
corr_23 = 0.10

table2 = f"""
|             |Rentabilidad                 |Riesgo                         |
|-------------|-----------------------------|-------------------------------|
|**Bono 1**   |{round(bond.ret * 100, 1)} % |{round(bond.volat * 100, 1)} % |
|**Acción**   |{round(stock.ret * 100, 1)} %|{round(stock.volat * 100, 1)} %|
|**Bono 2**   |{round(bond2.ret * 100, 1)} %|{round(bond2.volat * 100, 1)} %|
"""
text = """
Ignoremos por ahora las **correlaciones**. 

Para que se vea claramente vamos a representar 10.000 carteras combinando los tres activos con diferentes pesos:
"""
# - $\\rho_{12} = -${-corr_12}
# - $\\rho_{13}$ = ${corr_13}
# - $\\rho_{23}$ = ${corr_23}
# """

st.markdown(table2)
st.markdown(text)
st.write(" ")

fig0, ax0 = plt.subplots(figsize=(9, 6))

ax0 = ef3.plot_3a_frontier(
    asset_1=bond,
    asset_2=stock,
    asset_3=bond2,
    correlation_12=corr_12,
    correlation_13=corr_13,
    correlation_23=corr_23,
    ax=ax0,
    plot_title="Frontera Eficiente",
    efficient_label="Carteras eficientes",
    inefficient_label="Carteras ineficientes",
    min_variance_label="Cartera de riesgo mínimo",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
)

st.pyplot(fig0, use_container_width=True)


text = """
Al aumentar el número de activos vemos claramente por qué tiene sentido el concepto de **frontera** eficiente.

Hemos hecho un ejercicio variando los pesos de los activos para entender lo que es la frontera eficiente. Hagamos ahora otro ejercicio más interesante desde el punto de vista de un inversor.

## Cartera de riesgo limitado

Para simplificar, volvamos a nuestra cartera de sólo dos activos: acción y bono, con los mismos parámetros de rentabilidad, riesgo y correlación. 
"""
st.markdown(text)
st.markdown(table1)

text = """
Primero tenemos que plantearnos el riesgo que estamos dispuestos a asumir en nuestra cartera de inversión. A partir de ahí las ecuaciones de Markowitz nos dirán los pesos de los activos para formar la cartera óptima, la cartera que está en la frontera eficiente. Y también nos dirán la rentabilidad esperada de esa cartera.
"""
st.markdown(text)
st.write(" ")

# --- CHART 2 -------------------------------------------
# Global Minimum Volatility
gvm, *_ = ef2.min_vol_2a_pf(bond.volat, stock.volat, corr)

# st.write("###### Riesgo")
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    pf_risk = st.slider(
        label="**Riesgo asumible**",
        min_value=gvm * 100,
        max_value=stock.volat * 100,
        value=(stock.volat * 100) / 2,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    pf_risk /= 100

w1, w2 = ef2.weights_2a_pf_long_only(bond, stock, corr, pf_risk)
pf_ret = w1 * bond.ret + w2 * stock.ret

with col_a2:
    st.metric("**Riesgo cartera**", f"{round(pf_risk * 100, 1)} %")
with col_a3:
    # st.metric("**Rentabilidad**", f"{round(ret * 100, 1)} %")
    st.metric("**Rent. cartera**", f"{round(pf_ret * 100, 1)} %")

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef2.plot_2a_frontier(
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

text = """
Como es lógico, conforme aumentamos el riesgo que estamos dispuestos a asumir, aumenta la proporción de acciones (el activo de mayor riesgo) en la cartera. Y también aumenta la rentabilidad esperada. 

Y si disminuimos el riesgo de la cartera, se reduce el peso de las acciones, así hasta llegar a la cartera de mínimo riesgo, que tiene un porcentaje pequeño de acciones, pero no cero. Ya hemos explicado que esto es debido a que los activos elegidos tienen correlación negativa. Pero en la siguiente sección veremos esto en detalle.
"""
st.markdown(text)


# --- CHART 3 -------------------------------------------
text04 = """
## Correlación

Veamos cómo afecta la correlación a la frontera eficiente. Para ello vamos a suponer que tenemos dos carteras, ambas con dos activos, y estos activos con rentabilidad y riesgo iguales, pero con distinta correlación según seleccionemos con los deslizadores.
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

ax3 = ef2.plot_2a_frontier(
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

ax3 = ef2.plot_2a_frontier(
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

text05 = """
El efecto fundamental es que la frontera eficiente se curva hacia la izquierda a medida que la correlación disminuye. Interpretación: cuando se reduce la correlación podemos conseguir la misma rentabilidad pero con menos riesgo. Esto es la esencia de la diversificación.

Diversificar no es solamente tener dos (o más) activos en la cartera, sino también tener en cuenta su correlación. Si nos marcamos un riesgo máximo, como en el apartado anterior, podemos conseguir una rentabilidad mayor si los activos tienen correlación negativa que si tienen correlación positiva. 
"""
st.markdown(text05)


# --- EQUALIZER ------------------------------------------------
st.markdown("## Equalizer")

data_path = "data/"
asset_files = {
    "Letras del tesoro": data_path + "xtnd_US1-3MO_m.csv",
    "Oro": data_path + "xtnd_Gold_m.csv",
    "Bono 10 años": data_path + "xtnd_US10Y_m.csv",
    "SP500": data_path + "xtnd_SP500_m.csv",
}
selected_assets = st.segmented_control(
    label="Selecciona activos para la cartera",
    options=list(asset_files.keys()),
    selection_mode="multi",
    # required=True,
    default=list(asset_files.keys()),
)

if len(selected_assets) == 0:
    st.warning("Please select at least one asset.")
    st.stop()

selected_asset_files = {
    asset_name: asset_files[asset_name] for asset_name in selected_assets
}

# load asset returns
asset_weights = [0.25, 0.25, 0.25, 0.25]
asset_rets = utils.load_asset_returns(asset_files=selected_asset_files)
asset_rets = asset_rets.loc["1970":]


fig4, ax4 = plt.subplots(figsize=(9, 6))
ax4 = plot_annual_returns(asset_rets, None, ax4, pf_label="Cartera")

st.pyplot(fig4, use_container_width=True)

# TESTING SOME NUMBERS
# (1 + returns).groupby(returns.index.year).prod(min_count=1) - 1


# --- COPYRIGHT & DISCLAIMER
utils.footnote()
