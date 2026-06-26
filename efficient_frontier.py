# ruff: noqa: E501

import matplotlib.pyplot as plt
import streamlit as st

# import src.three_assets_portfolio as ef3
import src.two_assets_portfolio as ef2
from src import utils
from src.asset import Asset
from src.gpt_equalizer import plot_annual_returns

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


# --- Prepare Data
data_path = "data/"

asset_files = {
    "SP500": data_path + "xtnd_SP500_m.csv",
    "Bond 10Y": data_path + "xtnd_US10Y_m.csv",
    "Oro": data_path + "xtnd_Gold_m.csv",
    "Small Caps": data_path + "xtnd_Rus2000_m.csv",
    "Bond 20Y": data_path + "xtnd_US20Y_m.csv",
    # "Letras del tesoro": data_path + "xtnd_US1-3MO_m.csv",
}
monthly_prices = utils.load_asset_prices(asset_files)
monthly_returns = monthly_prices.pct_change()
annual_returns = (
    monthly_returns.loc["1970":]
    .resample("YE")
    .apply(lambda x: (x + 1).prod() - 1)
)
annual_returns.index = annual_returns.index.year


# Intro
st.title("Entendiendo a Markowitz: diversificación, correlación y riesgo")

text = """
El concepto de diversificación siempre ha estado presente en la mente del inversor de manera intuitiva. Hasta mitad del siglo XX la filosofía de inversión dominante fue seleccionar buenos activos (acciones, bonos, propiedades o negocios) y mantenerlos durante largos periodos. De esta forma el riesgo quedaba distribuido.

Fue Henry Markowitz (premio Nobel de economía en 1990) quien, en un artículo publicado en 1952, formalizó matemáticamente el concepto de riesgo y dio un nuevo significado al concepto de diversificación, propiciando el nacimiento de la Teoría Moderna de Carteras. 

Vamos a internarnos poco a poco en los conceptos que nos propone Markowitz.
"""
st.markdown(text)

# Two asset portfolio definitions
pf_2a = annual_returns.loc[:, ["Bond 10Y", "SP500"]]
pf_2a_avg = pf_2a.mean().rename("Rent Esperada")
pf_2a_std = pf_2a.std().rename("Volatilidad")
pf_2a_corr = pf_2a.corr()
pf_2a_corr.index.names = ["Correlación"]
corr_2a = pf_2a_corr.iloc[0, 1]

# Bond and Stock Assets definition
bond10y = Asset(
    name="Bono 10Y",
    avg=pf_2a_avg.loc["Bond 10Y"],
    std=pf_2a_std.loc["Bond 10Y"],
)
sp500 = Asset(
    name="S&P 500",
    avg=pf_2a_avg.loc["SP500"],
    std=pf_2a_std.loc["SP500"],
)


# === Asset Risk & Return ==========================================
text = """
## Rentabilidad y riesgo de un activo
Consideremos los siguientes dos activos financieros: 

- el Bono a 10 años del Tesoro de los Estados Unidos, un bono de referencia mundial
- el S&P 500, el índice bursátil más importante del mundo

Estas han sido sus rentabilidades anuales desde 1970:
"""
st.markdown(text)
st.dataframe((pf_2a.T * 100).round(1), width="content")

text = f"""
Markowitz utilizó las matemáticas estadísticas para definir la rentabilidad y el riesgo de un activo. La **rentabilidad esperada** de un activo sería la media de las rentabilidades históricas, representada por la letra griega mu ($\\mu$):

Por otro lado, las rentabilidades anuales no coinciden con la media, algunas son mayores y otras menores. Esa dispersión se mide con una variable estadística llamada *desviación estándar*, representa por la letra griega sigma ($\\sigma$). 

Markowitz definió el **riesgo** de un activo financiero como su desviación estándar, dando así una medida matemática del riesgo. En el mundo de las inversiones a la desviación estándar se le suele llamar **volatilidad**, una palabra que seguro has oido muchas veces. 

Para nuestros activos tendríamos los siguientes valores:

|            |Rentabilidad                   |Riesgo                         |
|------------|-------------------------------|-------------------------------|
|**Bono 10Y**|{round(bond10y.avg * 100, 1)} %|{round(bond10y.std * 100, 1)} %|
|**S&P 500** |{round(sp500.avg * 100, 1)} %  |{round(sp500.std * 100, 1)} %  |

Ahora formemos una cartera de inversión con ambos activos.
"""
st.markdown(text)


# === Portfolio Risk & Return ==========================================
text = """
## Rentabilidad y riesgo de una cartera con dos activos

Para formar una cartera tenemos que asignar pesos ($w$) a cada uno de los activos. Con esto, nuestros activos tendrían estas características:

- Bono 10Y --> $\\mu_1$, $\\sigma_1$, $w_1$
- S&P 500 ---> $\\mu_2$, $\\sigma_2$, $w_2$

Aplicando la matemática estadísticas, Markowitz introdujo estas fórmulas para calcular la rentabilidad esperada y el riesgo de una cartera formada por dos activos:
"""
st.markdown(text)

st.latex(r"""\mu_p = w_1 \cdot \mu_1 + w_2 \cdot \mu_2""")
st.latex(r"""\sigma_p^2 = w_1^2 \cdot \sigma_1^2 + w_2^2 \cdot \sigma_2^2 
+ 2 \cdot w_1 \cdot w_2 \cdot \rho_{12} \cdot \sigma_1 \cdot \sigma_2""")

text = """
La rentabilidad esperada de la cartera es bastante obvia, depende de la rentabilidad de cada activo en proporción a su peso en la cartera. Pero lo realmente interesante es el riesgo de la cartera. Vamos a traducir la fórmula a conceptos para verlo mejor:
"""
st.markdown(text)

conceptual = r"""
[riesgo_{cartera}] \propto [riesgo_1]^2 + [riesgo_2]^2 + [correlación (\rho_{12})] \cdot [riesgo_1] \cdot [riesgo_2]
"""
st.latex(conceptual)

text = """
El riesgo de la cartera depende de los riesgo de los activos y de una nueva variable, $\\rho_{12}$ (letra griega rho), que representa la **correlación** entre los activos 1 y 2 de la cartera. Y este nuevo factor es muy importante.
"""
text = """
El riesgo de la cartera depende de los riesgo de los activos y de una nueva variable, $\\rho_{12}$ (letra griega rho), que representa la **correlación** entre los activos 1 y 2 de la cartera. Y este nuevo factor es muy importante.
"""
st.markdown(text)


# === Correlation ==========================================
text = """
### Correlación

La correlación es uno de los conceptos más importantes en la teoría de carteras. Es una variable estadística que indica en qué medida  dos activos se mueven a la par, tanto en dirección como en magnitud relativa. Su valor está dentro del intervalo [-1, 1] y se interpreta de la siguiente manera:

- $\\rho = +1$: movimiento perfectamente sincronizado. Si el activo 1 sube un 2%, el activo 2 sube proporcionalmente siempre. No hay ningún beneficio de combinarlos.
- $\\rho = 0$: los movimientos son independientes. Saber qué hizo el activo 1 no te dice nada sobre qué hará el activo 2.
- $\\rho = -1$: movimiento perfectamente opuesto. Cuando el activo 1 sube, el activo 2 baja en la misma proporción. Combinados en la proporción correcta, el riesgo se anula por completo.

En la práctica, los activos financieros rara vez tienen correlaciones extremas. Los valores se mueven, aproximadamente, en estos rangos:

- correlaciones altas: [0.60, 0.9]
- correlaciones moderadas: [0.30, 0.60]
- activos descorrelacionados: [-0.2, +0.25]
- correlaciones negativas: [-0.6, -0.2] 
"""
st.markdown(text)


# === Efficient Frontier ========================================
text = """
#### Cálculos para la cartera

Volviendo a Markowitz, ahora sabemos que el riesgo de una cartera no depende solamente de los riesgos de sus activos, sino también de la correlación entre ellos. Diversificar ya no es solamente tener activos distintos en la cartera, hay que tener en cuenta sus correlaciones.

Recordemos las varibles de nuestros activos, añadiendo ahora la correlación:
"""
st.markdown(text)


col_a, col_b, col_c = st.columns([3, 3, 3])
with col_a:
    st.dataframe((pf_2a_avg * 100).round(1), width="content")
with col_b:
    st.dataframe((pf_2a_std * 100).round(1), width="content")
with col_c:
    st.dataframe((pf_2a_corr).round(2), width="content")

text = """
Ya tenemos casi todos los datos que necesitamos para calcular el riesgo y la rentabilidad esperada de nuestra cartera con el Bono 10Y y el S&P 500. Sólo nos queda decidir qué peso le asignamos a cada activo en la cartera y ya podemos resolver las fórmulas. 

Por ejemplo, si asignamos Bono 10Y un peso $w_1$ = 40% y al S&P 500 un peso $w_2$ = 60%, las ecuaciones de Markowitz nos darían:

- $\\mu_p$ = 9.4 %
- $\\sigma_p$ = 10.9 %

Pero vamos a hacerlo más interesante, vamos a representar en una gráfica todas las combinaciones de pesos posibles.
"""
st.markdown(text)


# === Efficient Frontier ========================================
text = """
### La frontera eficiente

En el eje X tenemos el riesgo de la cartera ($\\sigma_p$) y en el eje Y la rentabilidad esperada ($\\mu_p$). Cada punto de la línea azul y de la línea gris discontinua representa una cartera posible, es decir, una combinación de pesos $w_1$ y $w_2$ para los activos de la cartera.

La línea gris discontinua representa todas las **carteras ineficientes**. Son ineficientes porque en la linea azul hay una cartera que, asumiendo el mismo riesgo, proporciona una rentabilidad mayor. La línea azul, por tanto representa las **carteras eficientes**.

Puedes seleccionar tu propia cartera cambiando los pesos de los activos con el deslizador para ver rentabilidad y riesgo obtendrías.
"""
st.markdown(text)

# --- CHART 1: efficient frotier modifying weights
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    st.write(" ")
    weight = st.slider(
        label=f"**Peso {bond10y.name}**",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        format="%d%%",
        # label_visibility="collapsed",
    )

with col_a2:
    st.metric(f"**{bond10y.name}**", f"{weight} %")
with col_a3:
    st.metric(f"**{sp500.name}**", f"{100 - weight} %")

b_weight = weight / 100  # bond asset weight
s_weight = 1 - b_weight  # stock asset weight

fig1, ax1 = plt.subplots(figsize=(9, 6))

ef2.plot_2a_frontier(
    asset_1=bond10y,
    asset_2=sp500,
    correlation=corr_2a,
    ax=ax1,
    plot_title="Frontera Eficiente",
    efficient_label="Carteras eficientes",
    inefficient_label="Carteras ineficientes",
    min_variance_label="Cartera de mínimo riesgo",
    x_axis_label="Riesgo ($\\sigma_p$)",
    y_axis_label="Rentabilidad ($\\mu_p$)",
)
ax1.set_xlim(0.0)

ef2.scatter_2a_portfolio(
    asset_1=bond10y,
    asset_2=sp500,
    weight_1=b_weight,
    weight_2=s_weight,
    correlation=corr_2a,
    ax=ax1,
    point_label="Cartera seleccionada",
)

# calculate risk and return for the selected weights
pf_2a_risk, pf_2a_return = ef2.risk_return_2a_pf(
    asset_1=bond10y,
    asset_2=sp500,
    weight_1=b_weight,
    weight_2=s_weight,
    correlation=corr_2a,
)

y_min, y_max = ax1.get_ylim()
x_min, x_max = ax1.get_xlim()
bbox = {"boxstyle": "square", "fc": "orange", "pad": 0.3, "color": "white"}

txt = f"{round(pf_2a_risk * 100, 1)}%"
ax1.text(x=pf_2a_risk, y=y_min, s=txt, ha="center", bbox=bbox)

txt = f"{round(pf_2a_return * 100, 1)}%"
ax1.text(x=x_min, y=pf_2a_return, s=txt, ha="center", bbox=bbox)

st.pyplot(fig1, width="stretch")

text = """
El punto azul representa la cartera de mínimo riesgo. No es posible reducir más el riesgo. **_Puede que te llame la atención que esta cartera de mínimo riesgo no está formada por bonos al 100%. Esto es debido, justamente, a la correlación._**
"""
st.markdown(text)
st.write(" ")


# === Correlation and Efficient Frontier =================================
text = f"""
## Correlación y frontera eficiente

Hemos hecho un ejercicio variando los pesos de los activos para entender cómo cambia la rentabilidad y el riesgo de la cartera. Hagamos ahora otro ejercicio más interesante. ¿Cómo cambia la frontera eficiente en función de la correlación de los activos?

Vamos a imaginar que la correlación de nuestros dos activos no es la que hemos calculado ($\\rho$ = {round(corr_2a, 2)}), sino otra que vamos a variar a voluntad. Veamos qué pasaría con la frontera eficiente de la cartera. 
"""
st.markdown(text)
st.write("")

_, col1, _ = st.columns([1, 4, 1])
with col1:
    corr1 = st.slider(
        label="**Correlación**",
        min_value=-1.0,
        max_value=1.0,
        value=corr_2a,
        step=0.01,
        format="%0.2f",
        label_visibility="visible",
    )

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef2.plot_2a_frontier(
    asset_1=bond10y,
    asset_2=sp500,
    correlation=corr_2a,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {round(corr_2a, 2)}",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
    efficient_color="darkgrey",
    min_variance_color="darkgrey",
)

ax2 = ef2.plot_2a_frontier(
    asset_1=bond10y,
    asset_2=sp500,
    correlation=corr1,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {round(corr1, 2)}",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="Cartera de mínima volatilidad",
    efficient_color="darkorange",
)
ax2.legend()
ax2.set_xlim(0.0)

st.pyplot(fig2, width="stretch")

text = """
El efecto fundamental es que, cuando la correlación disminuye, la frontera eficiente se curva hacia la izquierda y hacia arriba. Interpretación: cuando se reduce la correlación podemos conseguir la **misma rentabilidad** pero con **menos riesgo**, o **mayor rentabilidad** con el **mismo riesgo**. 

Esta es la esencia de la diversificación. Diversificar no es solamente tener varios activos en la cartera, sino también tener en cuenta su correlación. Los activos descorrelacionados o correlacionados negativamente nos protegen. 
"""
st.markdown(text)


# === Limited Risk Portfolio ========================================
text = """
## Limitación de riesgo en las carteras

Los ejercicios anteriores nos han servido para entender el papel que juegan los pesos de los activos y su correlación en los resultados de la cartera pero, como inversores, lo que realmente nos interesa es limitar el riesgo que estamos dispuesto a asumir.

A partir de ahí, podemos usar las ecuaciones de Markowitz para calcular los pesos óptimos para ese nivel de riesgo. Y con los pesos podemos calcular la rentabilidad esperada de nuestra cartera.
"""
st.markdown(text)
st.write(" ")

# --- CHART 2: Max volatility portfolio
# calculate Global Minimum Volatility
gvm, *_ = ef2.min_vol_2a_pf(bond10y.std, sp500.std, corr_2a)

col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    max_risk = st.slider(
        label="**Máximo riesgo asumible**",
        min_value=gvm * 100,
        max_value=sp500.std * 100,
        value=gvm * 100 * 1.1,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    max_risk /= 100

# w1, w2 = ef2.weights_2a_pf_long_only(bond10y, sp500, corr_2a, max_risk)
w1, w2 = ef2.optimal_two_asset_weights_long_only(
    bond10y, sp500, corr_2a, max_risk
)
pf_ret = w1 * bond10y.avg + w2 * sp500.avg

with col_a2:
    st.metric("**Riesgo cartera**", f"{round(max_risk * 100, 1)} %")
with col_a3:
    # st.metric("**Rentabilidad**", f"{round(ret * 100, 1)} %")
    st.metric("**Rent. cartera**", f"{round(pf_ret * 100, 1)} %")

fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3 = ef2.plot_2a_frontier(
    asset_1=bond10y,
    asset_2=sp500,
    correlation=corr_2a,
    ax=ax3,
    plot_title="Pesos en función del riesgo",
    efficient_label="",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
)
ax3.set_xlim(0.0)

ax3.scatter(
    # max_risk,
    max_risk,
    pf_ret,
    c="darkorange",
    marker="D",
    zorder=6,
    label=f"Cartera riesgo {round(max_risk * 100, 1)} %",
)
weights_text = f"""
{bond10y.name} = {round(w1 * 100)} %
{sp500.name} = {round(w2 * 100)} %
"""
ax3.annotate(
    weights_text,
    xy=(max_risk, pf_ret),
    xycoords="data",
    xytext=(0.7, 0.1),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "ec": "grey"},
    bbox={"boxstyle": "square", "fc": "white", "ec": "darkorange", "pad": 0.3},
)

ax3.legend()
st.pyplot(fig3, width="stretch")

text = """
Como es lógico, conforme aumentamos el riesgo que estamos dispuestos a asumir, aumenta la proporción de acciones (el activo de mayor riesgo) en la cartera. Y también aumenta la rentabilidad esperada. 

Y si disminuimos el riesgo de la cartera, se reduce el peso de las acciones, así hasta llegar a la cartera de mínimo riesgo, que tiene un porcentaje pequeño de acciones, pero no cero. Ya hemos explicado que esto es debido a que los activos elegidos tienen correlación negativa. Pero en la siguiente sección veremos esto en detalle.
"""
st.markdown(text)


# === EQUALIZER ==================================================
st.markdown("## Equalizer")

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
asset_weights = [0.25, 0.25, 0.25]
monthly_rets = utils.calculate_asset_returns(asset_files=selected_asset_files)
monthly_rets = monthly_rets.loc["1970":]

fig4, ax4 = plt.subplots(figsize=(9, 6))
ax4 = plot_annual_returns(monthly_rets, None, ax4, pf_label="Cartera")

st.pyplot(fig4, width="stretch")


# --- COPYRIGHT & DISCLAIMER
utils.footnote()
