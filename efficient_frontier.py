# ruff: noqa: E501

import matplotlib.pyplot as plt
import streamlit as st

import src.plot_3assets as pf3
import src.three_assets_portfolio as ef3
import src.three_assets_weights as w3a
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
    # "Bill 3M": data_path + "xtnd_US1-3MO_m.csv",
    # "Bond 10Y": data_path + "xtnd_US10Y_m.csv",
    "Bond 20Y": data_path + "xtnd_US20Y_m.csv",
    # "SP500": data_path + "xtnd_SP500_m.csv",
    "SmallCaps": data_path + "xtnd_Rus2000_m.csv",
    "Gold": data_path + "xtnd_Gold_m.csv",
}
monthly_prices = utils.load_asset_prices(asset_files)
monthly_returns = monthly_prices.pct_change()
annual_returns = (
    monthly_returns.loc["1970":]
    .resample("YE")
    .apply(lambda x: (x + 1).prod() - 1)
)
annual_returns.index = annual_returns.index.year

pf_avg = annual_returns.mean().rename("Rentabilidad")
pf_std = annual_returns.std().rename("Volatilidad")
pf_corr = annual_returns.corr()
pf_corr.index.names = ["Correlación"]

# asset order is bond20y - smallcaps - gold
corr12 = round(pf_corr.iloc[1, 0], 2)
corr13 = round(pf_corr.iloc[2, 0], 2)
corr23 = round(pf_corr.iloc[2, 1], 2)

# bill3m = Asset(
#     name="Letras 3M",
#     avg=pf_avg.loc["Bill 3M"],
#     std=pf_std.loc["Bill 3M"],
# )
# bond10y = Asset(
#     name="Bono 10Y",
#     avg=pf_avg.loc["Bond 10Y"],
#     std=pf_std.loc["Bond 10Y"],
# )
bond20y = Asset(
    name="Bono 20Y",
    avg=pf_avg.loc["Bond 20Y"],
    std=pf_std.loc["Bond 20Y"],
)
# sp500 = Asset(
#     name="SP500",
#     avg=pf_avg.loc["SP500"],
#     std=pf_std.loc["SP500"],
# )
smallcaps = Asset(
    name="SmallCaps",
    avg=pf_avg.loc["SmallCaps"],
    std=pf_std.loc["SmallCaps"],
)
gold = Asset(
    name="Oro",
    avg=pf_avg.loc["Gold"],
    std=pf_std.loc["Gold"],
)

# Intro
st.title("Teoría Moderna de Carteras: diversificación, correlación y riesgo")

text = f"""
El concepto de diversificación siempre ha estado presente en la mente del inversor de manera intuitiva. Hasta mitad del siglo XX la filosofía de inversión dominante fue seleccionar buenos activos (acciones, bonos, propiedades o negocios) y mantenerlos durante largos periodos. De esta forma el riesgo quedaba distribuido. Incluso hoy, muchos inversores particulares crean sus carteras siguiendo esta filosofía.

Pero hasta los años 50 del siglo XX no hubo ninguna forma de cuantificar el riesgo de las inversiones. Fue en 1952 cuando, Henry Markowitz (premio Nobel de economía en 1990), propuso la primera formalización matemática del concepto de riesgo y dio un nuevo significado al concepto de diversificación.

Propuso un método para crear carteras eficientes (_MVO - Mean Variance Optimization_) que dio lugar al nacimiento de lo que se ha llamado la **Teoría Moderna de Carteras** (_Modern Portfolio Theory_). Una cartera eficiente es la que consigue una rentabilidad objetivo con el mínimo riesgo posible. O también, la que consigue la máxima rentabilidad posible con un riesgo dado.

### La frontera eficiente

Para entender mejor el concepto de cartera eficiente, vamos a considerar tres activos financieros y vamos a representar en una gráfica una muestra de carteras creadas asignando distintos pesos a estos activos. 

Los activos seleccionados son los siguientes:

- Bonos LP: Bono a largo plazo del Gobierno EE.UU. (más de 20 años); dentro del mundo de la renta fija, los bonos de mayor duración son los más volátiles
- SmallCaps: Empresas de EE.UU. de pequeña capitalización, que en renta variable son empresas más volátiles que las de gran capitalización
- Oro: un activo alternativo con una volátilidad mayor que los activos anteriores

Estas son sus métricas:

|         |Rentabilidad                    |Volatilidad                       |
|---------|--------------------------------|----------------------------------|
|Bonos LP |{round(bond20y.avg * 100, 1)}%  |{round(bond20y.std * 100, 1)}%    |
|SmallCaps|{round(smallcaps.avg * 100, 1)}%|{round(smallcaps.std * 100, 1)}%  |
|Oro      |{round(gold.avg * 100, 1)}%     |{round(gold.std * 100, 1)}%       |

Para cada activo las rentabilidad mostrada corresponden a la media de las rentabilidades anuales desde 1970 hasta 2025. Y la volatilidad está calculada, estadísticamente, con la desviación estándar:
"""
st.markdown(text)
st.write("")

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1 = ef3.plot_3a_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    asset_3=gold,
    correlation_12=corr12,
    correlation_13=corr13,
    correlation_23=corr23,
    ax=ax1,
    num_portfolios=10_000,
    plot_title="Frontera Eficiente",
    efficient_label="Frontera eficiente",
    inefficient_label="Carteras ineficientes",
    min_variance_label="Cartera de mínimo riesgo",
    x_axis_label="Riesgo (Volatilidad)",
    y_axis_label="Rentabilidad",
    show_assets=True,
    asset_labels=["Bonos LP 100%", "Small Caps 100%", "Oro 100%"],
)

st.pyplot(fig1, width="stretch")

text = """
La línea azul se llama **frontera eficiente** y representa las carteras óptimas o eficientes, las que consiguen la máxima rentabilidad para un riesgo dado, o las que consiguen una rentabilidad dada con un riesgo mínimo. El resto de carteras son carteras posibles, pero no son eficientes.

Por último, el punto azul representa la cartera con menor riesgo que podemos crear con estos tres activos. 
"""
st.markdown(text)

# === Correlation ==========================================
text = f"""
### Correlación

Uno de los conceptos más importantes en la Teoría Moderna de Carteras es la correlación entre activos. El modelo de Markowitz nos enseña que el riesgo de una cartera no solo depende de los riesgo de sus activos, sino también de la correlación entre ellos. 

La correlación es una variable estadística que indica en qué medida dos activos se mueven a la par, tanto en dirección como en magnitud relativa. Su valor está dentro del intervalo [-1, 1] y se interpreta de la siguiente manera:

- $\\rho = +1$: movimiento perfectamente sincronizado. Si el activo 1 sube un 2%, el activo 2 sube proporcionalmente siempre. No hay ningún beneficio de combinarlos.
- $\\rho = 0$: los movimientos son independientes. Saber qué hizo el activo 1 no te dice nada sobre qué hará el activo 2.
- $\\rho = -1$: movimiento perfectamente opuesto. Cuando el activo 1 sube, el activo 2 baja en la misma proporción. Combinados en la proporción correcta, el riesgo se anula por completo.

En la práctica, los activos financieros rara vez tienen correlaciones extremas. Los valores se mueven, aproximadamente, en estos rangos:

- correlaciones altas: [0.60, 0.9]
- correlaciones moderadas: [0.30, 0.60]
- activos descorrelacionados: [-0.2, +0.25]
- correlaciones negativas: [-0.6, -0.2]

Por ejemplo, los tres activos seleccionados en nuestra cartera tienen las siguientes correlaciones entre ellos:

|CORRELACIÓN  |Bonos LP |SmallCaps|Oro|
|-------------|---------|---------|--------|
|**Bonos LP** |1        |{corr12} |{corr13}|
|**SmallCaps**|{corr12} |1        |{corr23}|
|**Oro**      |{corr13} |{corr23} |1       |

La correlación es muy importante porque nos ayuda a reducir el riesgo de nuestra cartera de inversión. Vamos a verlo con la frontera eficiente de una cartera con sólo dos activos: Bonos LP y SmallCaps.
"""
st.markdown(text)


# === Correlation and Efficient Frontier =================================
text = f"""
### Correlación y frontera eficiente

La correlación entre los Bonos LP y las SmallCaps es de {corr12}. ¿Cómo cambiaría la frontera eficiente si la correlación entre estos dos activos fuera distinta? Compruébalo con el deslizador.
"""
st.write(text)
st.write("")

_, col1, _ = st.columns([1, 4, 1])
with col1:
    corr_select = st.slider(
        label="**Correlación**",
        min_value=-1.0,
        max_value=1.0,
        value=corr12,
        step=0.01,
        format="%0.2f",
        label_visibility="visible",
    )

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2 = ef2.pf_2a_plot_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    correlation=corr12,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {corr12}",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
    efficient_color="darkgrey",
    min_variance_color="darkgrey",
)

ax2 = ef2.pf_2a_plot_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    correlation=corr_select,
    ax=ax2,
    plot_title="Efecto de la correlación",
    efficient_label=f"Cartera $\\rho$ = {corr_select}",
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

Esta es la esencia de la diversificación. Diversificar no es solamente tener varios activos en la cartera, sino también tener en cuenta su correlación. Los activos descorrelacionados o correlacionados negativamente nos protegen de la volatilidad. 

Hagamos ahora un ejercicio sencillo con nuestra cartera de tres activos. Vamos a dibujar la frontera eficiente en dos casos (sin las nubes de carteras ineficientes): 

- **Caso 1**: correlaciones reales
- **Caso 2**: mismos valores de correlación pero los hacemos positivos
"""
st.markdown(text)
st.write("")

fig11, ax11 = plt.subplots(figsize=(9, 6))

ax11 = ef3.plot_3a_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    asset_3=gold,
    correlation_12=corr12,
    correlation_13=corr13,
    correlation_23=corr23,
    ax=ax11,
    num_portfolios=10_000,
    plot_title="",
    efficient_label="Caso 1",
    inefficient_label="",
    min_variance_label="",
    x_axis_label="",
    y_axis_label="",
    show_assets=False,
    show_portfolio_cloud=False,
    frontier_color="steelblue",
)

ax11 = ef3.plot_3a_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    asset_3=gold,
    correlation_12=-corr12,
    correlation_13=-corr13,
    correlation_23=-corr23,
    ax=ax11,
    num_portfolios=10_000,
    plot_title="Frontera Eficiente y Correlación",
    efficient_label="Caso 2",
    inefficient_label="Carteras ineficientes",
    min_variance_label="Cartera de mínimo riesgo",
    x_axis_label="Riesgo (Volatilidad)",
    y_axis_label="Rentabilidad",
    show_assets=False,
    show_portfolio_cloud=False,
    frontier_color="darkorange",
)
st.pyplot(fig11, width="stretch")


# === Limited Risk Portfolio ========================================
text = """
## Limitación del riesgo

La que, quizás, es la pregunta más importante que un inversor debe hacerse es esta: ¿qué nivel de riesgo soy capaz de tolerar? De hecho, en la Unión Europea, cualquier profesional o entidad financiera que ofrezca asesoramiento en materia de inversiones está obligado por ley a realizar un perfil de riesgo al cliente antes de recomendarle cualquier producto financiero.

Las ecuaciones de Markowitz nos pueden servir para encontrar los pesos adecuados de los activos para no superar un nivel de riesgo específico. Vamos a verlo con la cartera de Bonos LP y Small Caps. Luego hablaremos de carteras con más activos.

#### Cartera 2 activos: Bonos LP y Small Caps

Fijemos el máximo riesgo que queremos asumir y veamos cuáles serían los pesos óptimos de nuestros activos. Tengamos en cuenta que el menor riesgo posible es el de la Cartera de Mínima Volatilidad. Y el mayor es el del activo de mayor riesgo, en este caso las Small Caps.
"""
st.markdown(text)
st.write(" ")

# --- CHART 2: Max volatility portfolio
# calculate Global Minimum Volatility
gvm, *_ = ef2.pf_2a_min_volatility(bond20y.std, smallcaps.std, corr12)

col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    max_risk = st.slider(
        label="**Máximo riesgo asumible**",
        min_value=gvm * 100,
        max_value=smallcaps.std * 100,
        value=gvm * 100 * 1.1,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    max_risk /= 100

w1, w2 = ef2.pf_2a_optimal_weights(bond20y, smallcaps, corr12, max_risk)
pf_ret = ef2.pf_2a_return(w1, bond20y.avg, smallcaps.avg)

with col_a2:
    st.metric("**Riesgo cartera**", f"{round(max_risk * 100, 1)} %")
with col_a3:
    # st.metric("**Rentabilidad**", f"{round(ret * 100, 1)} %")
    st.metric("**Rent. cartera**", f"{round(pf_ret * 100, 1)} %")

fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3 = ef2.pf_2a_plot_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    correlation=corr12,
    ax=ax3,
    plot_title="Pesos en función del riesgo",
    efficient_label="",
    inefficient_label="",
    x_axis_label="Riesgo",
    y_axis_label="Rentabilidad",
    min_variance_label="",
)

ax3.scatter(
    # max_risk,
    max_risk,
    pf_ret,
    c="darkorange",
    marker="D",
    zorder=6,
    # label=f"Cartera riesgo {round(max_risk * 100, 1)} %",
)

weights_text = f"""{bond20y.name} = {round(w1 * 100)} % \n{smallcaps.name} = {round(w2 * 100)} %"""

ax3.annotate(
    weights_text,
    xy=(max_risk, pf_ret),
    xycoords="data",
    xytext=(0.7, 0.1),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "ec": "grey"},
    bbox={"boxstyle": "square", "fc": "white", "ec": "darkorange", "pad": 0.3},
)

st.pyplot(fig3, width="stretch")

text = """
Como es lógico, conforme aumentamos el riesgo que estamos dispuestos a asumir, aumenta el peso del activo de mayor riesgo. Y también debería aumentar la rentabilidad esperada de la cartera porque, en caso contrario, tenemos un activo con más riesgo pero menos rentabilidad que el otro. Eso sería absurdo. 

Vamos a repetir el ejercicio con nuestra cartera de tres activos completa y vamos a buscar los pesos óptimos para un nivel de riesgo. Y al mismo tiempo, vamos a representar una cartera muy habitual, la cartera equiponderada (mismos pesos para los tres activos) y a ver lo cerca o lejos que está de la frontera eficiente. 

#### Cartera 3 activos: Bonos LP, Small Caps y Oro
"""
st.markdown(text)
st.write("")

# Three Asset Portfolio
gvm_3a = w3a.global_min_volatility_3a(
    stds=[bond20y.std, smallcaps.std, gold.std],
    correlations=[corr12, corr13, corr23],
)
col_a1, _, col_a2, col_a3 = st.columns([3, 0.5, 1, 1])
with col_a1:
    # st.write("**Riesgo**")
    max_3a_risk = st.slider(
        label="**Máximo riesgo**",
        min_value=gvm_3a * 100,
        max_value=smallcaps.std * 100,
        value=(gvm_3a + abs(gvm_3a - smallcaps.std) / 2) * 100,
        step=0.1,
        format="%1.1f%%",
        label_visibility="visible",
    )
    max_3a_risk /= 100

solution_exists = True
try:
    data_dict = w3a.weights_3a_with_max_volatility(
        assets=[bond20y, smallcaps, gold],
        correlations=[corr12, corr13, corr23],
        max_volatility=max_3a_risk,
    )
    pf_ret = data_dict["portfolio_return"]
    w1 = data_dict["weights"][0]
    w2 = data_dict["weights"][1]
    w3 = data_dict["weights"][2]
except RuntimeError:
    solution_exists = False
    chart_text = "No hay solución para la cartera"
    pf_ret = 0.0
    w1, w2, w3 = 0.0, 0.0, 0.0

with col_a2:
    st.metric("**Riesgo cartera**", f"{round(max_3a_risk * 100, 1)} %")
with col_a3:
    st.metric("**Rent. cartera**", f"{round(pf_ret * 100, 1)} %")

fig3, ax3 = plt.subplots(figsize=(9, 6))

ax3 = ef3.plot_3a_frontier(
    asset_1=bond20y,
    asset_2=smallcaps,
    asset_3=gold,
    correlation_12=corr12,
    correlation_13=corr13,
    correlation_23=corr23,
    ax=ax3,
    num_portfolios=10_000,
    plot_title="",
    efficient_label="",
    inefficient_label="",
    min_variance_label="",
    x_axis_label="",
    y_axis_label="",
    show_assets=False,
    show_portfolio_cloud=False,
    frontier_color="steelblue",
)

weights_text = f"""{bond20y.name} = {round(w1 * 100)} % \n{smallcaps.name} = {round(w2 * 100)} % \n{gold.name} = {round(w3 * 100)} %"""

if solution_exists:
    ax3.scatter(
        # max_risk,
        max_3a_risk,
        pf_ret,
        c="darkorange",
        marker="D",
        zorder=6,
    )

    ax3.annotate(
        weights_text,
        xy=(max_3a_risk, pf_ret),
        xycoords="data",
        xytext=(0.7, 0.1),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "ec": "grey"},
        ha="right",
        bbox={
            "boxstyle": "square",
            "fc": "white",
            "ec": "darkorange",
            "pad": 0.3,
        },
    )

else:
    ax3.text(x=0.1, y=0.15, s=chart_text)
    ax3.text(x=0.1, y=0.1, s=weights_text)

equipf_ret, equipf_std = pf3.calc_pf3a_metrics(
    assets=[bond20y, smallcaps, gold],
    weights=[1 / 3, 1 / 3, 1 / 3],
    correlations=[corr12, corr13, corr23],
)
equi_label = f"Cartera equiponderada\n   Riesgo = {round(equipf_std * 100, 1)}%\n   Rentab = {round(equipf_ret * 100, 1)}%"
ax3.scatter(
    # max_risk,
    equipf_std,
    equipf_ret,
    c="magenta",
    marker="s",
    zorder=6,
    label=equi_label,
)
ax3.legend()
st.pyplot(fig3, width="stretch")

st.write("Recordemos las métricas de los activos que componen la cartera:")

metrics_table = f"""
|         |Rentabilidad                    |Volatilidad                       |
|---------|--------------------------------|----------------------------------|
|Bonos LP |{round(bond20y.avg * 100, 1)}%  |{round(bond20y.std * 100, 1)}%    |
|SmallCaps|{round(smallcaps.avg * 100, 1)}%|{round(smallcaps.std * 100, 1)}%  |
|Oro      |{round(gold.avg * 100, 1)}%     |{round(gold.std * 100, 1)}%       |
"""

corr_table = f"""
|CORRELACIÓN  |Bonos LP |SmallCaps|Oro|
|-------------|---------|---------|--------|
|**Bonos LP** |1        |{corr12} |{corr13}|
|**SmallCaps**|{corr12} |1        |{corr23}|
|**Oro**      |{corr13} |{corr23} |1       |
"""
col_a, col_b = st.columns([3, 4])
with col_a:
    st.markdown(metrics_table)
with col_b:
    st.markdown(corr_table)

text = """
Fijémonos en una consecuencia importante de las carteras Markowitz: el hecho de tener correlaciones negativas nos permite construir una cartera con un riesgo como el de los Bonos LP pero con rentabilidades como las de las Small Caps o el Oro.
"""
st.write("")
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

fig5, ax5 = plt.subplots(figsize=(9, 6))
ax5 = plot_annual_returns(monthly_rets, None, ax5, pf_label="Cartera")

st.pyplot(fig5, width="stretch")


# --- COPYRIGHT & DISCLAIMER
utils.footnote()


# ===================================================================
# # === MARKOWITZ CALCULATOR ========================================
# ===================================================================
# st.markdown("## Calculadora Markowitz")
# col_a1, _, col_a2 = st.columns([3, 0.5, 3])
# with col_a1:
#     # st.write(" ")
#     std_a = st.slider(
#         label="**Volatilidad Activo A**",
#         min_value=1,
#         max_value=50,
#         value=10,
#         step=1,
#         format="%d%%",
#         # label_visibility="collapsed",
#     )
#     std_b = st.slider(
#         label="**Volatilidad Activo B**",
#         min_value=1,
#         max_value=50,
#         value=25,
#         step=1,
#         format="%d%%",
#         # label_visibility="collapsed",
#     )
#     corr_ab = st.slider(
#         label="**Correlación A y B**",
#         min_value=-1.0,
#         max_value=1.0,
#         value=0.5,
#         step=0.1,
#         format="%0.1f",
#         # label_visibility="collapsed",
#     )

# with col_a2:
#     # st.write(" ")
#     avg_a = st.slider(
#         label="**Rentabilidad Activo A**",
#         min_value=1,
#         max_value=30,
#         value=10,
#         step=1,
#         format="%d%%",
#         # label_visibility="collapsed",
#     )
#     avg_b = st.slider(
#         label="**Rentabilidad Activo B**",
#         min_value=1,
#         max_value=30,
#         value=25,
#         step=1,
#         format="%d%%",
#         # label_visibility="collapsed",
#     )
#     min_std, *_ = ef2.pf_2a_min_volatility(std_a, std_b, corr_ab)
#     max_std = max(std_a, std_b)
#     pf_max_std = st.slider(
#         label="**Cartera: máxima volatilidad permitida**",
#         min_value=round(min_std),
#         max_value=round(max_std),
#         value=round((min_std + max_std) / 2),
#         step=1,
#         format="%d%%",
#         # label_visibility="collapsed",
#     )

# asset_a = Asset(name="Activo A", avg=avg_a / 100, std=std_a / 100)
# asset_b = Asset(name="Activo B", avg=avg_b / 100, std=std_b / 100)

# w_a, w_b = ef2.pf_2a_optimal_weights(
#     asset1=asset_a, asset2=asset_b, corr=corr_ab, target_vol=pf_max_std / 100
# )

# std_ab, avg_ab = ef2.pf_2a_risk_return(w_a, asset_a, asset_b, corr_ab)

# fig4, ax4 = plt.subplots(figsize=(9, 6))

# ax4 = ef2.pf_2a_plot_frontier(
#     asset_1=asset_a,
#     asset_2=asset_b,
#     correlation=corr_ab,
#     ax=ax4,
#     plot_title="Calculadora Markowitz",
#     efficient_label="",
#     inefficient_label="",
#     x_axis_label="Riesgo",
#     y_axis_label="Rentabilidad",
#     min_variance_label="",
# )
# ax3.set_xlim(0.0)

# ax4.scatter(
#     std_ab,
#     avg_ab,
#     c="darkorange",
#     marker="D",
#     zorder=6,
#     # label=f"Cartera riesgo {round(max_risk * 100, 1)} %",
# )

# weights_text = f"""{asset_a.name} = {round(w_a * 100)} % \n{asset_b.name} = {round(w_b * 100)} %"""

# ax4.annotate(
#     weights_text,
#     xy=(std_ab, avg_ab),
#     xycoords="data",
#     xytext=(0.7, 0.1),
#     textcoords="axes fraction",
#     arrowprops={"arrowstyle": "->", "ec": "grey"},
#     bbox={"boxstyle": "square", "fc": "white", "ec": "darkorange", "pad": 0.3},
# )

# st.pyplot(fig4, width="stretch")
