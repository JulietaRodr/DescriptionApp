import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Portafolio Bonito", layout="centered")
st.title("🌸 Portafolio de Inversión 🌸")
st.markdown("Bienvenida a tu espacio personalizado para visualizar tus inversiones 💼")
st.markdown("Aquí podrás analizar el comportamiento histórico y proyectado de tu portafolio. Perfecto para mujeres emprendedoras y futuras inversionistas. 💖")

# Entrada de activos
assets_input = st.text_input("🔍 Escribe los símbolos de los activos separados por comas (ej. PG, AAPL, ^GSPC):", "PG, ^GSPC")
assets = [a.strip().upper() for a in assets_input.split(",")]
start_date = st.date_input("📅 Fecha de inicio de análisis:", pd.to_datetime("2010-01-01"))

# Descarga de datos
pf_data = pd.DataFrame()
for asset in assets:
    st.write(f"🔄 Descargando datos para: {asset}")
    data = yf.download(asset, start=start_date.strftime('%Y-%m-%d'))

    if data.empty:
        st.warning(f"❌ No hay datos para {asset}")
        continue

    if 'Adj Close' in data.columns:
        pf_data[asset] = data['Adj Close']
    elif 'Close' in data.columns:
        pf_data[asset] = data['Close']

# Mostrar datos
if not pf_data.empty:
    st.subheader("📊 Datos históricos de precios")
    st.markdown("Estos son los precios ajustados (o cierre) de tus activos seleccionados. Reflejan el valor de cada activo a lo largo del tiempo.")
    st.dataframe(pf_data.tail())

    st.subheader("📈 Desempeño normalizado desde el inicio (base 100)")
    st.markdown("Esta gráfica muestra cómo ha cambiado el valor de cada activo desde la fecha de inicio. Todos comienzan en 100, para que puedas comparar su evolución de forma clara.")
    norm_data = (pf_data / pf_data.iloc[0]) * 100
    st.line_chart(norm_data)

    # Cálculo de log-returns
    log_returns = np.log(pf_data / pf_data.shift(1)).dropna()

    st.subheader("📉 Rendimiento promedio anual 📅")
    st.markdown("Aquí puedes ver cuánto rinde en promedio cada activo al año, basado en el rendimiento diario. Te da una idea del crecimiento esperado.")
    st.write(log_returns.mean() * 250)

    cov_matrix = log_returns.cov() * 250
    st.subheader("🤝 Matriz de covarianza anualizada")
    st.markdown("La covarianza indica cómo se mueven los activos juntos. Si dos activos tienen covarianza alta, suelen subir y bajar al mismo tiempo.")
    st.dataframe(cov_matrix)

    st.subheader("📊 Matriz de correlación")
    st.markdown("La correlación va de -1 a 1. Valores cercanos a 1 significan que los activos se mueven parecido, y cercanos a -1 que se mueven de forma opuesta. Ideal para diversificar tu portafolio.")
    st.dataframe(log_returns.corr())

    # Simulación de portafolios
    st.subheader("🎯 Frontera eficiente (simulación de 1000 portafolios)")
    st.markdown("Aquí simulamos 1000 combinaciones de inversión. Cada punto representa un portafolio con diferente mezcla de activos. Buscamos el mejor equilibrio entre **riesgo** (volatilidad) y **rendimiento**.")

    num_assets = len(assets)
    pfolio_returns = []
    pfolio_volatilities = []

    for _ in range(1000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
        pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    portfolios = pd.DataFrame({
        'Return': pfolio_returns,
        'Volatility': pfolio_volatilities
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(portfolios['Volatility'], portfolios['Return'], alpha=0.5, c='orchid')
    ax.set_xlabel('Volatilidad Esperada')
    ax.set_ylabel('Rendimiento Esperado')
    ax.set_title('Frontera Eficiente 🌷')
    st.pyplot(fig)

    st.markdown("💡 Busca los puntos más arriba y hacia la izquierda: alto rendimiento y bajo riesgo. ¡Eso es lo ideal!")

else:
    st.info("Introduce símbolos válidos para comenzar.")
