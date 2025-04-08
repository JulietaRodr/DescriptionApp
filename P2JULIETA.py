import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolio", layout="centered")
st.title("📈 Análisis de Portafolio con Datos de Yahoo Finance")

# Selección de activos
assets = ['PG', '^GSPC']
st.write(f"Activos seleccionados: {assets}")

pf_data = pd.DataFrame()

# Descargar datos
for asset in assets:
    st.write(f"🔄 Descargando datos para: {asset}")
    data = yf.download(asset, start='2010-01-01')

    if data.empty:
        st.error(f"❌ No hay datos para {asset}")
        continue

    # Revisar si es MultiIndex y contiene 'Adj Close'
    if isinstance(data.columns, pd.MultiIndex):
        if ('Adj Close', asset) in data.columns:
            pf_data[asset] = data[('Adj Close', asset)]
        elif ('Close', asset) in data.columns:
            st.warning(f"⚠️ 'Adj Close' no disponible para {asset}, usando 'Close'")
            pf_data[asset] = data[('Close', asset)]
        else:
            st.warning(f"⚠️ Ni 'Adj Close' ni 'Close' disponibles para {asset}")
    else:
        if 'Adj Close' in data.columns:
            pf_data[asset] = data['Adj Close']
        elif 'Close' in data.columns:
            st.warning(f"⚠️ 'Adj Close' no disponible para {asset}, usando 'Close'")
            pf_data[asset] = data['Close']

# Mostrar datos
st.subheader("✅ Datos de Precios")
st.dataframe(pf_data.tail())

# Normalizar y graficar
st.subheader("📊 Desempeño normalizado desde 2010 (base 100)")
pf_data_norm = (pf_data / pf_data.iloc[0]) * 100
pf_data_norm.plot(figsize=(10,5))
plt.title('Desempeño normalizado desde 2010 (base 100)')
plt.ylabel('Índice (base 100)')
plt.grid(True)
st.pyplot(plt.gcf())

# Calcular log-returns
log_returns = np.log(pf_data / pf_data.shift(1)).dropna()

# Matriz de covarianza anualizada
st.subheader("📈 Matriz de covarianza anualizada")
cov_matrix = log_returns.cov() * 250
st.dataframe(cov_matrix)

# Simulación de portafolios
num_assets = len(assets)
pfolio_returns = []
pfolio_volatilities = []

for _ in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))))

pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

portfolios = pd.DataFrame({
    'Return': pfolio_returns,
    'Volatility': pfolio_volatilities
})

# Gráfica de frontera eficiente
st.subheader("💡 Simulación de portafolios aleatorios")
portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6), alpha=0.5)
plt.xlabel('Volatilidad Esperada')
plt.ylabel('Retorno Esperado')
plt.title('Simulación de 1000 portafolios')
st.pyplot(plt.gcf())
