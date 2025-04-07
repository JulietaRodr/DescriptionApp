import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lista de activos
assets = ['PG', '^GSPC']
pf_data = pd.DataFrame()

# Descargar datos para cada activo
for asset in assets:
    print(f"🔄 Descargando datos para: {asset}")
    data = yf.download(asset, start='2010-01-01')

    if data.empty:
        print(f"❌ No hay datos para {asset}")
        continue

    if isinstance(data.columns, pd.MultiIndex):
        if ('Adj Close', asset) in data.columns:
            pf_data[asset] = data[('Adj Close', asset)]
        elif ('Close', asset) in data.columns:
            print(f"⚠️ 'Adj Close' no disponible para {asset}, usando 'Close'")
            pf_data[asset] = data[('Close', asset)]
        else:
            print(f"⚠️ Ni 'Adj Close' ni 'Close' disponibles para {asset}")
    else:
        if 'Adj Close' in data.columns:
            pf_data[asset] = data['Adj Close']
        elif 'Close' in data.columns:
            print(f"⚠️ 'Adj Close' no disponible para {asset}, usando 'Close'")
            pf_data[asset] = data['Close']

# Mostrar los primeros datos descargados
print("\n✅ Datos finales:")
print(pf_data.head())

# Normalización de datos (base 100)
pf_data_normalized = (pf_data / pf_data.iloc[0] * 100)
pf_data_normalized.plot(figsize=(10, 5))
plt.title('Desempeño normalizado desde 2010 (base 100)')
plt.ylabel('Índice (base 100)')
plt.grid(True)
plt.show()

# Calcular log-returns
log_returns = np.log(pf_data / pf_data.shift(1)).dropna()

# Calcular matriz de covarianza anualizada
cov_matrix = log_returns.cov() * 250
print("\nMatriz de Covarianza:")
print(cov_matrix)

# Calcular correlación de los log-returns
print("\nCorrelación de los activos:")
print(log_returns.corr())

# Número de activos
num_assets = len(assets)
print(f"\nNúmero de activos: {num_assets}")

# Ejemplo de aleatorización de pesos para portafolios
arr = np.random.random(2)
print("\nPesos aleatorios para dos activos:")
print(arr)

# Verificar la suma de los pesos
print("\nSuma de los pesos aleatorios:", arr[0] + arr[1])

# Calcular el retorno esperado de un portafolio con pesos aleatorios
weights = np.random.random(num_assets)
weights /= np.sum(weights)
print("\nPesos normalizados del portafolio:")
print(weights)

# Verificar la suma de los pesos normalizados
print("\nSuma de los pesos normalizados:", weights[0] + weights[1])

# Calcular el retorno esperado del portafolio
pfolio_return = np.sum(weights * log_returns.mean()) * 250
print(f"\nRetorno esperado del portafolio: {pfolio_return}")

# Calcular la volatilidad del portafolio
pfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights)))
print(f"\nVolatilidad esperada del portafolio: {pfolio_volatility}")

# Generar portafolios aleatorios y calcular su retorno y volatilidad
pfolio_returns = []
pfolio_volatilities = []

for x in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))))

# Convertir a arrays de numpy
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

# Crear un DataFrame con los resultados de los portafolios
portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})

# Mostrar las primeras filas del DataFrame
print("\nPrimeras filas de los portafolios:")
print(portfolios.head())

# Mostrar las últimas filas del DataFrame
print("\nÚltimas filas de los portafolios:")
print(portfolios.tail())

# Gráfico de los portafolios generados
portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Portafolios Aleatorios')
plt.grid(True)
plt.show()
