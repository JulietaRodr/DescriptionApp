#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install yfinance


# In[ ]:


pip install --upgrade yfinance


# In[79]:


import yfinance as yf
import pandas as pd

assets = ['PG', '^GSPC']
pf_data = pd.DataFrame()

for asset in assets:
    print(f"🔄 Descargando datos para: {asset}")
    data = yf.download(asset, start='2010-01-01')

    if data.empty:
        print(f"❌ No hay datos para {asset}")
        continue

    # Revisar si es MultiIndex y contiene 'Adj Close'
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

# Mostrar resultados
print("\n✅ Datos finales:")
print(pf_data.head())


# In[81]:


pf_data.tail()


# In[85]:


(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10,5))
plt.title('Desempeño normalizado desde 2010 (base 100)')
plt.ylabel('Índice (base 100)')
plt.grid(True)
plt.show()


# In[87]:


log_returns = np.log(pf_data/pf_data.shift(1))


# In[89]:


log_returns.mean()*250


# In[93]:


import numpy as np

# 1. Calcular log-returns
log_returns = np.log(pf_data / pf_data.shift(1)).dropna()

# 2. Calcular matriz de covarianza anualizada
cov_matrix = log_returns.cov() * 250

# 3. Mostrarla
print(cov_matrix)


# In[99]:


log_returns.cov() * 250


# In[95]:


log_returns.corr()


# In[101]:


num_assets=len(assets)


# In[103]:


num_assets


# In[113]:


arr = np.random.random(2)
arr


# In[115]:


arr[0] + arr[1]


# In[121]:


weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights


# In[123]:


weights[0] + weights[1]


# In[125]:


np.sum(weights*log_returns.mean())*250


# In[127]:


np.dot(weights.T, np.dot(log_returns.cov() *250, weights))


# In[129]:


np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 250, weights)))


# In[133]:


pfolio_returns = []
pfolio_volatilities = []

for x in range (1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 250, weights))))

pfolio_returns, pfolio_volatilities
    


# In[135]:


pfolio_returns = []
pfolio_volatilities = []

for x in range (1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 250, weights))))

pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

pfolio_returns, pfolio_volatilities
    


# In[141]:


portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility' : pfolio_volatilities})


# In[143]:


portfolios.head()


# In[145]:


portfolios.tail()


# In[157]:


import matplotlib.pyplot as plt

print(portfolios.columns)

portfolios_subset = portfolios[['Volatility', 'Return']]

portfolios_subset.plot(x='Volatility', y='Return', kind='scatter', figsize=(10, 6))

plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')

# Mostrar la gráfica
plt.show()


# In[ ]:




