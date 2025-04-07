import streamlit as st
prompt= st.text_area("Escribe tu prompt")
st.text(prompt)
print = "hello"

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
