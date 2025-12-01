
import streamlit as st
import pandas as pd
import numpy as np
import sys, platform

st.set_page_config(page_title="Diagnóstico - CrediFast", layout="wide")

st.title("🔧 Diagnóstico do Ambiente • CrediFast")
st.write("Este app mínimo serve para confirmar que o ambiente está OK antes de adicionar ML.")

# Mostrar versões (diagnóstico)
st.subheader("Versões de pacotes e ambiente")
st.write({
    "Python": sys.version,
    "Platform": platform.platform(),
    "streamlit": st.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__
})

# Upload de CSV
st.subheader("Upload de CSV (apenas pré-visualização)")
csv = st.file_uploader("Envie o arquivo credit_risk_dataset.csv", type=["csv"])
if csv is not None:
    df = pd.read_csv(csv)
    st.write("Shape:", df.shape)
    st.dataframe(df.head(15), use_container_width=True)
else:
    st.info("Envie o CSV para ver a prévia.")
