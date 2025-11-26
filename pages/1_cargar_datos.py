# pages/2_📁_Cargar_Datos.py
import streamlit as st
from backend.data_loader import load_data, get_basic_summary

if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'raw_data_summary' not in st.session_state:
    st.session_state['raw_data_summary'] = None 

st.title("📁 Cargar datos")

st.markdown("Sube tu archivo de datos en formato **.csv** o **.xlsx**.")

uploaded_file = st.file_uploader(
    "Selecciona tu archivo",
    type=["csv", "xlsx", "xls"],
    help="Archivos típicos exportados de Excel, instrumentos, etc.",
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.session_state["raw_data"] = df
        st.session_state["raw_data_summary"] = get_basic_summary(df)
        st.success("✅ Archivo cargado correctamente.")
    except Exception as e:
        st.error(f"❌ Ocurrió un error al leer el archivo: {e}")

st.markdown("---")

df = st.session_state["raw_data"]
summary = st.session_state["raw_data_summary"]

st.markdown('### 👀 Vista previa')
if df is None:
    st.info("Aquí se mostrará la vista previa cuando cargues un archivo válido.")
else:
    st.dataframe(df.head())
    
    st.write(f"**Filas:** {summary['n_rows']}")
    st.write(f"**Columnas:** {summary['n_cols']}")

# Bloques extra con más detalle
if df is not None and summary is not None:
    st.markdown("---")

    with st.expander('🔢 Tipos de datos'):
         st.dataframe(
            summary["dtypes"].to_frame("dtype"),
            height=200,
            use_container_width=True
        )
    with st.expander("🔍 Valores faltantes por columna"):
        st.write(summary["na_counts"])

    with st.expander("📊 Resumen estadístico"):
        st.write(summary["describe"])
