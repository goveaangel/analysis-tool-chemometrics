# pages/3_🧼_Preprocesamiento.py
import streamlit as st

st.title("🧼 Preprocesamiento de datos")
st.markdown("Configura la **limpieza y transformación** de tus datos.")

st.subheader("1.Creacion de plantil")
st.caption("Más adelante aquí usaremos las columnas reales del dataset.")

variable_cols = st.multiselect(
    "Variables numéricas para PCA y clustering",
    options=[],
    help="Cuando carguemos datos reales, aparecerán las columnas numéricas aquí.",
)

label_col = st.selectbox(
    "Columna de etiquetas (opcional, categórica)",
    options=["(ninguna disponible todavía)"],
    help="Se usará para colorear gráficos (tipo de muestra, lote, etc.).",
)

st.markdown("---")
st.subheader("2. Manejo de valores faltantes")

missing_option = st.radio(
    "Estrategia para NaNs:",
    options=[
        "Eliminar filas con NaNs",
        "Imputar con media",
        "Imputar con mediana",
    ],
)

st.markdown("---")
st.subheader("3. Escalado / Normalización")

scaling_option = st.radio(
    "Escalado de variables:",
    options=[
        "Sin escalado",
        "Estandarización (media 0, varianza 1)",
        "Normalización Min-Max [0, 1]",
    ],
)

st.markdown("---")
st.subheader("4. Vista rápida de distribuciones")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Histogramas")
    st.info("Aquí mostraremos histogramas por variable seleccionada.")

with col2:
    st.markdown("#### Boxplots")
    st.info("Aquí mostraremos boxplots para detectar outliers.")

st.markdown("---")
st.subheader("5. Heatmap de correlación")
st.info("Aquí colocaremos un heatmap interactivo de correlación entre variables.")

st.markdown("---")
if st.button("✅ Aplicar preprocesamiento (placeholder)"):
    st.success("Más adelante este botón aplicará el preprocesamiento real a tus datos.")