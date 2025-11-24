# pages/4_📉_PCA.py
import streamlit as st

st.title("📉 Análisis de Componentes Principales (PCA)")
st.markdown(
    "Explora la **varianza explicada**, los **scores** y el **biplot** de tu PCA."
)

st.subheader("1. Configuración de PCA")

col1, col2 = st.columns(2)

with col1:
    n_components = st.slider(
        "Número de componentes principales",
        min_value=2,
        max_value=10,
        value=3,
        help="Más adelante limitaremos según el número de variables disponibles.",
    )

with col2:
    color_by = st.selectbox(
        "Colorear puntos por:",
        options=["Ninguno", "Etiqueta categórica", "Clúster (cuando esté disponible)"],
    )

st.markdown("---")
st.subheader("2. Varianza explicada")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Tabla de varianza explicada")
    st.info("Aquí mostraremos una tabla con varianza y varianza acumulada por componente.")

with col2:
    st.markdown("#### Scree plot / gráfica de codo")
    st.info("Aquí irá la gráfica de varianza explicada (Scree plot).")

st.markdown("---")
st.subheader("3. Scatter plot de scores")

show_3d = st.checkbox("Usar plot 3D (PC1, PC2, PC3 si existen)")

if show_3d:
    st.info("Aquí colocaremos un scatter plot 3D interactivo con Plotly.")
else:
    st.info("Aquí colocaremos un scatter plot 2D (por ejemplo PC1 vs PC2).")

st.markdown("---")
st.subheader("4. Biplot")

st.info(
    "Aquí mostraremos un biplot para visualizar **scores** de muestras y **cargas** de variables.\n\n"
    "Se usará para interpretar qué variables explican mejor la separación de muestras."
)