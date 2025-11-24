# pages/5_🧬_Clustering.py
import streamlit as st

st.title("🧬 Clustering (K-means & Jerárquico)")
st.markdown("Configura y visualiza **clústers** en el espacio de las PCs.")

st.subheader("1. Opciones de clustering")

method = st.radio(
    "Método de clustering:",
    options=["K-means", "Clúster jerárquico"],
)

if method == "K-means":
    k = st.slider(
        "Número de clústers (k)",
        min_value=2,
        max_value=10,
        value=3,
    )
    init_reps = st.number_input(
        "Número de inicializaciones (repeticiones)",
        min_value=1,
        max_value=50,
        value=10,
    )
    st.caption("Más adelante usaremos esto para estabilidad del resultado de K-means.")
else:
    linkage = st.selectbox(
        "Tipo de liga (linkage)",
        options=["ward", "complete", "average", "single"],
        index=0,
    )
    st.caption("Se usará al construir el dendrograma y la matriz de distancias.")

st.markdown("---")
st.subheader("2. Visualización de clústers en espacio de PCs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Scatter plot de clústers")
    st.info("Aquí mostraremos PC1 vs PC2 coloreados por clúster.")

with col2:
    st.markdown("#### Parámetros gráficos")
    palette = st.selectbox(
        "Paleta de colores",
        options=["Viridis", "Plasma", "Cividis", "Categorical"],
    )
    point_size = st.slider("Tamaño de puntos", 3, 20, 8)
    alpha = st.slider("Transparencia (alpha)", 0.1, 1.0, 0.8)

st.markdown("---")
st.subheader("3. Dendrograma (para clustering jerárquico)")

if method == "Clúster jerárquico":
    st.info("Aquí irá un dendrograma interactivo; podrás seleccionar un corte para definir clústers.")
else:
    st.caption("Cambia a 'Clúster jerárquico' para ver el área del dendrograma.")

st.markdown("---")
st.subheader("4. Métricas de calidad del clustering")

st.info(
    "Aquí mostraremos métricas como **silhouette score** y, opcionalmente, un **silhouette plot**.\n\n"
    "También podemos agregar inercia (K-means) u otras medidas."
)

st.markdown("---")
st.subheader("5. Resumen de clústers")

st.info(
    "En esta sección resumiremos:\n"
    "- Tamaño de cada clúster\n"
    "- Medias de variables en cada clúster\n"
    "- Centroides (para K-means)\n"
    "Todo esto se llenará cuando conectemos el backend."
)