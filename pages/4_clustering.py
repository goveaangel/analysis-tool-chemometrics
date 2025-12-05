# pages/5_🧬_Clustering.py
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA 
from backend.clustering import (
    run_kmeans,
    run_hierarchical,
    cluster_scatter_pcs,
    dendrogram_from_pcs,
    cluster_summary_table,
)

st.title("🧬 Clustering (K-means & Jerárquico)")
st.markdown("Configura y visualiza **clústers** en el espacio de las PCs.")

# ============================
# Recuperar datos base
# ============================

# Ajusta la clave según como lo guardaste tú:
raw_df = st.session_state.get("raw_df") or st.session_state.get("raw_data")

if raw_df is None:
    st.warning("No se encontraron datos originales en sesión (raw_df/raw_data). "
               "Sube un dataset en la pestaña de carga antes de usar clustering.")
    st.stop()

# ============================
# Generar PCs fake si aún no hay PCA
# ============================
if "pca_scores" not in st.session_state:
    st.info("No se encontraron scores de PCA. Generando PCs temporales para pruebas...")

    # Tomamos solo columnas numéricas
    num_df = raw_df.select_dtypes(include="number")

    if num_df.shape[1] < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para calcular PCA y clustering.")
        st.stop()

    # El número de componentes será hasta 3 o el máximo posible
    n_components = min(3, num_df.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    scores_array = pca.fit_transform(num_df)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    scores = pd.DataFrame(scores_array, index=num_df.index, columns=pc_cols)

    st.session_state["pca_scores"] = scores
else:
    scores = st.session_state["pca_scores"]

if scores is None:
    st.warning("Primero necesitas calcular el PCA en la pestaña correspondiente.")
    st.stop()

# Detectar columnas de PCs (por si cambian nombres)
pc_cols = [c for c in scores.columns if c.upper().startswith("PC")]
if len(pc_cols) < 2:
    st.warning("Se requieren al menos PC1 y PC2 para visualizar el clustering.")
    st.stop()

# Usaremos hasta las primeras 3 PCs para el modelo
pc_model_cols = pc_cols[:3]
pc_plot_x = pc_cols[0]
pc_plot_y = pc_cols[1]

# ============================
# 1. Opciones de clustering
# ============================
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
    st.caption("Más inicializaciones pueden dar una solución más estable de K-means.")

    # Ejecutar K-means en el backend
    kmeans_result = run_kmeans(
        scores_df=scores[pc_model_cols],
        n_clusters=k,
        n_init=init_reps,
        random_state=42,
    )
    labels = kmeans_result["labels"]

else:
    linkage = st.selectbox(
        "Tipo de liga (linkage)",
        options=["ward", "complete", "average", "single"],
        index=0,
    )
    n_clusters = st.slider(
        "Número de clústers (para jerárquico)",
        min_value=2,
        max_value=10,
        value=3,
    )
    st.caption("El número de clústers se usará al cortar el dendrograma.")

    # Ejecutar clustering jerárquico
    hier_result = run_hierarchical(
        scores_df=scores[pc_model_cols],
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",
    )
    labels = hier_result["labels"]

st.markdown("---")

# ============================
# 2. Visualización de clústers
# ============================
st.subheader("2. Visualización de clústers en espacio de PCs")

col1, col2 = st.columns(2)

with col2:
    st.markdown("#### Parámetros gráficos")
    palette = st.selectbox(
        "Paleta de colores",
        options=["Viridis", "Plasma", "Cividis", "Categorical"],
    )
    point_size = st.slider("Tamaño de puntos", 3, 20, 8)
    alpha = st.slider("Transparencia (alpha)", 0.1, 1.0, 0.8)

with col1:
    st.markdown("#### Scatter plot de clústers")
    fig_scatter = cluster_scatter_pcs(
        scores_df=scores[pc_model_cols],
        labels=labels,
        palette=palette,
        point_size=point_size,
        alpha=alpha,
        pc_x=pc_plot_x,
        pc_y=pc_plot_y,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================
# 3. Dendrograma
# ============================
st.subheader("3. Dendrograma (para clustering jerárquico)")

if method == "Clúster jerárquico":
    fig_dend = dendrogram_from_pcs(scores_df=scores[pc_model_cols])
    st.plotly_chart(fig_dend, use_container_width=True)
else:
    st.caption("Cambia a 'Clúster jerárquico' para ver el dendrograma.")

st.markdown("---")

# ============================
# 4. Métricas de calidad
# ============================
st.subheader("4. Métricas de calidad del clustering")

if method == "K-means":
    sil = kmeans_result.get("silhouette", float("nan"))
    inertia = kmeans_result.get("inertia", float("nan"))
    st.write(f"Silhouette score: **{sil:.3f}**")
    st.write(f"Inercia (SSE): **{inertia:.2f}**")
else:
    sil = hier_result.get("silhouette", float("nan"))
    st.write(f"Silhouette score: **{sil:.3f}**")

st.markdown("---")

# ============================
# 5. Resumen de clústers
# ============================
st.subheader("5. Resumen de clústers")

if raw_df is None:
    st.caption("Conecta el DataFrame original en sesión para mostrar el resumen de clústers.")
else:
    summary = cluster_summary_table(original_df=raw_df, labels=labels)

    st.markdown("**Tamaño de cada clúster**")
    st.dataframe(summary["sizes"].to_frame("n_observaciones"))

    st.markdown("**Medias de variables numéricas por clúster**")
    st.dataframe(summary["means"])