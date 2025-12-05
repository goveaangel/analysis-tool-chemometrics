# pages/5_🧬_Clustering.py
import streamlit as st
import numpy as np

from backend.clustering import (
    run_kmeans,
    run_hierarchical,
    cluster_scatter_pcs,
    dendrogram_from_pcs,
    cluster_summary_table,
)

st.title("🧬 Clustering (K-means & Jerárquico)")
st.markdown(
    """
En esta sección agrupamos las **muestras** en clústers usando sus coordenadas en el
espacio de **componentes principales (PCs)**.

La idea es encontrar grupos de muestras que se comportan de forma **similar** en las
variables originales (temperaturas, concentraciones, etc.), pero trabajando en un
espacio reducido y más interpretable.
"""
)

st.caption(
    "Primero se aplica PCA para reducir la dimensión; después, sobre esos scores de PCA, "
    "se ejecutan los algoritmos de clustering."
)

# ============================
# 0. Recuperar datos base
# ============================

clean_data = st.session_state.get("clean_data", None)
raw_df_state = st.session_state.get("raw_df", None)
raw_data_state = st.session_state.get("raw_data", None)

if clean_data is not None:
    raw_df = clean_data
elif raw_df_state is not None:
    raw_df = raw_df_state
elif raw_data_state is not None:
    raw_df = raw_data_state
else:
    raw_df = None

if raw_df is None:
    st.warning(
        "No se encontraron datos en sesión. "
        "Primero carga y preprocesa un dataset en las pestañas anteriores."
    )
    st.stop()

# ============================
# 0.5 Recuperar info de PCA
# ============================

pca_info = st.session_state.get("pca_info", None)

if pca_info is None or "scores" not in pca_info:
    st.info(
        "Ve a la pestaña **📉 PCA**, ejecuta el análisis y presiona "
        "el botón **'✅ Guardar información PCA'** antes de usar clustering."
    )

scores = pca_info["scores"]

if scores is None or scores.empty:
    st.error("Los scores de PCA están vacíos. Revisa la configuración en la pestaña de PCA.")
    st.stop()

# Detectar columnas de PCs
pc_cols = [c for c in scores.columns if c.upper().startswith("PC")]
if len(pc_cols) < 2:
    st.warning("Se requieren al menos PC1 y PC2 para visualizar el clustering.")
    st.stop()

# Usaremos hasta las primeras 3 PCs para el modelo (si existen)
pc_model_cols = pc_cols[: min(3, len(pc_cols))]
pc_plot_x = pc_model_cols[0]
pc_plot_y = pc_model_cols[1] if len(pc_model_cols) > 1 else pc_model_cols[0]

# ============================
# 1. Opciones de clustering
# ============================
st.subheader("1. Opciones de clustering")

st.markdown(
    """
Selecciona el tipo de **algoritmo de agrupamiento** que quieres usar:

- **K-means**: crea clústers “esféricos” alrededor de centroides.
- **Clúster jerárquico**: construye una jerarquía (árbol) de agrupamientos.
"""
)

method = st.radio(
    "Método de clustering:",
    options=["K-means", "Clúster jerárquico"],
    help=(
        "K-means busca k grupos compactos en el espacio de las PCs.\n"
        "El clustering jerárquico construye un árbol de similitud entre muestras."
    ),
)

if method == "K-means":
    st.markdown("#### Parámetros de K-means")
    st.caption(
        "K-means reparte las muestras en k grupos, de manera que cada muestra quede "
        "cerca del centro (centroide) de su clúster."
    )

    k = st.slider(
        "Número de clústers (k)",
        min_value=2,
        max_value=10,
        value=3,
        help="Número de grupos que quieres formar en los datos.",
    )
    init_reps = st.number_input(
        "Número de inicializaciones (repeticiones)",
        min_value=1,
        max_value=50,
        value=10,
        help=(
            "K-means depende del punto de partida. Probar varias inicializaciones "
            "ayuda a encontrar una solución más estable."
        ),
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
    st.markdown("#### Parámetros del clustering jerárquico")
    st.caption(
        "El clustering jerárquico no parte de un número fijo de grupos: construye un árbol "
        "de similitud y luego se corta ese árbol para obtener k clústers."
    )

    linkage = st.selectbox(
        "Tipo de liga (linkage)",
        options=["ward", "complete", "average", "single"],
        index=0,
        help=(
            "ward: minimiza el aumento de varianza dentro de los clústers.\n"
            "complete: usa la distancia máxima entre puntos de dos clústers.\n"
            "average: usa la distancia promedio.\n"
            "single: usa la distancia mínima (tiende a generar cadenas)."
        ),
    )
    n_clusters = st.slider(
        "Número de clústers (para jerárquico)",
        min_value=2,
        max_value=10,
        value=3,
        help="Número de grupos que se cortarán a partir del dendrograma.",
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

st.markdown(
    """
El siguiente gráfico muestra las **muestras proyectadas en las PCs** (por ejemplo PC1 vs PC2),
coloreadas según el clúster al que pertenecen.

- Puntos **del mismo color** pertenecen al **mismo clúster**.
- Puntos **cercanos** indican muestras con comportamiento similar.
- Grupos separados sugieren **patrones diferentes** de proceso o calidad.
"""
)

col1, col2 = st.columns(2)

with col2:
    st.markdown("#### Parámetros gráficos")
    palette = st.selectbox(
        "Paleta de colores",
        options=["Viridis", "Plasma", "Cividis", "Categorical"],
        help="Cambia la paleta de colores para diferenciar mejor los clústers.",
    )
    point_size = st.slider(
        "Tamaño de puntos", 3, 20, 8, help="Controla el tamaño de cada muestra en el scatter."
    )
    alpha = st.slider(
        "Transparencia (alpha)",
        0.1,
        1.0,
        0.8,
        help="Valores más bajos hacen los puntos más transparentes (útil si hay muchos).",
    )

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

st.caption(
    f"Cada punto es una muestra en el plano formado por {pc_plot_x} y {pc_plot_y}. "
    "Los colores indican a qué clúster pertenece cada muestra."
)

st.markdown("---")

# ============================
# 3. Dendrograma
# ============================
st.subheader("3. Dendrograma (para clustering jerárquico)")

st.markdown(
    """
El **dendrograma** muestra cómo se van fusionando las muestras en clústers
cuando aumenta el nivel de similitud. Es útil para:

- Ver si hay grupos bien separados.
- Decidir un número razonable de clústers.
"""
)

if method == "Clúster jerárquico":
    fig_dend = dendrogram_from_pcs(scores_df=scores[pc_model_cols])
    st.plotly_chart(fig_dend, use_container_width=True)
    st.caption(
        "Las uniones más bajas indican muestras muy similares. Cortar el árbol a una cierta altura "
        "equivale a elegir un número de clústers."
    )
else:
    st.caption("Cambia a 'Clúster jerárquico' para ver el dendrograma.")

st.markdown("---")

# ============================
# 4. Métricas de calidad
# ============================
st.subheader("4. Métricas de calidad del clustering")

st.markdown(
    """
Estas métricas ayudan a evaluar qué tan “bien definidos” están los clústers:

- **Silhouette score**: mide qué tan separado está cada clúster de los demás.
  Valores cercanos a 1 indican clústers compactos y bien separados; cercanos a 0,
  clústers poco claros.
- **Inercia (SSE)**: solo para K-means. Es la suma de distancias al centroide;
  valores más bajos indican clústers más compactos (pero siempre comparando con el mismo dataset).
"""
)

if method == "K-means":
    sil = kmeans_result.get("silhouette", float("nan"))
    inertia = kmeans_result.get("inertia", float("nan"))
    st.write(f"Silhouette score: **{sil:.3f}**")
    st.write(f"Inercia (SSE): **{inertia:.2f}**")
else:
    sil = hier_result.get("silhouette", float("nan"))
    st.write(f"Silhouette score: **{sil:.3f}**")

st.caption(
    "Estas métricas no sustituyen el criterio del experto, pero permiten comparar configuraciones "
    "diferentes de clústers (por ejemplo, distintos valores de k)."
)

st.markdown("---")

# ============================
# 5. Resumen de clústers
# ============================
st.subheader("5. Resumen de clústers")

st.markdown(
    """
Aquí se muestra un resumen de **tamaño de cada clúster** y las **medias de las variables**
dentro de cada grupo. Esto ayuda a interpretar qué caracteriza a cada clúster:

- Clústers con **mayor promedio** en cierta variable pueden asociarse, por ejemplo, a
  condiciones de proceso más calientes, mayores concentraciones, etc.
- Diferencias entre clústers sugieren **regímenes de operación** o **tipos de muestra** distintos.
"""
)

if raw_df is None:
    st.caption("Conecta el DataFrame original en sesión para mostrar el resumen de clústers.")
else:
    summary = cluster_summary_table(original_df=raw_df, labels=labels)

    st.markdown("**Tamaño de cada clúster**")
    st.dataframe(summary["sizes"].to_frame("n_observaciones"))

    st.markdown("**Medias de variables numéricas por clúster**")
    st.dataframe(summary["means"])

# ============================
# 6. Guardar información de clustering en session_state
# ============================

cluster_info = {
    "method": method,
    "pc_model_cols": pc_model_cols,
    "labels": labels,
    "n_obs": len(labels),
}

if method == "K-means":
    cluster_info["n_clusters"] = int(k)
    cluster_info["silhouette"] = float(sil) if not np.isnan(sil) else None
    cluster_info["inertia"] = float(inertia) if not np.isnan(inertia) else None
else:
    cluster_info["n_clusters"] = int(n_clusters)
    cluster_info["silhouette"] = float(sil) if not np.isnan(sil) else None
    cluster_info["linkage"] = linkage

cluster_info["cluster_sizes"] = summary["sizes"]
cluster_info["cluster_means"] = summary["means"]

cluster_figs = {
    "scatter": fig_scatter,
}
if method == "Clúster jerárquico" and "fig_dend" in locals():
    cluster_figs["dendrogram"] = fig_dend

st.markdown("---")
st.subheader("6. Guardar información de clustering")

st.markdown(
    """
Al guardar, esta pestaña almacena en la sesión:

- La configuración del algoritmo (método, número de clústers, linkage, etc.).
- Las etiquetas de clúster para cada muestra.
- El resumen de tamaños y medias.
- Las figuras principales (scatter y dendrograma).

Esto permitirá reutilizar estos resultados en la pestaña de **Resultados** o en un reporte.
"""
)

if st.button("✅ Guardar información de clustering", use_container_width=True):
    try:
        st.session_state["cluster_info"] = cluster_info
        st.session_state["cluster_figs"] = cluster_figs
        st.success("Información y gráficas de clustering guardadas correctamente.")
    except Exception as e:
        st.error(f"Error al guardar la información de clustering: {e}")