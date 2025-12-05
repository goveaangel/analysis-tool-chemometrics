import math
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage as sch_linkage


# ---------- CAPA DE MODELOS ----------

def run_kmeans(scores_df, n_clusters=3, n_init=10, random_state=42):
    """
    Ejecuta K-means sobre el espacio de PCs.

    Parámetros
    ----------
    scores_df : pd.DataFrame
        DataFrame con columnas ['PC1', 'PC2', 'PC3', ...] (scores del PCA).
    n_clusters : int
        Número de clústers.
    n_init : int
        Número de inicializaciones de K-means.
    random_state : int
        Semilla para reproducibilidad.

    Devuelve
    --------
    result : dict
        {
            "model": objeto KMeans entrenado,
            "labels": np.ndarray con etiquetas de clúster,
            "inertia": float (SSE),
            "silhouette": float,
            "centroids": np.ndarray con centroides en el espacio de PCs
        }
    """
    X = scores_df.values

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(X)

    inertia = float(model.inertia_)
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        sil = float(silhouette_score(X, labels))
    else:
        sil = np.nan

    return {
        "model": model,
        "labels": labels,
        "inertia": inertia,
        "silhouette": sil,
        "centroids": model.cluster_centers_,
    }


def run_hierarchical(scores_df, n_clusters=3,
                     linkage="ward", metric="euclidean"):
    """
    Ejecuta clustering jerárquico sobre el espacio de PCs.

    Parámetros
    ----------
    scores_df : pd.DataFrame
        DataFrame con columnas ['PC1', 'PC2', 'PC3', ...].
    n_clusters : int
        Número de clústers.
    linkage : str
        'ward', 'complete', 'average', 'single'.
    metric : str
        Métrica de distancia (se ignora si linkage='ward').

    Devuelve
    --------
    result : dict
        {
            "model": objeto AgglomerativeClustering,
            "labels": np.ndarray con etiquetas de clúster,
            "silhouette": float
        }
    """
    X = scores_df.values

    if linkage == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )

    labels = model.fit_predict(X)

    if n_clusters > 1 and len(np.unique(labels)) > 1:
        sil = float(silhouette_score(X, labels))
    else:
        sil = np.nan

    return {
        "model": model,
        "labels": labels,
        "silhouette": sil,
    }


# ---------- CAPA DE VISUALIZACIÓN ----------

def cluster_scatter_pcs(scores_df,
                        labels,
                        palette="Viridis",
                        point_size=8,
                        alpha=0.8,
                        pc_x="PC1",
                        pc_y="PC2"):
    """
    Scatter plot PCx vs PCy coloreado por clúster.

    scores_df : DataFrame con scores del PCA (PC1, PC2, PC3, ...).
    labels : array-like con etiquetas de clúster (mismo largo que scores_df).
    """
    plot_df = scores_df.copy()
    plot_df["cluster"] = pd.Series(labels, index=plot_df.index).astype(str)

    palette_map = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Cividis": px.colors.sequential.Cividis,
        "Categorical": px.colors.qualitative.Set2,
    }
    color_seq = palette_map.get(palette, px.colors.qualitative.Set2)

    fig = px.scatter(
        plot_df,
        x=pc_x,
        y=pc_y,
        color="cluster",
        color_discrete_sequence=color_seq,
    )

    fig.update_traces(
        marker=dict(size=point_size, opacity=alpha),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        xaxis_title=pc_x,
        yaxis_title=pc_y,
        title=f"Clústers en espacio de {pc_x} vs {pc_y}",
        legend_title="Clúster",
    )

    return fig


def dendrogram_from_pcs(scores_df, max_samples=200, fig_height=600, label_threshold=40):
    """
    Construye un dendrograma a partir de los scores de PCs usando Plotly.

    - Si hay muchas observaciones, se toma una muestra aleatoria (max_samples).
    - Si el número de observaciones visibles es grande, se ocultan las etiquetas del eje X.
    """
    X = scores_df.values
    n = X.shape[0]

    # Muestreo para no matar el dendrograma
    if n > max_samples:
        idx = np.random.choice(n, size=max_samples, replace=False)
        X = X[idx, :]
        labels = scores_df.index[idx].astype(str).tolist()
    else:
        labels = scores_df.index.astype(str).tolist()

    n_visible = len(labels)

    fig = ff.create_dendrogram(
        X,
        labels=labels,
        orientation="top",
    )

    # Layout general
    fig.update_layout(
        title="Dendrograma en espacio de PCs",
        xaxis_title="Observaciones",
        yaxis_title="Distancia",
        height=fig_height,
    )

    # 👇 Si hay demasiadas observaciones, no mostramos etiquetas en el eje X
    if n_visible > label_threshold:
        fig.update_xaxes(showticklabels=False)

    return fig
# ---------- CAPA DE RESUMEN ----------

def cluster_summary_table(original_df, labels):
    """
    Construye tablas de resumen de clústers:

    - tamaños de clúster
    - medias de variables numéricas por clúster

    Devuelve
    --------
    summary : dict
        {
            "sizes": Serie con tamaño de cada clúster,
            "means": DataFrame con medias por clúster
        }
    """
    labels = pd.Series(labels, name="cluster")

    df_with_clusters = original_df.copy()
    df_with_clusters["cluster"] = labels.values

    sizes = df_with_clusters["cluster"].value_counts().sort_index()
    means = df_with_clusters.groupby("cluster").mean(numeric_only=True)

    return {
        "sizes": sizes,
        "means": means,
    }