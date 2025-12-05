# backend/pca.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


def run_pca_full(df):
    """
    Ejecuta PCA sobre las columnas numéricas del DataFrame preprocesado.
    Calcula todas las componentes posibles.

    Devuelve un dict con, entre otros, explained_variance_ratio.
    """
    X_num = df.select_dtypes(include="number").copy()

    if X_num.shape[1] == 0:
        raise ValueError("No hay columnas numéricas para aplicar PCA.")

    n_features = X_num.shape[1]
    X_values = X_num.values

    pca = PCA(n_components=n_features)
    scores = pca.fit_transform(X_values)

    scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(n_features)],
        index=df.index,
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_num.columns,
        columns=scores_df.columns,
    )

    return {
        "X_num": X_num,
        "scores": scores_df,
        "loadings": loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_model": pca,
        "columns": list(X_num.columns),
    }


def get_explained_variance_df(explained_variance_ratio):
    """
    DataFrame con varianza explicada y acumulada.
    """
    pcs = [f"PC{i+1}" for i in range(len(explained_variance_ratio))]
    var_exp = explained_variance_ratio
    var_exp_acum = np.cumsum(var_exp)

    df_var = pd.DataFrame(
        {
            "PC": pcs,
            "Varianza explicada": var_exp,
            "Varianza acumulada": var_exp_acum,
        }
    )
    return df_var

def plot_cumulative_variance(df_var_full, n_components):
    """
    Crea gráfica de varianza acumulada (Scree plot acumulado)
    y marca el punto correspondiente al número de componentes seleccionado.

    Parámetros
    ----------
    df_var_full : DataFrame
        Debe tener columnas: "n", "PC", "Varianza acumulada"
    n_components : int
        Punto a resaltar en la curva

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """

    # Obtener valor acumulado exacto
    selected_row = df_var_full.iloc[n_components - 1]
    var_acum = selected_row["Varianza acumulada"]

    # Línea base
    fig = px.line(
        df_var_full,
        x="n",
        y="Varianza acumulada",
        markers=True,
        title="Varianza acumulada por número de componentes",
    )

    # Eje X con etiquetas PC1, PC2, PC3, …
    fig.update_xaxes(
        tickmode="array",
        tickvals=df_var_full["n"],
        ticktext=df_var_full["PC"],
        title_text="Componentes principales",
    )

    # Punto resaltado
    fig.add_scatter(
        x=[n_components],
        y=[var_acum],
        mode="markers+text",
        marker=dict(size=10),
        text=[f"{var_acum * 100:.1f}%"],
        textposition="top center",
        name="Selección actual",
    )

    return fig

# backend/pca_plots.py

import plotly.express as px


import plotly.express as px
import plotly.graph_objects as go

def plot_scree(df_var, n_components=None, title=""):
    """
    Crea la gráfica de codo (Scree plot) y marca en rojo la componente seleccionada.

    Parámetros
    ----------
    df_var : pd.DataFrame
        Columns: ["PC", "Varianza explicada"]
    n_components : int or None
        Punto a resaltar (PC_n_components).
    title : str
        Título de la gráfica.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """

    # Selección de componentes
    if n_components is not None:
        df_plot = df_var.iloc[:n_components+2].copy()
    else:
        df_plot = df_var.copy()

    # Gráfica base (línea + marcadores)
    fig = px.line(
        df_plot,
        x="PC",
        y="Varianza explicada",
        markers=True,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Componentes principales",
        yaxis_title="Varianza explicada",
        height=400
    )

    # Agregar el punto rojo
    if n_components is not None and n_components-1 < len(df_var):
        selected_row = df_var.iloc[n_components - 1]
        fig.add_trace(
            go.Scatter(
                x=[selected_row["PC"]],
                y=[selected_row["Varianza explicada"]],
                mode="markers",
                marker=dict(color="red", size=12),
                name=f"PC{n_components} seleccionado",
            )
        )

    return fig

def get_scores_subset(pca_results, n_components):
    """
    Devuelve las primeras n componentes de los scores como DataFrame.
    """
    scores = pca_results["scores"]
    max_pcs = scores.shape[1]
    n = min(n_components, max_pcs)
    pcs = [f"PC{i+1}" for i in range(n)]
    return scores[pcs].copy()


def plot_scores_2d(scores_df, pc_x="PC1", pc_y="PC2", title=None):
    """
    Crea un scatter plot 2D de scores PCA con estilo formateado.

    Parámetros
    ----------
    scores_df : DataFrame
        Debe contener las columnas pc_x y pc_y.
    pc_x : str
        Nombre de la componente para el eje X (ej. 'PC1').
    pc_y : str
        Nombre de la componente para el eje Y (ej. 'PC2').
    title : str or None
        Título de la gráfica.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """
    if pc_x not in scores_df.columns or pc_y not in scores_df.columns:
        raise ValueError(f"Se requieren las columnas {pc_x} y {pc_y} para el scatter 2D.")

    if title is None:
        title = f"Scores PCA ({pc_x} vs {pc_y})"

    # Figura base
    fig = px.scatter(
        scores_df,
        x=pc_x,
        y=pc_y,
        title=title,
    )

    # Estilo de puntos
    fig.update_traces(
        marker=dict(
            size=7,
            opacity=0.75,
            line=dict(width=0.5, color="black"),
        )
    )

    # Estilo de fondo y ejes
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(
            title=pc_x,
            showgrid=True,
            gridcolor="#e8e8e8",
        ),
        yaxis=dict(
            title=pc_y,
            showgrid=True,
            gridcolor="#e8e8e8",
        ),
        title=dict(x=0.5, xanchor="center"),
        height=500,
    )

    return fig

def plot_scores_3d(scores_df, pc_x="PC1", pc_y="PC2", pc_z="PC3", title=None):
    """
    Crea un scatter plot 3D de scores PCA con estilo formateado.

    Parámetros
    ----------
    scores_df : DataFrame
        Debe contener las columnas pc_x, pc_y y pc_z.
    pc_x, pc_y, pc_z : str
        Nombres de las componentes para los ejes X, Y y Z.
    title : str or None
        Título de la gráfica.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """
    for col in [pc_x, pc_y, pc_z]:
        if col not in scores_df.columns:
            raise ValueError(
                f"Se requieren las columnas {pc_x}, {pc_y} y {pc_z} para el scatter 3D."
            )

    if title is None:
        title = f"Scores PCA ({pc_x} vs {pc_y} vs {pc_z})"

    # Figura base
    fig = px.scatter_3d(
        scores_df,
        x=pc_x,
        y=pc_y,
        z=pc_z,
        title=title,
    )

    # Estilo de puntos
    fig.update_traces(
        marker=dict(
            size=4,
            opacity=0.7,
            line=dict(width=0.3, color="black"),
        )
    )

    # Estilo de escena 3D
    fig.update_layout(
        scene=dict(
            xaxis_title=pc_x,
            yaxis_title=pc_y,
            zaxis_title=pc_z,
            xaxis=dict(showgrid=True, gridcolor="#e8e8e8", backgroundcolor="white"),
            yaxis=dict(showgrid=True, gridcolor="#e8e8e8", backgroundcolor="white"),
            zaxis=dict(showgrid=True, gridcolor="#e8e8e8", backgroundcolor="white"),
        ),
        title=dict(x=0.5, xanchor="center"),
        height=650,
    )

    return fig

def plot_biplot_2d(scores_df, loadings_df, pc_x="PC1", pc_y="PC2", title=None):
    """
    Biplot 2D de PCA: muestra scores (observaciones) y loadings (variables) en las PCs seleccionadas.

    Parámetros
    ----------
    scores_df : DataFrame
        Scores de las observaciones. Debe contener columnas pc_x y pc_y.
    loadings_df : DataFrame
        Loadings de las variables. Debe contener columnas pc_x y pc_y.
    pc_x, pc_y : str
        Nombres de los componentes para los ejes (ej. 'PC1', 'PC2').
    title : str or None
        Título de la gráfica.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """

    if pc_x not in scores_df.columns or pc_y not in scores_df.columns:
        raise ValueError(f"Scores no contienen columnas {pc_x} y/o {pc_y}.")
    if pc_x not in loadings_df.columns or pc_y not in loadings_df.columns:
        raise ValueError(f"Loadings no contienen columnas {pc_x} y/o {pc_y}.")

    if title is None:
        title = f"Biplot PCA ({pc_x} vs {pc_y})"

    # Scores en las PCs seleccionadas
    x_scores = scores_df[pc_x]
    y_scores = scores_df[pc_y]

    # Loadings en las PCs seleccionadas
    loadings_sub = loadings_df[[pc_x, pc_y]].copy()

    # Escalado de loadings para que quepan en el mismo plano que los scores
    max_score = max(x_scores.abs().max(), y_scores.abs().max())
    max_loading = max(loadings_sub[pc_x].abs().max(), loadings_sub[pc_y].abs().max())
    scale = 0.7 * max_score / max_loading if max_loading != 0 else 1.0

    loadings_sub_scaled = loadings_sub * scale

    # Construir figura
    fig = go.Figure()

    # 1) Puntos de las observaciones (scores)
    fig.add_trace(
        go.Scatter(
            x=x_scores,
            y=y_scores,
            mode="markers",
            name="Observaciones",
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color="black"),
            ),
        )
    )

    # 2) Flechas de las variables (loadings)
    arrow_x = []
    arrow_y = []
    for var_name, row in loadings_sub_scaled.iterrows():
        arrow_x.extend([0, row[pc_x], None])
        arrow_y.extend([0, row[pc_y], None])

    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
        )
    )

    # 3) Etiquetas de las variables en la punta de cada flecha
    fig.add_trace(
        go.Scatter(
            x=loadings_sub_scaled[pc_x],
            y=loadings_sub_scaled[pc_y],
            mode="text+markers",
            text=loadings_sub_scaled.index,
            textposition="top center",
            marker=dict(size=4, color="red"),
            name="Variables",
        )
    )

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        plot_bgcolor="white",
        xaxis=dict(
            title=pc_x,
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=True,
            zerolinecolor="#bbbbbb",
        ),
        yaxis=dict(
            title=pc_y,
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=True,
            zerolinecolor="#bbbbbb",
        ),
        height=600,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # aspecto 1:1

    return fig