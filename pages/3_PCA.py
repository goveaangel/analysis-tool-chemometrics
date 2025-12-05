# pages/4_📉_PCA.py
import streamlit as st
import numpy as np
import plotly.express as px

from backend.pca import (
    run_pca_full,
    get_explained_variance_df,
    plot_cumulative_variance,
    plot_scree,
    get_scores_subset,
    plot_scores_2d,
    plot_scores_3d,
    plot_biplot_2d
)

st.title("📉 Análisis de Componentes Principales (PCA)")
st.markdown("Explora la **varianza explicada**, los **scores** y el **biplot** de tu PCA.")

#Inicializar claves session state
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

if "clean_data" not in st.session_state:
    st.session_state["clean_data"] = None

if "preprocessing_report" not in st.session_state:
    st.session_state["preprocessing_report"] = None

if "clean_data" not in st.session_state:
    st.session_state["clean_data"] = None

data = st.session_state['clean_data']

if data is None:
    st.info('''
        Primero aplica tecnicas de prepocesamiento en la seccion de **🧼 Preprocesamiento de datos**. Despues regresa a esta
        seccion para aplicar **PCA**.
    ''')
    st.stop()

# ----------------------------
# 0) Configuración de PCA
# ----------------------------

# Ejecutar PCA completo una sola vez
try:
    pca_results = run_pca_full(data)
except ValueError as e:
    st.error(str(e))
    st.stop()

var_ratio_full = pca_results["explained_variance_ratio"]
df_var_full = get_explained_variance_df(var_ratio_full)

# Agregar índice numérico para el eje x
df_var_full["n"] = np.arange(1, len(df_var_full) + 1)
max_components = len(df_var_full)

# ----------------------------
# 1) Seleccion Componentes Principales
# ----------------------------
st.markdown("#### 1️⃣ Seleccion de Componentes Principales - Varianza acumulada vs número de componentes")
col1, col2, col3 = st.columns([0.7,0.15,0.15])
# 0) Botón PCA automático (solo dispara evento)
with col3:
    pca_auto_pressed = st.button(
        "🚨 PCA automático",
        use_container_width=True,
        help="Selecciona automáticamente el menor número de componentes que expliquen al menos el 90% de la varianza."
    )

# 1) Inicializar valor por defecto en session_state si no existe
if "pca_n_components" not in st.session_state:
    st.session_state["pca_n_components"] = min(3, max_components)

# 2) Si se presionó el botón, calculamos n_auto y ACTUALIZAMOS session_state ANTES del slider
if pca_auto_pressed:
    threshold = 0.90
    mask = df_var_full["Varianza acumulada"] >= threshold

    if mask.any():
        first_idx = mask.idxmax()
        n_auto = first_idx + 1
    else:
        n_auto = len(df_var_full)

    st.session_state["pca_n_components"] = n_auto

    st.success(
        f"PCA automático seleccionado: **{n_auto} componentes** "
        f"(≈ {df_var_full.loc[n_auto - 1, 'Varianza acumulada']*100:.2f}% de varianza)."
    )

# 3) Ahora sí, creamos el slider usando el valor de session_state
with col1:
    n_components = st.slider(
        "Número de componentes",
        min_value=1,
        max_value=max_components,
        key="pca_n_components",  # el valor lo controla session_state
        help="Selecciona cuántas componentes considerar. La gráfica muestra la varianza acumulada.",
    )

# 4) Métrica de varianza acumulada
with col2:
    selected_row = df_var_full.iloc[n_components - 1]
    var_acum = selected_row["Varianza acumulada"]

    st.metric(
        label="Varianza acumulada",
        value=f"{var_acum * 100:.2f} %",
        help="Porcentaje de varianza explicada al usar las primeras n componentes.",
    )

fig_var_acum = plot_cumulative_variance(df_var_full, n_components)

st.plotly_chart(fig_var_acum, use_container_width=True)

st.caption(
    "La varianza acumulada indica cuánta información del dataset se conserva al incluir un número creciente de componentes principales. "
    "El punto seleccionado muestra cuánta variabilidad total queda explicada con la cantidad de componentes elegida."
)

st.markdown("---")

#----------------------
# 2) Varianza Explicada
#----------------------
st.subheader("2️⃣ Varianza explicada")

col1, col2 = st.columns([0.35,0.65])

# Usamos solo las primeras n_components
df_var_subset = df_var_full.iloc[:n_components+2].copy()

with col1:
    st.markdown("#### Tabla de varianza explicada")
    st.write('''
        Muestra **cuánto aporta cada componente principal** a la variabilidad total del dataset.
        La columna **varianza acumulada** indica el **porcentaje de información capturada** al considerar las primeras n PCs. La fila resaltada corresponde a la selección actual.
    ''')

    selected_idx = n_components - 1  # porque las filas empiezan en 0

    styled_df = (
        df_var_subset[["PC", "Varianza explicada", "Varianza acumulada"]]
        .style
        .format({
            "Varianza explicada": "{:.3f}",
            "Varianza acumulada": "{:.3f}",
        })
        .apply(
            lambda row: ["background-color: #d8e6ff" if row.name == selected_idx else ""]
            * len(row),
            axis=1,
        )
    )

    st.dataframe(styled_df)

with col2:
    st.markdown("#### Grafica de Codo - Varianza explicada por componente")
    st.write('''
        Visualiza la varianza explicada por cada componente. El **punto rojo** marca la PC seleccionada.
        El *codo* ayuda a identificar **cuántos componentes son suficientes** antes de que la ganancia de información se vuelva mínima.
    ''')
    fig_scree = plot_scree(df_var_full, n_components=n_components)
    st.plotly_chart(fig_scree, use_container_width=True)

st.markdown("---")

# Diccionarios para guardar info y figuras de PCA
pca_info = {
    "n_components": n_components,
    "explained_variance_ratio": var_ratio_full,
    "df_var_full": df_var_full,
    "df_var_subset": df_var_subset,
    "scores": pca_results["scores"],
    "loadings": pca_results["loadings"],
    # scatter_mode lo agregamos más abajo cuando ya existe `mode`
}

pca_figs = {
    "fig_var_acum": fig_var_acum,
    "fig_scree": fig_scree,
}

#----------------------
# 3) Scatter Plots 2D/3D
#----------------------
st.subheader("3️⃣ Scatter plot de scores")

st.markdown(
    """
    Este gráfico muestra cómo se **distribuyen las observaciones** en el espacio de los 
    componentes principales seleccionados. **Puntos cercanos** representan **observaciones 
    con patrones similares** en las variables originales. **Puntos alejados** indican 
    **diferencias** importantes o posibles **outliers**.
    """
)

# Tomamos scores hasta las primeras n componentes
scores_subset = get_scores_subset(pca_results, n_components)

# PCs disponibles según n_components
pc_options = [f"PC{i+1}" for i in range(scores_subset.shape[1])]

# ------------------ Botones 2D / 3D con estado ------------------
if "scatter_mode" not in st.session_state:
    st.session_state["scatter_mode"] = "2D"  # modo por defecto

col1, col2 = st.columns(2)
with col1:
    btn_2d = st.button("2D Scatter Plot", use_container_width=True)
with col2:
    btn_3d = st.button("3D Scatter Plot", use_container_width=True)

# Actualizar modo según botón
if btn_2d:
    st.session_state["scatter_mode"] = "2D"

if btn_3d:
    if len(pc_options) < 3:
        st.warning("Selecciona al menos 3 componentes en la sección 1 para poder usar el scatter 3D.")
    else:
        st.session_state["scatter_mode"] = "3D"

mode = st.session_state["scatter_mode"]

st.markdown("#### Selección de componentes para el scatter")

if mode == "2D":
    # 1) Selección de PCs para ejes X e Y
    col_pc1, col_pc2 = st.columns(2)
    with col_pc1:
        pc_x = st.selectbox(
            "PC eje X",
            pc_options,
            index=0,
            key="pc_x_2d"
        )
    with col_pc2:
        pc_y = st.selectbox(
            "PC eje Y",
            pc_options,
            index=1 if len(pc_options) > 1 else 0,
            key="pc_y_2d"
        )

    # Evitar que X e Y sean exactamente la misma PC
    if pc_x == pc_y and len(pc_options) > 1:
        st.warning("Selecciona PCs diferentes para los ejes X e Y.")
    else:
        # 2) Calcular porcentaje de varianza explicada de cada PC
        pc_x_num = int(pc_x.replace("PC", ""))
        pc_y_num = int(pc_y.replace("PC", ""))

        var_x = var_ratio_full[pc_x_num - 1] * 100
        var_y = var_ratio_full[pc_y_num - 1] * 100

        # 3) Construir figura (estilo ya viene desde el backend)
        fig_scores_2d = plot_scores_2d(
            scores_subset,
            pc_x=pc_x,
            pc_y=pc_y,
            title=f"Scores PCA ({pc_x} vs {pc_y})"
        )

        # 4) Mostrar gráfico
        st.plotly_chart(fig_scores_2d, use_container_width=True)

        # 5) Texto interpretativo
        st.markdown(f'''
            **{pc_x}** explica aproximadamente **{var_x:.1f}% de la varianza**; **{pc_y}** explica **{var_y:.1f}%**. 
            Los puntos cercanos representan observaciones con patrones similares en las variables originales.
        ''')

        st.caption(
            "La visualización 2D permite examinar cómo se distribuyen las observaciones en las dos "
            "direcciones principales de variación. Puntos cercanos indican patrones similares en las "
            "variables originales, mientras que puntos alejados pueden revelar diferencias importantes "
            "entre observaciones o posibles outliers."
        )

        # 6) Guardar informacion
        pca_figs["fig_scores_2d"] = fig_scores_2d
        pca_info["pc_x_2d"] = pc_x
        pca_info["pc_y_2d"] = pc_y

else:  # mode == "3D"
    if len(pc_options) < 3:
        st.warning("Selecciona al menos 3 componentes en la sección 1 para poder mostrar el scatter 3D.")
    else:
        # 1) Selección de PCs para ejes X, Y y Z
        col_pc1, col_pc2, col_pc3 = st.columns(3)
        with col_pc1:
            pc_x = st.selectbox("PC eje X", pc_options, index=0, key="pc_x_3d")
        with col_pc2:
            pc_y = st.selectbox("PC eje Y", pc_options, index=1, key="pc_y_3d")
        with col_pc3:
            pc_z = st.selectbox("PC eje Z", pc_options, index=2, key="pc_z_3d")

        # Validar que todas sean diferentes
        if len({pc_x, pc_y, pc_z}) < 3:
            st.warning("Selecciona tres PCs diferentes para los ejes X, Y y Z.")
        else:
            # 2) Calcular porcentajes de varianza explicada
            pc_x_num = int(pc_x.replace("PC", ""))
            pc_y_num = int(pc_y.replace("PC", ""))
            pc_z_num = int(pc_z.replace("PC", ""))

            var_x = var_ratio_full[pc_x_num - 1] * 100
            var_y = var_ratio_full[pc_y_num - 1] * 100
            var_z = var_ratio_full[pc_z_num - 1] * 100

            # 3) Construir figura (estilo ya viene del backend)
            fig_scores_3d = plot_scores_3d(
                scores_subset,
                pc_x=pc_x,
                pc_y=pc_y,
                pc_z=pc_z,
                title=f"Scores PCA ({pc_x} vs {pc_y} vs {pc_z})",
            )

            # 4) Mostrar gráfico
            st.plotly_chart(fig_scores_3d, use_container_width=True)

            # 5) Texto interpretativo
            st.markdown(
                f"**{pc_x}** explica aproximadamente **{var_x:.1f}% de la varianza**, "
                f"**{pc_y}** explica **{var_y:.1f}%** y "
                f"**{pc_z}** explica **{var_z:.1f}%**."
            )
            st.caption(
                "La visualización 3D permite explorar cómo se distribuyen las observaciones en tres "
                "direcciones principales de variación. Puntos cercanos representan observaciones con "
                "patrones similares; puntos alejados pueden indicar diferencias marcadas u outliers."
            )

            # 6) Guardar Informacion
            pca_figs["fig_scores_3d"] = fig_scores_3d
            pca_info["pc_x_3d"] = pc_x
            pca_info["pc_y_3d"] = pc_y
            pca_info["pc_z_3d"] = pc_z

st.markdown("---")

#----------------------
# 4) BIPLOT
#----------------------
st.subheader("4️⃣ Biplot")


st.markdown('''
    El biplot muestra, en un mismo gráfico, tanto las **observaciones** como las **variables** del PCA. 
    Cada **punto** representa una observación proyectada en las componentes principales, mientras que 
    cada **flecha** representa una variable original. La **dirección** de la flecha indica hacia dónde 
    aumenta esa variable y su **longitud** refleja qué tanta influencia tiene en la variación del plano. 
    Si varios puntos están alineados con una flecha, significa que esas observaciones tienen **valores altos** 
    en esa variable.
''')

# PCs disponibles (todas las calculadas por el PCA)
pc_options_biplot = [f"PC{i+1}" for i in range(len(var_ratio_full))]

col1, col2 = st.columns(2)

with col1:
    pc_x_bi = st.selectbox(
        "PC eje X (biplot)",
        pc_options_biplot,
        index=0,
        key="pc_x_biplot"
    )
with col2:
    pc_y_bi = st.selectbox(
        "PC eje Y (biplot)",
        pc_options_biplot,
        index=1 if len(pc_options_biplot) > 1 else 0,
        key="pc_y_biplot"
    )

if pc_x_bi == pc_y_bi and len(pc_options_biplot) > 1:
    st.warning("Selecciona PCs diferentes para los ejes X e Y del biplot.")
else:
    fig_biplot = plot_biplot_2d(
        pca_results["scores"],
        pca_results["loadings"],
        pc_x=pc_x_bi,
        pc_y=pc_y_bi,
        title=f"Biplot PCA ({pc_x_bi} vs {pc_y_bi})",
    )
    st.plotly_chart(fig_biplot, use_container_width=True)

    # Texto interpretativo corto
    st.caption(
        "Las flechas indican la dirección en la que cada variable aumenta. "
        "Variables con flechas largas y cercanas al eje de una PC contribuyen más a esa componente. "
        "Puntos alineados con una flecha comparten valores altos en esa variable."
    )

    # Guardar Informacion
    pca_figs["fig_biplot"] = fig_biplot
    pca_info["pc_x_biplot"] = pc_x_bi
    pca_info["pc_y_biplot"] = pc_y_bi

st.markdown('---')

#----------------------
# 4) Guardar informacion PCA
#----------------------

if st.button("✅ Guardar información PCA", use_container_width=True):
    try:
        st.session_state["pca_info"] = pca_info
        st.session_state["pca_figs"] = pca_figs
        st.success("Información y gráficas de PCA guardadas correctamente")
    except Exception as e:
        st.error(f"Error al guardar la información de PCA: {e}")

#----------------------
# 5) Información técnica: qué se guarda del PCA
#----------------------
st.markdown("### ℹ️ Información técnica del PCA (para el equipo)")

st.markdown("""
Este apartado resume **qué se guarda en `st.session_state`** después de usar esta pestaña
y cómo se puede reutilizar en otras partes de la aplicación.
""")

with st.expander("Ver detalle de objetos guardados en la sesión"):
    st.markdown("#### 📁 Claves principales en `st.session_state`")

    st.markdown("""
- `\"pca_info\"`: diccionario con **información numérica y estructural** del PCA.
- `\"pca_figs\"`: diccionario con las **figuras Plotly** generadas en esta pestaña.
    """)

    st.markdown("#### 🧠 Contenido de `st.session_state[\"pca_info\"]`")

    st.markdown("""
Cuando se guarda la información de PCA, `pca_info` contiene al menos:

- `\"n_components\"`: número de componentes principales seleccionados.
- `\"explained_variance_ratio\"`: arreglo con la varianza explicada por cada PC.
- `\"df_var_full\"`: DataFrame con:
    - `PC` (PC1, PC2, ...),
    - `Varianza explicada`,
    - `Varianza acumulada`,
    - `n` (índice 1..k).
- `\"df_var_subset\"`: subconjunto de `df_var_full` hasta las primeras `n_components` (y algunas extra si aplica).
- `\"scores\"`: DataFrame de scores del PCA, con columnas `PC1`, `PC2`, ..., una fila por observación.
- `\"loadings\"`: DataFrame de loadings, con filas = variables originales y columnas = `PC1`, `PC2`, ...

Dependiendo de lo que el usuario haya usado en la interfaz, también puede incluir:

- `\"pc_x_2d\"`, `\"pc_y_2d\"`: PCs usadas en el scatter plot 2D.
- `\"pc_x_3d\"`, `\"pc_y_3d\"`, `\"pc_z_3d\"`: PCs usadas en el scatter plot 3D.
- `\"pc_x_biplot\"`, `\"pc_y_biplot\"`: PCs usadas en el biplot.
    """)

    st.markdown("#### 📊 Contenido de `st.session_state[\"pca_figs\"]`")

    st.markdown("""
Este diccionario guarda las figuras de Plotly ya construidas:

- `\"fig_var_acum\"`: gráfica de varianza acumulada.
- `\"fig_scree\"`: gráfica de codo (Scree plot).
- `\"fig_scores_2d\"` (opcional): scatter plot 2D de scores (si se generó).
- `\"fig_scores_3d\"` (opcional): scatter plot 3D de scores (si se generó).
- `\"fig_biplot\"` (opcional): biplot PCA (si se generó).
    """)

    st.markdown("#### 🛠 Ejemplo de uso en otra pestaña")

    st.code("""
pca_info = st.session_state.get("pca_info", {})
pca_figs = st.session_state.get("pca_figs", {})

# Ejemplo: obtener scores reducidos a las primeras n_components
n_components = pca_info.get("n_components", 2)
scores = pca_info.get("scores")

if scores is not None:
    scores_reducidos = scores.iloc[:, :n_components]

# Ejemplo: volver a mostrar la gráfica de varianza acumulada
fig_var_acum = pca_figs.get("fig_var_acum")
if fig_var_acum is not None:
    st.plotly_chart(fig_var_acum, use_container_width=True)
""", language="python")

    # Opcional: mostrar un resumen en vivo de lo que hay
    if "pca_info" in st.session_state:
        st.markdown("#### 🔍 Vista rápida de `pca_info` (tipos de cada entrada)")
        summary_info = {k: type(v).__name__ for k, v in st.session_state["pca_info"].items()}
        st.json(summary_info)

    if "pca_figs" in st.session_state:
        st.markdown("#### 🔍 Claves disponibles en `pca_figs`")
        st.json(list(st.session_state["pca_figs"].keys()))