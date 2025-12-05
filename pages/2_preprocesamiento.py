import streamlit as st

from backend.preprocesing import (
    basic,
    run_manual_preprocessing,
    correlation_heatmap,
    boxplot_variables_grid,
    histogram_variables_grid,
    detect_column_types,  
)

# Inicializar claves mínimas en session_state
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

if "clean_data" not in st.session_state:
    st.session_state["clean_data"] = None

if "preprocessing_report" not in st.session_state:
    st.session_state["preprocessing_report"] = None

st.title("🧼 Preprocesamiento de datos")

st.markdown(
    """
Aquí configuras cómo se van a **limpiar y preparar los datos** antes de aplicar PCA y clustering.

Puedes usar una **plantilla automática** (recomendado para empezar) o configurar todo de forma **manual** en las secciones siguientes.
"""
)

raw_df = st.session_state["raw_data"]

if raw_df is None:
    st.info(
        "Primero carga un dataset en la sección **📁 Cargar datos**. "
        "Después regresa aquí para preprocesarlo."
    )
    st.stop()

#Detección rápida de tipos
numeric_cols = list(raw_df.select_dtypes(include="number").columns)
categorical_cols = list(raw_df.select_dtypes(include=["object", "category"]).columns)

st.success("Datos crudos disponibles desde **📁 Cargar datos**.")
st.caption(f"Dimensiones actuales del dataset: `{raw_df.shape[0]} filas × {raw_df.shape[1]} columnas`")

st.markdown("---")

# ==========================================================
# 0) PLANTILLAS AUTOMÁTICAS
# ==========================================================
with st.expander('🚨 Plantillas de Preprocesamiento'):
    st.header("1️⃣ Plantillas de preprocesamiento")

    st.markdown(
        """
    Selecciona una **plantilla automática** para aplicar un preprocesamiento recomendado
    (Selección de variables, NaNs, escalado, etc.).  
    """
    )

    plantilla = st.selectbox(
        "Elige una plantilla:",
        [
            "Ninguna",
            "Plantilla básica",
        ],
        help="Las plantillas aplican un pipeline predefinido de preprocesamiento.",
    )
    st.caption("Resumen rápido de la plantilla seleccionada:")
    if plantilla == "Ninguna":
        st.write("- No se aplicará ningún preprocesamiento automático.")
    elif plantilla == "Plantilla básica":
        st.write("- Deteccion automatica columnas numercias y categoricas.")
        st.write("- Elimina columnas y filas con muchos valores faltantes.")
        st.write("- Imputa los NAs restantes con la mediana.")
        st.write("- Elimina columnas con varianza casi nula.")
        st.write("- Escala los datos (z-score).")

    aplicar_plantilla = st.button("⚙️ Aplicar plantilla seleccionada")

    if aplicar_plantilla:
        if plantilla == "Ninguna":
            st.info("No se aplicó ninguna plantilla. Puedes configurar el preprocesamiento manualmente en las secciones siguientes.")
        elif plantilla == "Plantilla básica":
            try:
                # Llamamos a la plantilla básica del backend
                clean_df, report = basic(raw_df)

                # Recuperar info de columnas desde el reporte
                numeric_cols_basic = report.get("pure_numeric_cols", list(clean_df.columns))
                categorical_cols_basic = report.get("categorical_cols", [])

                st.session_state["selected_numeric_vars"] = numeric_cols_basic
                st.session_state["selected_categorical_vars"] = categorical_cols_basic

                if categorical_cols_basic:
                    st.session_state["categorical_data"] = raw_df[categorical_cols_basic].copy()
                else:
                    st.session_state["categorical_data"] = None

                # Guardamos resultados en session_state
                st.session_state["clean_data"] = clean_df
                st.session_state["preprocessing_report"] = report

                st.success("✅ Plantilla básica aplicada correctamente.")

                st.markdown("### 👀 Vista previa de los datos preprocesados")
                st.dataframe(clean_df.head(15), use_container_width=True)

                st.markdown("### 🧾 Resumen rápido del preprocesamiento")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.write("**Filas antes / después:**")
                    st.write(f"{report['rows_before']} → {report['rows_after']}")
                    st.write("**Columnas antes / después:**")
                    st.write(f"{report['cols_before']} → {report['cols_after']}")
                    st.write("**Columnas eliminadas por NaNs:**")
                    st.write(report["dropped_nan_columns"] or "Ninguna")
                    st.write("**Columnas eliminadas por baja varianza:**")
                    st.write(report["dropped_low_var_columns"] or "Ninguna")

                with col_b:
                    st.write("**Estrategia de NaNs:**", report["nan_strategy"])
                    st.write("**Método de escalado:**", report["scaling_method"])
                    st.write("**Outliers (método / acción):**")
                    st.write(f"{report['outlier_method']} / {report['outlier_action']}")
                    st.write("**Transformaciones:**")
                    if report["transform_method"] == "none":
                        st.write("No se aplicaron transformaciones.")
                    else:
                        st.write(
                            f"{report['transform_method']} en {report['transformed_columns']}"
                        )

                with st.expander("🔍 Ver reporte completo (JSON)"):
                    st.json(report)

            except Exception as e:
                st.error(f"❌ Ocurrió un error al aplicar la plantilla básica: {e}")

# ==========================================================
# 0.5) GRAFICAS Y VISUALIZACIONES
# ==========================================================
with st.expander('📊 Graficas y visualizaciones'):

    st.subheader('Heatmap Correlación')
    st.write(
        """
    Un *heatmap de correlación* muestra qué tan relacionadas están tus variables.
    Los valores van de **-1** a **+1**:

    - **+1 (correlación positiva fuerte):** ambas variables aumentan o disminuyen juntas.  
    - **0 (sin correlación):** no hay relación lineal clara.  
    - **-1 (correlación negativa fuerte):** cuando una sube, la otra baja.

    Los colores más **intensos** indican relaciones más fuertes.  
    Esto ayuda a detectar variables redundantes, posibles agrupamientos y señales de problemas como multicolinealidad.
    """
    )
    
    
    fig_raw = correlation_heatmap(raw_df, method="pearson")
    st.plotly_chart(fig_raw, use_container_width=True)
    

    st.subheader("Visualizaciones de Distribución y Rango por Variable")
    st.caption(
        "Selecciona una o varias variables numéricas para explorar su distribución, "
        "rango y posibles valores atípicos."
    )

    if len(numeric_cols) == 0:
        st.warning("El dataset no tiene variables numéricas para generar boxplots.")
    else:
        selected_vars = []
        # Umbral para decidir si usamos checkboxes o multiselect
        max_checkbox_vars = 20
        if len(numeric_cols) <= max_checkbox_vars:
            # --- Versión UX pro con checkboxes en grid ---
            st.caption("Marca las variables que quieras visualizar:")
            # Número de columnas del grid de checkboxes
            if len(numeric_cols) <= 6:
                n_cols = 2
            elif len(numeric_cols) <= 12:
                n_cols = 3
            else:
                n_cols = 4

            cols = st.columns(n_cols)

            for i, col_name in enumerate(numeric_cols):
                with cols[i % n_cols]:
                    checked = st.checkbox(
                        col_name,
                        key=f"boxvar_{col_name}",
                    )
                    if checked:
                        selected_vars.append(col_name)
        else:
            # --- Si hay muchísimas variables, usamos multiselect ---
            selected_vars = st.multiselect(
                "Variables numéricas disponibles",
                options=numeric_cols,
                help="Selecciona las variables que quieres visualizar en los boxplots.",
            )

        # Feedback rápido de lo que eligió el usuario
        st.caption(f"Variables seleccionadas: {len(selected_vars)}")

        st.subheader("Boxplots de las variables seleccionadas")
        st.write("""
            Los *boxplots* muestran **cómo se distribuye cada variable** y permiten **detectar valores atípicos**.  
            La caja representa **el rango donde se concentra la mayor parte de los datos**, la línea central es la
            **mediana**, y los puntos fuera de los **bigotes** indican **posibles outliers**.  
            Esto ayuda a identificar **alta dispersión**, **asimetría** o **comportamientos anómalos** que pueden 
            afectar el **análisis multivariado**.
        """)
        # Generar figura solo si hay selección
        fig_box = boxplot_variables_grid(raw_df, variables=selected_vars)

        if not selected_vars or fig_box is None:
            st.info("Selecciona al menos una variable para visualizar sus boxplots.")
        else:
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Histogramas de las variables seleccionadas")
        st.write("""
            Los histogramas muestran **cómo se distribuyen los valores** de cada variable.  
            Permiten identificar si los datos son **simétricos**, si están **concentrados en ciertos rangos**, o si
            existen **colas largas** y **valores atípicos**.  
            Son útiles para entender **la forma de la distribución** antes de aplicar **transformaciones** o 
            **modelos multivariados**.
        """)
        fig_hist = histogram_variables_grid(raw_df, variables=selected_vars, nbins=30)

        if not selected_vars or fig_hist is None:
            st.info("Selecciona al menos una variable para visualizar sus histogramas.")
        else:
            st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('---')

# ==========================================================
# 1) SELECCIÓN DE COLUMNAS
# ==========================================================
st.header("1️⃣ Selección de columnas")
st.markdown("Elige qué variables se usarán en el análisis.")

df = raw_df  

# Detección automática de tipos 

types_info = detect_column_types(df)

all_cols = types_info["all_cols"]
numeric_cols = types_info["numeric_cols"]
non_numeric_cols = types_info["non_numeric_cols"]
likely_cat_from_numeric = types_info["likely_cat_from_numeric"]
detected_categorical_cols = types_info["categorical_cols"]
pure_numeric_cols = types_info["pure_numeric_cols"]

col_num, col_cat = st.columns(2)

with col_num:
    st.subheader("Variables numéricas")

    st.markdown("Se detectaron automáticamente las siguientes variables:")
    st.caption(
        ", ".join(numeric_cols)
        if numeric_cols else "_No se detectaron variables numéricas._"
    )

    selected_vars = st.multiselect(
        "Selecciona variables numéricas:",
        options=numeric_cols,
        default=pure_numeric_cols,   # solo numéricas "puras"
    )

    st.caption(f"Has seleccionado **{len(selected_vars)}** variables numéricas.")

with col_cat:
    st.subheader("Variables categóricas")

    st.markdown("Se detectaron automáticamente las siguientes variables:")
    st.caption(
        ", ".join(detected_categorical_cols)
        if detected_categorical_cols else "_No se detectaron variables categóricas._"
    )

    selected_cats = st.multiselect(
        "Selecciona variables categóricas:",
        options=all_cols,
        default=detected_categorical_cols,
    )

    st.caption(f"Has seleccionado **{len(selected_cats)}** variables categóricas.")

# Guardar selección en session_state
st.session_state["selected_numeric_vars"] = selected_vars
st.session_state["selected_categorical_vars"] = selected_cats

# También guardamos un DataFrame solo con las categóricas seleccionadas
if selected_cats:
    st.session_state["categorical_data"] = raw_df[selected_cats].copy()
else:
    st.session_state["categorical_data"] = None

st.markdown('---')

# ==========================================================
# 2) MANEJO DE NaNs
# ==========================================================
st.header("2️⃣ Manejo de valores faltantes (NaNs)")

st.markdown(
    """
Configura cómo tratar los **valores faltantes** en las variables seleccionadas.
Esto es importante para que PCA y clustering funcionen correctamente.
"""
)

col_nan1, col_nan2 = st.columns(2)

with col_nan1:
    nan_strategy = st.radio(
        "Estrategia principal:",
        options=[
            "Eliminar filas con NaNs",
            "Imputar con media",
            "Imputar con mediana",
        ],
    )

with col_nan2:
    max_nan_col = st.slider(
        "Eliminar columnas con más de este porcentaje de NaNs:",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Columnas con un porcentaje de NaNs mayor a este valor se eliminarán.",
    )

    max_nan_row = st.slider(
        "Eliminar filas con más de este porcentaje de NaNs:",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Filas con demasiados NaNs pueden distorsionar el análisis.",
    )

with st.expander("🔍 Vista rápida de NaNs por columna"):
    st.write(raw_df[selected_vars].isna().sum())

st.markdown("---")

# ==========================================================
# 3) ESCALADO / NORMALIZACIÓN
# ==========================================================
st.header("3️⃣ Escalado y normalización")

st.markdown(
    """
El escalado controla cómo contribuye cada variable al análisis multivariado.  
En quimiometría, es común usar **autoscaling (z-score)** para PCA.
"""
)

scaling_method = st.radio(
    "Selecciona el método de escalado:",
    options=[
        "Sin escalado",
        "Centrado a la media",
        "Autoscaling (z-score)",
        "Min–Max [0, 1]",
        "Pareto (quimiometría)",
    ],
)

st.markdown("---")

# ==========================================================
# 4) OUTLIERS
# ==========================================================
st.header("4️⃣ Detección y tratamiento de outliers")

st.markdown(
    """
Los **outliers** pueden rotar fuertemente los componentes principales y alterar clústers.  
Aquí podrás detectarlos y decidir qué hacer con ellos.
"""
)

col_out1, col_out2 = st.columns(2)

with col_out1:
    outlier_method = st.selectbox(
        "Método de detección:",
        options=[
            "Ninguno",
            "Z-score (|z| > 3)",
            "IQR (1.5×IQR)",
        ],
    )

with col_out2:
    outlier_action_ui = st.selectbox(
        "Acción a tomar:",
        options=[
            "No hacer nada (solo diagnóstico)",
            "Solo marcar outliers",
            "Excluir filas outliers",
        ],
    )

st.markdown("---")

# ==========================================================
# 5) TRANSFORMACIONES DE VARIABLES
# ==========================================================
st.header("5️⃣ Transformaciones de variables")

st.markdown(
    """
Las transformaciones pueden ayudar a **reducir sesgos** y a que el PCA refleje mejor la estructura química real.
"""
)

col_tr1, col_tr2 = st.columns(2)

with col_tr1:
    vars_to_transform = st.multiselect(
        "Variables a transformar (opcional):",
        options=selected_vars,
    )

with col_tr2:
    transform_type = st.selectbox(
        "Tipo de transformación:",
        options=[
            "Ninguna",
            "Log10 (solo valores > 0)",
            "Log natural (ln)",
            "Raíz cuadrada",
            # Futuro: "SNV (Standard Normal Variate)", "MSC", etc.
        ],
    )

if vars_to_transform and transform_type != "Ninguna":
    st.caption(
        f"Se aplicará **{transform_type}** a: "
        + ", ".join(vars_to_transform)
    )
else:
    st.caption("No se ha configurado ninguna transformación por ahora.")

st.markdown("---")

# ==========================================================
# 6) GENERAR REPORTE Y CONFIRMAR PREPROCESAMIENTO
# ==========================================================
st.header("6️⃣ Generar reporte y confirmar preprocesamiento")

st.markdown(
    """
Revisa un resumen de la configuración de preprocesamiento y genera el conjunto final de datos
que se usará en las secciones de **PCA** y **Clustering**.
"""
)

st.markdown("**Resumen de configuración (solo UI, sin aplicar aún):**")

st.write("- Plantilla seleccionada:", plantilla)
st.write("- Nº de variables seleccionadas:", len(selected_vars))
st.write("- Estrategia de NaNs:", nan_strategy)
st.write(f"- Umbral NaNs por columna: {max_nan_col}%")
st.write(f"- Umbral NaNs por fila: {max_nan_row}%")
st.write("- Método de escalado:", scaling_method)
st.write("- Método de outliers:", outlier_method)
st.write("- Acción sobre outliers:", outlier_action_ui)
st.write("- Transformación:", transform_type if transform_type != "Ninguna" else "Ninguna")
st.write("- Variables transformadas:", ", ".join(vars_to_transform) if vars_to_transform else "Ninguna")

generar = st.button("✅ Generar datos preprocesados")

if generar:
    try:
        clean_df, report = run_manual_preprocessing(
            raw_df,
            selected_vars=selected_vars,
            nan_strategy=nan_strategy,
            max_nan_col=max_nan_col,
            max_nan_row=max_nan_row,
            scaling_method=scaling_method,
            outlier_method=outlier_method,
            outlier_action_ui=outlier_action_ui,
            transform_type=transform_type,
            vars_to_transform=vars_to_transform,
        )

        report["selected_numeric_vars"] = selected_vars
        report["selected_categorical_vars"] = selected_cats

        # Guardar en session_state para usar en PCA / Clustering
        st.session_state["clean_data"] = clean_df
        st.session_state["preprocessing_report"] = report

        st.success("✅ Pipeline manual aplicado correctamente.")

        st.markdown("### 👀 Vista previa de los datos preprocesados")
        st.dataframe(clean_df.head(15), use_container_width=True)

        st.markdown("### 🧾 Resumen rápido del preprocesamiento aplicado")
        col_a, col_b = st.columns(2)

        with col_a:
            st.write("**Filas antes / después:**")
            st.write(f"{report['rows_before']} → {report['rows_after']}")
            st.write("**Columnas antes / después:**")
            st.write(f"{report['cols_before']} → {report['cols_after']}")
            st.write("**Columnas eliminadas por NaNs:**")
            st.write(report["dropped_nan_columns"] or "Ninguna")
            st.write("**Columnas eliminadas por baja varianza:**")
            st.write(report["dropped_low_var_columns"] or "Ninguna")

        with col_b:
            st.write("**Estrategia de NaNs interna:**", report["nan_strategy"])
            st.write("**Método de escalado:**", report["scaling_method"])
            st.write("**Outliers (método / acción):**")
            st.write(f"{report['outlier_method']} / {report['outlier_action']}")
            st.write("**Transformaciones:**")
            if report["transform_method"] == "none":
                st.write("No se aplicaron transformaciones.")
            else:
                st.write(
                    f"{report['transform_method']} en {report['transformed_columns']}"
                )

        with st.expander("🔍 Ver reporte completo (JSON)"):
            st.json(report)

    except Exception as e:
        st.error(f"❌ Ocurrió un error al generar los datos preprocesados: {e}")

