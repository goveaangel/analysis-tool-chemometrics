import streamlit as st
from backend.preprocesing import basic, run_manual_preprocessing

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
else:
    st.success("Datos crudos disponibles desde **📁 Cargar datos**.")
    st.caption(f"Dimensiones actuales del dataset: `{raw_df.shape[0]} filas × {raw_df.shape[1]} columnas`")

    st.markdown("---")

    # ==========================================================
    # 1) PLANTILLAS AUTOMÁTICAS
    # ==========================================================
    st.header("1️⃣ Plantillas de preprocesamiento")

    st.markdown(
        """
Selecciona una **plantilla automática** para aplicar un preprocesamiento recomendado
(Selección de variables, NaNs, escalado, etc.).  
Más adelante podrás refinar estos pasos manualmente.
"""
    )

    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        plantilla = st.selectbox(
            "Elige una plantilla:",
            [
                "Ninguna",
                "Plantilla básica (PCA estándar)",
            ],
            help="Las plantillas aplican un pipeline predefinido de preprocesamiento.",
        )

    with col_p2:
        st.caption("Resumen rápido de la plantilla seleccionada:")
        if plantilla == "Ninguna":
            st.write("- No se aplicará ningún preprocesamiento automático.")
        elif plantilla == "Plantilla básica (PCA estándar)":
            st.write("- Selección de variables numéricas.")
            st.write("- Manejo simple de NaNs.")
            st.write("- Eliminación de columnas con poca información.")
            st.write("- Escalado tipo autoscaling (z-score).")

    aplicar_plantilla = st.button("⚙️ Aplicar plantilla seleccionada")

    if aplicar_plantilla:
        if plantilla == "Ninguna":
            st.info("No se aplicó ninguna plantilla. Puedes configurar el preprocesamiento manualmente en las secciones siguientes.")
        elif plantilla == "Plantilla básica (PCA estándar)":
            try:
                # Llamamos a la plantilla básica del backend
                clean_df, report = basic(raw_df)

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

    st.markdown("---")

    # ==========================================================
    # 2) SELECCIÓN DE COLUMNAS
    # ==========================================================
    st.header("2️⃣ Selección de columnas")

    st.markdown('Elige qué variables se usarán en el análisis.')

    # Detección rápida de tipos (solo UI, sin aplicar nada todavía)
    numeric_cols = list(raw_df.select_dtypes(include="number").columns)
    categorical_cols = list(raw_df.select_dtypes(include=["object", "category"]).columns)

    st.markdown("**Variables numéricas detectadas:**")
    st.caption(", ".join(numeric_cols) if numeric_cols else "_No se detectaron columnas numéricas._")

    selected_vars = st.multiselect(
        "Selecciona variables numéricas para el análisis:",
        options=numeric_cols,
        default=numeric_cols,  # por ahora seleccionamos todas por defecto
    )

    st.caption(f'Has seleccionado **{len(selected_vars)}** variables numéricas')

    st.markdown("---")

    # ==========================================================
    # 3) MANEJO DE NaNs
    # ==========================================================
    st.header("3️⃣ Manejo de valores faltantes (NaNs)")

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
    # 4) ESCALADO / NORMALIZACIÓN
    # ==========================================================
    st.header("4️⃣ Escalado y normalización")

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
    # 5) OUTLIERS
    # ==========================================================
    st.header("5️⃣ Detección y tratamiento de outliers")

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
                "Solo marcar outliers",
                "Excluir filas outliers",
                "No hacer nada (solo diagnóstico)",
            ],
        )

    st.caption("⚠️ Por ahora esto es solo configuración de UI. La lógica se conectará en el backend.")

    st.markdown("---")

    # ==========================================================
    # 6) TRANSFORMACIONES DE VARIABLES
    # ==========================================================
    st.header("6️⃣ Transformaciones de variables")

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
    # 7) GENERAR REPORTE Y CONFIRMAR PREPROCESAMIENTO
    # ==========================================================
    st.header("7️⃣ Generar reporte y confirmar preprocesamiento")

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

    generar = st.button("✅ Generar datos preprocesados (pendiente de backend)")

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

