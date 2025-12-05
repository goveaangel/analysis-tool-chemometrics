# pages/6_📂_Resultados_Exportacion.py

import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score

from backend.preprocesing import correlation_heatmap

# ====================================
# Helpers
# ====================================

def fig_to_bytes(fig, fmt="png"):
    """
    Convierte una figura Plotly a bytes de imagen (png o svg).
    Requiere kaleido instalado para fig.to_image().
    """
    if fig is None:
        return None, None
    try:
        if fmt == "png":
            mime = "image/png"
        elif fmt == "svg":
            mime = "image/svg+xml"
        else:
            raise ValueError("Formato no soportado.")
        data = fig.to_image(format=fmt)
        return data, mime
    except Exception as e:
        st.warning(f"No se pudo exportar una figura (¿falta kaleido?): {e}")
        return None, None


def get_silhouette(scores_df, labels, sil_from_info=None):
    """
    Usa el silhouette guardado si existe; si no, intenta recalcularlo.
    """
    if sil_from_info is not None:
        return sil_from_info

    if scores_df is None or labels is None:
        return None

    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return None

    try:
        return float(silhouette_score(scores_df.values, labels))
    except Exception:
        return None


# ====================================
# Título y carga de session_state
# ====================================

st.title("📂 Resultados y exportación")
st.markdown(
    """
Esta pestaña resume todo el flujo:

1. **Datos y preprocesamiento**  
2. **PCA** (scores, loadings, varianza explicada)  
3. **Clustering** (si lo corriste)

Desde aquí puedes **descargar CSVs, figuras** y un **reporte de interpretación**.
"""
)

# Datos base
raw_df = st.session_state.get("raw_data", None)
clean_df = st.session_state.get("clean_data", None)
prep_report = st.session_state.get("preprocessing_report", None)

# PCA
pca_info = st.session_state.get("pca_info", None)
pca_figs = st.session_state.get("pca_figs", {})

# Clustering
cluster_info = st.session_state.get("cluster_info", None)
cluster_figs = st.session_state.get("cluster_figs", {})

# ====================================
# 1️⃣ Datos y preprocesamiento
# ====================================

st.markdown("---")
st.subheader("1️⃣ Datos y preprocesamiento")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Descarga de datasets**")

    if raw_df is not None:
        raw_csv = raw_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Datos crudos (raw_data.csv)",
            data=raw_csv,
            file_name="raw_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay `raw_data` en sesión (ve a 📁 Cargar Datos).")

    if clean_df is not None:
        clean_csv = clean_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Datos preprocesados (clean_data.csv)",
            data=clean_csv,
            file_name="clean_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay `clean_data` en sesión (ve a 🧼 Preprocesamiento).")

with col_b:
    st.markdown("**Reporte de preprocesamiento (JSON)**")
    if prep_report is not None:
        report_json_str = json.dumps(
        prep_report,
        indent=2,
        ensure_ascii=False,
        default=str,   # 👈 clave para convertir int64, float64, etc.
        )

        st.download_button(
            "⬇️ preprocessing_report.json",
            data=report_json_str.encode("utf-8"),
            file_name="preprocessing_report.json",
            mime="application/json",
        )

    else:
        st.info("No hay `preprocessing_report` en sesión (généralo en 🧼 Preprocesamiento).")

# Heatmap de correlación (recalculado aquí para poder exportar)
fig_corr = None
if raw_df is not None:
    try:
        fig_corr = correlation_heatmap(raw_df, method="pearson")
    except Exception as e:
        st.warning(f"No se pudo generar el heatmap de correlación: {e}")
else:
    st.info("No hay datos crudos para calcular el heatmap de correlación.")

# ====================================
# 2️⃣ Resultados de PCA
# ====================================

st.markdown("---")
st.subheader("2️⃣ Resultados de PCA")

if pca_info is None:
    st.info("No se encontró `pca_info` en sesión. Ve a 📉 PCA y pulsa **'✅ Guardar información PCA'**.")
else:
    scores = pca_info.get("scores", None)
    loadings = pca_info.get("loadings", None)
    df_var_full = pca_info.get("df_var_full", None)
    var_ratio_full = np.array(pca_info.get("explained_variance_ratio", []))

    col1, col2, col3 = st.columns(3)

    # Scores
    with col1:
        if scores is not None:
            csv_scores = scores.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Scores PCA (pca_scores.csv)",
                data=csv_scores,
                file_name="pca_scores.csv",
                mime="text/csv",
            )
        else:
            st.info("Scores no encontrados en `pca_info`.")

    # Loadings
    with col2:
        if loadings is not None:
            csv_loadings = loadings.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Loadings PCA (pca_loadings.csv)",
                data=csv_loadings,
                file_name="pca_loadings.csv",
                mime="text/csv",
            )
        else:
            st.info("Loadings no encontrados en `pca_info`.")

    # Varianza explicada
    with col3:
        if df_var_full is not None:
            csv_var = df_var_full.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Varianza explicada (pca_variance.csv)",
                data=csv_var,
                file_name="pca_variance.csv",
                mime="text/csv",
            )
        else:
            st.info("Tabla de varianza no encontrada en `pca_info`.")

    st.markdown("#### Interpretación rápida del PCA")

    if var_ratio_full.size > 0:
        n_pcs = len(var_ratio_full)
        var_pc1 = var_ratio_full[0] * 100
        var_pc2 = var_ratio_full[1] * 100 if n_pcs >= 2 else 0.0
        var_pc3 = var_ratio_full[2] * 100 if n_pcs >= 3 else 0.0
        var_acum_2 = var_ratio_full[:2].sum() * 100 if n_pcs >= 2 else var_pc1
        var_acum_3 = var_ratio_full[:3].sum() * 100 if n_pcs >= 3 else var_acum_2

        st.markdown(
            f"""
- El PCA calculó **{n_pcs} componentes principales**.
- **PC1** explica ≈ **{var_pc1:.1f}%** de la varianza total.
- **PC2** explica ≈ **{var_pc2:.1f}%**.
- Juntas, **PC1–PC2** capturan ≈ **{var_acum_2:.1f}%** de la variabilidad.
- Si consideras hasta **PC3**, la varianza acumulada sube a ≈ **{var_acum_3:.1f}%**.

Lectura rápida:
- Muestras cercanas en el scatter de scores tienen perfiles de variables similares.
- Las cargas grandes (positivas o negativas) indican qué variables más influyen
  en la separación a lo largo de cada componente.
"""
        )
    else:
        st.info("No se encontró `explained_variance_ratio` en `pca_info`.")

# ====================================
# 3️⃣ Resultados de clustering
# ====================================

st.markdown("---")
st.subheader("3️⃣ Resultados de clustering")

if cluster_info is None:
    st.info("No se encontró `cluster_info` en sesión. Ve a 🧬 Clustering y pulsa **'✅ Guardar información de clustering'**.")
else:
    method = cluster_info.get("method", "Desconocido")
    labels = cluster_info.get("labels", None)
    n_obs = cluster_info.get("n_obs", None)
    n_clusters = cluster_info.get("n_clusters", None)
    sil_stored = cluster_info.get("silhouette", None)

    cluster_sizes = cluster_info.get("cluster_sizes", None)
    cluster_means = cluster_info.get("cluster_means", None)

    st.markdown(f"**Método utilizado:** `{method}`")

    base_df = raw_df if raw_df is not None else clean_df

    if cluster_sizes is not None and cluster_means is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tamaño de cada clúster**")
            sizes_df = cluster_sizes.to_frame("n_observaciones")
            st.dataframe(sizes_df)
        with col2:
            st.markdown("**Medias por clúster (variables numéricas)**")
            means_df = cluster_means
            st.dataframe(means_df)

        # Descargas
        col3, col4 = st.columns(2)
        with col3:
            csv_sizes = sizes_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Tamaños de clúster (cluster_sizes.csv)",
                data=csv_sizes,
                file_name="cluster_sizes.csv",
                mime="text/csv",
            )
        with col4:
            csv_means = means_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Medias de clúster (cluster_means.csv)",
                data=csv_means,
                file_name="cluster_means.csv",
                mime="text/csv",
            )
    else:
        st.info("No se encontró el resumen de clústers en `cluster_info`.")

    # Silhouette global
    scores_for_sil = None
    if pca_info is not None:
        scores_for_sil = pca_info.get("scores", None)

    sil_val = get_silhouette(scores_for_sil, labels, sil_from_info=sil_stored)

    st.markdown("#### Interpretación rápida de los clústers")

    if n_clusters is not None and n_obs is not None:
        st.markdown(f"- Se formaron **{n_clusters} clústers** a partir de **{n_obs} observaciones**.")

    texto_sil = (
        f"El **silhouette global** es ≈ **{sil_val:.3f}**."
        if sil_val is not None
        else "No se pudo calcular el silhouette (se necesitan al menos 2 clústers bien definidos)."
    )
    st.markdown(
        f"""
- La tabla de tamaños indica cuántas observaciones caen en cada clúster.
- Las medias por clúster permiten ver en qué variables se diferencian los grupos.
- {texto_sil}

Regla general:
- Silhouette cercano a **1** → clústers bien separados.
- Cerca de **0** → clústers solapados.
- Negativo → observaciones potencialmente mal asignadas.
"""
    )

# ====================================
# 4️⃣ Exportación de figuras
# ====================================

st.markdown("---")
st.subheader("4️⃣ Exportación de figuras")

st.markdown(
    """
Desde aquí puedes descargar las figuras principales del análisis en formato de imagen
(útiles para reportes, presentaciones, artículos, etc.).
"""
)

img_format = st.radio(
    "Formato de imagen",
    options=["png", "svg"],
    index=0,
    horizontal=True,
)

# Mapa etiqueta → figura
figures = {}

# PCA figs
if isinstance(pca_figs, dict):
    figures["Varianza acumulada (PCA)"] = pca_figs.get("fig_var_acum", None)
    figures["Scree plot (PCA)"] = pca_figs.get("fig_scree", None)
    figures["Scores 2D (PCA)"] = pca_figs.get("fig_scores_2d", None)
    figures["Scores 3D (PCA)"] = pca_figs.get("fig_scores_3d", None)
    figures["Biplot (PCA)"] = pca_figs.get("fig_biplot", None)

# Clustering figs
if isinstance(cluster_figs, dict):
    figures["Scatter clústers (PCs)"] = cluster_figs.get("scatter", None)
    figures["Dendrograma (PCs)"] = cluster_figs.get("dendrogram", None)

# Heatmap correlación
figures["Heatmap de correlación"] = fig_corr

for label, fig in figures.items():
    if fig is None:
        st.info(f"{label}: figura no disponible en esta sesión.")
        continue
    img_bytes, mime = fig_to_bytes(fig, fmt=img_format)
    if img_bytes is not None:
        st.download_button(
            label=f"⬇️ {label} ({img_format.upper()})",
            data=img_bytes,
            file_name=f"{label.replace(' ', '_').lower()}.{img_format}",
            mime=mime,
        )

# ====================================
# 5️⃣ Reporte global (.txt) y ZIP
# ====================================

st.markdown("---")
st.subheader("5️⃣ Reporte global y exportación completa")

# ----- Reporte de texto -----
report_lines = []

# Preprocesamiento
if prep_report is not None:
    report_lines.append("=== PREPROCESAMIENTO ===")
    report_lines.append(
        f"Filas antes / después: {prep_report['rows_before']} → {prep_report['rows_after']}"
    )
    report_lines.append(
        f"Columnas antes / después: {prep_report['cols_before']} → {prep_report['cols_after']}"
    )
    report_lines.append(f"Estrategia de NaNs: {prep_report['nan_strategy']}")
    report_lines.append(f"Método de escalado: {prep_report['scaling_method']}")
    report_lines.append(
        f"Método de outliers / acción: {prep_report['outlier_method']} / {prep_report['outlier_action']}"
    )
    report_lines.append(
        f"Transformaciones aplicadas: {prep_report['transform_method']} "
        f"en {prep_report['transformed_columns'] if prep_report['transformed_columns'] else 'ninguna'}"
    )
    report_lines.append("")

# PCA
if pca_info is not None and "explained_variance_ratio" in pca_info:
    report_lines.append("=== PCA ===")
    vr = np.array(pca_info["explained_variance_ratio"])
    n_pcs = len(vr)
    var_pc1 = vr[0] * 100
    var_pc2 = vr[1] * 100 if n_pcs >= 2 else 0.0
    var_pc3 = vr[2] * 100 if n_pcs >= 3 else 0.0
    var_acum_2 = vr[:2].sum() * 100 if n_pcs >= 2 else var_pc1
    var_acum_3 = vr[:3].sum() * 100 if n_pcs >= 3 else var_acum_2

    report_lines.append(f"Número total de PCs: {n_pcs}")
    report_lines.append(f"PC1 explica ≈ {var_pc1:.2f}% de la varianza.")
    report_lines.append(f"PC2 explica ≈ {var_pc2:.2f}%.")
    report_lines.append(f"PC1–PC2 acumulan ≈ {var_acum_2:.2f}% de la varianza.")
    report_lines.append(f"PC1–PC3 acumulan ≈ {var_acum_3:.2f}% de la varianza.")
    report_lines.append(
        "En el espacio de scores, observaciones cercanas representan patrones similares "
        "en las variables originales; observaciones alejadas pueden indicar perfiles muy "
        "distintos u outliers."
    )
    report_lines.append("")

# Clustering
if cluster_info is not None:
    report_lines.append("=== CLUSTERING ===")
    labels = cluster_info.get("labels", None)
    n_clusters = cluster_info.get("n_clusters", None)
    n_obs = cluster_info.get("n_obs", None)

    if n_clusters is not None and n_obs is not None:
        report_lines.append(f"Se ajustaron {n_clusters} clústers sobre {n_obs} observaciones.")

    cluster_sizes = cluster_info.get("cluster_sizes", None)
    if cluster_sizes is not None:
        report_lines.append("Tamaño de clústeres:")
        for idx, val in cluster_sizes.items():
            report_lines.append(f"  - Clúster {idx}: {int(val)} observaciones")

    scores_for_sil = None
    if pca_info is not None:
        scores_for_sil = pca_info.get("scores", None)
    sil_val = get_silhouette(scores_for_sil, labels, sil_from_info=cluster_info.get("silhouette", None))

    if sil_val is not None:
        report_lines.append(f"Silhouette global ≈ {sil_val:.3f}.")
        report_lines.append(
            "Valores cercanos a 1 indican clústers bien separados; valores cercanos a 0 "
            "indican solapamiento; valores negativos sugieren posibles asignaciones incorrectas."
        )
    report_lines.append("")

report_text = "\n".join(report_lines) if report_lines else "No hay información suficiente en sesión para generar un reporte."

st.markdown("#### Vista previa del reporte")
st.text_area("Reporte generado", value=report_text, height=250)

st.download_button(
    "⬇️ Descargar reporte (.txt)",
    data=report_text.encode("utf-8"),
    file_name="reporte_quimiometria.txt",
    mime="text/plain",
)

# ----- Exportar TODO en ZIP -----
st.markdown("#### Exportar todo en un solo archivo (.zip)")

if st.button("💾 Generar paquete completo (.zip)"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Datos
        if raw_df is not None:
            zf.writestr("raw_data.csv", raw_df.to_csv(index=True))
        if clean_df is not None:
            zf.writestr("clean_data.csv", clean_df.to_csv(index=True))
        if prep_report is not None:
            zf.writestr(
                "preprocessing_report.json",
                     json.dumps(
                                prep_report,
                                indent=2,
                                ensure_ascii=False,
                                default=str,
                            ),
                        )

        # PCA
        if pca_info is not None:
            if "scores" in pca_info and pca_info["scores"] is not None:
                zf.writestr("pca_scores.csv", pca_info["scores"].to_csv(index=True))
            if "loadings" in pca_info and pca_info["loadings"] is not None:
                zf.writestr("pca_loadings.csv", pca_info["loadings"].to_csv(index=True))
            if "df_var_full" in pca_info and pca_info["df_var_full"] is not None:
                zf.writestr("pca_variance.csv", pca_info["df_var_full"].to_csv(index=False))

        # Clustering
        if cluster_info is not None:
            labels = cluster_info.get("labels", None)
            if labels is not None:
                if pca_info is not None and "scores" in pca_info and pca_info["scores"] is not None:
                    idx = pca_info["scores"].index
                else:
                    idx = pd.Index(range(len(labels)), name="sample")

                labels_df = pd.DataFrame({"cluster": labels}, index=idx)
                zf.writestr("cluster_labels.csv", labels_df.to_csv(index=True))

            cluster_sizes = cluster_info.get("cluster_sizes", None)
            cluster_means = cluster_info.get("cluster_means", None)
            if cluster_sizes is not None:
                zf.writestr(
                    "cluster_sizes.csv",
                    cluster_sizes.to_frame("n_observaciones").to_csv(index=True),
                )
            if cluster_means is not None:
                zf.writestr(
                    "cluster_means.csv",
                    cluster_means.to_csv(index=True),
                )

        # Figuras (si kaleido está disponible)
        all_figs_for_zip = {}

        if isinstance(pca_figs, dict):
            all_figs_for_zip["pca_var_acum"] = pca_figs.get("fig_var_acum")
            all_figs_for_zip["pca_scree"] = pca_figs.get("fig_scree")
            all_figs_for_zip["pca_scores_2d"] = pca_figs.get("fig_scores_2d")
            all_figs_for_zip["pca_scores_3d"] = pca_figs.get("fig_scores_3d")
            all_figs_for_zip["pca_biplot"] = pca_figs.get("fig_biplot")

        if isinstance(cluster_figs, dict):
            all_figs_for_zip["cluster_scatter"] = cluster_figs.get("scatter")
            all_figs_for_zip["cluster_dendrogram"] = cluster_figs.get("dendrogram")

        all_figs_for_zip["corr_heatmap"] = fig_corr

        for name, fig in all_figs_for_zip.items():
            if fig is None:
                continue
            img_bytes, _mime = fig_to_bytes(fig, fmt="png")
            if img_bytes is not None:
                zf.writestr(f"{name}.png", img_bytes)

    buffer.seek(0)
    st.download_button(
        "⬇️ Descargar paquete completo (.zip)",
        data=buffer.getvalue(),
        file_name="resultados_completos.zip",
        mime="application/zip",
    )
