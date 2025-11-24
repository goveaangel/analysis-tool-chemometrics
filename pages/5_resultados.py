# pages/6_📂_Resultados_Exportacion.py
import streamlit as st

st.title("📂 Resultados y exportación")
st.markdown(
    "Descarga resultados numéricos (scores, loadings, labels de clústers) y figuras."
)

st.subheader("1. Resultados numéricos")

st.markdown("- Scores de PCA (coordenadas de las muestras en las PCs)")
st.markdown("- Cargas de PCA (contribución de cada variable)")
st.markdown("- Labels de clústers para cada muestra")

st.info("Aquí agregaremos botones para descargar archivos .csv con estos resultados.")

st.markdown("---")
st.subheader("2. Exportación de gráficas")

st.markdown("- Scree plot")
st.markdown("- Scatter plot de scores (2D / 3D)")
st.markdown("- Biplot")
st.markdown("- Dendrograma")
st.markdown("- Silhouette plot")
st.markdown("- Heatmap de correlación")

st.info("Aquí agregaremos botones para guardar las figuras en .png o .svg.")

st.markdown("---")
if st.button("💾 Exportar todo (placeholder)"):
    st.success("Más adelante este botón generará un paquete de archivos descargables.")