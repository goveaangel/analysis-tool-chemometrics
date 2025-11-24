# app.py
import streamlit as st

st.set_page_config(
    page_title="Laboratorio de PCA y Clustering para Quimiometría",
    layout="wide",
)

st.title("Laboratorio de PCA y Clustering para Quimiometría")
st.subheader("Bienvenido al **Laboratorio de PCA y Clustering para Quimiometría**")

st.markdown(
    """
Esta aplicación está diseñada para que **estudiantes de química** puedan:

- Subir sus conjuntos de datos experimentales  
- Aplicar **preprocesamiento básico** (NaNs, escalado, selección de variables)  
- Ejecutar **PCA** y visualizar varianza, scree plots, biplots y scores  
- Realizar **K-means** y **clustering jerárquico** en el espacio de PCs  
- Exportar resultados y usar una **guía integrada de interpretación**  

---

### 🔁 Flujo de trabajo sugerido

1. **Cargar datos** → subir `.csv` / `.xlsx`  
2. **Preprocesamiento** → seleccionar variables, manejar NaNs, escalado  
3. **PCA** → elegir número de componentes, ver scree plot y biplot  
4. **Clustering** → aplicar K-means o jerárquico, visualizar clústers en PCs  
5. **Resultados & exportación** → descargar scores, clústers y figuras  

Usa la navegación en la **barra lateral izquierda** para moverte entre las secciones.
"""
)

with st.expander("📁 Dataset de ejemplo (idea)"):
    st.markdown(
        """
Más adelante podemos añadir:
- Un pequeño dataset de ejemplo (p. ej., espectros o composiciones químicas)  
- Un botón **“Cargar dataset de ejemplo”** para demostraciones rápidas en clase.
"""
    )