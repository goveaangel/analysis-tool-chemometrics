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

- Subir sus conjuntos de datos  
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

st.markdown('---')

with st.expander('🧹 ¿Por qué es importante limpiar los datos?'):
    st.write("""
    Limpiar los datos garantiza que las conclusiones que obtengas sean confiables. 
    Si tu información contiene errores, valores faltantes, duplicados o formatos 
    inconsistentes, cualquier análisis puede volverse engañoso. La limpieza corrige 
    estos problemas para que los datos reflejen la realidad con mayor precisión.
    """)

with st.expander('🔧 ¿Qué es el preprocesamiento y por qué lo necesito?'):
    st.write("""
    El preprocesamiento prepara los datos para que los algoritmos puedan entenderlos 
    correctamente. Esto incluye convertir texto en números, normalizar escalas, 
    codificar categorías o estandarizar valores. Sin este paso, las comparaciones 
    entre variables no son justas y los métodos estadísticos no funcionan adecuadamente.
    """)

with st.expander('🧠 ¿Qué es PCA (Análisis de Componentes Principales)?'):
    st.write("""
    PCA es una técnica matemática que toma un conjunto grande de variables y las transforma 
    en un conjunto más pequeño que conserva la mayor parte de la información. Es como crear 
    “resúmenes” que capturan las variaciones más importantes del dataset, permitiendo 
    visualizar y analizar datos complejos de forma más simple.
    """)

with st.expander('✨ ¿Por qué usar PCA?'):
    st.write("""
    PCA te ayuda a simplificar datos sin perder lo esencial. Reduce el ruido, elimina 
    redundancias y permite visualizar patrones que serían difíciles de percibir entre 
    muchas columnas. También mejora la eficiencia y claridad de otros análisis, como el clustering.
    """)

with st.expander('🗂️ ¿Qué es el clustering o agrupamiento?'):
    st.write("""
    El clustering es una técnica que agrupa automáticamente tus datos según sus similitudes. 
    Cada grupo representa elementos que se comportan de manera parecida. No necesitas conocer 
    de antemano cuántos grupos hay: el algoritmo encuentra patrones y los organiza por ti.
    """)

with st.expander('📘 ¿Por qué usar clustering?'):
    st.write("""
    Porque te permite descubrir estructuras ocultas en tu información. Puedes identificar 
    grupos con comportamientos similares, segmentar productos, detectar anomalías o entender 
    mejor la variabilidad en tus datos. Es útil cuando quieres explorar sin tener etiquetas predefinidas.
    """)

with st.expander('🤝 ¿Por qué combinar PCA + clustering?'):
    st.write("""
    PCA reduce la complejidad y deja solo la información más relevante; clustering encuentra 
    grupos dentro de esa versión simplificada. Al combinarlos, los grupos se vuelven más claros, 
    más definidos y más fáciles de interpretar. PCA limpia el camino y el clustering revela los patrones.
    """)

with st.expander('🎯 ¿Para qué me sirve esta aplicación?'):
    st.write("""
    La aplicación te ayuda a entender tus datos aunque no tengas conocimientos técnicos. Te guía 
    paso a paso para limpiarlos, simplificarlos y descubrir patrones. El objetivo es transformar 
    tus datos en ideas claras que te ayuden a tomar mejores decisiones.
    """)

with st.expander('🔎 ¿Cómo interpreto los resultados?'):
    st.write("""
    Cada gráfica muestra relaciones importantes entre tus datos. Puntos cercanos representan 
    comportamientos similares; puntos alejados indican diferencias importantes. Los colores 
    muestran los grupos encontrados, y las explicaciones te ayudan a comprender qué significa 
    cada patrón y cómo usarlo.
    """)
