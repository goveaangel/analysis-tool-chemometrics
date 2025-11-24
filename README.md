# 🔬 Multivariate Analysis Tool for Chemometrics
A Streamlit-based PCA & Clustering Laboratory

---

## 📂 Repository Structure

```bashso
CHEMOMETRICS/
│
├── app.py                     # Main Streamlit entry point
│
├── frontend/                  # All Streamlit pages (multi-page app)
│   ├── 1_cargar_datos.py
│   ├── 2_preprocesamiento.py
│   ├── 3_PCA.py
│   ├── 4_clustering.py
│   └── 5_resultados.py
│
├── backend/                   # Core logic for loading and preprocessing
│   ├── __init__.py
│   └── data_loader.py
│
├── data/                      # Example datasets (if needed)
│   └── chemometrics_example.xlsx
│
├── models/                    # Future ML/PCA/clustering models will go here
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Project Overview
This project is a web-based interactive platform built with Streamlit for performing multivariate data analysis in a chemometrics context.
It is designed for students and researchers to:

- Upload chemical or experimental datasets
- Clean and preprocess data
- Run Principal Component Analysis (PCA)
- Perform K-Means and Hierarchical Clustering
- Visualize results dynamically
- Export cleaned datasets and scores
- Follow a guided workflow inspired by chemometric methodology

The application aims to be a teaching tool and a practical analysis assistant.

---

## 📊 Methodology

1. Load data 
2. Preprocessing
- Select variables
- Handle NaNs
- Automatic templates
- Scale/normalize
- Detect outliers
- Apply transformations
3. PCA
- Compute components
- Visualize variance
- Scatter/biplots
4. Clustering
- K-Means / Hierarchical
- Silhouette score
- Different linkage methods
5.  Output
- Export clean data
- Export cluster labels
- Export PCA scores
- Generate automated report

---

## 📈 Results Summary

---

## 🧠 Key Insights

---

## 🧩 Technologies Used
- Python 3.10+
- Streamlit — UI framework
- Pandas — Data handling
- NumPy — Numerical operations
- Scikit-learn (upcoming) — PCA & clustering
- SciPy (upcoming) — Hierarchical clustering
- Plotly & Matplotlib (upcoming) — Visualizations


---

## 📘 Reports

---

## 👥 Authors

- **Diego Vértiz Padilla**  
- **José Ángel Govea García**  
- **Daniel Alberto Sánchez Fortiz**  
- **Augusto Ley Rodríguez**  
- **Ángel Esparza Enríquez**

Tecnológico de Monterrey, School of Engineering and Sciences  
Guadalajara, Jalisco — México  

---

## 🔒 Confidentiality
This project is intended for academic and instructional purposes.
No confidential or proprietary datasets should be uploaded into the tool unless explicitly permitted.
