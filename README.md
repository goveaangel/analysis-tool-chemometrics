# 🔬 Multivariate Analysis Tool for Chemometrics
A Streamlit-based PCA & Clustering Laboratory

---

## 📂 Repository Structure

```bash
CHEMOMETRICS/
│
├── Inicio.py                     # Main Streamlit entry point
│
├── pages/                        # Streamlit multi-page interface
│   ├── 1_cargar_datos.py
│   ├── 2_preprocesamiento.py
│   ├── 3_PCA.py
│   ├── 4_clustering.py
│   └── 5_resultados.py
│
├── backend/                      # Core logic for PCA, clustering & preprocessing
│   ├── __init__.py
│   ├── data_loader.py
│   ├── pca.py
│   ├── preprocessing.py
│   └── clustering.py             
│
├── data/
│   └── chemometrics_example.xlsx  # Example dataset
│
├── requirements.txt
└── README.md           
```

---

## ⚙️ Project Overview

This project is a web-based interactive platform built with Streamlit for performing multivariate data analysis in a chemometrics context.  
It is designed for students, analysts, and researchers who need to:

- Upload chemical or experimental datasets  
- Clean and preprocess data  
- Perform Principal Component Analysis (PCA)  
- Apply K-Means and Hierarchical Clustering  
- Visualize results through interactive plots  
- Export cleaned datasets, PCA scores, and clustering results  
- Follow a guided workflow aligned with standard chemometric practices  

The application functions both as an educational tool and as a practical assistant for exploratory multivariate analysis.

---

## 🚀 Quick Start

1. Clone the repository
```bash
git clone https://github.com/goveaangel/analysis-tool-chemometrics
cd analysis-tool-chemometrics
```

2. Create a virtual environment
```bash
python -m venv chemometrics
```

3. Activate environment

Mac/Linux:
```bash
source chemometrics/bin/activate
```

Windows:
```bash
chemometrics\Scripts\activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run Streamlit
```bash
streamlit run app.py
```

---

## 📊 Methodology

| Step | Description |
|------|-------------|
| **1. [Load Data](#📥-input-requirements)** | Import datasets (CSV, XLS, XLSX) and automatically detect numerical and categorical variables. |
| **2. [Preprocessing](#🧼-preprocessing-workflow)** | Variable selection, NaN handling, preprocessing templates, normalization, outlier checks, and transformations. |
| **3. [PCA](#📈-pca-module)** | Compute principal components, visualize explained and cumulative variance, and generate score plots and biplots. |
| **4. [Clustering](#🧭-clustering-module)** | Apply K-Means or hierarchical clustering, evaluate clusters via silhouette score, and choose linkage strategies. |
| **5. [Output](#📈-results-summary)** | Export the cleaned dataset, PCA scores, cluster labels, and view summarized analytical results. |

---

## 📥 Input Requirements

The application accepts datasets in CSV, XLS, and XLSX formats. Each file should follow a standard tabular structure where rows represent observations and columns represent variables. While categorical variables may be included, multivariate methods such as PCA and clustering require at least two valid numerical variables.

Datasets may contain missing values, as these are handled automatically during preprocessing. However, users are advised to avoid columns containing free text, redundant identifiers, or constant values, as these provide little analytical value and may introduce noise.

---

## 🧼 Preprocessing Workflow

Before performing any multivariate analysis, the dataset undergoes a structured cleaning and transformation pipeline. This workflow includes:

- Detecting and removing columns with excessive missing values  
- Excluding rows that exceed a missing-value threshold  
- Median imputation for numerical variables  
- Filtering out features with near-zero variance  
- Scaling all variables using z-score normalization to ensure comparable influence across dimensions  

These steps produce a consistent and standardized dataset suitable for PCA, clustering, and other multivariate techniques.

---

## 📈 PCA Module

The PCA module enables exploration of the underlying structure of the dataset by reducing its dimensionality. The system computes the principal components, displays both the explained variance and the cumulative variance, and allows the user to choose the appropriate number of components.

Visual outputs include 2D and 3D score plots and a biplot that illustrates how variables and observations relate within the component space. Users may optionally color observations based on a selected categorical variable to enhance interpretability. All PCA results—including scores, loadings, and model information—can be exported for reporting or further analysis.a

---

## 🧭 Clustering Module

## 🧭 Clustering Module

The clustering module enables users to identify natural groupings within the dataset by leveraging the PCA scores as input features. The tool supports two clustering methods:

- **K-Means**, which partitions observations into k clusters based on centroid optimization.  
- **Hierarchical clustering**, which builds a tree-like structure of relationships and allows flexible exploration through different linkage criteria.

Users can configure the number of clusters, choose the linkage method for hierarchical models, and visualize the resulting groups directly in the PCA space. The module includes several interactive visual outputs:

- **2D scatter plots** of clusters using PC1 and PC2  
- **Dendrograms** for hierarchical clustering, with automatic sampling for large datasets  
- **Quality metrics**, including silhouette score and inertia (SSE for K-Means)  
- **Cluster summaries**, showing cluster sizes and mean values of numerical variables  

These capabilities provide a comprehensive framework for interpreting multivariate structure and assessing how observations group together after dimensionality reduction.

---

## 📈 Results Summary

---

## 🧠 Key Insights

---

## 🧩 Technologies Used
- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Plotly

---

## 👥 Authors

- **José Ángel Govea García**
- **Diego Vértiz Padilla**    
- **Daniel Alberto Sánchez Fortiz**  
- **Augusto Ley Rodríguez**  

Tecnológico de Monterrey, School of Engineering and Sciences  
Guadalajara, Jalisco — México  