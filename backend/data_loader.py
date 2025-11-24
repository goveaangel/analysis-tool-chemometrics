import pandas as pd
from typing import Dict, Any

def load_data(file):

    name = file.name.lower()

    if name.endswith('.csv'):
        try:
            return pd.read_csv(file)
        except Exception:
            return pd.read_csv(file, sep=';')
    
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)

    else:
        raise ValueError("Formato no soportado. Usa .csv o .xlsx.")

def get_basic_summary(df):
    
    summary = {
    "n_rows": df.shape[0],
    "n_cols": df.shape[1],
    "dtypes": df.dtypes.astype(str),
    "na_counts": df.isna().sum(),
    "describe": df.describe(include="all"),
    }
    return summary




