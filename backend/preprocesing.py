import pandas as pd
import numpy as np

# SELECCION INICIAL

def select_numeric(df):

    numeric_df = df.select_dtypes(include = 'number').copy()
    return numeric_df

def select_columns(df, selected_cols):

    df_selected_cols = df[selected_cols].copy()
    return df_selected_cols

# MANEJO DE NANS

def drop_nan_columns(df, max_nan):
    
    nan_fraction = df.isna().mean()
    cols_to_drop = nan_fraction[nan_fraction > max_nan].index.tolist()
    df_out = df.drop(columns = cols_to_drop).copy()

    return df_out, cols_to_drop

def drop_nan_rows(df, max_nan):

    nan_fraction = df.isna().mean(axis=1)
    mask = nan_fraction <= max_nan
    n_dropped = (~mask).sum()
    df_out = df.loc[mask].copy()
    
    return df_out, n_dropped

def impute_nans(df, strategy):

    df_out = df.copy()
    
    if strategy == 'mean':
        values = df_out.mean()
        df_out = df_out.fillna(values)

    elif strategy == 'median':
        values = df_out.median()
        df_out = df_out.fillna(values)

    else:
        raise ValueError(f'Estrategia de imputacion no soportada: {strategy}')
    
    return df_out

#FILTRADO

def drop_low_variance(df, var_thresh):
    
    variances = df.var()
    low_var_cols = variances[variances <= var_thresh].index.tolist()

    if low_var_cols:
        df_out = df.drop(columns = low_var_cols).copy()
    else:
        df_out = df.copy()

    return df_out, low_var_cols
    

def detect_outliers(df, method='zscore', z_thres=3.0, iqr_factor=1.5):
    
    df_num = df.select_dtypes(include='number')

    if df_num.empty:
       
        return pd.Series([False] * len(df), index=df.index)

    elif method == "none":

        return pd.Series([False] * len(df), index=df.index)    
    
    elif method == 'zscore':

        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0,1)
        zcores = (df_num - means) / stds
        outlier_mask = (zcores.abs() > z_thres).any(axis=1)

        return outlier_mask
    
    elif method == 'iqr':

        Q1 = df_num.quantile(0.25)
        Q3 = df_num.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        outlier_mask = ((df_num < lower) | (df_num > upper)).any(axis=1)

        return outlier_mask
    
    else:
        raise ValueError(f'Metodo de outliers no soportado: {method}')

def outlier_action(df, outlier_mask, action='mark'):
    
    df_out = df.copy()

    outlier_mask = outlier_mask.reindex(df_out.index)
    outlier_mask = outlier_mask.fillna(False)

    if action == 'none':

        return df_out
    
    elif action == 'mark':

        df_out['is_outlier'] = outlier_mask
        return df_out
    
    elif action == 'exclude':

        df_out = df_out.loc[~outlier_mask].copy()
        return df_out
    
    else:
        raise ValueError(f'Acción de outliers no soportada: {action}')

# TRANSFORMACIONES

def transform(df, columns, method='none'):

    df_out = df.copy()
    applied = []

    if method == 'none' or not columns:

        return df_out, applied
    
    for col in columns:
        if col not in df_out.columns:
            raise ValueError(f'Columna no encontrada para transofrmar: {col}')
        
    if method == 'log':

        for col in columns:
            if (df_out[col] <= 0).any():
                raise ValueError(f'No se puede aplicar log a {col} porque contiene valores <= 0')
            
            df_out[col] = np.log(df_out[col])
            applied.append(col)
        
        return df_out, applied
    
    elif method == 'log10':

        for col in columns:
            if (df_out[col] <= 0).any():
                raise ValueError(f'No se puede aplicar log10 a {col} porque contiene valores <= 0')
            
            df_out[col] = np.log10(df_out[col])
            applied.append(col)
        
        return df_out, applied       

    elif method == 'sqrt':

        for col in columns:
            if (df_out[col] < 0).any():
                raise ValueError(f'No se puede aplicar sqrt a {col} porque contiene valores < 0')
            
            df_out[col] = np.sqrt(df_out[col])
            applied.append(col)
        
        return df_out, applied 
    
    elif method == 'autoscale':

        for col in columns:
            
            mean = df_out[col].mean()
            std = df_out[col].std()

            if std == 0:
                raise ValueError(f'No se puede autoscale la columna {col} porque tiene desviación estándar 0')
            
            df_out[col] = (df_out[col] - mean) / std
            applied.append(col)
        
        return df_out, applied 
    
# ESCALADO

def scale(df, method='none'):
    
    df_scaled = df.copy()
    df_num = df_scaled.select_dtypes(include='number')
    params = {'method': method}

    if method == 'none' or df_num.empty:
        return df_scaled, params
    
    elif method == 'mean center':

        means = df_num.mean()
        df_scaled[df_num.columns] = df_num - means
        params['means'] = means.to_dict()

        return df_scaled, params

    elif method == "zscore":

        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0, 1)
        df_scaled[df_num.columns] = (df_num - means) / stds
        params["means"] = means.to_dict()
        params["stds"] = stds.to_dict()

        return df_scaled, params

    elif method == "minmax":

        mins = df_num.min()
        maxs = df_num.max()
        ranges = (maxs - mins).replace(0, 1)  
        df_scaled[df_num.columns] = (df_num - mins) / ranges
        params["mins"] = mins.to_dict()
        params["maxs"] = maxs.to_dict()

        return df_scaled, params

    elif method == "pareto":

        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0, 1)
        denom = np.sqrt(stds).replace(0, 1)
        df_scaled[df_num.columns] = (df_num - means) / denom
        params["means"] = means.to_dict()
        params["stds"] = stds.to_dict()
        params["denominator"] = denom.to_dict()

        return df_scaled, params

    else:
        raise ValueError(f"Método de escalado no soportado: {method}")

#  REPORTE

def build_report(
    *,
    template_name,
    n_rows_before,
    n_rows_after,
    n_cols_before,
    n_cols_after,
    
    # NANS
    nan_strategy,
    nan_col_threshold,
    nan_row_threshold,
    dropped_nan_columns,
    dropped_nan_rows,
    
    # VARIANZA
    variance_threshold,
    dropped_low_var_columns,
    
    # OUTLIERS
    outlier_method,
    outlier_action_type,
    n_outliers_detected,
    
    # TRANSFORMACIONES
    transform_method,
    transformed_columns,
    
    # ESCALADO
    scaling_method,
    scaling_params,
    
    # FINAL
    final_columns
):
    """
    Construye un diccionario limpio y organizado con TODO el registro
    del preprocesamiento aplicado.

    Se usa para UI, exportación, reproducibilidad y debugging.
    """
    
    report = {
        "template_name": template_name,
        
        "rows_before": n_rows_before,
        "rows_after": n_rows_after,
        "cols_before": n_cols_before,
        "cols_after": n_cols_after,

        # NANS
        "nan_strategy": nan_strategy,
        "nan_col_threshold": nan_col_threshold,
        "nan_row_threshold": nan_row_threshold,
        "dropped_nan_columns": dropped_nan_columns,
        "dropped_nan_rows": dropped_nan_rows,

        # VARIANZA
        "variance_threshold": variance_threshold,
        "dropped_low_var_columns": dropped_low_var_columns,

        # OUTLIERS
        "outlier_method": outlier_method,
        "outlier_action": outlier_action_type,
        "n_outliers_detected": n_outliers_detected,

        # TRANSFORMACIONES
        "transform_method": transform_method,
        "transformed_columns": transformed_columns,

        # ESCALADO
        "scaling_method": scaling_method,
        "scaling_params": scaling_params,

        # FINAL
        "final_columns": final_columns,
    }

    return report

# PIPELINE

def run_manual_preprocessing(
    df,
    selected_vars,
    nan_strategy,
    max_nan_col,
    max_nan_row,
    scaling_method,
    outlier_method,
    outlier_action_ui,
    transform_type,
    vars_to_transform,
):
    """
    Ejecuta el pipeline de preprocesamiento manual usando los helpers existentes.

    Parámetros:
        df: DataFrame original
        selected_vars: lista de columnas numéricas seleccionadas para el análisis
        nan_strategy: string de la UI
            - "Eliminar filas con NaNs"
            - "Imputar con media"
            - "Imputar con mediana"
        max_nan_col: porcentaje (0-100) de NaNs permitido por columna
        max_nan_row: porcentaje (0-100) de NaNs permitido por fila
        scaling_method: string de la UI
            - "Sin escalado"
            - "Centrado a la media"
            - "Autoscaling (z-score)"
            - "Min–Max [0, 1]"
            - "Pareto (quimiometría)"
        outlier_method:
            - "Ninguno"
            - "Z-score (|z| > 3)"
            - "IQR (1.5×IQR)"
        outlier_action:
            - "Solo marcar outliers"
            - "Excluir filas outliers"
            - "No hacer nada (solo diagnóstico)"
        transform_type:
            - "Ninguna"
            - "Log10 (solo valores > 0)"
            - "Log natural (ln)"
            - "Raíz cuadrada"
        vars_to_transform: lista de columnas a transformar (puede ser vacía)

    Devuelve:
        df_final: DataFrame preprocesado listo para PCA / clustering
        report: diccionario con el resumen del preprocesamiento
    """

    # -------------------------------------------------
    # 0) Validaciones básicas
    # -------------------------------------------------
    if not selected_vars:
        raise ValueError("No se seleccionaron variables numéricas para el preprocesamiento.")

    # Nos quedamos SOLO con las columnas seleccionadas
    df_work = select_columns(df, selected_vars)
    n_rows_before, n_cols_before = df_work.shape

    # Convertimos % → fracción
    nan_col_thresh = max_nan_col / 100.0
    nan_row_thresh = max_nan_row / 100.0

    # -------------------------------------------------
    # 1) Manejo de NaNs (por columnas y filas)
    # -------------------------------------------------
    # 1.1 Eliminar columnas con demasiados NaNs
    df_cols, dropped_nan_cols = drop_nan_columns(df_work, max_nan=nan_col_thresh)

    # 1.2 Eliminar filas con demasiados NaNs
    df_rows, dropped_nan_rows = drop_nan_rows(df_cols, max_nan=nan_row_thresh)

    # 1.3 Estrategia principal sobre NaNs restantes
    if nan_strategy == "Eliminar filas con NaNs":
        df_nans = df_rows.dropna(axis=0)
        nan_strategy_internal = "drop_rows"
    elif nan_strategy == "Imputar con media":
        df_nans = impute_nans(df_rows, strategy="mean")
        nan_strategy_internal = "mean"
    elif nan_strategy == "Imputar con mediana":
        df_nans = impute_nans(df_rows, strategy="median")
        nan_strategy_internal = "median"
    else:
        raise ValueError(f"Estrategia de NaNs no soportada: {nan_strategy}")

    # -------------------------------------------------
    # 2) Filtro de baja varianza
    # -------------------------------------------------
    # Para el pipeline manual, usamos un umbral suave (por ejemplo 0.0 para eliminar solo constantes)
    variance_threshold = 0.0
    df_filtered, low_var_cols = drop_low_variance(df_nans, var_thresh=variance_threshold)

    # -------------------------------------------------
    # 3) Transformaciones de variables
    # -------------------------------------------------
    # Mapear texto de la UI → método interno
    if transform_type == "Ninguna":
        transform_method = "none"
    elif transform_type == "Log10 (solo valores > 0)":
        transform_method = "log10"
    elif transform_type == "Log natural (ln)":
        transform_method = "log"
    elif transform_type == "Raíz cuadrada":
        transform_method = "sqrt"
    else:
        raise ValueError(f"Tipo de transformación no soportado: {transform_type}")

    df_transformed, transformed_columns = transform(
        df_filtered,
        columns=vars_to_transform,
        method=transform_method,
    )

    # -------------------------------------------------
    # 4) Escalado / normalización
    # -------------------------------------------------
    # Mapear texto de la UI → método interno de scale()
    if scaling_method == "Sin escalado":
        scaling_method = "none"
    elif scaling_method == "Centrado a la media":
        scaling_method = "mean center"
    elif scaling_method == "Autoscaling (z-score)":
        scaling_method = "zscore"
    elif scaling_method == "Min–Max [0, 1]":
        scaling_method = "minmax"
    elif scaling_method == "Pareto (quimiometría)":
        scaling_method = "pareto"
    else:
        raise ValueError(f"Método de escalado no soportado: {scaling_method}")

    df_scaled, scaling_params = scale(df_transformed, method=scaling_method)

    # -------------------------------------------------
    # 5) Detección y acción sobre outliers
    # -------------------------------------------------
    # Método interno
    if outlier_method == "Ninguno":
        outlier_method = "none"
    elif outlier_method == "Z-score (|z| > 3)":
        outlier_method = "zscore"
    elif outlier_method == "IQR (1.5×IQR)":
        outlier_method = "iqr"
    else:
        raise ValueError(f"Método deç outliers no soportado: {outlier_method}")

    # Acción interna
    if outlier_action_ui == "Solo marcar outliers":
        outlier_action_type = "mark"
    elif outlier_action_ui == "Excluir filas outliers":
        outlier_action_type = "exclude"
    elif outlier_action_ui == "No hacer nada (solo diagnóstico)":
        outlier_action_type = "none"
    else:
        raise ValueError(f"Acción de outliers no soportada: {outlier_action}")

    # Si no hay detección de outliers, la máscara es todo False
    if outlier_method == "none":
        outlier_mask = pd.Series([False] * len(df_scaled), index=df_scaled.index)
        n_outliers_detected = 0
    else:
        outlier_mask = detect_outliers(df_scaled, method=outlier_method)
        n_outliers_detected = int(outlier_mask.sum())

    df_final = outlier_action(df_scaled, outlier_mask, action=outlier_action_type)

    # -------------------------------------------------
    # 6) Dimensiones finales
    # -------------------------------------------------
    n_rows_after, n_cols_after = df_final.shape

    # -------------------------------------------------
    # 7) Construir reporte
    # -------------------------------------------------
    report = build_report(
        template_name="Preprocesamiento manual",
        n_rows_before=n_rows_before,
        n_rows_after=n_rows_after,
        n_cols_before=n_cols_before,
        n_cols_after=n_cols_after,

        nan_strategy=nan_strategy_internal,
        nan_col_threshold=nan_col_thresh,
        nan_row_threshold=nan_row_thresh,
        dropped_nan_columns=dropped_nan_cols,
        dropped_nan_rows=dropped_nan_rows,

        variance_threshold=variance_threshold,
        dropped_low_var_columns=low_var_cols,

        outlier_method=outlier_method,
        outlier_action_type=outlier_action_type,
        n_outliers_detected=n_outliers_detected,

        transform_method=transform_method,
        transformed_columns=transformed_columns,

        scaling_method=scaling_method,
        scaling_params=scaling_params,

        final_columns=list(df_final.columns),
    )

    return df_final, report

# ==========================================================
#  PLANTILLAS 
# ==========================================================

def basic(
    df,
    nan_col_thresh=0.4,
    nan_row_thresh=0.5,
    var_thresh=1e-8,
    scaling_method="zscore"
):
    
    n_rows_before, n_cols_before = df.shape

    df_num = df.select_dtypes(include='number')
    if df_num.shape[1] == 0:
        raise ValueError('La plantilla básica no encontró columnas numéricas en el dataset.')
    
    df_cols, dropped_nan_cols = drop_nan_columns(df_num, max_nan=nan_col_thresh)
    df_rows, dropped_nan_rows = drop_nan_rows(df_cols, max_nan=nan_row_thresh)

    df_imputed = impute_nans(df_rows, strategy='median')

    df_filtered, low_var_cols = drop_low_variance(df_imputed, var_thresh=var_thresh)

    df_scaled, scaling_params = scale(df_filtered, method=scaling_method)

    n_rows_after, n_cols_after = df_scaled.shape

    outlier_method = "none"
    outlier_action_type = "none"
    n_outliers_detected = 0

    transform_method = "none"
    transformed_columns = []

    # CONTRUIR REPORTE
    report = build_report(
        template_name="Plantilla básica (PCA estándar)",
        n_rows_before=n_rows_before,
        n_rows_after=n_rows_after,
        n_cols_before=n_cols_before,
        n_cols_after=n_cols_after,

        nan_strategy="median",
        nan_col_threshold=nan_col_thresh,
        nan_row_threshold=nan_row_thresh,
        dropped_nan_columns=dropped_nan_cols,
        dropped_nan_rows=dropped_nan_rows,

        variance_threshold=var_thresh,
        dropped_low_var_columns=low_var_cols,

        outlier_method=outlier_method,
        outlier_action_type=outlier_action_type,
        n_outliers_detected=n_outliers_detected,

        transform_method=transform_method,
        transformed_columns=transformed_columns,

        scaling_method=scaling_method,
        scaling_params=scaling_params,

        final_columns=list(df_scaled.columns),
    )

    return df_scaled, report