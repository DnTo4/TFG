import pandas as pd
import numpy as np
import joblib
import os

"""Módulo para la limpieza, acotación y validación de contraejemplos.

Proporciona funciones para acotar variables a rangos realistas,
redondear valores enteros de forma coherente y filtrar aquellos puntos que dejen
de ser contraejemplos.
"""

CONFIGURACIONES = {
    'diabetes': {
        'RANGOS_BIOLOGICOS': {
            'Pregnancies': [0, 10],
            'Glucose': [40, 400],
            'BloodPressure': [40, 200],
            'SkinThickness': [0, 100],
            'Insulin': [0, 900],
            'BMI': [10.0, 70.0],
            'DiabetesPedigreeFunction': [0.0, 3.0],
            'Age': [21, 100]
        },
        'VARS_ENTERAS': ['Pregnancies', 'Age']
    },
    'iris': {
        'RANGOS_BIOLOGICOS': {
            'sepal_length': [0.1, 15.0],
            'sepal_width': [0.1, 10.0],
            'petal_length': [0.1, 15.0],
            'petal_width': [0.1, 10.0]
        },
        'VARS_ENTERAS': []
    }
}

def limpiar_y_validar(file_path_in, file_path_out, model_path="modelos/modelo.joblib"):
    """Aplicar restricciones y validar contraejemplos con el modelo.

    Carga el fichero de contraejemplos, detecta el dataset, aplica límites
    y redondea variables enteras. Posteriormente, usa el modelo para comprobar
    si las instancias modificadas siguen cambiando de clase.
    """
    if not os.path.exists(file_path_in):
        print(f"Error: No se encontró el archivo {file_path_in}")
        return
        
    df = pd.read_csv(file_path_in)
    print(f"Cargados {len(df)} registros de {file_path_in}")
    
    # Identificar columnas con variables modificadas
    cols_ce = [col for col in df.columns if col.startswith('ce_')]
    cols_orig = [col.replace('ce_', '') for col in cols_ce]
    
    if not cols_ce:
        print("No se encontraron columnas de contraejemplos.")
        return
        
    # Detectar el conjunto de datos implícito
    if 'sepal_length' in cols_orig:
        dataset_name = 'iris'
        print("Dataset detectado: Iris")
    elif 'Pregnancies' in cols_orig:
        dataset_name = 'diabetes'
        print("Dataset detectado: Diabetes")
    else:
        print("Dataset desconocido. Se omiten las restricciones de dominio.")
        df.to_csv(file_path_out, index=False)
        return
        
    config = CONFIGURACIONES[dataset_name]
    rangos_biologicos = config['RANGOS_BIOLOGICOS']
    vars_enteras = config['VARS_ENTERAS']
        
    # Acotar y redondear variables
    for orig_col, ce_col in zip(cols_orig, cols_ce):
        if orig_col in rangos_biologicos:
            lim_inf, lim_sup = rangos_biologicos[orig_col]
            df[ce_col] = df[ce_col].clip(lower=lim_inf, upper=lim_sup)
            
        if orig_col in vars_enteras:
            df[ce_col] = df[ce_col].round().astype(int)
    
    # Acotar y redondear variables
    for orig_col in cols_orig:
        if orig_col in rangos_biologicos:
            lim_inf, lim_sup = rangos_biologicos[orig_col]
            df[orig_col] = df[orig_col].clip(lower=lim_inf, upper=lim_sup)
        if orig_col in vars_enteras:
            df[orig_col] = df[orig_col].round().astype(int)
            
    dist_l2_list = []
    # Recalcular variaciones e indicadores de cambio
    for index, row in df.iterrows():
        for orig_col, ce_col in zip(cols_orig, cols_ce):
            delta_col = f'delta_{orig_col}'
            changed_col = f'changed_{orig_col}'
            
            if delta_col in df.columns:
                delta = row[ce_col] - row[orig_col]
                df.at[index, delta_col] = delta
                if changed_col in df.columns:
                    df.at[index, changed_col] = 1 if delta != 0 else 0
                    
        # Recalcular la distancia euclidiana final
        vec_orig = row[cols_orig].values.astype(float)
        vec_ce = row[cols_ce].values.astype(float)
        dist = np.linalg.norm(vec_ce - vec_orig)
        dist_l2_list.append(dist)
        
    if 'dist_l2' in df.columns:
        df['dist_l2'] = dist_l2_list

    # Validar que las instancias siguen forzando un cambio de predicción
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            modelo = bundle["modelo"]
            
            # Obtener predicciones sobre los nuevos puntos acotados
            X_ce = df[cols_ce].copy()
            X_ce.columns = cols_orig
            
            # Alinear características con el formato esperado por el modelo
            if hasattr(modelo, "feature_names_in_"):
                modelo_features = list(modelo.feature_names_in_)
                missing_cols = [c for c in modelo_features if c not in X_ce.columns]
                unseen_cols = [c for c in X_ce.columns if c not in modelo_features]
                
                if missing_cols or unseen_cols:
                    raise ValueError(
                        f"Discrepancia de características detectada.\n"
                        f"El modelo cargado de '{model_path}' espera características: {modelo_features}\n"
                        f"Pero el archivo de contraejemplos proporciona características: {list(X_ce.columns)}\n"
                    )
                X_ce = X_ce[modelo_features]
            
            y_ce_new = modelo.predict(X_ce)
            
            # Descartar registros que no alteran la predicción original
            if 'pred_orig' in df.columns:
                mask_contraejemplo_valido = y_ce_new != df['pred_orig'].values
                
                num_invalidos = len(df) - mask_contraejemplo_valido.sum()
                df = df[mask_contraejemplo_valido].copy()
                
                print(f"Se descartaron {num_invalidos} registros.")
            else:
                print("No existe la columna 'pred_orig' en el CSV. No se pudo verificar el cambio de clase.")
                
        except Exception as e:
            print(f"Error al verificar contraejemplos con el modelo: {e}")
    else:
        print(f"No se encontró el modelo en {model_path}.")

    # Guardar resultados
    df.to_csv(file_path_out, index=False)
    print(f"\nFinalizado. {len(df)} contraejemplos válidos guardados en {file_path_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Limpia y valida biológicamente contraejemplos generados.")
    parser.add_argument("--entrada", type=str, default="datos/procesados/contraejemplos_factibles.csv", help="Ruta al archivo CSV de entrada.")
    parser.add_argument("--salida", type=str, default="datos/procesados/contraejemplos_limpios.csv", help="Ruta al archivo CSV de salida.")
    parser.add_argument("--modelo", type=str, default="modelos/modelo.joblib", help="Ruta al modelo oráculo.")
    args = parser.parse_args()

    limpiar_y_validar(args.entrada, args.salida, model_path=args.modelo)
