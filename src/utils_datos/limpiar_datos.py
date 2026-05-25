"""
Script para la limpieza y validación biológica de datasets y contraejemplos.

Este módulo toma un conjunto de datos y aplica
restricciones basadas en el conocimiento del dominio
para garantizar que los valores producidos sean humanamente posibles y
matemáticamente coherentes con su naturaleza.
"""
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuración del Limpiador por Dataset ---
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
# ---------------------------------------------

def limpiar_y_validar(file_path_in, file_path_out, model_path="modelos/modelo.joblib"):
    """
    Toma un CSV con contraejemplos, recorta los valores a rangos humanamente posibles,
    redondea las variables enteras y verifica que sigan siendo contraejemplos.
    
    Args:
        file_path_in (str): Ruta al archivo CSV original.
        file_path_out (str): Ruta donde guardar el CSV modificado.
        model_path (str): Ruta al modelo oráculo.
    """
    if not os.path.exists(file_path_in):
        print(f"Error: No se encontró el archivo {file_path_in}")
        return
        
    df = pd.read_csv(file_path_in)
    print(f"Cargados {len(df)} registros de {file_path_in}")
    
    # Identificar columnas de características originales
    cols_ce = [col for col in df.columns if col.startswith('ce_')]
    cols_orig = [col.replace('ce_', '') for col in cols_ce]
    
    if not cols_ce:
        print("No se encontraron columnas de contraejemplos (ce_...).")
        return
        
    # Detectar el dataset
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
        
    # Aplicar recortes y redondeos
    for orig_col, ce_col in zip(cols_orig, cols_ce):
        if orig_col in rangos_biologicos:
            lim_inf, lim_sup = rangos_biologicos[orig_col]
            df[ce_col] = df[ce_col].clip(lower=lim_inf, upper=lim_sup)
            
        if orig_col in vars_enteras:
            df[ce_col] = df[ce_col].round().astype(int)
    
    for orig_col in cols_orig:
        if orig_col in rangos_biologicos:
            lim_inf, lim_sup = rangos_biologicos[orig_col]
            df[orig_col] = df[orig_col].clip(lower=lim_inf, upper=lim_sup)
        if orig_col in vars_enteras:
            df[orig_col] = df[orig_col].round().astype(int)
            
    # Recalcular deltas y distancias L2 tras las modificaciones
    dist_l2_list = []
    for index, row in df.iterrows():
        # Recalcular deltas
        for orig_col, ce_col in zip(cols_orig, cols_ce):
            delta_col = f'delta_{orig_col}'
            changed_col = f'changed_{orig_col}'
            
            if delta_col in df.columns:
                delta = row[ce_col] - row[orig_col]
                df.at[index, delta_col] = delta
                if changed_col in df.columns:
                    df.at[index, changed_col] = 1 if delta != 0 else 0
                    
        # Recalcular distancia euclídea
        vec_orig = row[cols_orig].values.astype(float)
        vec_ce = row[cols_ce].values.astype(float)
        dist = np.linalg.norm(vec_ce - vec_orig)
        dist_l2_list.append(dist)
        
    if 'dist_l2' in df.columns:
        df['dist_l2'] = dist_l2_list

    # Comprobar que siguen siendo contraejemplos
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            modelo = bundle["modelo"]
            
            # Predicción de los nuevos puntos
            X_ce = df[cols_ce].copy()
            X_ce.columns = cols_orig
            
            # Alinear columnas con las que el modelo fue entrenado
            if hasattr(modelo, "feature_names_in_"):
                modelo_features = list(modelo.feature_names_in_)
                missing_cols = [c for c in modelo_features if c not in X_ce.columns]
                unseen_cols = [c for c in X_ce.columns if c not in modelo_features]
                
                if missing_cols or unseen_cols:
                    raise ValueError(
                        f"Discrepancia de características detectada.\n"
                        f"El modelo cargado de '{model_path}' espera características: {modelo_features}\n"
                        f"Pero el archivo de contraejemplos proporciona características: {list(X_ce.columns)}\n"
                        f"Asegúrese de estar evaluando contraejemplos que correspondan al mismo dataset del modelo."
                    )
                # Reordenar columnas para que coincidan exactamente
                X_ce = X_ce[modelo_features]
            
            y_ce_new = modelo.predict(X_ce)
            
            # Comparar con la predicción original
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

    # Guardar archivo limpio
    df.to_csv(file_path_out, index=False)
    print(f"\nFinalizado. {len(df)} contraejemplos válidos guardados en {file_path_out}")

if __name__ == "__main__":
    # Puedes cambiar los archivos de entrada y salida aquí
    ARCHIVO_ENTRADA = "datos/procesados/contraejemplos_factibles.csv"
    ARCHIVO_SALIDA = "datos/procesados/contraejemplos_limpios.csv"

    limpiar_y_validar(ARCHIVO_ENTRADA, ARCHIVO_SALIDA)
