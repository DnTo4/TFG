import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""Entrenamiento de Máquinas de Vectores de Soporte (SVM).

Proporciona funciones para procesar conjuntos de datos, estructurar características
y etiquetas, y ajustar un clasificador SVM con kernel RBF.
"""

def parse_df(df, target_col):
    """Extraer y organizar las variables del DataFrame.

    Integra instancias de entrenamiento y contraejemplos infiriendo las predicciones
    en caso de que existan.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Descartar columnas de análisis no estructuradas
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed", "mse_reconstruccion"]]
        
        # Extraer características e instancias
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Extraer y estructurar columnas del contraejemplo
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Obtener las etiquetas del contraejemplo
        try:
            import joblib
            bundle = joblib.load("modelos/modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"No se pudo cargar modelo.joblib para inferir las clases de los contraejemplos ({e}).")
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Unificar conjuntos en un único DataFrame
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Asumir la última columna si no se especifica la variable objetivo
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """Cargar y codificar las variables de los datasets.

    Ejecuta sobre las características y alinea su orden y cantidad
    entre el conjunto de entrenamiento y prueba.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Fusionar temporalmente para mantener dimensiones idénticas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    # Deshacer fusión conservando las variables procesadas
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """Entrenar un clasificador SVM con kernel radial (RBF).

    Aplica escalado estándar, entrena el SVM y calcula la precisión sobre
    el conjunto de test.
    """
    # Cargar y estructurar los conjuntos de datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    )

    # Entrenar SVM
    modelo.fit(X_train, y_train)

    # Medir la precisión final
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    nombres = X_train.columns.tolist()

    return modelo, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador SVM.")
    parser.add_argument("--train", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de entrenamiento (default: datos/originales/diabetes.csv)")
    parser.add_argument("--test", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de prueba (default: datos/originales/diabetes.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo (default: última columna)")
    parser.add_argument("--salida", type=str, default="modelos/modelo.joblib", help="Ruta para exportar el modelo entrenado (default: modelos/modelo.joblib)")
    
    args = parser.parse_args()
    
    try:
        print(f"Entrenando SVM con datos de '{args.train}'...")
        modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
            args.train, args.test, target_column=args.target
        )
        print(f"Precisión de test: {acc:.4f}")
        
        # Exportar el modelo
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        joblib.dump({"modelo": modelo, "nombres": nombres}, args.salida)
        print(f"Modelo exportado a '{args.salida}'")
    except Exception as e:
        print(f"No se pudo completar el entrenamiento: {e}")
