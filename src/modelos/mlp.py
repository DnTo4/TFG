import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

"""Entrenamiento de Perceptrón Multicapa (MLP).

Proporciona funciones para cargar, estructurar y normalizar conjuntos de datos,
entrenar un MLP y guardar el modelo resultante.
"""

def parse_df(df, target_col):
    """Extraer y estructurar las variables.

    Permite diferenciar e integrar datos originales y contraejemplos reconstruyendo
    la predicción si corresponde.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Identificar columnas de variables originales
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed", "mse_reconstruccion"]]
        
        # Extraer variables independientes y clases del original
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Extraer variables correspondientes al contraejemplo
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Estimar la clase objetivo para el contraejemplo
        try:
            import joblib
            bundle = joblib.load("modelos/modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"No se pudo cargar modelo.joblib para inferir las clases de los contraejemplos ({e}).")
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Combinar los dos conjuntos de instancias
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Extraer la última columna si no se define explícitamente
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """Cargar conjuntos de datos y formatear variables categóricas.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Fusionar temporalmente para codificar de forma coherente
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    # Separar los conjuntos de entrenamiento y prueba
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """Entrenar un MLP.

    Define la estructura de las capas, escala las variables
    y estima la exactitud del clasificador resultante.
    """
    # Cargar y preprocesar los conjuntos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Estructurar MLP
    modelo = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=0)
    )

    # Entrenar el clasificador
    modelo.fit(X_train, y_train)

    # Evaluar precisión
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    nombres = X_train.columns.tolist()

    return modelo, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador MLP.")
    parser.add_argument("--train", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de entrenamiento (default: datos/originales/diabetes.csv)")
    parser.add_argument("--test", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de prueba (default: datos/originales/diabetes.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo (default: última columna)")
    parser.add_argument("--salida", type=str, default="modelos/modelo.joblib", help="Ruta para exportar el modelo entrenado (default: modelos/modelo.joblib)")
    
    args = parser.parse_args()
    
    try:
        print(f"Entrenando MLP con datos de '{args.train}'...")
        modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
            args.train, args.test, target_column=args.target
        )
        print(f"Precisión de test: {acc:.4f}")
        
        # Exportar el modelo
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        joblib.dump({"modelo": modelo, "nombres": nombres}, args.salida)
        print(f"Modelo exportado a '{args.salida}'.")
    except Exception as e:
        print(f"No se pudo completar el entrenamiento: {e}")
