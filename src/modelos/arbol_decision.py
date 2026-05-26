import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

"""Entrenamiento y optimización de un árbol de decisión.
"""

def parse_df(df, target_col):
    """Extraer y separar las características y clases del DataFrame.

    Distingue si el DataFrame corresponde a contraejemplos exportados (con prefijo 'ce_')
    o a conjuntos de datos originales.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Filtrar columnas originales
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed", "mse_reconstruccion"]]
        
        # Obtener características y clases
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Extraer características del contraejemplo
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Estimar la clase del contraejemplo
        try:
            import joblib
            bundle = joblib.load("modelos/modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"No se pudo cargar modelo.joblib para inferir clases: {e}")
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Fusionar datos originales y contraejemplos
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Extraer la última columna
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """Cargar conjuntos de datos de entrenamiento y prueba.

    Asegura que las características resultantes sean idénticas en ambos conjuntos
    para garantizar dimensiones coherentes antes de entrenar el clasificador.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Codificar variables categóricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    # Volver a dividir en entrenamiento y prueba
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """Entrenar el árbol de decisión mediante Grid Search.

    Ajusta hiperparámetros y retorna el mejor estimador entrenado y sus métricas.
    """
    # Cargar y preprocesar los conjuntos de datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Estructurar el flujo
    pipeline = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(random_state=42)
    )

    # Configurar la rejilla de parámetros a explorar
    param_grid = {
        'decisiontreeclassifier__max_depth': [3, 4, 5, 6, 8, 10, None],
        'decisiontreeclassifier__min_samples_split': [2, 5, 10],
        'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 8],
        'decisiontreeclassifier__criterion': ['gini', 'entropy']
    }

    # Definir el número de divisiones para validación cruzada
    cv_folds = min(5, len(X_train) // 2)
    if cv_folds < 2:
        cv_folds = 2

    # Ejecutar la búsqueda de rejilla
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Estimar la exactitud final
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    nombres = X_train.columns.tolist()

    return best_model, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Entrenamiento de Árbol de Decisión.")
    parser.add_argument("--train", type=str, default="datos/procesados/contraejemplos.csv", help="Ruta al dataset de entrenamiento (default: datos/procesados/contraejemplos.csv)")
    parser.add_argument("--test", type=str, default="datos/procesados/contraejemplos.csv", help="Ruta al dataset de prueba (default: datos/procesados/contraejemplos.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo (default: última columna)")
    parser.add_argument("--salida", type=str, default="modelos/modelo.joblib", help="Ruta para exportar el modelo entrenado (default: modelos/modelo.joblib)")
    
    args = parser.parse_args()
    
    try:
        print(f"Entrenando Árbol de Decisión con datos de '{args.train}'...")
        modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
            args.train, args.test, target_column=args.target
        )
        print(f"Precision de test: {acc:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        joblib.dump({"modelo": modelo, "nombres": nombres}, args.salida)
        print(f"Modelo exportado a '{args.salida}'")
    except Exception as e:
        print(f"No se pudo completar el entrenamiento: {e}")
