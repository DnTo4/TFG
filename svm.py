import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_df(df, target_col):
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Modo: entrenar con contraejemplos
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and not c.startswith("delta_") and not c.startswith("changed_") and c not in ["pred_orig", "dist_l2", "num_features_changed"]]
        
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Para multiclase (como iris), necesitamos saber a qué clase pertenece el contraejemplo.
        # En lugar de asumir binario, le preguntamos al modelo original (oráculo) que dictó ese contraejemplo.
        try:
            import joblib
            bundle = joblib.load("modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"Aviso: No se pudo cargar modelo.joblib para inferir las clases de los contraejemplos ({e}).")
            # Fallback por si fallara (sólo funciona si es numérico binario)
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Convertir variables categóricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline SVM
    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    )

    # Entrenar
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar nombres de columnas
    nombres = X_train.columns.tolist()

    # Devolver DataFrames para mantener compatibilidad con pipeline
    return modelo, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    
    # Configuración para ejecución directa
    TRAIN_PATH = "diabetes.csv"
    TEST_PATH = "diabetes.csv"
    TARGET_COLUMN = None  # Ajustar si la variable objetivo no es la última columna
    OUTPUT_MODEL = "modelo.joblib"
    
    print(f"Entrenando SVM con datos de '{TRAIN_PATH}'...")
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
        TRAIN_PATH, TEST_PATH, target_column=TARGET_COLUMN
    )
    
    print(f"Precisión de test: {acc:.4f}")
    
    # Exportar el modelo en el formato que espera analisis.py
    joblib.dump({"modelo": modelo, "nombres": nombres}, OUTPUT_MODEL)
    print(f"Modelo exportado exitosamente a '{OUTPUT_MODEL}'. Analizable con analisis.py.")
