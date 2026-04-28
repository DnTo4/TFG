import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_df(df, target_col):
    """
    Analiza y prepara el DataFrame, detectando si se trata de un dataset estándar 
    o un archivo de contraejemplos.

    Si detecta contraejemplos, expande el dataset duplicando las filas: una con 
    los valores originales y otra con los valores del contraejemplo ('ce_'), 
    asignando las clases correspondientes.

    Args:
        df (pd.DataFrame): El DataFrame cargado desde el CSV.
        target_col (str): Nombre de la columna objetivo. Si es None, se asume la última.

    Returns:
        tuple: (X, y) listos para el procesamiento o entrenamiento.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Extraemos las columnas originales descartando métricas de análisis
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed"]]
        
        # Separar datos originales
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Preparar datos de contraejemplos (renombrando columnas para que coincidan)
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Intentar determinar la clase del contraejemplo usando el modelo original
        try:
            import joblib
            bundle = joblib.load("modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"Aviso: No se pudo cargar modelo.joblib para inferir clases: {e}")
            # Fallback: Inversión binaria si el modelo no está disponible
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Concatenar ambos sets para formar el dataset de entrenamiento
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Modo estándar: Dataset de clasificación tradicional
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """
    Carga los archivos de entrenamiento y prueba, aplicando el parseo y preprocesamiento.

    Realiza la conversión de variables categóricas
    de manera consistente entre ambos conjuntos de datos.

    Args:
        train_path (str): Ruta al archivo CSV de entrenamiento.
        test_path (str): Ruta al archivo CSV de prueba.
        target_column (str, optional): Nombre de la columna etiqueta.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Convertir categóricas a numéricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    # Separar de nuevo en entrenamiento y prueba
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """
    Entrena un perceptrón utilizando un pipeline con escalado estándar.

    Args:
        train_path (str): Ruta al archivo de entrenamiento.
        test_path (str): Ruta al archivo de prueba.
        target_column (str, optional): Nombre de la columna objetivo.

    Returns:
        tuple: (modelo, datasets_tupla, precision, nombres_columnas)
    """
    # Cargar y preprocesar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline: Escalado + modelo lineal
    model = make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=1000, tol=1e-3, random_state=0)
    )

    # Fase de ajuste del modelo
    model.fit(X_train, y_train)

    # Evaluación de desempeño
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Lista de nombres de características
    nombres = X_train.columns.tolist()

    return model, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    
    # --- Configuración de ejecución ---
    TRAIN_PATH = "contraejemplos.csv"
    TEST_PATH = "contraejemplos.csv"
    TARGET_COLUMN = None  # Ajustar según el dataset
    OUTPUT_MODEL = "modelo.joblib"
    
    print(f"Entrenando Perceptron con datos de '{TRAIN_PATH}'...")
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
        TRAIN_PATH, TEST_PATH, target_column=TARGET_COLUMN
    )
    
    print(f"Precision de test: {acc:.4f}")
    
    # Exportar el modelo en el formato estandarizado del proyecto
    joblib.dump({"modelo": modelo, "nombres": nombres}, OUTPUT_MODEL)
    print(f"Modelo exportado  a '{OUTPUT_MODEL}'")

