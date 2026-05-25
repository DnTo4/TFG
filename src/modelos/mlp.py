import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def parse_df(df, target_col):
    """
    Procesa un DataFrame para separar características y etiquetas.

    Si el DataFrame contiene columnas con el prefijo 'ce_', la función asume que 
    se trata de un archivo de contraejemplos y reconstruye un dataset aumentado 
    combinando los puntos originales con sus contraejemplos.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna objetivo. Si es None, se usa la última.

    Returns:
        tuple: (X, y) donde X son las características y y las etiquetas.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Identificar columnas originales excluyendo métricas y prefijos de contraejemplos
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed", "mse_reconstruccion"]]
        
        # Extraer puntos originales y sus etiquetas predichas por el modelo
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Extraer contraejemplos (columnas con prefijo 'ce_') y renombrarlas a las originales
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Determinar la clase del contraejemplo consultando el modelo original
        try:
            import joblib
            bundle = joblib.load("modelos/modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"Aviso: No se pudo cargar modelo.joblib para inferir las clases de los contraejemplos ({e}).")
            # Fallback: Inversión de clase (asume clasificación binaria 0/1)
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Concatenar puntos originales y contraejemplos para crear un dataset robusto
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Modo estándar: Carga de dataset tradicional
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """
    Carga los archivos de entrenamiento y prueba y realiza el preprocesamiento básico.

    Incluye la conversión de variables categóricas a numéricas.

    Args:
        train_path (str): Ruta al archivo de entrenamiento.
        test_path (str): Ruta al archivo de prueba.
        target_column (str, optional): Nombre de la columna objetivo.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) procesados.
    """
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Convertir variables categóricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    # Volver a separar tras el encoding
    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """
    Entrena un Perceptrón Multicapa (MLP) utilizando un pipeline de estandarización.

    Args:
        train_path (str): Ruta al archivo de entrenamiento.
        test_path (str): Ruta al archivo de prueba.
        target_column (str, optional): Nombre de la columna objetivo.

    Returns:
        tuple: (modelo, (X_train, y_train, X_test, y_test), precisión, nombres_columnas)
    """
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline: Estandarización de datos + (MLP)
    # hidden_layer_sizes=(100, 50) define dos capas ocultas de 100 y 50 neuronas.
    modelo = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=0)
    )

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Evaluar el rendimiento en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar nombres de columnas
    nombres = X_train.columns.tolist()

    return modelo, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    
    # --- Configuración para ejecución directa ---
    TRAIN_PATH = "datos/originales/diabetes.csv"
    TEST_PATH = "datos/originales/diabetes.csv"
    TARGET_COLUMN = None  # Cambiar por el nombre real si no es la última columna
    OUTPUT_MODEL = "modelos/modelo.joblib"
    
    print(f"Entrenando MLP (Red Neuronal) con datos de '{TRAIN_PATH}'...")
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
        TRAIN_PATH, TEST_PATH, target_column=TARGET_COLUMN
    )
    
    print(f"Precisión de test: {acc:.4f}")
    
    # Exportar el modelo en formato joblib
    joblib.dump({"modelo": modelo, "nombres": nombres}, OUTPUT_MODEL)
    print(f"Modelo exportado a '{OUTPUT_MODEL}'.")
