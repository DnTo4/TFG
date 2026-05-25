import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_df(df, target_col):
    """
    Analiza un DataFrame para identificar si contiene datos estándar o un conjunto
    de contraejemplos .

    Si se detectan contraejemplos (columnas con prefijo 'ce_'), la función aumenta 
    el dataset: incluye los puntos originales y los contraejemplos.

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.
        target_col (str): Nombre de la columna objetivo. Si es None, usa la última.

    Returns:
        tuple: (X, y) con las características y etiquetas procesadas.
    """
    if "pred_orig" in df.columns and any(c.startswith("ce_") for c in df.columns):
        # Filtrar columnas auxiliares y de métricas para obtener solo las dimensiones reales
        orig_cols = [c for c in df.columns if not c.startswith("ce_") and 
                     not c.startswith("delta_") and 
                     not c.startswith("changed_") and 
                     c not in ["pred_orig", "dist_l2", "num_features_changed", "mse_reconstruccion"]]
        
        # Extraer características y etiquetas originales
        X_orig = df[orig_cols]
        y_orig = df["pred_orig"]
        
        # Extraer contraejemplos y renombrar columnas para que coincidan con originales
        X_ce = df[[f"ce_{c}" for c in orig_cols]]
        X_ce.columns = orig_cols
        
        # Obtener las etiquetas de los contraejemplos mediante el modelo
        try:
            import joblib
            bundle = joblib.load("modelos/modelo.joblib")
            modelo_oraculo = bundle["modelo"]
            y_ce = pd.Series(modelo_oraculo.predict(X_ce))
        except Exception as e:
            print(f"Aviso: No se pudo cargar modelo.joblib para inferir las clases de los contraejemplos ({e}).")
            # Fallback: Asume clasificación binaria e invierte la etiqueta original
            y_ce = 1 - pd.to_numeric(y_orig, errors='coerce').fillna(0).astype(int)
        
        # Unificar datos originales y contraejemplos
        X = pd.concat([X_orig, X_ce], axis=0).reset_index(drop=True)
        y = pd.concat([y_orig, y_ce], axis=0).reset_index(drop=True)
        return X, y
    else:
        # Modo estándar: Carga de dataset convencional
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

def load_data(train_path, test_path, target_column=None):
    """
    Carga los archivos de entrenamiento y prueba, aplicando el parseo 
    y preprocesamiento de variables categóricas.

    Args:
        train_path (str): Ruta al archivo de entrenamiento.
        test_path (str): Ruta al archivo de prueba.
        target_column (str, optional): Nombre de la columna objetivo.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) listos para el modelo.
    """
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = parse_df(df_train, target_column)
    X_test, y_test = parse_df(df_test, target_column)

    # Convertir variables categóricas a numéricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    """
    Entrena un clasificador SVM con kernel RBF y escalado de características.

    Args:
        train_path (str): Ruta al archivo de entrenamiento.
        test_path (str): Ruta al archivo de prueba.
        target_column (str, optional): Nombre de la columna objetivo.

    Returns:
        tuple: (modelo_entrenado, datasets_tupla, precision, nombres_columnas)
    """
    # Cargar y preparar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline: Estandarización + SVM RBF
    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    )

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Evaluación de precisión
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Almacenar nombres de columnas
    nombres = X_train.columns.tolist()

    return modelo, (X_train, y_train, X_test, y_test), acc, nombres

if __name__ == "__main__":
    import joblib
    
    # --- Configuración para ejecución directa ---
    TRAIN_PATH = "datos/originales/diabetes.csv"
    TEST_PATH = "datos/originales/diabetes.csv"
    TARGET_COLUMN = None  # Ajustar si la variable objetivo no es la última columna
    OUTPUT_MODEL = "modelos/modelo.joblib"
    
    print(f"Entrenando SVM con datos de '{TRAIN_PATH}'...")
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_model(
        TRAIN_PATH, TEST_PATH, target_column=TARGET_COLUMN
    )
    
    print(f"Precisión de test: {acc:.4f}")
    
    # Exportar el modelo
    joblib.dump({"modelo": modelo, "nombres": nombres}, OUTPUT_MODEL)
    print(f"Modelo exportado a '{OUTPUT_MODEL}'")
