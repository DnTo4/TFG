import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

# --- Hiperparámetros del Autoencoder ---
AE_ACTIVATION = 'relu'
AE_SOLVER = 'adam'
AE_MAX_ITER = 1000
AE_RANDOM_STATE = 42
AE_PERCENTIL_UMBRAL = 99
# ---------------------------------------

def load_and_prepare_data(file_path, target_col=None):
    """
    Carga un dataset y separa las características (X) de la etiqueta (y).
    
    Args:
        file_path (str): Ruta al archivo CSV con los datos originales.
        target_col (str, optional): Nombre de la columna objetivo. Si es None, 
            asume que es la última columna.
            
    Returns:
        pd.DataFrame: DataFrame solo con las características numéricas originales.
    """
    df = pd.read_csv(file_path)
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    # Opcional: asegurarnos de usar solo variables numéricas
    X = X.select_dtypes(include=[np.number])
    return X

def train_autoencoder(X):
    """
    Entrena un Autoencoder basado en Perceptrón Multicapa (MLPRegressor) para 
    aprender la representación subyacente de los datos originales.
    
    Args:
        X (pd.DataFrame): DataFrame de características originales estandarizadas o sin estandarizar.
        
    Returns:
        tuple: (modelo MLPRegressor, StandardScaler entrenado, umbral de error calculado).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configuramos el MLPRegressor como Autoencoder
    # La arquitectura de cuello de botella comprimirá y luego reconstruirá la información
    n_features = X.shape[1]
    hidden_layer_sizes = (max(int(n_features * 1.5), 2), max(int(n_features // 2), 2), max(int(n_features * 1.5), 2))
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=AE_ACTIVATION,
        solver=AE_SOLVER,
        max_iter=AE_MAX_ITER,
        random_state=AE_RANDOM_STATE
    )
    
    autoencoder.fit(X_scaled, X_scaled)
    
    # Calcular el error de reconstrucción (MSE) para los datos de entrenamiento
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    
    # Definir el umbral como el percentil correspondiente del error de los datos reales
    umbral = np.percentile(mse, AE_PERCENTIL_UMBRAL)
    
    return autoencoder, scaler, umbral

def filter_counterfactuals(autoencoder, scaler, umbral, file_path_ce, output_path):
    """
    Filtra los contraejemplos basándose en su error de reconstrucción usando
    el autoencoder previamente entrenado.
    
    Args:
        autoencoder (MLPRegressor): Modelo autoencoder ya entrenado.
        scaler (StandardScaler): Escalador entrenado con los datos originales.
        umbral (float): Valor máximo de error MSE permitido.
        file_path_ce (str): Ruta al archivo de contraejemplos original.
        output_path (str): Ruta donde se guardarán los contraejemplos filtrados.
        
    Returns:
        pd.DataFrame: DataFrame con los contraejemplos factibles (que pasaron el filtro).
    """
    if not os.path.exists(file_path_ce):
        print(f"No se encontró el archivo: {file_path_ce}")
        return None
        
    df_ce = pd.read_csv(file_path_ce)
    
    # Extraer las columnas que corresponden al contraejemplo generado ('ce_...')
    cols_ce = [col for col in df_ce.columns if col.startswith('ce_')]
    X_ce = df_ce[cols_ce]
    
    if X_ce.empty:
        print("No se encontraron columnas con el prefijo 'ce_' en el archivo de contraejemplos.")
        return None
        
    # Renombrar las columnas para que coincidan con las originales usadas en el fit del scaler
    X_ce.columns = [col.replace('ce_', '') for col in X_ce.columns]
        
    # Estandarizar los contraejemplos
    X_ce_scaled = scaler.transform(X_ce)
    
    # Reconstruir usando el autoencoder
    X_ce_pred = autoencoder.predict(X_ce_scaled)
    
    # Calcular el error de reconstrucción para cada contraejemplo
    mse_ce = np.mean(np.power(X_ce_scaled - X_ce_pred, 2), axis=1)
    
    # Filtrar los que estén por debajo (o igual) del umbral
    mask_factibles = mse_ce <= umbral
    df_factibles = df_ce[mask_factibles].copy()
    
    # Opcional: Agregar el error de reconstrucción al CSV de salida para referencia
    df_factibles['mse_reconstruccion'] = mse_ce[mask_factibles]
    
    # Guardar en CSV
    df_factibles.to_csv(output_path, index=False)
    
    print(f"\n--- Resultados del Filtro de Veracidad ---")
    print(f"Umbral de error (MSE): {umbral:.4f}")
    print(f"Total de contraejemplos evaluados: {len(df_ce)}")
    print(f"Contraejemplos factibles: {len(df_factibles)}")
    print(f"Porcentaje factible: {(len(df_factibles)/len(df_ce))*100:.2f}%")
    print(f"Archivo guardado en: {output_path}")
    
    return df_factibles

if __name__ == "__main__":
    # Rutas de los archivos (ajustables según la estructura del proyecto)
    DATASET_ORIGINAL = "datos/originales/diabetes.csv"
    CONTRAEJEMPLOS = "datos/procesados/contraejemplos.csv"
    CONTRAEJEMPLOS_FACTIBLES = "datos/procesados/contraejemplos_factibles.csv"
    
    print("Iniciando el filtro de veracidad con Autoencoder (MLP)...")
    
    # 1. Cargar y preparar datos originales
    print("Cargando dataset original...")
    X_train = load_and_prepare_data(DATASET_ORIGINAL)
    
    # 2. Entrenar el Autoencoder
    print(f"Entrenando Autoencoder sobre {X_train.shape[1]} características...")
    modelo, scaler, umbral = train_autoencoder(X_train)
    
    # 3. Filtrar los contraejemplos
    print("Filtrando contraejemplos...")
    filter_counterfactuals(modelo, scaler, umbral, CONTRAEJEMPLOS, CONTRAEJEMPLOS_FACTIBLES)
