import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

"""Entrenamiento de autoencoder para evaluación de realismo.

Entrena un MLP para aprender la distribución del dataset original y
filtrar contraejemplos con errores de reconstrucción elevados.
"""

def cargar_y_preparar_datos(file_path, target_col=None):
    """Cargar y preparar los datos de entrada.

    Descarta la etiqueta de clase y conserva únicamente las variables numéricas.
    """
    df = pd.read_csv(file_path)
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    return X

def entrenar_autoencoder(X, activation='relu', solver='adam', max_iter=1000, random_state=42, percentil_umbral=99):
    """Ajustar un MLP sobre los datos escalados.

    Calcula la dimensión oculta en base al número de características de entrada,
    entrena el MLP y estima el umbral del error de reconstrucción (MSE).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_features = X.shape[1]
    # Configurar dimensiones ocultas
    hidden_layer_sizes = (max(int(n_features * 1.5), 2), max(int(n_features // 2), 2), max(int(n_features * 1.5), 2))
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )
    
    autoencoder.fit(X_scaled, X_scaled)
    
    # Calcular el error de reconstrucción
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    
    # Definir el umbral basándose en un percentil
    umbral = np.percentile(mse, percentil_umbral)
    
    return autoencoder, scaler, umbral

def filtrar_contraejemplos(autoencoder, scaler, umbral, file_path_ce, output_path):
    """Filtrar contraejemplos usando el autoencoder.

    Alinea las características, calcula sus errores de reconstrucción y
    conserva únicamente aquellos que quedan por debajo del umbral de tolerancia.
    """
    if not os.path.exists(file_path_ce):
        print(f"No se encontró el archivo: {file_path_ce}")
        return None
        
    df_ce = pd.read_csv(file_path_ce)
    
    # Separar las columnas con prefijo 'ce_'
    cols_ce = [col for col in df_ce.columns if col.startswith('ce_')]
    X_ce = df_ce[cols_ce]
    
    if X_ce.empty:
        print("No se encontraron columnas con el prefijo 'ce_' en el archivo de contraejemplos.")
        return None
        
    # Renombrar las columnas
    X_ce = X_ce.copy()
    X_ce.columns = [col.replace('ce_', '') for col in X_ce.columns]
    
    # Validar coherencia de las características
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
        missing_cols = [c for c in scaler_features if c not in X_ce.columns]
        unseen_cols = [c for c in X_ce.columns if c not in scaler_features]
        
        if missing_cols or unseen_cols:
            raise ValueError(
                f"Discrepancia de características detectada para el Autoencoder.\n"
                f"El Scaler espera características: {scaler_features}\n"
                f"Pero los contraejemplos proporcionan características: {list(X_ce.columns)}\n"
            )
        # Ordenar columnas
        X_ce = X_ce[scaler_features]
        
    # Estandarizar y proyectar usando el autoencoder
    X_ce_scaled = scaler.transform(X_ce)
    X_ce_pred = autoencoder.predict(X_ce_scaled)
    
    # Calcular el error del contraejemplo
    mse_ce = np.mean(np.power(X_ce_scaled - X_ce_pred, 2), axis=1)
    
    # Filtrar contraejemplos factibles
    mask_factibles = mse_ce <= umbral
    df_factibles = df_ce[mask_factibles].copy()
    
    df_factibles['mse_reconstruccion'] = mse_ce[mask_factibles]
    
    # Exportar resultados a un CSV
    df_factibles.to_csv(output_path, index=False)
    
    print(f"\n--- Resultados del Filtro de Veracidad ---")
    print(f"Umbral de error (MSE): {umbral:.4f}")
    print(f"Total de contraejemplos evaluados: {len(df_ce)}")
    print(f"Contraejemplos factibles: {len(df_factibles)}")
    print(f"Porcentaje factible: {(len(df_factibles)/len(df_ce))*100:.2f}%")
    print(f"Archivo guardado en: {output_path}")
    
    return df_factibles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filtro de veracidad y realismo con Autoencoder.")
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset original para entrenar el autoencoder (default: datos/originales/diabetes.csv)")
    parser.add_argument("--contraejemplos", type=str, default="datos/procesados/contraejemplos.csv", help="Ruta al archivo CSV con los contraejemplos a evaluar (default: datos/procesados/contraejemplos.csv)")
    parser.add_argument("--salida", type=str, default="datos/procesados/contraejemplos_factibles.csv", help="Ruta para guardar los contraejemplos válidos/factibles (default: datos/procesados/contraejemplos_factibles.csv)")
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo del dataset original (default: última columna)")
    parser.add_argument("--percentil", type=float, default=99.0, help="Percentil de error para definir el umbral (default: 99.0)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Iteraciones máximas del Autoencoder (default: 1000)")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla del generador aleatorio (default: 42)")
    
    args = parser.parse_args()
    
    print("Iniciando el filtro de veracidad...")
    
    # Cargar y preprocesar los datos originales
    print("Cargando dataset original...")
    X_train = cargar_y_preparar_datos(args.dataset, target_col=args.target)
    
    # Entrenar el Autoencoder
    print(f"Entrenando Autoencoder sobre {X_train.shape[1]} características...")
    modelo, scaler, umbral = entrenar_autoencoder(
        X_train, 
        max_iter=args.max_iter, 
        random_state=args.random_state, 
        percentil_umbral=args.percentil
    )
    
    # Filtrar contraejemplos y almacenar
    print("Filtrando contraejemplos...")
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    filtrar_contraejemplos(modelo, scaler, umbral, args.contraejemplos, args.salida)
