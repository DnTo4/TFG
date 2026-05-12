import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Ruta del dataset
RUTA_DATASET = "datos/originales/iris.data"

def buscar_minima_distancia_dataset(ruta):
    """
    Identifica y exporta los dos puntos más cercanos entre sí que pertenecen a clases distintas.
    
    Este proceso es útil para entender los límites de decisión (decision boundaries) 
    naturales de los datos, encontrando el "par crítico" donde la distinción entre 
    clases es mínima según la distancia Euclídea.

    Args:
        ruta (str): Ruta del archivo que contiene el dataset.
    """
    print(f"Cargando dataset: '{ruta}' ...")
    
    # Lectura de datos
    try:
        df = pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {ruta}")
        return

    # Separación de características (X) y etiquetas (y)
    # Se asume que la variable objetivo es la última columna
    target_col = df.columns[-1]
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # Preprocesamiento: Conversión de variables categóricas a numéricas
    X = pd.get_dummies(X_raw)
    
    X_val = X.values
    y_val = y.values
    
    # Calcular matriz de distancias Euclidianas entre todos los pares
    distancias = cdist(X_val, X_val, metric='euclidean')
    
    # Filtrado por clase: Solo interesan pares con etiquetas diferentes
    y_cols = np.repeat(y_val[:, np.newaxis], len(y_val), axis=1)
    y_rows = np.repeat(y_val[np.newaxis, :], len(y_val), axis=0)
    mascara_distinta_clase = (y_cols != y_rows)
    
    # Ignorar pares de la misma clase asignándoles una distancia infinita
    distancias[~mascara_distinta_clase] = np.inf
    
    # Búsqueda del valor mínimo global en la matriz filtrada
    if np.isinf(distancias.min()):
        print("Error: No se encontró ningún par con clases distintas.")
        return
        
    # Obtener índices de los dos puntos más cercanos
    idx_a, idx_b = np.unravel_index(np.argmin(distancias), distancias.shape)
    min_dist = distancias[idx_a, idx_b]
    
    # Reporte de resultados
    print(f"\nEJEMPLO (A) | Clase Original: '{y_val[idx_a]}'")
    for col in X.columns:
        print(f"    - {col}: {X.iloc[idx_a][col]}")
        
    print(f"\nEJEMPLO (B) | Clase Original: '{y_val[idx_b]}'")
    for col in X.columns:
        print(f"    - {col}: {X.iloc[idx_b][col]}")
    
    print(f"\nDistancia Euclidiana mínima entre clases: {min_dist:.6f}\n")
    
    # Exportación del "par crítico"
    pd.DataFrame(
        [df.iloc[idx_a], df.iloc[idx_b]], 
        index=["Ejemplo_A", "Ejemplo_B"]
    ).to_csv("datos/procesados/par_minimo.csv")
    print("Se ha generado el archivo 'par_minimo.csv'.\n")

if __name__ == "__main__":
    buscar_minima_distancia_dataset(RUTA_DATASET)
