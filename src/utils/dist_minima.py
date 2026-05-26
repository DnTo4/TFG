import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

"""Calcula la distancia mínima entre instancias de diferentes clases de un dataset.

Permite localizar los dos ejemplos más cercanos pertenecientes a clases distintas.
"""

def buscar_minima_distancia_dataset(ruta):
    """Buscar y exportar los dos puntos de clases distintas con menor distancia euclidiana.

    Carga el conjunto de datos, procesa variables, calcula la matriz
    de distancias y exporta el par a un fichero CSV.
    """
    print(f"Cargando dataset: '{ruta}' ...")
    
    # Leer el conjunto de datos
    try:
        df = pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {ruta}")
        return

    # Separar características y etiquetas
    target_col = df.columns[-1]
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # Codificar variables categóricas a numéricas
    X = pd.get_dummies(X_raw)
    
    X_val = X.values
    y_val = y.values
    
    # Calcular la matriz de distancias Euclidianas
    distancias = cdist(X_val, X_val, metric='euclidean')
    
    # Filtrar por pares que pertenecen a clases distintas
    y_cols = np.repeat(y_val[:, np.newaxis], len(y_val), axis=1)
    y_rows = np.repeat(y_val[np.newaxis, :], len(y_val), axis=0)
    mascara_distinta_clase = (y_cols != y_rows)
    
    # Penalizar los pares que pertenecen a la misma clase
    distancias[~mascara_distinta_clase] = np.inf
    
    # Obtener el par de puntos con la distancia mínima
    if np.isinf(distancias.min()):
        print("Error: No se encontró ningún par con clases distintas.")
        return
        
    # Extraer los índices de las dos instancias más cercanas
    idx_a, idx_b = np.unravel_index(np.argmin(distancias), distancias.shape)
    min_dist = distancias[idx_a, idx_b]
    
    # Presentar informe de resultados
    print(f"\nEJEMPLO (A) | Clase Original: '{y_val[idx_a]}'")
    for col in X.columns:
        print(f"    - {col}: {X.iloc[idx_a][col]}")
        
    print(f"\nEJEMPLO (B) | Clase Original: '{y_val[idx_b]}'")
    for col in X.columns:
        print(f"    - {col}: {X.iloc[idx_b][col]}")
    
    print(f"\nDistancia Euclidiana mínima entre clases: {min_dist:.6f}\n")
    
    # Exportar el par mínimo obtenido
    pd.DataFrame(
        [df.iloc[idx_a], df.iloc[idx_b]], 
        index=["Ejemplo_A", "Ejemplo_B"]
    ).to_csv("datos/procesados/par_minimo.csv")
    print("Se ha generado el archivo 'par_minimo.csv'.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Identifica y exporta los dos puntos más cercanos entre sí que pertenecen a clases distintas.")
    parser.add_argument("--ruta", type=str, default="datos/originales/iris.data", help="Ruta del archivo que contiene el dataset")
    args = parser.parse_args()
    
    buscar_minima_distancia_dataset(args.ruta)
