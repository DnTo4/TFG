import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Ruta por defecto
RUTA_DATASET = "iris.data"

def buscar_minima_distancia_dataset(ruta):
    print(f"Cargando dataset: '{ruta}' ...")
    
    # Cargamos el CSV simplemente. 
    df = pd.read_csv(ruta)
    
    # Asumimos que la variable objetivo es la última columna (común en datasets)
    target_col = df.columns[-1]
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # One-hot encoding por si el dataset tiene variables categóricas
    X = pd.get_dummies(X_raw)
    
    X_val = X.values
    y_val = y.values
    
    # Calcular matriz de distancias
    distancias = cdist(X_val, X_val, metric='euclidean')
    
    # Matriz booleana: True si (i, j) tienen distinta clase
    y_cols = np.repeat(y_val[:, np.newaxis], len(y_val), axis=1)
    y_rows = np.repeat(y_val[np.newaxis, :], len(y_val), axis=0)
    mascara_distinta_clase = (y_cols != y_rows)
    
    # Reemplazar con infinito las distancias donde la clase es la misma
    distancias[~mascara_distinta_clase] = np.inf
    
    # Encontrar la posición del mínimo valor en toda la matriz
    if np.isinf(distancias.min()):
        print("Error: No se encontró ningún par con clases distintas.")
        return
        
    idx_a, idx_b = np.unravel_index(np.argmin(distancias), distancias.shape)
    min_dist = distancias[idx_a, idx_b]
    
    # Imprimir los resultados
    print(f"\nEJEMPLO | Clase Original: '{y_val[idx_a]}'")
    for col in X.columns:
        print(f"   - {col}: {X.iloc[idx_a][col]}")
        
    print(f"\nCONTRAEJEMPLO | Clase Original: '{y_val[idx_b]}'")
    for col in X.columns:
        print(f"   - {col}: {X.iloc[idx_b][col]}")
    
    print(f"\nDistancia Euclidiana: {min_dist:.6f}\n")
    
    # Exportar datos
    pd.DataFrame(
        [df.iloc[idx_a], df.iloc[idx_b]], 
        index=["Ejemplo_A", "Ejemplo_B"]
    ).to_csv("par_minimo.csv")
    print("Se ha generado el archivo 'par_minimo.csv'.\n")

if __name__ == "__main__":
    buscar_minima_distancia_dataset(RUTA_DATASET)
