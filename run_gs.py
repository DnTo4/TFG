import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.modelos.perceptron import train_model as train_pt_model
from src.modelos.svm import train_model as train_svm_model
from src.modelos.mlp import train_model as train_mlp_model
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

# ---- PARÁMETROS DE CONFIGURACIÓN ----
MODELO = "mlp"        # Tipo de clasificador: "perceptron", "svm" o "mlp"

N_INICIALES = 40      # Número de puntos de origen seleccionados del dataset
SEEDS = 2             # Cantidad de semillas (intentos) por cada punto de inicio
MUESTRAS = 500        # Puntos generados por cada capa (cáscara) en Growing Spheres
ANCHO_BANDA = 0.5     # Radio inicial (eta) para la búsqueda esferal
MAX_ITERS = 50        # Límite de iteraciones para encontrar un contraejemplo
UMBRAL = 1e-3         # Distancia mínima para considerar un contraejemplo como "nuevo"
RANDOM_STATE = 42     # Semilla global para reproducibilidad

TRAIN = "datos/originales/iris.data"   # Ruta del dataset de entrenamiento
TEST  = "datos/originales/iris.data"   # Ruta del dataset de prueba
TARGET  = None        # Nombre de la columna objetivo (None usa la última)

FS = True             # Activar feature selection
GUARDAR_CSV = True    # Habilitar la exportación de resultados a un archivo
CSV_PATH   = "datos/procesados/contraejemplos.csv"
MODEL_PATH = "modelos/modelo.joblib"
# -------------------------------------

def nuevo(x, cEjs, umbral):
    """
    Verifica si un contraejemplo recién encontrado es lo suficientemente distinto 
    a los ya almacenados en la lista de resultados.

    Args:
        x (np.ndarray): El contraejemplo a evaluar.
        cEjs (list): Lista de contraejemplos ya encontrados.
        umbral (float): Distancia mínima euclidiana requerida.

    Returns:
        bool: True si el contraejemplo es único o la lista está vacía.
    """
    if len(cEjs) == 0:
        return True
    dist = np.linalg.norm(np.asarray(cEjs) - x, axis=1)
    return np.all(dist > umbral)

def contraejemplos(modelo, entrada_df):
    """
    Coordina la generación masiva de contraejemplos para múltiples puntos de inicio.

    Itera sobre una selección aleatoria del dataset y aplica Growing Spheres 
    para encontrar el punto más cercano que cambie la predicción del modelo.

    Args:
        modelo: El clasificador entrenado.
        entrada_df (pd.DataFrame): Dataset completo (X) para extraer puntos de inicio.

    Returns:
        tuple: (contraejemplos_encontrados, puntos_inicio, etiquetas_originales)
    """
    rng = np.random.default_rng(RANDOM_STATE)
    predict_fn = modelo.predict
    
    # Selección aleatoria de índices de inicio
    idx = rng.permutation(len(entrada_df))[:N_INICIALES]
    seeds = rng.integers(0, 10_000_000, size=SEEDS)
    cEjs, inic, labels_orig = [], [], []

    for i in idx:
        x0 = entrada_df.iloc[[i]] # Extraer punto original como DataFrame
        for s in seeds:
            try:
                # Generar contraejemplo
                cEj = growing_spheres_generacion(
                    predict_fn=predict_fn,
                    x=x0,
                    muestras=MUESTRAS,
                    ancho_banda=ANCHO_BANDA,
                    max_iters=MAX_ITERS,
                    random_state=int(s),
                )
                
                # Refinar contraejemplo (minimizar características cambiadas)
                if FS:
                    cEj = feature_selection(predict_fn, x0, cEj)

                # Convertir a arrays planos para procesamiento
                cEj_arr = np.asarray(cEj, dtype=float).flatten()
                x0_arr  = np.asarray(x0, dtype=float).flatten()

                # Almacenar si es un resultado nuevo y válido
                if nuevo(cEj_arr, cEjs, UMBRAL):
                    cEjs.append(cEj_arr)
                    inic.append(x0_arr)
                    labels_orig.append(predict_fn(x0)[0])
                    
            except RuntimeError:
                # Se lanza si no se encuentra un contraejemplo en MAX_ITERS
                continue

    if not cEjs:
        return np.zeros((0, entrada_df.shape[1])), np.zeros((0, entrada_df.shape[1])), np.array([])
    
    return np.array(cEjs, dtype=float), np.array(inic, dtype=float), np.array(labels_orig)

def guardarCSV(cEjs, starts, labels_orig, nombres):
    """
    Exporta los resultados a un archivo CSV.

    Calcula las diferencias (deltas), las distancias L2 y marca qué variables 
    específicas fueron modificadas por el algoritmo.

    Args:
        cEjs (np.ndarray): Matriz de contraejemplos.
        starts (np.ndarray): Matriz de puntos originales.
        labels_orig (np.ndarray): Predicciones originales del modelo.
        nombres (list): Nombres de las características del dataset.
    """
    if len(cEjs) == 0:
        print("No hay contraejemplos para guardar.")
        return

    starts = np.asarray(starts, dtype=float)
    cEjs   = np.asarray(cEjs,  dtype=float)
    
    # Métricas de distancia y cambio
    dists   = np.linalg.norm(cEjs - starts, axis=1)
    difs    = cEjs - starts
    changed = np.abs(difs) > 1e-9 # Indica si hubo cambio significativo

    data = {}
    for i, col in enumerate(nombres):
        data[f"{col}"]         = starts[:,i]      # Valor original
        data[f"ce_{col}"]      = cEjs[:,i]        # Valor contraejemplo
        data[f"delta_{col}"]   = difs[:,i]        # Magnitud del cambio
        data[f"changed_{col}"] = changed[:,i].astype(int) # Binario (0/1) si cambió

    data["pred_orig"]            = labels_orig
    data["num_features_changed"] = changed.sum(axis=1)
    data["dist_l2"]              = dists

    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV guardado: {CSV_PATH}")

def main():
    """
    Flujo principal: 
    1. Entrena el modelo seleccionado.
    2. Guarda el modelo para análisis posterior.
    3. Genera contraejemplos basados en los datos de entrada.
    4. Exporta los resultados a CSV.
    """
    modelos = {
        "perceptron": train_pt_model,
        "svm": train_svm_model,
        "mlp": train_mlp_model
    }

    if MODELO not in modelos:
        raise ValueError(f"Modelo no soportado: {MODELO}")
        
    # Entrenamiento del clasificador
    train_func = modelos[MODELO]
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_func(TRAIN, TEST, TARGET)

    print(f"Accuracy test: {acc:.4f}")

    # Guarda el modelo
    joblib.dump({"modelo": modelo, "nombres": nombres}, MODEL_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")

    # Unión de datos
    entrada_df = pd.concat([X_train, X_test], axis=0)

    # Ejecución de Growing Spheres
    cEjs, starts, labels_orig = contraejemplos(modelo, entrada_df)
    print(f"Contraejemplos encontrados: {len(cEjs)}")

    # Exportación de datos
    if GUARDAR_CSV and len(cEjs) > 0:
        guardarCSV(cEjs, starts, labels_orig, nombres)

if __name__ == "__main__":
    main()
