import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from perceptron import train_model as train_pt_model
from svm import train_model as train_svm_model
from mlp import train_model as train_mlp_model
from growing_spheres import growing_spheres_generacion, feature_selection

# ---- parámetros ----
MODELO = "mlp"  # "perceptron", "svm" o "mlp"

N_INICIALES = 40      # n puntos de inicio
SEEDS = 2             # semillas por punto
MUESTRAS = 500       # muestras por cáscara
ANCHO_BANDA = 0.5     # eta inicial
MAX_ITERS = 50       # iteraciones máximas por fase
UMBRAL = 1e-3
RANDOM_STATE = 42

TRAIN = "iris.data"
TEST  = "iris.data"
TARGET  = None

FS = True              # aplicar feature selection
GUARDAR_CSV = True     # guardar contraejemplos en CSV
CSV_PATH   = "contraejemplos.csv"
MODEL_PATH = "modelo.joblib"
# ----------------------------

def nuevo(x, cEjs, umbral):
    # Verifica si un contraejemplo es nuevo
    if len(cEjs) == 0:
        return True
    dist = np.linalg.norm(np.asarray(cEjs) - x, axis=1)
    return np.all(dist > umbral)

def contraejemplos(modelo, entrada_df):
    rng = np.random.default_rng(RANDOM_STATE)
    predict_fn = modelo.predict
    idx = rng.permutation(len(entrada_df))[:N_INICIALES]
    seeds = rng.integers(0, 10_000_000, size=SEEDS)
    cEjs, inic, labels_orig = [], [], []

    for i in idx:
        x0 = entrada_df.iloc[[i]]
        for s in seeds:
            try:
                cEj = growing_spheres_generacion(
                    predict_fn=predict_fn,
                    x=x0,
                    muestras=MUESTRAS,
                    ancho_banda=ANCHO_BANDA,
                    max_iters=MAX_ITERS,
                    random_state=int(s),
                )
                if FS:
                    cEj = feature_selection(predict_fn, x0, cEj)

                cEj_arr = np.asarray(cEj, dtype=float).flatten()
                x0_arr  = np.asarray(x0, dtype=float).flatten()

                if nuevo(cEj_arr, cEjs, UMBRAL):
                    cEjs.append(cEj_arr)
                    inic.append(x0_arr)
                    labels_orig.append(predict_fn(x0)[0])
            except RuntimeError:
                continue

    if not cEjs:
        return np.zeros((0, entrada_df.shape[1])), np.zeros((0, entrada_df.shape[1])), np.array([])
    return np.array(cEjs, dtype=float), np.array(inic, dtype=float), np.array(labels_orig)

def guardarCSV(cEjs, starts, labels_orig, nombres):
    # Guarda los contraejemplos en CSV
    if len(cEjs) == 0:
        print("No hay contraejemplos para guardar.")
        return

    starts = np.asarray(starts, dtype=float)
    cEjs   = np.asarray(cEjs,   dtype=float)
    dists   = np.linalg.norm(cEjs - starts, axis=1)
    difs    = cEjs - starts
    changed = np.abs(difs) > 1e-9

    data = {}
    for i, col in enumerate(nombres):
        data[f"{col}"]         = starts[:,i]
        data[f"ce_{col}"]      = cEjs[:,i]
        data[f"delta_{col}"]   = difs[:,i]
        data[f"changed_{col}"] = changed[:,i].astype(int)

    data["pred_orig"]            = labels_orig
    data["num_features_changed"] = changed.sum(axis=1)
    data["dist_l2"]              = dists

    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV guardado: {CSV_PATH}")

def main():

    modelos = {
        "perceptron": train_pt_model,
        "svm": train_svm_model,
        "mlp": train_mlp_model
    }

    if MODELO not in modelos:
        raise ValueError(f"Modelo no soportado: {MODELO}")
        
    train_func = modelos[MODELO]
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_func(TRAIN, TEST, TARGET)

    print(f"Accuracy test: {acc:.4f}")

    # Guardar modelo para analizar.py
    joblib.dump({"modelo": modelo, "nombres": nombres}, MODEL_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")

    # Combinar DataFrames
    entrada_df = pd.concat([X_train, X_test], axis=0)
    labels     = pd.concat([y_train, y_test], axis=0)

    # Generación de contraejemplos
    cEjs, starts, labels_orig = contraejemplos(modelo, entrada_df)
    print(f"Contraejemplos encontrados: {len(cEjs)}")

    if GUARDAR_CSV and len(cEjs) > 0:
        guardarCSV(cEjs, starts, labels_orig, nombres)

if __name__ == "__main__":
    main()