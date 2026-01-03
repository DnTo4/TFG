# run_gs.py  (adaptado al pseudocódigo + feature_selection opcional, sin bounds)
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import train_model
from growing_spheres import growing_spheres_generation, feature_selection

# ---- parámetros simples ----
N_INICIALES = 70      # n de puntos de inicio
SEEDS = 5       # semillas por punto
MUESTRAS = 500    # muestras por cáscara
ANCHO_BANDA = 0.5       # eta inicial
MAX_ITERS = 200     # iteraciones máximas por fase
UMBRAL = 1e-3
RANDOM_STATE = 42

FS = False    # aplicar feature selection
GUARDAR_CSV = True     # guardar contraejemplos en CSV
PATH = "contraejemplos.csv"
# ----------------------------

# Verifica si un contraejemplo es nuevo (distancia > umbral)
def nuevo(x, cEjs, umbral):
    if not cEjs:
        return True
    
    dist = np.linalg.norm(np.asarray(cEjs) - x, axis=1)
    return np.all(dist > umbral)

def contraejemplos(modelo, entrada):
    # Aleatoriedad reproducible
    rng = np.random.default_rng(RANDOM_STATE)

    # Funcion de prediccion del modelo
    predict_fn = modelo.predict

    # Seleccion aleatoria de N_INICIALES indices de X
    idx = rng.permutation(len(entrada))[:N_INICIALES]

    # Para cada modelo prueba SEEDS muestreos por punto
    seeds = rng.integers(0, 10_000_000, size=SEEDS)

    # Almacenar contraejemplos y puntos iniciales
    cEjs, starts = [], []

    # Para cada punto inicial prueba varias semillas
    for i in idx:
        x0 = entrada[i]
        for s in seeds:
            try:
                cEj = growing_spheres_generation(
                    predict_fn=predict_fn,
                    x=x0,
                    muestras=MUESTRAS,
                    ancho_banda=ANCHO_BANDA,
                    max_iters=MAX_ITERS,
                    random_state=int(s),
                )
                # Aplicar feature selection si esta activado
                if FS:
                    cEj = feature_selection(predict_fn, x0, cEj)

                # Añadir si es nuevo
                if nuevo(cEj, cEjs, UMBRAL):
                    cEjs.append(cEj); starts.append(x0)
            except RuntimeError:
                continue
    # Si no hay contraejemplos, devolver arrays vacíos
    if not cEjs:
        return np.zeros((0, entrada.shape[1])), np.zeros((0, entrada.shape[1]))
    
    return np.array(cEjs), np.array(starts)

# Graficar resultados
def plot(modelo, entrada, y, cEjs, starts):
    predict_fn   = modelo.predict
    has_proba    = hasattr(modelo, "predict_proba")
    has_decision = hasattr(modelo, "decision_function")

    pad = 0.8
    x_min, x_max = entrada[:,0].min()-pad, entrada[:,0].max()+pad
    y_min, y_max = entrada[:,1].min()-pad, entrada[:,1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    plt.figure(figsize=(7,6))
    if has_proba:
        probs = modelo.predict_proba(grid)[:,1].reshape(xx.shape)
        plt.contour(xx, yy, probs, levels=[0.5])
    elif has_decision:
        scores = modelo.decision_function(grid).reshape(xx.shape)
        plt.contour(xx, yy, scores, levels=[0.0])
    else:
        y_grid = predict_fn(grid).reshape(xx.shape)
        plt.contour(xx, yy, y_grid, levels=[0.5])

    plt.scatter(entrada[:,0], entrada[:,1], s=10)
    if len(cEjs) > 0:
        for a, b in zip(starts, cEjs):
            plt.plot([a[0], b[0]], [a[1], b[1]])
        plt.scatter(starts[:,0], starts[:,1], marker='o', s=60)
        plt.scatter(cEjs[:,0],    cEjs[:,1],    marker='X', s=80)
    plt.title(f"Contraejemplos: {len(cEjs)}")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.show()

def main():
    modelo, (X_train, y_train, X_test, y_test), acc = train_model()
    print(f"Accuracy test: {acc:.4f}")

    entrada = np.vstack([X_train, X_test])
    labels = np.concatenate([y_train, y_test])

    cEjs, starts = contraejemplos(modelo, entrada)
    print(f"Contraejemplos encontrados: {len(cEjs)}")

    # Guardar contraejemplos en CSV
    if GUARDAR_CSV and len(cEjs) > 0:
        dists = np.linalg.norm(cEjs - starts, axis=1)
        df = pd.DataFrame({
            "x0_1": starts[:,0], "x0_2": starts[:,1],
            "ce_1": cEjs[:,0],    "ce_2": cEjs[:,1],
            "dist_l2": dists
        })
        df.to_csv(PATH, index=False)
        print(f"CSV Guardado: {PATH}")

    # Graficar si es 2D
    if entrada.shape[1] == 2:
        plot(modelo, entrada, labels, cEjs, starts)
    else:
        print("Plot omitido")

if __name__ == "__main__":
    main()
