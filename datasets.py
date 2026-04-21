import numpy as np
import pandas as pd

# Funciones extraídas de las fuentes [1] y [2]
def vectores_unitarios(n, d, rng):
    vec = rng.normal(size=(n, d))
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec

def muestrea_cascara(centro, r_inf, r_sup, n, rng):
    d = centro.size
    radios = rng.uniform(r_inf, r_sup, size=n).reshape(-1, 1)
    return centro + vectores_unitarios(n, d, rng) * radios

def crear_dataset_no_lineal(nombre_archivo, n_puntos=100, seed=42):
    rng = np.random.default_rng(seed)
    centro = np.array([0.0, 0.0])
    n_clase = n_puntos // 2

    # Clase 0: Círculo central (Radio de 0 a 2)
    c0 = muestrea_cascara(centro, 0.0, 2.0, n_clase, rng)
    y0 = np.zeros(n_clase)

    # Clase 1: Anillo exterior (Radio de 4 a 6)
    c1 = muestrea_cascara(centro, 4.0, 6.0, n_clase, rng)
    y1 = np.ones(n_clase)

    # Combinar y guardar
    X = np.vstack([c0, c1])
    y = np.concatenate([y0, y1])
    
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y.astype(int)
    
    # Mezclar datos antes de guardar
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(nombre_archivo, index=False)
    print(f"Archivo '{nombre_archivo}' (Círculos concéntricos) generado.")

def crear_moons(nombre_archivo, n_puntos=200, noise=0.15, seed=42):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_puntos, noise=noise, random_state=seed)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    df.to_csv(nombre_archivo, index=False)
    print(f"Archivo '{nombre_archivo}' (Lunas entrelazadas) generado.")

def crear_lineal_2d(nombre_archivo, n_puntos=200, seed=42):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_puntos, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, random_state=seed, 
                               class_sep=1.5) # class_sep alto para q sea mas limpio
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    df.to_csv(nombre_archivo, index=False)
    print(f"Archivo '{nombre_archivo}' (Separación Lineal 2D) generado.")

if __name__ == "__main__":
    # Generar círculos concéntricos (train y test)
    crear_dataset_no_lineal("train_nolineal.csv", n_puntos=150, seed=42)
    crear_dataset_no_lineal("test_nolineal.csv", n_puntos=50, seed=123)

    # Generar lunas
    crear_moons("train_moons.csv", n_puntos=200, seed=42)
    
    # Generar lineal
    crear_lineal_2d("train_lineal.csv", n_puntos=200, seed=42)