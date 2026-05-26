import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.modelos.arbol_decision import train_model as train_arbol_model
from src.modelos.svm import train_model as train_svm_model
from src.modelos.mlp import train_model as train_mlp_model

"""Generación y optimización local de contraejemplos mediante Growing Spheres.

Implementa una búsqueda local basada en el muestreo concéntrico en forma de cáscaras esféricas
alrededor de una instancia para localizar el contraejemplo más cercano. Se llega a una
modificación mínima de características mediante selección de variables.
"""

def vectores_unitarios(n, d, rng):
    """Generar n vectores unitarios aleatorios de dimensión d.

    Utiliza una distribución normal estándar para el muestreo de direcciones y
    las normaliza dividiendo por su norma L2.
    """
    vec = rng.normal(size=(n, d))
    # Normalizar a longitud
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12) # El +1e-12 evita división por cero
    return vec

def muestrea_cascara(centro, r_inf, r_sup, n, rng):
    """Muestrear n puntos de forma uniforme dentro de una cáscara esférica.

    Genera radios aleatorios dentro del intervalo definido por los radios inferior
    y superior y los escala por vectores unitarios.
    """
    d = centro.size
    radios = rng.uniform(r_inf, r_sup, size=n).reshape(-1, 1)
    # Escalar vectores unitarios
    return centro + vectores_unitarios(n, d, rng) * radios

def growing_spheres_generacion(predict_fn, x, *, muestras=512, ancho_banda=0.5, max_iters=200, random_state=0):
    """Busca el contraejemplo local más cercano utilizando esferas crecientes.

    Muestrea concéntricamente alrededor de la instancia de entrada, reduciendo o
    ampliando progresivamente el radio de exploración hasta localizar una región
    que contenga predicciones de otra clase.
    """
    columnas = list(x.columns) if hasattr(x, 'columns') else None

    # Definir envoltura interna para predicciones
    def predict_wrapped(arr):
        if columnas is not None:
            return predict_fn(pd.DataFrame(arr, columns=columnas))
        return predict_fn(arr)

    # Reducir a vector unidimensional
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(random_state)
    y_x = predict_wrapped(x.reshape(1, -1))[0]

    # Crear candidatos en un radio inicial eta
    eta = float(ancho_banda)
    iter = 0
    cand = muestrea_cascara(x, 0.0, eta, muestras, rng)

    # Reducir eta a la mitad si la esfera inicial ya contiene contraejemplos
    while np.any(predict_wrapped(cand) != y_x) and iter < max_iters:
        eta *= 0.5
        cand = muestrea_cascara(x, 0.0, eta, muestras, rng)
        iter += 1

    # Definir una banda exterior y buscar hacia afuera
    a0, a1 = eta, 2 * eta
    iter = 0
    cand = muestrea_cascara(x, a0, a1, muestras, rng)

    # Incrementar el radio de la cáscara hasta localizar contraejemplos
    while not np.any(predict_wrapped(cand) != y_x) and iter < max_iters:
        a0 = a1
        a1 = a1 + eta
        cand = muestrea_cascara(x, a0, a1, muestras, rng)
        iter += 1

    # Extraer los índices de candidatos válidos en la última cáscara
    labels = predict_wrapped(cand)
    idx = np.where(labels != y_x)[0]

    if idx.size == 0:
        raise RuntimeError("No se encontro contraejemplo")
    
    # Seleccionar el candidato con la distancia euclidiana más pequeña
    i = idx[np.argmin(np.linalg.norm(cand[idx] - x, axis=1))]

    return cand[i]

def feature_selection(predict_fn, x, CEj):
    """Optimizar la dispersión del contraejemplo desactivando características.

    Evalúa las características ordenadas de menor a mayor cambio,
    e intenta restaurar sus valores originales sin alterar el cambio de clase.
    """
    columnas = list(x.columns) if hasattr(x, 'columns') else None

    def predict_wrapped(arr):
        if columnas is not None:
            return predict_fn(pd.DataFrame(arr, columns=columnas))
        return predict_fn(arr)

    x = np.asarray(x, float).ravel()
    cEj = np.asarray(CEj, float).ravel().copy()
    y_x = predict_wrapped(x.reshape(1, -1))[0]

    # Intentar reducir la variación de la características
    while predict_wrapped(cEj.reshape(1, -1))[0] != y_x:
        dif = np.abs(cEj - x)
        # Ignorar variables idénticas
        dif[dif == 0.0] = np.inf

        # Seleccionar la característica de cambio mínimo
        k = int(np.argmin(dif))

        # Evaluar la devolución de la característica k a su valor original
        prueba = cEj.copy()
        prueba[k] = x[k]

        # Comprobar que el cambio de clase se mantiene
        if predict_wrapped(prueba.reshape(1, -1))[0] != y_x:
            cEj = prueba
        else:
            break

    return cEj

def nuevo(x, cEjs, umbral):
    """Validar si un contraejemplo es nuevo con respecto a los ya aceptados.

    Estima la distancia L2 con respecto a todos los puntos almacenados y retorna
    True si queda por encima del umbral de tolerancia.
    """
    if len(cEjs) == 0:
        return True
    dist = np.linalg.norm(np.asarray(cEjs) - x, axis=1)
    return np.all(dist > umbral)

def contraejemplos(modelo, entrada_df, n_iniciales, seeds_count, muestras, ancho_banda, max_iters, random_state, umbral, fs):
    """Generar y consolidar contraejemplos locales para múltiples puntos iniciales.

    Muestrea de forma aleatoria índices de inicio, itera sobre las semillas aleatorias
    y devuelve matrices de resultados.
    """
    rng = np.random.default_rng(random_state)
    predict_fn = modelo.predict
    
    # Seleccionar de forma aleatoria las instancias de partida
    idx = rng.permutation(len(entrada_df))[:n_iniciales]
    seeds = rng.integers(0, 10_000_000, size=seeds_count)
    cEjs, inic, labels_orig = [], [], []

    # Iterar sobre las instancias seleccionadas
    for i in idx:
        x0 = entrada_df.iloc[[i]]
        # Ejecutar búsquedas con múltiples semillas para cada punto
        for s in seeds:
            try:
                # Generar el contraejemplo
                cEj = growing_spheres_generacion(
                    predict_fn=predict_fn,
                    x=x0,
                    muestras=muestras,
                    ancho_banda=ancho_banda,
                    max_iters=max_iters,
                    random_state=int(s),
                )
                
                # Minimizar variables modificadas mediante selección de características
                if fs:
                    cEj = feature_selection(predict_fn, x0, cEj)

                cEj_arr = np.asarray(cEj, dtype=float).flatten()
                x0_arr  = np.asarray(x0, dtype=float).flatten()

                # Guardar el contraejemplo si es nuevo
                if nuevo(cEj_arr, cEjs, umbral):
                    cEjs.append(cEj_arr)
                    inic.append(x0_arr)
                    labels_orig.append(predict_fn(x0)[0])
                    
            except RuntimeError:
                continue

    if not cEjs:
        return np.zeros((0, entrada_df.shape[1])), np.zeros((0, entrada_df.shape[1])), np.array([])
    
    return np.array(cEjs, dtype=float), np.array(inic, dtype=float), np.array(labels_orig)

def guardarCSV(cEjs, starts, labels_orig, nombres, csv_path):
    """Construir y almacenar un fichero CSV con los contraejemplos.

    Calcula diferencias, variaciones y métricas de distancia, mapea los nombres
    de variables originales y las variables modificadas, y exporta en filas planas.
    """
    if len(cEjs) == 0:
        print("No hay contraejemplos para guardar.")
        return

    starts = np.asarray(starts, dtype=float)
    cEjs   = np.asarray(cEjs,  dtype=float)
    
    # Calcular métricas de distancia y variaciones
    dists   = np.linalg.norm(cEjs - starts, axis=1)
    difs    = cEjs - starts
    changed = np.abs(difs) > 1e-9

    data = {}
    # Estructurar registros de salida
    for i, col in enumerate(nombres):
        data[f"{col}"]         = starts[:,i]
        data[f"ce_{col}"]      = cEjs[:,i]
        data[f"delta_{col}"]   = difs[:,i]
        data[f"changed_{col}"] = changed[:,i].astype(int)

    data["pred_orig"]            = labels_orig
    data["num_features_changed"] = changed.sum(axis=1)
    data["dist_l2"]              = dists

    # Almacenar en formato CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV guardado: {csv_path}")

def main():
    """Ejecutar la automatización de Growing Spheres como script."""
    parser = argparse.ArgumentParser(description="Ejecución independiente de Growing Spheres.")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["arbol_decision", "svm", "mlp"], help="Tipo de clasificador (default: mlp)")
    parser.add_argument("--n-iniciales", type=int, default=40, help="Número de puntos de origen seleccionados del dataset (default: 40)")
    parser.add_argument("--seeds", type=int, default=2, help="Cantidad de semillas por cada punto de inicio (default: 2)")
    parser.add_argument("--muestras", type=int, default=500, help="Puntos generados por capa en Growing Spheres (default: 500)")
    parser.add_argument("--ancho-banda", type=float, default=0.5, help="Radio inicial (eta) para la búsqueda esferal (default: 0.5)")
    parser.add_argument("--max-iters", type=int, default=50, help="Límite de iteraciones para encontrar un contraejemplo (default: 50)")
    parser.add_argument("--umbral", type=float, default=1e-3, help="Distancia mínima para considerar un contraejemplo como nuevo (default: 1e-3)")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla global para reproducibilidad (default: 42)")
    parser.add_argument("--train", type=str, default="datos/originales/iris.data", help="Ruta del dataset de entrenamiento (default: datos/originales/iris.data)")
    parser.add_argument("--test", type=str, default="datos/originales/iris.data", help="Ruta del dataset de prueba (default: datos/originales/iris.data)")
    parser.add_argument("--target", type=str, default=None, help="Nombre de la columna objetivo (default: última columna)")
    parser.add_argument("--no-fs", action="store_true", help="Desactivar feature selection")
    parser.add_argument("--no-guardar-csv", action="store_true", help="Deshabilitar la exportación de resultados a un archivo")
    parser.add_argument("--csv-path", type=str, default="datos/procesados/contraejemplos.csv", help="Ruta del archivo CSV de salida (default: datos/procesados/contraejemplos.csv)")
    parser.add_argument("--model-path", type=str, default="modelos/modelo.joblib", help="Ruta de guardado del modelo entrenado (default: modelos/modelo.joblib)")
    
    args = parser.parse_args()

    modelos = {
        "arbol_decision": train_arbol_model,
        "svm": train_svm_model,
        "mlp": train_mlp_model
    }

    if args.modelo not in modelos:
        raise ValueError(f"Modelo no soportado: {args.modelo}")
        
    # Entrenar el clasificador base
    train_func = modelos[args.modelo]
    modelo, (X_train, y_train, X_test, y_test), acc, nombres = train_func(args.train, args.test, args.target)

    print(f"Accuracy test: {acc:.4f}")

    # Guardar modelo entrenado
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump({"modelo": modelo, "nombres": nombres}, args.model_path)
    print(f"Modelo guardado: {args.model_path}")

    # Unir conjuntos de datos
    entrada_df = pd.concat([X_train, X_test], axis=0)

    # Cargar hiperparámetros
    from src.utils.hiperparametros import obtener_hiperparametros
    _, params_gs = obtener_hiperparametros(args.train, args.modelo)

    muestras_final = args.muestras if args.muestras != 500 else params_gs["muestras"]
    ancho_banda_final = args.ancho_banda if args.ancho_banda != 0.5 else params_gs["ancho_banda"]
    max_iters_final = args.max_iters if args.max_iters != 50 else params_gs["max_iters"]

    # Ejecutar Growing Spheres
    print(f"Ejecutando GS con Muestras={muestras_final}, AB={ancho_banda_final:.2f}, Max Iters={max_iters_final}...")
    cEjs, starts, labels_orig = contraejemplos(
        modelo=modelo, 
        entrada_df=entrada_df, 
        n_iniciales=args.n_iniciales,
        seeds_count=args.seeds,
        muestras=muestras_final,
        ancho_banda=ancho_banda_final,
        max_iters=max_iters_final,
        random_state=args.random_state,
        umbral=args.umbral,
        fs=not args.no_fs
    )
    print(f"Contraejemplos encontrados: {len(cEjs)}")

    # Exportar resultados
    if not args.no_guardar_csv and len(cEjs) > 0:
        guardarCSV(cEjs, starts, labels_orig, nombres, args.csv_path)

if __name__ == "__main__":
    main()
