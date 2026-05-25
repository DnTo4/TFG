"""
Comparativa de generadores de contraejemplos.
Enfrenta el Algoritmo Genético, el Memético y Growing Spheres
"""
import time
import argparse
import numpy as np
import pandas as pd
import joblib

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection
from src.visualizacion.analisis import plot_contraejemplos

def ejecutar_genetico(modelo, limites, nombres_dim, scaler, ruta_salida, modelo_str, dataset_path):
    print("\nEjecutando Algoritmo Genético...")
    
    from src.utils.hiperparametros import obtener_hiperparametros
    params_ga, _ = obtener_hiperparametros(dataset_path, modelo_str)
    
    pares_ga, _ = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=params_ga["tamano_poblacion"],
        generaciones=params_ga["generaciones"],
        tasa_mutacion=params_ga["tasa_mutacion"],
        num_pares=30
    )
    
    if len(pares_ga) > 0:
        exportar_resultados(np.array(pares_ga), modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def ejecutar_memetico(modelo, limites, nombres_dim, scaler, ruta_salida, modelo_str, dataset_path):
    print("\nEjecutando Algoritmo Memético...")
    
    from src.utils.hiperparametros import obtener_hiperparametros
    params_ga, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
    
    # Búsqueda global
    print("Búsqueda Global...")
    pares_ga, _ = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=params_ga["tamano_poblacion"],
        generaciones=params_ga["generaciones"],
        tasa_mutacion=params_ga["tasa_mutacion"],
        num_pares=20
    )
    
    if len(pares_ga) == 0:
        print("El AG no encontró semillas viables.")
        return None
        
    print("Búsqueda Local GS...")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    predict_fn = modelo.predict
    
    for idx, par in enumerate(pares_ga):
        x_orig = par[:d_dim]
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        try:
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=params_gs["muestras"],
                ancho_banda=params_gs["ancho_banda"],
                max_iters=params_gs["max_iters"],
                random_state=42 + idx
            )
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except RuntimeError:
            continue

    if len(pares_memeticos) > 0:
        exportar_resultados(np.array(pares_memeticos), modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def ejecutar_gs_puro(modelo, X_train, nombres_dim, ruta_salida, modelo_str, dataset_path):
    print("\nEjecutando Growing Spheres...")
    
    from src.utils.hiperparametros import obtener_hiperparametros
    _, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
    
    predict_fn = modelo.predict
    rng = np.random.default_rng(42)
    
    # Seleccionamos 20 puntos aleatorios del dataset para iniciar
    idx_seleccionados = rng.permutation(len(X_train))[:20]
    
    cEjs, inic = [], []
    for i in idx_seleccionados:
        x0_dict = {col: [X_train[i, j]] for j, col in enumerate(nombres_dim)}
        df_orig = pd.DataFrame(x0_dict)
        
        try:
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=params_gs["muestras"],
                ancho_banda=params_gs["ancho_banda"],
                max_iters=params_gs["max_iters"],
                random_state=int(rng.integers(0, 10000))
            )
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            
            x0_arr = df_orig.values.flatten()
            ce_arr = ce_gs.flatten()
            
            # Comprobar unicidad simple
            if len(cEjs) == 0 or np.all(np.linalg.norm(np.asarray(cEjs) - ce_arr, axis=1) > 1e-3):
                cEjs.append(ce_arr)
                inic.append(x0_arr)
        except RuntimeError:
            continue

    if len(cEjs) > 0:
        # Ensamblamos la matriz [x_orig, ce_gs] para la función de exportación
        pares_gs = np.hstack([np.array(inic), np.array(cEjs)])
        exportar_resultados(pares_gs, modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def main(modelo_str, dataset_path, algoritmo):
    print("=== Comparativa de Generadores ===")
    print(f"Dataset : {dataset_path}")
    print(f"Modelo  : {modelo_str.upper()}")
    
    # Entrenar el modelo
    print("Entrenando modelo...")
    modelo, limites, nombres_dim, scaler = entrenar_clasificador(
        dataset_path, dataset_path, tipo_modelo=modelo_str
    )
    
    # Cargar datos puros para GS (necesita puntos base reales)
    df = pd.read_csv(dataset_path)
    if "Outcome" in df.columns:
        X_df = df.drop(columns=["Outcome"])
    elif "class" in df.columns:
        X_df = df.drop(columns=["class"])
    else:
        X_df = df.iloc[:, :-1]
    X_train_raw = X_df.values

    # 2. Ejecutar según selección
    dfs = {}
    tiempos = {}
    if algoritmo in ["genetico", "todos"]:
        ruta = "datos/procesados/contraejemplos_genetico.csv"
        t0 = time.time()
        dfs["Genético"] = ejecutar_genetico(modelo, limites, nombres_dim, scaler, ruta, modelo_str, dataset_path)
        tiempos["Genético"] = time.time() - t0
        
    if algoritmo in ["memetico", "todos"]:
        ruta = "datos/procesados/contraejemplos_memetico.csv"
        t0 = time.time()
        dfs["Memético"] = ejecutar_memetico(modelo, limites, nombres_dim, scaler, ruta, modelo_str, dataset_path)
        tiempos["Memético"] = time.time() - t0
        
    if algoritmo in ["gs", "todos"]:
        ruta = "datos/procesados/contraejemplos_gs.csv"
        t0 = time.time()
        dfs["GrowingSpheres"] = ejecutar_gs_puro(modelo, X_train_raw, nombres_dim, ruta, modelo_str, dataset_path)
        tiempos["GrowingSpheres"] = time.time() - t0

    # 3. Mostrar Resumen Final
    print("\n--- RESUMEN DE LA COMPARATIVA ---")
    for nombre, df_res in dfs.items():
        if df_res is not None and "dist_l2" in df_res.columns:
            dist_media = df_res["dist_l2"].mean()
            dist_min = df_res["dist_l2"].min()
            t_exec = tiempos.get(nombre, 0.0)
            print(f"- {nombre:15} | Tiempo: {t_exec:6.2f}s | Distancia media: {dist_media:.4f} | Mejor Distancia: {dist_min:.4f} | Contraejemplos: {len(df_res)}")
        else:
            print(f"- {nombre:15} | Falló o no encontró resultados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparativa de Algoritmos.")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp", "perceptron"])
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv")
    parser.add_argument("--algoritmo", type=str, default="todos", choices=["genetico", "memetico", "gs", "todos"])
    args = parser.parse_args()
    
    # Eliminamos las advertencias para salida más limpia
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args.modelo, args.dataset, args.algoritmo)
