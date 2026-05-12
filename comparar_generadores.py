"""
Fase 2: Comparativa de generadores de contraejemplos.
Enfrenta el Algoritmo Genético (AG), el Memético y Growing Spheres (GS)
utilizando el mismo modelo y dataset para asegurar una comparativa justa.
"""
import time
import argparse
import numpy as np
import pandas as pd
import joblib

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection
from src.visualizacion.analisis import plot_contraejemplos

def ejecutar_genetico(modelo, limites, nombres_dim, scaler, ruta_salida):
    print("\n[AG] Ejecutando Algoritmo Genético Puro...")
    start_time = time.time()
    
    pares_ga, _ = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=150,
        generaciones=60,
        tasa_mutacion=0.20,
        num_pares=30
    )
    
    elapsed = time.time() - start_time
    print(f"[AG] Finalizado en {elapsed:.2f}s. Encontrados {len(pares_ga)} pares.")
    
    if len(pares_ga) > 0:
        exportar_resultados(np.array(pares_ga), modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def ejecutar_memetico(modelo, limites, nombres_dim, scaler, ruta_salida):
    print("\n[MEM] Ejecutando Algoritmo Memético (AG + GS)...")
    start_time = time.time()
    
    # Búsqueda global (AG)
    print("  [MEM] Fase 1: Búsqueda Global (Semillas)...")
    pares_ga, _ = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=100,
        generaciones=40,
        tasa_mutacion=0.20,
        num_pares=20
    )
    
    if len(pares_ga) == 0:
        print("  [MEM] El AG no encontró semillas viables.")
        return None
        
    print(f"  [MEM] Fase 2: Búsqueda Local GS sobre {len(pares_ga)} semillas...")
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
                muestras=500,
                ancho_banda=0.5,
                max_iters=50,
                random_state=42 + idx
            )
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except RuntimeError:
            continue

    elapsed = time.time() - start_time
    print(f"[MEM] Finalizado en {elapsed:.2f}s. Encontrados {len(pares_memeticos)} pares refinados.")
    
    if len(pares_memeticos) > 0:
        exportar_resultados(np.array(pares_memeticos), modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def ejecutar_gs_puro(modelo, X_train, nombres_dim, ruta_salida):
    print("\n[GS] Ejecutando Growing Spheres Puro...")
    start_time = time.time()
    
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
                muestras=500,
                ancho_banda=0.5,
                max_iters=50,
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

    elapsed = time.time() - start_time
    print(f"[GS] Finalizado en {elapsed:.2f}s. Encontrados {len(cEjs)} pares.")
    
    if len(cEjs) > 0:
        # Ensamblamos la matriz [x_orig, ce_gs] para la función de exportación
        pares_gs = np.hstack([np.array(inic), np.array(cEjs)])
        exportar_resultados(pares_gs, modelo, nombres_dim, archivo_csv=ruta_salida)
        return pd.read_csv(ruta_salida)
    return None

def main(modelo_str, dataset_path, algoritmo):
    print("=== Fase 2: Comparativa de Generadores ===")
    print(f"[*] Dataset : {dataset_path}")
    print(f"[*] Modelo  : {modelo_str.upper()}")
    
    # 1. Entrenar el modelo
    print("[*] Entrenando modelo...")
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
    if algoritmo in ["genetico", "todos"]:
        ruta = "datos/procesados/contraejemplos_genetico.csv"
        dfs["Genético"] = ejecutar_genetico(modelo, limites, nombres_dim, scaler, ruta)
        
    if algoritmo in ["memetico", "todos"]:
        ruta = "datos/procesados/contraejemplos_memetico.csv"
        dfs["Memético"] = ejecutar_memetico(modelo, limites, nombres_dim, scaler, ruta)
        
    if algoritmo in ["gs", "todos"]:
        ruta = "datos/procesados/contraejemplos_gs.csv"
        dfs["GrowingSpheres"] = ejecutar_gs_puro(modelo, X_train_raw, nombres_dim, ruta)

    # 3. Mostrar Resumen Final
    print("\n--- RESUMEN DE LA COMPARATIVA ---")
    for nombre, df_res in dfs.items():
        if df_res is not None and "dist_l2" in df_res.columns:
            dist_media = df_res["dist_l2"].mean()
            print(f"- {nombre:15} | Distancia media (L2): {dist_media:.4f} | Contraejemplos: {len(df_res)}")
        else:
            print(f"- {nombre:15} | Falló o no encontró resultados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar la Fase 2: Comparativa de Algoritmos.")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp", "perceptron"])
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv")
    parser.add_argument("--algoritmo", type=str, default="todos", choices=["genetico", "memetico", "gs", "todos"])
    args = parser.parse_args()
    
    # Eliminamos las advertencias para salida más limpia
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args.modelo, args.dataset, args.algoritmo)
