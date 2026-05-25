"""
Análisis de Compromiso entre Escalabilidad y Calidad de los Algoritmos de Contraejemplos.
Compara el Algoritmo Genético, Memético y Growing Spheres variando el número de contraejemplos solicitados.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

# Desactivar advertencias
warnings.simplefilter("ignore")

def benchmark_genetico(modelo, limites, nombres_dim, scaler, n_pares, modelo_str, dataset_path):
    start = time.time()
    try:
        from src.utils.hiperparametros import obtener_hiperparametros
        params_ga, _ = obtener_hiperparametros(dataset_path, modelo_str)
        
        pares, _ = algoritmo_genetico(
            modelo=modelo,
            limites=limites,
            nombres_caracteristicas=nombres_dim,
            scaler=scaler,
            tamano_poblacion=params_ga["tamano_poblacion"],
            generaciones=params_ga["generaciones"],
            tasa_mutacion=params_ga["tasa_mutacion"],
            num_pares=n_pares
        )
        tiempo = time.time() - start
        
        if len(pares) > 0:
            d_dim = len(nombres_dim)
            dists = []
            for par in pares:
                p_a = par[:d_dim]
                p_b = par[d_dim:]
                dists.append(np.linalg.norm(p_b - p_a))
            l2_media = np.mean(dists)
            exito = len(pares) / n_pares
        else:
            l2_media = np.nan
            exito = 0.0
    except Exception as e:
        print(f"  [!] Error AG: {e}")
        tiempo = time.time() - start
        l2_media = np.nan
        exito = 0.0
        
    return tiempo, l2_media, exito

def benchmark_gs(modelo, X_train_raw, nombres_dim, n_pares, modelo_str, dataset_path):
    start = time.time()
    predict_fn = modelo.predict
    rng = np.random.default_rng(42)
    
    from src.utils.hiperparametros import obtener_hiperparametros
    _, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
    
    # Asegurar que no pedimos más puntos de los disponibles en el dataset
    max_puntos = min(len(X_train_raw), n_pares)
    idx_seleccionados = rng.permutation(len(X_train_raw))[:max_puntos]
    
    cEjs, dists = [], []
    for i in idx_seleccionados:
        x0_dict = {col: [X_train_raw[i, j]] for j, col in enumerate(nombres_dim)}
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
            
            dist = np.linalg.norm(ce_arr - x0_arr)
            dists.append(dist)
            cEjs.append(ce_arr)
        except RuntimeError:
            continue
            
    tiempo = time.time() - start
    l2_media = np.mean(dists) if len(dists) > 0 else np.nan
    exito = len(cEjs) / n_pares
    
    return tiempo, l2_media, exito

def benchmark_memetico(modelo, limites, nombres_dim, scaler, n_pares, modelo_str, dataset_path):
    start = time.time()
    try:
        from src.utils.hiperparametros import obtener_hiperparametros
        params_ga, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
        
        # Búsqueda global reducida para buscar semillas rápidamente
        pares_ga, _ = algoritmo_genetico(
            modelo=modelo,
            limites=limites,
            nombres_caracteristicas=nombres_dim,
            scaler=scaler,
            tamano_poblacion=params_ga["tamano_poblacion"],
            generaciones=params_ga["generaciones"],
            tasa_mutacion=params_ga["tasa_mutacion"],
            num_pares=n_pares
        )
        
        if len(pares_ga) == 0:
            tiempo = time.time() - start
            return tiempo, np.nan, 0.0
            
        d_dim = len(nombres_dim)
        dists = []
        cEjs = []
        predict_fn = modelo.predict
        
        # Búsqueda local refinada mediante GS para cada semilla
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
                
                dist = np.linalg.norm(ce_gs.flatten() - x_orig)
                dists.append(dist)
                cEjs.append(ce_gs.flatten())
            except RuntimeError:
                # Si falla la búsqueda local, conservamos el par genético original
                p_b = par[d_dim:]
                dist = np.linalg.norm(p_b - x_orig)
                dists.append(dist)
                cEjs.append(p_b)
                
        tiempo = time.time() - start
        l2_media = np.mean(dists) if len(dists) > 0 else np.nan
        exito = len(cEjs) / n_pares
    except Exception as e:
        print(f"  [!] Error Memético: {e}")
        tiempo = time.time() - start
        l2_media = np.nan
        exito = 0.0
        
    return tiempo, l2_media, exito

def main():
    print("=========================================================")
    print("  ESTUDIO COMPARATIVO DE COMPROMISO: AG vs GS vs MEMÉTICO")
    print("=========================================================")
    
    dataset_path = "datos/originales/train_moons.csv"
    modelo_str = "svm"
    
    print(f"[*] Entrenando clasificador base ({modelo_str.upper()}) sobre moons...")
    modelo, limites, nombres_dim, scaler = entrenar_clasificador(
        dataset_path, dataset_path, tipo_modelo=modelo_str
    )
    
    df = pd.read_csv(dataset_path)
    X_train_raw = df.drop(columns=[df.columns[-1]]).values
    
    # Rango de números de contraejemplos a evaluar
    valores_n = [5, 10, 20, 30, 40, 50]
    
    resultados = []
    
    for n in valores_n:
        print(f"\n>>> Evaluando para N = {n} contraejemplos...")
        
        # 1. Growing Spheres
        print("    [+] Ejecutando Growing Spheres...")
        t_gs, l2_gs, ex_gs = benchmark_gs(modelo, X_train_raw, nombres_dim, n, modelo_str, dataset_path)
        print(f"        Tiempo: {t_gs:.2f}s | L2: {l2_gs:.4f} | Éxito: {ex_gs*100:.1f}%")
        
        # 2. Algoritmo Genético
        print("    [+] Ejecutando Algoritmo Genético...")
        t_ga, l2_ga, ex_ga = benchmark_genetico(modelo, limites, nombres_dim, scaler, n, modelo_str, dataset_path)
        print(f"        Tiempo: {t_ga:.2f}s | L2: {l2_ga:.4f} | Éxito: {ex_ga*100:.1f}%")
        
        # 3. Algoritmo Memético
        print("    [+] Ejecutando Algoritmo Memético...")
        t_mem, l2_mem, ex_mem = benchmark_memetico(modelo, limites, nombres_dim, scaler, n, modelo_str, dataset_path)
        print(f"        Tiempo: {t_mem:.2f}s | L2: {l2_mem:.4f} | Éxito: {ex_mem*100:.1f}%")
        
        resultados.append({
            "N": n,
            "t_gs": t_gs, "l2_gs": l2_gs, "ex_gs": ex_gs,
            "t_ga": t_ga, "l2_ga": l2_ga, "ex_ga": ex_ga,
            "t_mem": t_mem, "l2_mem": l2_mem, "ex_mem": ex_mem
        })
        
    df_res = pd.DataFrame(resultados)
    
    # Exportar datos
    os.makedirs("resultados", exist_ok=True)
    df_res.to_csv("resultados/rendimiento_algoritmos.csv", index=False)
    print("\n[+] Datos guardados en 'resultados/rendimiento_algoritmos.csv'")
    
    # Generar Gráfico 1: Tiempo vs N (Escalabilidad y Punto de Inflexión)
    plt.figure(figsize=(10, 6))
    plt.plot(df_res["N"], df_res["t_gs"], marker='o', linewidth=2.5, color='#4CAF50', label="Growing Spheres")
    plt.plot(df_res["N"], df_res["t_ga"], marker='s', linewidth=2.5, color='#2196F3', label="Algoritmo Genético")
    plt.plot(df_res["N"], df_res["t_mem"], marker='^', linewidth=2.5, color='#9C27B0', label="Algoritmo Memético")
    
    plt.grid(color='#E0E0E0', linestyle='--')
    plt.title("Escalabilidad Temporal de los Algoritmos", fontweight='bold')
    plt.xlabel("Número de Contraejemplos Solicitados (N)")
    plt.ylabel("Tiempo de Ejecución Total (Segundos)")
    plt.legend(frameon=True, edgecolor='#E0E0E0')
    plt.tight_layout()
    plt.savefig("resultados/escalabilidad_tiempo.png", dpi=150)
    plt.close()
    print("[+] Gráfico de escalabilidad guardado en 'resultados/escalabilidad_tiempo.png'")
    
    # Generar Gráfico 2: Calidad (Distancia L2 media) vs N
    plt.figure(figsize=(10, 6))
    plt.plot(df_res["N"], df_res["l2_gs"], marker='o', linestyle='--', linewidth=2.0, color='#4CAF50', label="Growing Spheres")
    plt.plot(df_res["N"], df_res["l2_ga"], marker='s', linestyle='--', linewidth=2.0, color='#2196F3', label="Algoritmo Genético")
    plt.plot(df_res["N"], df_res["l2_mem"], marker='^', linestyle='-', linewidth=2.5, color='#9C27B0', label="Algoritmo Memético")
    
    plt.grid(color='#E0E0E0', linestyle='--')
    plt.title("Calidad del Contraejemplo (Distancia L2 Media)", fontweight='bold')
    plt.xlabel("Número de Contraejemplos Solicitados (N)")
    plt.ylabel("Distancia L2 Media")
    plt.legend(frameon=True, edgecolor='#E0E0E0')
    plt.tight_layout()
    plt.savefig("resultados/calidad_l2.png", dpi=150)
    plt.close()
    print("[+] Gráfico de calidad de distancia guardado en 'resultados/calidad_l2.png'")
    
    # Conclusiones impresas en pantalla
    print("\n==============================")
    print("       INFORME DE RENDIMIENTO")
    print("==============================")
    
    # Calcular punto de cruce aproximado de tiempo
    gs_tiempos = df_res["t_gs"].values
    ga_tiempos = df_res["t_ga"].values
    cruce = None
    for i in range(len(valores_n)):
        if gs_tiempos[i] > ga_tiempos[i]:
            cruce = valores_n[i]
            break
            
    print("EFICIENCIA:")
    if cruce:
        print(f"   -> EXISTE UN PUNTO DE CRUCE a partir de N = {cruce} contraejemplos.")
        print(f"      - Para N < {cruce}: Growing Spheres es más rápido")
        print(f"      - Para N >= {cruce}: El Algoritmo Genético/Memético es más eficiente")
    else:
        # Si no cruzó en el rango probado
        if gs_tiempos[-1] < ga_tiempos[-1]:
            print(f"   -> En el rango evaluado (hasta {valores_n[-1]}), Growing Spheres sigue siendo el más rápido.")
            print(f"      Sin embargo, observa cómo su pendiente de crecimiento lineal O(N) es más empinada.")
        else:
            print(f"   -> El Algoritmo Genético/Memético es más rápido en todo el rango evaluado.")
            
    print("\nCALIDAD DE LOS CONTRAEJEMPLOS (DISTANCIA L2):")
    l2_gs_mean = df_res["l2_gs"].mean()
    l2_ga_mean = df_res["l2_ga"].mean()
    l2_mem_mean = df_res["l2_mem"].mean()

    print(f"     * Growing Spheres : {l2_gs_mean:.4f}")
    print(f"     * Algoritmo Genético: {l2_ga_mean:.4f}")
    print(f"     * Algoritmo Memético: {l2_mem_mean:.4f}")

if __name__ == "__main__":
    main()
