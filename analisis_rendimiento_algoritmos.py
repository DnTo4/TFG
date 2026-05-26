import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

"""Análisis comparativo del rendimiento y escalabilidad de algoritmos de contraejemplos.

Proporciona funciones para medir el tiempo de ejecución, la distancia L2 media y la tasa de éxito
de los algoritmos Genético, Growing Spheres y Memético.
"""

warnings.simplefilter("ignore")

def rendimiento_genetico(modelo, limites, nombres_dim, scaler, n_pares, modelo_str, dataset_path):
    """Evaluar el rendimiento del algoritmo genético en la generación de contraejemplos.

    Calcula el tiempo de ejecución, la distancia euclidiana media de los contraejemplos válidos
    y la tasa de éxito respecto al total solicitado.
    """
    start = time.time()
    try:
        from src.utils.hiperparametros import obtener_hiperparametros
        # Cargar parámetros configurados para el dataset
        params_ga, _ = obtener_hiperparametros(dataset_path, modelo_str)
        
        # Ejecutar el algoritmo genético
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
        
        # Procesar los resultados obtenidos
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

def rendimiento_gs(modelo, X_train_raw, nombres_dim, n_pares, modelo_str, dataset_path):
    """Evaluar el rendimiento del algoritmo Growing Spheres en la generación de contraejemplos.

    Realiza búsquedas locales basadas en instancias del dataset de entrenamiento para medir
    el tiempo invertido, la calidad de la distancia y el porcentaje de convergencia.
    """
    start = time.time()
    predict_fn = modelo.predict
    rng = np.random.default_rng(42)
    
    from src.utils.hiperparametros import obtener_hiperparametros
    # Obtener parámetros óptimos de Growing Spheres
    _, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
    
    # Evitar solicitar más puntos de los disponibles en el conjunto
    max_puntos = min(len(X_train_raw), n_pares)
    idx_seleccionados = rng.permutation(len(X_train_raw))[:max_puntos]
    
    cEjs, dists = [], []
    # Generar contraejemplos para cada punto seleccionado
    for i in idx_seleccionados:
        x0_dict = {col: [X_train_raw[i, j]] for j, col in enumerate(nombres_dim)}
        df_orig = pd.DataFrame(x0_dict)
        
        try:
            # Ejecutar generación local mediante esferas crecientes
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=params_gs["muestras"],
                ancho_banda=params_gs["ancho_banda"],
                max_iters=params_gs["max_iters"],
                random_state=int(rng.integers(0, 10000))
            )
            # Simplificar contraejemplo reduciendo cambios en características
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

def rendimiento_memetico(modelo, limites, nombres_dim, scaler, n_pares, modelo_str, dataset_path):
    """Evaluar el rendimiento del algoritmo memético en la generación de contraejemplos.

    Combina la exploración global del algoritmo genético con la explotación local del
    algoritmo Growing Spheres para refinar los resultados.
    """
    start = time.time()
    try:
        from src.utils.hiperparametros import obtener_hiperparametros
        # Cargar configuraciones de hiperparámetros
        params_ga, params_gs = obtener_hiperparametros(dataset_path, modelo_str)
        
        # Ejecutar búsqueda global reducida para localizar candidatos iniciales
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
        
        # Refinar localmente cada semilla generada usando Growing Spheres
        for idx, par in enumerate(pares_ga):
            x_orig = par[:d_dim]
            df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
            
            try:
                # Aplicar esferas crecientes desde el punto de partida
                ce_gs = growing_spheres_generacion(
                    predict_fn=predict_fn,
                    x=df_orig,
                    muestras=params_gs["muestras"],
                    ancho_banda=params_gs["ancho_banda"],
                    max_iters=params_gs["max_iters"],
                    random_state=42 + idx
                )
                # Seleccionar características cambiadas
                ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
                
                dist = np.linalg.norm(ce_gs.flatten() - x_orig)
                dists.append(dist)
                cEjs.append(ce_gs.flatten())
            except RuntimeError:
                # Conservar el contraejemplo original si falla la refinación
                p_b = par[d_dim:]
                dist = np.linalg.norm(p_b - x_orig)
                dists.append(dist)
                cEjs.append(p_b)
                
        tiempo = time.time() - start
        l2_media = np.mean(dists) if len(dists) > 0 else np.nan
        exito = len(cEjs) / n_pares
    except Exception as e:
        print(f"Error en el Algoritmo Memético: {e}")
        tiempo = time.time() - start
        l2_media = np.nan
        exito = 0.0
        
    return tiempo, l2_media, exito

def main():
    """Ejecutar la comparación de rendimiento de los diferentes generadores.

    Configura argumentos de entrada, entrena el modelo base, ejecuta las simulaciones
    y guarda los resultados.
    """
    parser = argparse.ArgumentParser(description="Análisis comparativo de compromiso: AG vs GS vs Memético.")
    parser.add_argument("--dataset", type=str, default="datos/originales/train_moons.csv", help="Ruta al dataset original (default: datos/originales/train_moons.csv)")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp", "arbol_decision"], help="Tipo de clasificador base (default: svm)")
    parser.add_argument("--valores-n", type=str, default="5,10,20,30,40,50", help="Valores de N contraejemplos a evaluar separados por comas (default: 5,10,20,30,40,50)")
    
    args = parser.parse_args()
    
    # Validar y procesar parámetros numéricos de entrada
    try:
        valores_n = [int(x.strip()) for x in args.valores_n.split(",")]
    except ValueError:
        raise ValueError(f"Error procesando valores. Deben ser enteros separados por comas: {args.valores_n}")

    print("====================================================")
    print("  ESTUDIO COMPARATIVO: AG vs GS vs MEMÉTICO")
    print("====================================================")
    print(f"Dataset: {args.dataset}")
    print(f"Modelo : {args.modelo.upper()}")
    print(f"Rango N: {valores_n}")
    print("----------------------------------------------------")
    
    # Comprobar la existencia del archivo de entrada
    if not os.path.exists(args.dataset):
        print(f"Error: No se encontró el dataset en {args.dataset}")
        return

    # Entrenar clasificador y preparar variables globales
    print(f"Entrenando clasificador ({args.modelo.upper()})")
    modelo, limites, nombres_dim, scaler = entrenar_clasificador(
        args.dataset, args.dataset, tipo_modelo=args.modelo
    )
    
    df = pd.read_csv(args.dataset)
    X_train_raw = df.drop(columns=[df.columns[-1]]).values
    
    resultados = []
    
    # Iterar sobre las diferentes cantidades solicitadas de contraejemplos
    for n in valores_n:
        print(f"\nEvaluando para N = {n} contraejemplos...")
        
        # Ejecutar Growing Spheres
        print("Ejecutando Growing Spheres...")
        t_gs, l2_gs, ex_gs = rendimiento_gs(modelo, X_train_raw, nombres_dim, n, args.modelo, args.dataset)
        print(f"Tiempo: {t_gs:.2f}s | L2: {l2_gs:.4f} | Éxito: {ex_gs*100:.1f}%")
        
        # Ejecutar Algoritmo Genético
        print("Ejecutando Algoritmo Genético...")
        t_ga, l2_ga, ex_ga = rendimiento_genetico(modelo, limites, nombres_dim, scaler, n, args.modelo, args.dataset)
        print(f"Tiempo: {t_ga:.2f}s | L2: {l2_ga:.4f} | Éxito: {ex_ga*100:.1f}%")
        
        # Ejecutar Algoritmo Memético
        print("Ejecutando Algoritmo Memético...")
        t_mem, l2_mem, ex_mem = rendimiento_memetico(modelo, limites, nombres_dim, scaler, n, args.modelo, args.dataset)
        print(f"Tiempo: {t_mem:.2f}s | L2: {l2_mem:.4f} | Éxito: {ex_mem*100:.1f}%")
        
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
    print("\nDatos guardados en 'resultados/rendimiento_algoritmos.csv'")
    
    # Crear gráfica comparativa de tiempo
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
    print("Gráfico de escalabilidad guardado en 'resultados/escalabilidad_tiempo.png'")
    
    # Crear gráfica comparativa de calidad
    plt.figure(figsize=(10, 6))
    if not df_res["l2_gs"].isna().all():
        plt.plot(df_res["N"], df_res["l2_gs"], marker='o', linestyle='--', linewidth=2.0, color='#4CAF50', label="Growing Spheres")
    if not df_res["l2_ga"].isna().all():
        plt.plot(df_res["N"], df_res["l2_ga"], marker='s', linestyle='--', linewidth=2.0, color='#2196F3', label="Algoritmo Genético")
    if not df_res["l2_mem"].isna().all():
        plt.plot(df_res["N"], df_res["l2_mem"], marker='^', linestyle='-', linewidth=2.5, color='#9C27B0', label="Algoritmo Memético")
    
    plt.grid(color='#E0E0E0', linestyle='--')
    plt.title("Calidad del Contraejemplo (Distancia L2 Media)", fontweight='bold')
    plt.xlabel("Número de Contraejemplos Solicitados (N)")
    plt.ylabel("Distancia L2 Media")
    plt.legend(frameon=True, edgecolor='#E0E0E0')
    plt.tight_layout()
    plt.savefig("resultados/calidad_l2.png", dpi=150)
    plt.close()
    print("Gráfico de calidad de distancia guardado en 'resultados/calidad_l2.png'")
    
    # Generar conclusiones
    print("\n==============================")
    print("       INFORME DE RENDIMIENTO")
    print("==============================")
    
    # Localizar punto de cruce de eficiencia temporal
    gs_tiempos = df_res["t_gs"].values
    ga_tiempos = df_res["t_ga"].values
    cruce = None
    for i in range(len(valores_n)):
        if gs_tiempos[i] > ga_tiempos[i]:
            cruce = valores_n[i]
            break
            
    print("EFICIENCIA:")
    if cruce:
        print(f"   Existe un punto de cruce a partir de N = {cruce} contraejemplos.")
        print(f"      - Para N < {cruce}: Growing Spheres es más rápido")
        print(f"      - Para N >= {cruce}: El Algoritmo Memético es más eficiente")
    else:
        # Analizar comportamiento si no se cruzaron las curvas en el rango
        if gs_tiempos[-1] < ga_tiempos[-1]:
            print(f"En el rango evaluado (hasta {valores_n[-1]}), Growing Spheres sigue siendo el más rápido.")
        else:
            print(f"El Algoritmo Memético es más rápido en todo el rango evaluado.")
            
    print("\nCALIDAD DE LOS CONTRAEJEMPLOS (DISTANCIA L2):")
    l2_gs_mean = df_res["l2_gs"].mean()
    l2_ga_mean = df_res["l2_ga"].mean()
    l2_mem_mean = df_res["l2_mem"].mean()

    print(f"     * Growing Spheres : {l2_gs_mean:.4f}" if not pd.isna(l2_gs_mean) else "     * Growing Spheres : N/A")
    print(f"     * Algoritmo Genético: {l2_ga_mean:.4f}" if not pd.isna(l2_ga_mean) else "     * Algoritmo Genético: N/A")
    print(f"     * Algoritmo Memético: {l2_mem_mean:.4f}" if not pd.isna(l2_mem_mean) else "     * Algoritmo Memético: N/A")

if __name__ == "__main__":
    main()
