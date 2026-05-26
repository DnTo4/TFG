import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

import warnings
warnings.simplefilter("ignore")

"""Optimización de hiperparámetros de algoritmos de contraejemplos.

Realiza una exploración mediante Grid Search sobre diferentes configuraciones de
población, generaciones, tasas de mutación y número de muestras, evaluando la
convergencia, tiempo de cómputo y distancia en múltiples datasets.
"""

def parse_arguments():
    """Configurar y analizar los argumentos de línea de comandos.

    Define los parámetros de exploración para los algoritmos Genético, Growing Spheres
    y Memético, así como opciones de almacenamiento y robustez estadística.
    """
    parser = argparse.ArgumentParser(description="Estudio masivo de hiperparámetros de algoritmos de contraejemplos.")
    parser.add_argument("--algoritmo", type=str, default="ambos", choices=["ga", "gs", "memetico", "ambos"],
                        help="Algoritmo a estudiar: 'ga' (Genético), 'gs' (Growing Spheres), 'memetico' (Memético), 'ambos' (GA y GS).")
    
    # Hiperparámetros GA
    parser.add_argument("--pop-vals", type=str, default="50,100,200",
                        help="Valores de tamaño de población de GA separados por comas (para GA y Memético).")
    parser.add_argument("--gen-vals", type=str, default="40,80,120",
                        help="Valores de generaciones de GA separados por comas (para GA y Memético).")
    parser.add_argument("--mut-vals", type=str, default="0.05,0.15,0.30",
                        help="Valores de tasa de mutación de GA separados por comas (solo para GA).")
    
    # Hiperparámetros GS
    parser.add_argument("--muestras-vals", type=str, default="100,300,500",
                        help="Valores de número de muestras de GS separados por comas (para GS y Memético).")
    parser.add_argument("--ancho-banda-vals", type=str, default="0.2,0.5,0.8",
                        help="Valores de ancho de banda (eta) de GS separados por comas (solo para GS).")
    parser.add_argument("--max-iters-vals", type=str, default="20,40,60",
                        help="Valores de iteraciones máximas de GS separados por comas (solo para GS).")
    
    parser.add_argument("--num-pares", type=int, default=30,
                        help="Número de pares de contraejemplos deseados.")
    parser.add_argument("--repeticiones", type=int, default=2,
                        help="Número de repeticiones de cada configuración para robustez estadística.")
    parser.add_argument("--output-dir", type=str, default="resultados",
                        help="Directorio para guardar los resultados.")
    return parser.parse_args()

def calcular_convergencia(historial, umbral=0.95):
    """Calcular la generación en la que el algoritmo estabiliza su aprendizaje.

    Identifica el punto en el que se alcanza el porcentaje especificado de la mejora total
    del fitness a lo largo del historial de optimización.
    """
    if len(historial) <= 1:
        return 1
        
    inicial = historial[0]
    final = historial[-1]
    
    # Retornar el total de generaciones si no se encontraron pares válidos
    if final >= 900000:
        return len(historial)

    mejora_total = inicial - final
    if mejora_total <= 0:
        return 1
        
    umbral_mejora = inicial - umbral * mejora_total
    
    # Buscar la generación exacta que supera el umbral de mejora
    for gen, val in enumerate(historial):
        if val <= umbral_mejora:
            return gen + 1
            
    return len(historial)

def ejecutar_gs(modelo, X_train_raw, nombres_dim, n_pares, muestras, ancho_banda, max_iters, seed):
    """Ejecutar Growing Spheres bajo una configuración específica de parámetros.

    Realiza búsquedas locales repetidas a partir de instancias seleccionadas aleatoriamente
    del conjunto de entrenamiento.
    """
    print(f"Iniciando Growing Spheres (Muestras={muestras}, AB={ancho_banda:.2f}, Max Iters={max_iters})...")
    rng = np.random.default_rng(seed)
    predict_fn = modelo.predict
    
    # Limitar la muestra al total disponible si es menor que lo solicitado
    max_puntos = min(len(X_train_raw), n_pares)
    idx_seleccionados = rng.permutation(len(X_train_raw))[:max_puntos]
    
    pares = []
    
    # Procesar secuencialmente cada punto semilla seleccionado
    for i in idx_seleccionados:
        x0_dict = {col: [X_train_raw[i, j]] for j, col in enumerate(nombres_dim)}
        df_orig = pd.DataFrame(x0_dict)
        
        try:
            # Ejecutar exploración por esferas crecientes
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=muestras,
                ancho_banda=ancho_banda,
                max_iters=max_iters,
                random_state=int(rng.integers(0, 10000))
            )
            # Optimizar características cambiadas
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            
            x0_arr = df_orig.values.flatten()
            ce_arr = ce_gs.flatten()
            
            pares.append(np.concatenate([x0_arr, ce_arr]))
        except RuntimeError:
            continue
            
    return np.array(pares)

def ejecutar_memetico(modelo, limites, nombres_dim, scaler, n_pares, tamano_poblacion, generaciones, tasa_mutacion, muestras, ancho_banda, max_iters, seed):
    """Ejecutar el algoritmo memético para una configuración determinada de parámetros.

    Coordina la generación evolutiva global de semillas con la optimización local
    de esferas crecientes en cada una de ellas.
    """
    print(f"Memético: Iniciando búsqueda global (Pop={tamano_poblacion}, Gen={generaciones})...")
    np.random.seed(seed)
    try:
        # Generar semillas usando el algoritmo genético
        pares_ga, historial_ga = algoritmo_genetico(
            modelo=modelo,
            limites=limites,
            nombres_caracteristicas=nombres_dim,
            scaler=scaler,
            tamano_poblacion=tamano_poblacion,
            generaciones=generaciones,
            tasa_mutacion=tasa_mutacion,
            num_pares=n_pares
        )
    except Exception as e:
        print(f"Error en fase GA del Memético: {e}")
        return np.array([]), [999999.0]
        
    if len(pares_ga) == 0:
        return np.array([]), historial_ga
        
    d_dim = len(nombres_dim)
    pares_memeticos = []
    predict_fn = modelo.predict
    
    # Iniciar refinamiento
    print(f"Memético: Iniciando refinamiento Growing Spheres para {len(pares_ga)} pares semilla...")
    for idx, par in enumerate(pares_ga):
        x_orig = par[:d_dim]
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        try:
            # Ejecutar optimización local 
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=muestras,
                ancho_banda=ancho_banda,
                max_iters=max_iters,
                random_state=seed + idx
            )
            # Aplicar selección de características
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except RuntimeError:
            continue
            
    return np.array(pares_memeticos), historial_ga

def ejecutar_grid_search(args, modelo, limites, nombres_dim, scaler, X_train_raw, dataset_name, classifier_name):
    """Ejecutar la búsqueda de rejilla completa para las configuraciones especificadas.

    Ejecuta repeticiones independientes de cada combinación parametrizada de hiperparámetros
    y calcula el promedio de sus métricas de rendimiento y convergencia.
    """
    num_pares = args.num_pares
    resultados_detallados = []
    
    if args.algoritmo == "ga":
        pop_list = [int(x) for x in args.pop_vals.split(",")]
        gen_list = [int(x) for x in args.gen_vals.split(",")]
        mut_list = [float(x) for x in args.mut_vals.split(",")]
        
        total_configs = len(pop_list) * len(gen_list) * len(mut_list)
        print(f"\nIniciando Grid Search para GA con {total_configs} combinaciones...")
        print(f"    - Tamaños de población: {pop_list}")
        print(f"    - Generaciones: {gen_list}")
        print(f"    - Tasas de mutación: {mut_list}")
        
        config_idx = 1
        for pop in pop_list:
            for gen in gen_list:
                for mut in mut_list:
                    print(f"\n[{config_idx}/{total_configs}] Config: Población={pop}, Gen={gen}, Mutación={mut:.2f}")
                    
                    tiempos = []
                    num_pares_encontrados = []
                    distancias_l2 = []
                    features_cambiadas = []
                    convergencia_gens = []
                    historiales = []
                    
                    for rep in range(args.repeticiones):
                        np.random.seed(42 + rep)
                        t_inicio = time.time()
                        
                        try:
                            # Lanzar el algoritmo genético para la combinación actual
                            pares, historial = algoritmo_genetico(
                                modelo=modelo,
                                limites=limites,
                                nombres_caracteristicas=nombres_dim,
                                scaler=scaler,
                                tamano_poblacion=pop,
                                generaciones=gen,
                                tasa_mutacion=mut,
                                num_pares=num_pares
                            )
                        except Exception as e:
                            print(f"Error en ejecución: {e}")
                            continue
                            
                        elapsed = time.time() - t_inicio
                        cant_validos = len(pares)
                        
                        tiempos.append(elapsed)
                        num_pares_encontrados.append(cant_validos)
                        historiales.append(historial)
                        
                        if cant_validos > 0:
                            d_dim = len(nombres_dim)
                            dists_rep = []
                            feats_rep = []
                            # Evaluar la calidad de los contraejemplos válidos
                            for par in pares:
                                p_a = par[:d_dim]
                                p_b = par[d_dim:]
                                delta = p_b - p_a
                                dists_rep.append(np.linalg.norm(delta))
                                feats_rep.append(np.sum(np.abs(delta) > 1e-3))
                            
                            distancias_l2.append(np.mean(dists_rep))
                            features_cambiadas.append(np.mean(feats_rep))
                        else:
                            distancias_l2.append(np.nan)
                            features_cambiadas.append(np.nan)
                            
                        conv = calcular_convergencia(historial)
                        convergencia_gens.append(conv)
                        
                        print(f"Rep {rep+1}/{args.repeticiones}: Encontrados={cant_validos}/{num_pares} | L2={distancias_l2[-1]:.4f} | Tiempo={elapsed:.2f}s | Conv={conv} gen")
                    
                    # Promedios de la búsqueda
                    exito_medio = np.mean(num_pares_encontrados) / num_pares
                    tiempo_medio = np.mean(tiempos)
                    dist_media = np.nanmean(distancias_l2) if not np.all(np.isnan(distancias_l2)) else np.nan
                    feat_medio = np.nanmean(features_cambiadas) if not np.all(np.isnan(features_cambiadas)) else np.nan
                    conv_media = np.mean(convergencia_gens)
                    
                    # Homogeneizar longitudes
                    max_len = max(len(h) for h in historiales) if historiales else 1
                    historiales_completos = []
                    for h in historiales:
                        if len(h) < max_len:
                            historiales_completos.append(h + [h[-1]] * (max_len - len(h)))
                        else:
                            historiales_completos.append(h)
                    hist_promedio = np.mean(historiales_completos, axis=0).tolist() if historiales_completos else [999999.0]
                    
                    resultados_detallados.append({
                        "dataset": dataset_name,
                        "clasificador": classifier_name,
                        "tamano_poblacion": pop,
                        "generaciones": gen,
                        "tasa_mutacion": mut,
                        "tasa_exito": exito_medio,
                        "num_pares_medios": np.mean(num_pares_encontrados),
                        "dist_l2_media": dist_media,
                        "features_cambiadas_media": feat_medio,
                        "tiempo_ejecucion_medio": tiempo_medio,
                        "generaciones_convergencia_media": conv_media,
                        "historial_fitness": hist_promedio
                    })
                    config_idx += 1
                    
    elif args.algoritmo == "gs":
        muestras_list = [int(x) for x in args.muestras_vals.split(",")]
        ancho_banda_list = [float(x) for x in args.ancho_banda_vals.split(",")]
        max_iters_list = [int(x) for x in args.max_iters_vals.split(",")]
        
        total_configs = len(muestras_list) * len(ancho_banda_list) * len(max_iters_list)
        print(f"\nIniciando Grid Search para GS con {total_configs} combinaciones...")
        print(f"    - Muestras: {muestras_list}")
        print(f"    - Anchos de banda: {ancho_banda_list}")
        print(f"    - Iteraciones máximas: {max_iters_list}")
        
        config_idx = 1
        for muestras in muestras_list:
            for ab in ancho_banda_list:
                for max_it in max_iters_list:
                    print(f"\n[{config_idx}/{total_configs}] Config: Muestras={muestras}, Ancho Banda={ab:.2f}, Max Iters={max_it}")
                    
                    tiempos = []
                    num_pares_encontrados = []
                    distancias_l2 = []
                    features_cambiadas = []
                    
                    for rep in range(args.repeticiones):
                        t_inicio = time.time()
                        
                        # Ejecutar Growing Spheres para la combinación actual
                        pares = ejecutar_gs(
                            modelo=modelo,
                            X_train_raw=X_train_raw,
                            nombres_dim=nombres_dim,
                            n_pares=num_pares,
                            muestras=muestras,
                            ancho_banda=ab,
                            max_iters=max_it,
                            seed=42 + rep
                        )
                        
                        elapsed = time.time() - t_inicio
                        cant_validos = len(pares)
                        
                        tiempos.append(elapsed)
                        num_pares_encontrados.append(cant_validos)
                        
                        if cant_validos > 0:
                            d_dim = len(nombres_dim)
                            dists_rep = []
                            feats_rep = []
                            for par in pares:
                                p_a = par[:d_dim]
                                p_b = par[d_dim:]
                                delta = p_b - p_a
                                dists_rep.append(np.linalg.norm(delta))
                                feats_rep.append(np.sum(np.abs(delta) > 1e-3))
                            
                            distancias_l2.append(np.mean(dists_rep))
                            features_cambiadas.append(np.mean(feats_rep))
                        else:
                            distancias_l2.append(np.nan)
                            features_cambiadas.append(np.nan)
                            
                        print(f"      Rep {rep+1}/{args.repeticiones}: Encontrados={cant_validos}/{num_pares} | L2={distancias_l2[-1]:.4f} | Tiempo={elapsed:.2f}s")
                    
                    # Guardar resultados agregados
                    exito_medio = np.mean(num_pares_encontrados) / num_pares
                    tiempo_medio = np.mean(tiempos)
                    dist_media = np.nanmean(distancias_l2) if not np.all(np.isnan(distancias_l2)) else np.nan
                    feat_medio = np.nanmean(features_cambiadas) if not np.all(np.isnan(features_cambiadas)) else np.nan
                    
                    resultados_detallados.append({
                        "dataset": dataset_name,
                        "clasificador": classifier_name,
                        "muestras": muestras,
                        "ancho_banda": ab,
                        "max_iters": max_it,
                        "tasa_exito": exito_medio,
                        "num_pares_medios": np.mean(num_pares_encontrados),
                        "dist_l2_media": dist_media,
                        "features_cambiadas_media": feat_medio,
                        "tiempo_ejecucion_medio": tiempo_medio,
                        "generaciones_convergencia_media": 1.0,
                        "historial_fitness": [dist_media] * 10 if not np.isnan(dist_media) else [999999.0] * 10
                    })
                    config_idx += 1
                    
    elif args.algoritmo == "memetico":
        pop_list = [int(x) for x in args.pop_vals.split(",")]
        gen_list = [int(x) for x in args.gen_vals.split(",")]
        muestras_list = [int(x) for x in args.muestras_vals.split(",")]
        
        total_configs = len(pop_list) * len(gen_list) * len(muestras_list)
        print(f"\nIniciando Grid Search para Memético con {total_configs} combinaciones...")
        print(f"    - Tamaños de población GA: {pop_list}")
        print(f"    - Generaciones GA: {gen_list}")
        print(f"    - Muestras GS: {muestras_list}")
        
        tasa_mutacion_ga = 0.20
        ancho_banda_gs = 0.5
        max_iters_gs = 40
        
        config_idx = 1
        for pop in pop_list:
            for gen in gen_list:
                for muestras in muestras_list:
                    print(f"\n[{config_idx}/{total_configs}] Config: Población GA={pop}, Gen GA={gen}, Muestras GS={muestras}")
                    
                    tiempos = []
                    num_pares_encontrados = []
                    distancias_l2 = []
                    features_cambiadas = []
                    convergencia_gens = []
                    historiales = []
                    
                    for rep in range(args.repeticiones):
                        t_inicio = time.time()
                        
                        # Ejecutar el algoritmo memético
                        pares, historial_ga = ejecutar_memetico(
                            modelo=modelo,
                            limites=limites,
                            nombres_dim=nombres_dim,
                            scaler=scaler,
                            n_pares=num_pares,
                            tamano_poblacion=pop,
                            generaciones=gen,
                            tasa_mutacion=tasa_mutacion_ga,
                            muestras=muestras,
                            ancho_banda=ancho_banda_gs,
                            max_iters=max_iters_gs,
                            seed=42 + rep
                        )
                        
                        elapsed = time.time() - t_inicio
                        cant_validos = len(pares)
                        
                        tiempos.append(elapsed)
                        num_pares_encontrados.append(cant_validos)
                        
                        if cant_validos > 0:
                            d_dim = len(nombres_dim)
                            dists_rep = []
                            feats_rep = []
                            for par in pares:
                                p_a = par[:d_dim]
                                p_b = par[d_dim:]
                                delta = p_b - p_a
                                dists_rep.append(np.linalg.norm(delta))
                                feats_rep.append(np.sum(np.abs(delta) > 1e-3))
                            
                            distancias_l2.append(np.mean(dists_rep))
                            features_cambiadas.append(np.mean(feats_rep))
                        else:
                            distancias_l2.append(np.nan)
                            features_cambiadas.append(np.nan)
                            
                        # Almacenar historial normalizando penalizaciones
                        if cant_validos > 0 and len(historial_ga) > 0:
                            hist_ajustado = list(historial_ga)
                            hist_ajustado = [x if x < 900000 else np.nanmean(dists_rep)*1.5 for x in hist_ajustado]
                            hist_ajustado[-1] = np.mean(dists_rep)
                            historiales.append(hist_ajustado)
                        else:
                            historiales.append(historial_ga)
                            
                        conv = calcular_convergencia(historial_ga)
                        convergencia_gens.append(conv)
                        
                        print(f"Rep {rep+1}/{args.repeticiones}: Encontrados={cant_validos}/{num_pares} | L2={distancias_l2[-1]:.4f} | Tiempo={elapsed:.2f}s")
                    
                    # Datos del algoritmo memético
                    exito_medio = np.mean(num_pares_encontrados) / num_pares
                    tiempo_medio = np.mean(tiempos)
                    dist_media = np.nanmean(distancias_l2) if not np.all(np.isnan(distancias_l2)) else np.nan
                    feat_medio = np.nanmean(features_cambiadas) if not np.all(np.isnan(features_cambiadas)) else np.nan
                    conv_media = np.mean(convergencia_gens)
                    
                    max_len = max(len(h) for h in historiales) if historiales else 1
                    historiales_completos = []
                    for h in historiales:
                        if len(h) < max_len:
                            historiales_completos.append(h + [h[-1]] * (max_len - len(h)))
                        else:
                            historiales_completos.append(h)
                    hist_promedio = np.mean(historiales_completos, axis=0).tolist() if historiales_completos else [999999.0]
                    
                    resultados_detallados.append({
                        "dataset": dataset_name,
                        "clasificador": classifier_name,
                        "tamano_poblacion": pop,
                        "generaciones": gen,
                        "muestras": muestras,
                        "tasa_exito": exito_medio,
                        "num_pares_medios": np.mean(num_pares_encontrados),
                        "dist_l2_media": dist_media,
                        "features_cambiadas_media": feat_medio,
                        "tiempo_ejecucion_medio": tiempo_medio,
                        "generaciones_convergencia_media": conv_media,
                        "historial_fitness": hist_promedio
                    })
                    config_idx += 1
                    
    return pd.DataFrame(resultados_detallados)

def calcular_puntuacion_seleccion(df):
    """Puntuar las diferentes configuraciones en base a varios criterios.

    Calcula una puntuación ponderada que considera la tasa de éxito (40%),
    distancia L2 (35%), velocidad de cómputo (15%) y número de características
    cambiadas (10%) para guiar la selección del mejor hiperparámetro.
    """
    df_out = []
    # Agrupar por dataset para normalizaciones independientes
    for dataset_name, df_sub in df.groupby("dataset"):
        df_temp = df_sub.copy()
        max_dist = df_temp["dist_l2_media"].max()
        max_time = df_temp["tiempo_ejecucion_medio"].max()
        max_feats = df_temp["features_cambiadas_media"].max()
        
        if pd.isna(max_dist) or max_dist == 0: max_dist = 1.0
        if pd.isna(max_time) or max_time == 0: max_time = 1.0
        if pd.isna(max_feats) or max_feats == 0: max_feats = 1.0
        
        df_temp["dist_l2_media"] = df_temp["dist_l2_media"].fillna(max_dist)
        df_temp["features_cambiadas_media"] = df_temp["features_cambiadas_media"].fillna(max_feats)
        
        score_exito = df_temp["tasa_exito"]
        score_dist = 1.0 - (df_temp["dist_l2_media"] / max_dist)
        score_tiempo = 1.0 - (df_temp["tiempo_ejecucion_medio"] / max_time)
        score_feats = 1.0 - (df_temp["features_cambiadas_media"] / max_feats)
        
        score_dist = np.clip(score_dist, 0, 1)
        score_tiempo = np.clip(score_tiempo, 0, 1)
        score_feats = np.clip(score_feats, 0, 1)
        
        # Estimar la puntuación de utilidad
        df_sub["score_seleccion"] = (score_exito * 0.40) + (score_dist * 0.35) + (score_tiempo * 0.15) + (score_feats * 0.10)
        df_out.append(df_sub)
        
    return pd.concat(df_out, ignore_index=True)

def generar_grafico_dataset(df_dataset, output_dir, dataset_name):
    """Generar gráfica comparativa de convergencia para el conjunto de datos dado.

    Identifica la mejor configuración de algoritmo genético y Growing Spheres para
    cada arquitectura de clasificador base y visualiza su evolución de aptitud.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    clasificadores = ["svm", "mlp", "arbol_decision"]
    colors = {"svm": "#2196F3", "mlp": "#4CAF50", "arbol_decision": "#FF9800"}
    
    plt.figure(figsize=(11, 7))
    
    has_plotted = False
    
    # Graficar las curvas de cada clasificador
    for clf in clasificadores:
        df_clf = df_dataset[df_dataset["clasificador"] == clf]
        if df_clf.empty:
            continue
            
        df_ga = df_clf[df_clf["algoritmo"] == "ga"]
        df_gs = df_clf[df_clf["algoritmo"] == "gs"]
        
        row_ga = None
        row_gs = None
        
        if not df_ga.empty:
            row_ga = df_ga.sort_values(by="score_seleccion", ascending=False).iloc[0]
        if not df_gs.empty:
            row_gs = df_gs.sort_values(by="score_seleccion", ascending=False).iloc[0]
            
        # Dibujar la curva del Algoritmo Genético
        if row_ga is not None:
            hist = row_ga["historial_fitness"]
            if isinstance(hist, list):
                hist_filtrado = [x if x < 900000 else np.nan for x in hist]
                lbl_ga = f"{clf.upper()} - GA (Pop={row_ga['tamano_poblacion']}, Gen={row_ga['generaciones']}, Mut={row_ga['tasa_mutacion']:.2f})"
                plt.plot(hist_filtrado, label=lbl_ga, color=colors[clf], linewidth=2.5, linestyle="-")
                has_plotted = True
                
        # Dibujar la línea de referencia de Growing Spheres
        if row_gs is not None:
            l2_gs = row_gs["dist_l2_media"]
            if not pd.isna(l2_gs):
                lbl_gs = f"{clf.upper()} - GS (Muestras={row_gs['muestras']}, AB={row_gs['ancho_banda']:.1f}, Éxito={row_gs['tasa_exito']*100:.0f}%, {row_gs['tiempo_ejecucion_medio']:.2f}s)"
                plt.axhline(y=l2_gs, label=lbl_gs, color=colors[clf], linewidth=2.0, linestyle="--")
                has_plotted = True
                
    if not has_plotted:
        print(f"No se encontraron mejores configuraciones para graficar en {dataset_name}")
        plt.close()
        return
        
    plt.grid(color='#E8E8E8', linestyle='--', linewidth=0.8)
    plt.title(f"Comparativa GA vs GS - {dataset_name.capitalize()}", fontweight='bold', fontsize=14, pad=15)
    plt.xlabel("Generación (para GA)", fontsize=11, labelpad=10)
    plt.ylabel("Distancia L2 Media", fontsize=11, labelpad=10)
    
    plt.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0', loc='best', fontsize=9.5)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, f"convergencia_{dataset_name}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"Gráfico comparativo guardado en: {file_path}")

def main():
    """Ejecutar análisis de hiperparámetros.

    Coordina la carga de datos, el entrenamiento de múltiples clasificadores, la ejecución
    del Grid Search en paralelo para cada combinación, y la exportación de resultados y gráficas.
    """
    args = parse_arguments()
    
    print("================================")
    print(f"ESTUDIO DE HIPERPARÁMETROS")
    print("================================")
    
    datasets = {
        "iris": "datos/originales/iris.data",
        "diabetes": "datos/originales/diabetes.csv"
    }
    
    clasificadores = ["svm", "mlp", "arbol_decision"]
    
    # Inicializar las listas de algoritmos
    if args.algoritmo == "ambos":
        algoritmos_a_ejecutar = ["ga", "gs"]
    else:
        algoritmos_a_ejecutar = [args.algoritmo]
        
    print(f"Algoritmos seleccionados: {[alg.upper() for alg in algoritmos_a_ejecutar]}")
    
    all_results = []
    
    # Procesar cada conjunto de datos
    for ds_name, ds_path in datasets.items():
        print(f"\n====================================================")
        print(f"PROCESANDO DATASET: {ds_name.upper()} ({ds_path})")
        print(f"====================================================")
        
        try:
            df_data = pd.read_csv(ds_path)
            X_train_raw = df_data.drop(columns=[df_data.columns[-1]]).values
        except Exception as e:
            print(f"Error leyendo dataset {ds_path}: {e}")
            continue
            
        # Iterar sobre las topologías de clasificadores
        for clf_name in clasificadores:
            print(f"\nEntrenando clasificador base {clf_name.upper()} en {ds_name.upper()}...")
            try:
                modelo, limites, nombres_dim, scaler = entrenar_clasificador(
                    ds_path, ds_path, tipo_modelo=clf_name
                )
            except Exception as e:
                print(f"Error entrenando el modelo base: {e}")
                continue
                
            # Lanzar la búsqueda de hiperparámetros para cada algoritmo
            for alg in algoritmos_a_ejecutar:
                print(f"\nEJECUTANDO BÚSQUEDA: {alg.upper()} | Clasificador: {clf_name.upper()} | Dataset: {ds_name.upper()}")
                
                args_copia = argparse.Namespace(**vars(args))
                args_copia.algoritmo = alg
                
                df_res = ejecutar_grid_search(args_copia, modelo, limites, nombres_dim, scaler, X_train_raw, ds_name, clf_name)
                if not df_res.empty:
                    df_res["algoritmo"] = alg
                    all_results.append(df_res)
                    
    if not all_results:
        print("Error: No se obtuvieron resultados de la búsqueda.")
        return
        
    df_resultados_global = pd.concat(all_results, ignore_index=True)
    
    # Calcular puntuaciones de utilidad de cada hiperparámetro
    df_resultados_global = calcular_puntuacion_seleccion(df_resultados_global)
    
    # Clasificar resultados globales
    df_ordenado = df_resultados_global.sort_values(by="score_seleccion", ascending=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_out = os.path.join(args.output_dir, "estudio_hiperparametros.csv")
    
    df_guardar = df_ordenado.copy()
    df_guardar["historial_fitness"] = df_guardar["historial_fitness"].apply(lambda h: f"[{h[0]:.4f} ... {h[-1]:.4f}]" if isinstance(h, list) else str(h))
    
    # Reordenar campos antes de exportar
    cols = ["algoritmo", "dataset", "clasificador"] + [c for c in df_guardar.columns if c not in ["algoritmo", "dataset", "clasificador"]]
    df_guardar = df_guardar[cols]
    
    df_guardar.to_csv(csv_out, index=False)
    print(f"\nResultados globales exportados a: {csv_out}")
    
    # Generar gráficos comparativos
    print("Generando visualizaciones de comparación...")
    for ds_name in datasets.keys():
        df_ds = df_ordenado[df_ordenado["dataset"] == ds_name]
        if not df_ds.empty:
            generar_grafico_dataset(df_ds, args.output_dir, ds_name)
            
    print("\n====================================================")
    print(f"MEJORES CONFIGURACIONES POR ALGORITMO Y CLASIFICADOR")
    print("====================================================")
    
    for ds_name in datasets.keys():
        print(f"\n=============================")
        print(f"DATASET: {ds_name.upper()}")
        print(f"=============================")
        df_ds = df_ordenado[df_ordenado["dataset"] == ds_name]
        
        for alg in algoritmos_a_ejecutar:
            print(f"\n--- ALGORITMO: {alg.upper()} ---")
            df_alg = df_ds[df_ds["algoritmo"] == alg]
            
            for clf_name in clasificadores:
                df_clf = df_alg[df_alg["clasificador"] == clf_name]
                if not df_clf.empty:
                    row = df_clf.iloc[0]
                    print(f"\n  Ranking #1 para {clf_name.upper()} (Score: {row['score_seleccion']:.4f})")
                    if alg == "ga":
                        print(f"    - Parámetros     : Población={row['tamano_poblacion']}, Gen={row['generaciones']}, Mutación={row['tasa_mutacion']:.2f}")
                    elif alg == "gs":
                        print(f"    - Parámetros     : Muestras={row['muestras']}, AB={row['ancho_banda']:.2f}, Max It={row['max_iters']}")
                    else: # memetico
                        print(f"    - Parámetros     : Pop GA={row['tamano_poblacion']}, Gen GA={row['generaciones']}, Muestras GS={row['muestras']}")
                        
                    print(f"    - Rendimiento    : Éxito={row['tasa_exito']*100:.1f}% | L2 Media={row['dist_l2_media']:.4f} | Tiempo={row['tiempo_ejecucion_medio']:.2f}s")
                    if alg != "gs":
                        print(f"    - Convergencia   : ~{int(row['generaciones_convergencia_media'])} gen/pasos")

if __name__ == "__main__":
    main()
