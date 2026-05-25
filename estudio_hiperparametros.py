"""
Estudio de Hiperparámetros de Algoritmos de Contraejemplos
Optimiza y compara hiperparámetros para GA, GS y Memético en múltiples clasificadores y datasets.
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

# Desactivar advertencias para una salida limpia
import warnings
warnings.simplefilter("ignore")

def parse_arguments():
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
    """
    Calcula cuántas generaciones/iteraciones toma alcanzar el 95% del valor de mejora final.
    """
    if len(historial) <= 1:
        return 1
        
    inicial = historial[0]
    final = historial[-1]
    
    # Si no se encontraron pares válidos, retornar el total
    if final >= 900000:
        return len(historial)
        
    # La mejora total es la reducción en la distancia
    mejora_total = inicial - final
    if mejora_total <= 0:
        return 1
        
    umbral_mejora = inicial - umbral * mejora_total
    
    for gen, val in enumerate(historial):
        if val <= umbral_mejora:
            return gen + 1
            
    return len(historial)

def ejecutar_gs(modelo, X_train_raw, nombres_dim, n_pares, muestras, ancho_banda, max_iters, seed):
    """
    Genera contraejemplos usando Growing Spheres sobre una selección aleatoria del dataset.
    """
    print(f"      Iniciando búsqueda local con Growing Spheres (Muestras={muestras}, AB={ancho_banda:.2f}, Max Iters={max_iters})...")
    rng = np.random.default_rng(seed)
    predict_fn = modelo.predict
    
    # Asegurar que no pedimos más puntos de los disponibles
    max_puntos = min(len(X_train_raw), n_pares)
    idx_seleccionados = rng.permutation(len(X_train_raw))[:max_puntos]
    
    pares = []
    
    for i in idx_seleccionados:
        x0_dict = {col: [X_train_raw[i, j]] for j, col in enumerate(nombres_dim)}
        df_orig = pd.DataFrame(x0_dict)
        
        try:
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=muestras,
                ancho_banda=ancho_banda,
                max_iters=max_iters,
                random_state=int(rng.integers(0, 10000))
            )
            # Refinar contraejemplo (minimizar características cambiadas)
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            
            x0_arr = df_orig.values.flatten()
            ce_arr = ce_gs.flatten()
            
            pares.append(np.concatenate([x0_arr, ce_arr]))
        except RuntimeError:
            continue
            
    return np.array(pares)

def ejecutar_memetico(modelo, limites, nombres_dim, scaler, n_pares, tamano_poblacion, generaciones, tasa_mutacion, muestras, ancho_banda, max_iters, seed):
    """
    Genera contraejemplos usando el Algoritmo Memético (GA + GS).
    Retorna tanto los pares finales refinados como el historial de fitness del GA.
    """
    print(f"      [Memético] Paso 1: Iniciando búsqueda global con Algoritmo Genético (Pop={tamano_poblacion}, Gen={generaciones})...")
    # 1. Búsqueda Global (GA)
    np.random.seed(seed)
    try:
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
        print(f"      [!] Error en fase GA del Memético: {e}")
        return np.array([]), [999999.0]
        
    if len(pares_ga) == 0:
        return np.array([]), historial_ga
        
    d_dim = len(nombres_dim)
    pares_memeticos = []
    predict_fn = modelo.predict
    
    # 2. Búsqueda Local (GS) sobre las semillas de GA
    print(f"      [Memético] Paso 2: Iniciando refinamiento local con Growing Spheres para {len(pares_ga)} pares semilla...")
    for idx, par in enumerate(pares_ga):
        x_orig = par[:d_dim]
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        try:
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=muestras,
                ancho_banda=ancho_banda,
                max_iters=max_iters,
                random_state=seed + idx
            )
            # Refinar con Feature Selection
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except RuntimeError:
            # Si falla la optimización local, se omite.
            continue
            
    return np.array(pares_memeticos), historial_ga

def ejecutar_grid_search(args, modelo, limites, nombres_dim, scaler, X_train_raw, dataset_name, classifier_name):
    num_pares = args.num_pares
    resultados_detallados = []
    
    if args.algoritmo == "ga":
        pop_list = [int(x) for x in args.pop_vals.split(",")]
        gen_list = [int(x) for x in args.gen_vals.split(",")]
        mut_list = [float(x) for x in args.mut_vals.split(",")]
        
        total_configs = len(pop_list) * len(gen_list) * len(mut_list)
        print(f"\n[i] Iniciando Grid Search para GA con {total_configs} combinaciones...")
        print(f"    - Tamaños de población: {pop_list}")
        print(f"    - Generaciones: {gen_list}")
        print(f"    - Tasas de mutación: {mut_list}")
        
        config_idx = 1
        for pop in pop_list:
            for gen in gen_list:
                for mut in mut_list:
                    print(f"\n>>> [{config_idx}/{total_configs}] Config: Población={pop}, Gen={gen}, Mutación={mut:.2f}")
                    
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
                            print(f"      [!] Error en ejecución: {e}")
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
                        
                        print(f"      Rep {rep+1}/{args.repeticiones}: Encontrados={cant_validos}/{num_pares} | L2={distancias_l2[-1]:.4f} | Tiempo={elapsed:.2f}s | Conv={conv} gen")
                    
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
        print(f"\n[i] Iniciando Grid Search para GS con {total_configs} combinaciones...")
        print(f"    - Muestras: {muestras_list}")
        print(f"    - Anchos de banda: {ancho_banda_list}")
        print(f"    - Iteraciones máximas: {max_iters_list}")
        
        config_idx = 1
        for muestras in muestras_list:
            for ab in ancho_banda_list:
                for max_it in max_iters_list:
                    print(f"\n>>> [{config_idx}/{total_configs}] Config: Muestras={muestras}, Ancho Banda={ab:.2f}, Max Iters={max_it}")
                    
                    tiempos = []
                    num_pares_encontrados = []
                    distancias_l2 = []
                    features_cambiadas = []
                    
                    for rep in range(args.repeticiones):
                        t_inicio = time.time()
                        
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
        print(f"\n[i] Iniciando Grid Search para Memético con {total_configs} combinaciones...")
        print(f"    - Tamaños de población GA: {pop_list}")
        print(f"    - Generaciones GA: {gen_list}")
        print(f"    - Muestras GS: {muestras_list}")
        
        # Parámetros fijos razonables para el resto de variables
        tasa_mutacion_ga = 0.20
        ancho_banda_gs = 0.5
        max_iters_gs = 40
        
        config_idx = 1
        for pop in pop_list:
            for gen in gen_list:
                for muestras in muestras_list:
                    print(f"\n>>> [{config_idx}/{total_configs}] Config: Población GA={pop}, Gen GA={gen}, Muestras GS={muestras}")
                    
                    tiempos = []
                    num_pares_encontrados = []
                    distancias_l2 = []
                    features_cambiadas = []
                    convergencia_gens = []
                    historiales = []
                    
                    for rep in range(args.repeticiones):
                        t_inicio = time.time()
                        
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
                            
                        # Ajustar el historial de GA de fondo para que termine en la distancia de GS refinada
                        if cant_validos > 0 and len(historial_ga) > 0:
                            hist_ajustado = list(historial_ga)
                            # Reemplazar valores de penalización con la distancia media si correspondía
                            hist_ajustado = [x if x < 900000 else np.nanmean(dists_rep)*1.5 for x in hist_ajustado]
                            # El valor final de refinamiento local es la distancia real L2
                            hist_ajustado[-1] = np.mean(dists_rep)
                            historiales.append(hist_ajustado)
                        else:
                            historiales.append(historial_ga)
                            
                        conv = calcular_convergencia(historial_ga)
                        convergencia_gens.append(conv)
                        
                        print(f"      Rep {rep+1}/{args.repeticiones}: Encontrados={cant_validos}/{num_pares} | L2={distancias_l2[-1]:.4f} | Tiempo={elapsed:.2f}s")
                    
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
    """
    Calcula un score ponderado de 0 a 1 para seleccionar el mejor hiperparámetro.
    Calculado de manera agrupada por dataset para mayor precisión de escalas.
    """
    df_out = []
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
        
        df_sub["score_seleccion"] = (score_exito * 0.40) + (score_dist * 0.35) + (score_tiempo * 0.15) + (score_feats * 0.10)
        df_out.append(df_sub)
        
    return pd.concat(df_out, ignore_index=True)

def generar_grafico_dataset(df_dataset, output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    
    clasificadores = ["svm", "mlp", "arbol_decision"]
    colors = {"svm": "#2196F3", "mlp": "#4CAF50", "arbol_decision": "#FF9800"}
    
    plt.figure(figsize=(11, 7))
    
    has_plotted = False
    
    for clf in clasificadores:
        df_clf = df_dataset[df_dataset["clasificador"] == clf]
        if df_clf.empty:
            continue
            
        # 1. Obtener la mejor fila de GA para este clasificador
        df_ga = df_clf[df_clf["algoritmo"] == "ga"]
        # 2. Obtener la mejor fila de GS para este clasificador
        df_gs = df_clf[df_clf["algoritmo"] == "gs"]
        
        row_ga = None
        row_gs = None
        
        if not df_ga.empty:
            row_ga = df_ga.sort_values(by="score_seleccion", ascending=False).iloc[0]
        if not df_gs.empty:
            row_gs = df_gs.sort_values(by="score_seleccion", ascending=False).iloc[0]
            
        # Graficar GA (Línea de convergencia)
        if row_ga is not None:
            hist = row_ga["historial_fitness"]
            if isinstance(hist, list):
                hist_filtrado = [x if x < 900000 else np.nan for x in hist]
                lbl_ga = f"{clf.upper()} - GA (Pop={row_ga['tamano_poblacion']}, Gen={row_ga['generaciones']}, Mut={row_ga['tasa_mutacion']:.2f})"
                plt.plot(hist_filtrado, label=lbl_ga, color=colors[clf], linewidth=2.5, linestyle="-")
                has_plotted = True
                
        # Graficar GS (Línea horizontal constante de benchmark)
        if row_gs is not None:
            l2_gs = row_gs["dist_l2_media"]
            if not pd.isna(l2_gs):
                lbl_gs = f"{clf.upper()} - GS (Muestras={row_gs['muestras']}, AB={row_gs['ancho_banda']:.1f}, Éxito={row_gs['tasa_exito']*100:.0f}%, {row_gs['tiempo_ejecucion_medio']:.2f}s)"
                plt.axhline(y=l2_gs, label=lbl_gs, color=colors[clf], linewidth=2.0, linestyle="--")
                has_plotted = True
                
    if not has_plotted:
        print(f"      [!] No se encontraron mejores configuraciones para graficar en {dataset_name}")
        plt.close()
        return
        
    plt.grid(color='#E8E8E8', linestyle='--', linewidth=0.8)
    plt.title(f"Comparativa GA vs GS - {dataset_name.capitalize()}", fontweight='bold', fontsize=14, pad=15)
    plt.xlabel("Generación (para GA)", fontsize=11, labelpad=10)
    plt.ylabel("Distancia L2 Media (Fitness)", fontsize=11, labelpad=10)
    
    # Hacer la leyenda más legible e informativa
    plt.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0', loc='best', fontsize=9.5)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, f"convergencia_{dataset_name}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"[+] Gráfico comparativo guardado en: {file_path}")

def main():
    args = parse_arguments()
    
    print("====================================================")
    print(f"ESTUDIO COMPARATIVO MASIVO DE HIPERPARÁMETROS")
    print("====================================================")
    
    datasets = {
        "iris": "datos/originales/iris.data",
        "diabetes": "datos/originales/diabetes.csv"
    }
    
    clasificadores = ["svm", "mlp", "arbol_decision"]
    
    # Determinar algoritmos a ejecutar
    if args.algoritmo == "ambos":
        algoritmos_a_ejecutar = ["ga", "gs"]
    else:
        algoritmos_a_ejecutar = [args.algoritmo]
        
    print(f"[i] Algoritmos seleccionados: {[alg.upper() for alg in algoritmos_a_ejecutar]}")
    
    all_results = []
    
    for ds_name, ds_path in datasets.items():
        print(f"\n====================================================")
        print(f"PROCESANDO DATASET: {ds_name.upper()} ({ds_path})")
        print(f"====================================================")
        
        # Siempre leer X_train_raw por si se ejecuta GS
        try:
            df_data = pd.read_csv(ds_path)
            X_train_raw = df_data.drop(columns=[df_data.columns[-1]]).values
        except Exception as e:
            print(f"[-] Error leyendo dataset {ds_path}: {e}")
            continue
            
        for clf_name in clasificadores:
            print(f"\n[*] Entrenando clasificador base {clf_name.upper()} en {ds_name.upper()}...")
            try:
                modelo, limites, nombres_dim, scaler = entrenar_clasificador(
                    ds_path, ds_path, tipo_modelo=clf_name
                )
            except Exception as e:
                print(f"[-] Error fatal entrenando el modelo base: {e}")
                continue
                
            for alg in algoritmos_a_ejecutar:
                print(f"\n[+] EJECUTANDO BÚSQUEDA: {alg.upper()} | Clasificador: {clf_name.upper()} | Dataset: {ds_name.upper()}")
                
                # Crear una copia de los argumentos y sobreescribir el algoritmo específico
                args_copia = argparse.Namespace(**vars(args))
                args_copia.algoritmo = alg
                
                df_res = ejecutar_grid_search(args_copia, modelo, limites, nombres_dim, scaler, X_train_raw, ds_name, clf_name)
                if not df_res.empty:
                    df_res["algoritmo"] = alg
                    all_results.append(df_res)
                    
    if not all_results:
        print("[-] Error: No se obtuvieron resultados de la búsqueda.")
        return
        
    df_resultados_global = pd.concat(all_results, ignore_index=True)
    
    # 3. Calcular score ponderado agrupado por dataset
    df_resultados_global = calcular_puntuacion_seleccion(df_resultados_global)
    
    # 4. Determinar mejores configuraciones
    df_ordenado = df_resultados_global.sort_values(by="score_seleccion", ascending=False)
    
    # Crear carpeta de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    csv_out = os.path.join(args.output_dir, "estudio_hiperparametros.csv")
    
    # Guardar dataframe (sin la columna de historial completa para el CSV final por espacio)
    df_guardar = df_ordenado.copy()
    df_guardar["historial_fitness"] = df_guardar["historial_fitness"].apply(lambda h: f"[{h[0]:.4f} ... {h[-1]:.4f}]" if isinstance(h, list) else str(h))
    
    # Ordenar las columnas para que algoritmo quede al principio
    cols = ["algoritmo", "dataset", "clasificador"] + [c for c in df_guardar.columns if c not in ["algoritmo", "dataset", "clasificador"]]
    df_guardar = df_guardar[cols]
    
    df_guardar.to_csv(csv_out, index=False)
    print(f"\n[+] Resultados globales exportados a: {csv_out}")
    
    # 5. Generar y guardar gráficos individuales para cada dataset
    print("[*] Generando visualizaciones avanzadas de comparación...")
    for ds_name in datasets.keys():
        df_ds = df_ordenado[df_ordenado["dataset"] == ds_name]
        if not df_ds.empty:
            generar_grafico_dataset(df_ds, args.output_dir, ds_name)
            
    # 6. Mostrar resumen final de conclusiones
    print("\n====================================================")
    print(f"MEJORES CONFIGURACIONES POR ALGORITMO Y CLASIFICADOR")
    print("====================================================")
    
    for ds_name in datasets.keys():
        print(f"\n===============================")
        print(f">>> DATASET: {ds_name.upper()}")
        print(f"===============================")
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
