import numpy as np
import pandas as pd
import joblib

from genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from growing_spheres import growing_spheres_generacion, feature_selection

# ---- Parámetros Globales ----
RUTA_DATASET_TRAIN = "diabetes.csv"
RUTA_DATASET_TEST  = "diabetes.csv"
RUTA_MODELO  = "modelo.joblib"
RUTA_SALIDA  = "contraejemplos_memeticos.csv"

TIPO_MODELO = "mlp" # "svm", "mlp", "perceptron"

# ---- Parámetros Exploración GA (Global) ----
TAMANO_POBLACION_GA = 100
GENERACIONES_GA     = 60
TASA_MUTACION_GA    = 0.20
NUM_PARES_GA        = 25 # Número de zonas prometedoras (individuos) a extraer para refinar

# ---- Parámetros Explotación GS (Local) ----
MUESTRAS_GS = 500
ANCHO_BANDA_GS = 0.5
MAX_ITERS_GS = 50
USAR_FS = True # Aplicar feature selection tras growing spheres

def main():
    print("=====================================================")
    print("  ALGORITMO MEMÉTICO: GA (Global) + GS (Local)  ")
    print("=====================================================\n")
    
    # 1. Entrenamiento / Preparación del modelo oráculo
    print("Entrenando modelo clasificador base...")
    modelo_entrenado, limites_datos, nombres_dim, scaler_genetico = entrenar_clasificador(
        RUTA_DATASET_TRAIN, RUTA_DATASET_TEST, TIPO_MODELO
    )
    
    # Guardamos el modelo para analisis.py
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, RUTA_MODELO)
    print(f"Modelo complejo base preparado en '{RUTA_MODELO}'.\n")
    
    # 2. Búsqueda Global (Algoritmo Genético)
    print("--- Búsqueda Global (Explorando fronteras con GA) ---")
    pares_ga, historial = algoritmo_genetico(
        modelo=modelo_entrenado, 
        limites=limites_datos,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler_genetico,
        tamano_poblacion=TAMANO_POBLACION_GA, 
        generaciones=GENERACIONES_GA,
        tasa_mutacion=TASA_MUTACION_GA,
        num_pares=NUM_PARES_GA
    )
    
    if len(pares_ga) == 0:
        print("El GA no encontró ninguna transición/individuo válido en las zonas exploradas.")
        return
        
    print(f"\nExploracion completada. Extraidos {len(pares_ga)} individuos semilla.")
    
    # 3. Búsqueda Local Fina (Growing Spheres)
    print("\n--- Búsqueda Local (Minimizando distancia con GS) ---")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    
    # Función de predicción wrapper para pasar el DataFrame con nombres a Growing Spheres
    predict_fn = modelo_entrenado.predict
    
    for idx, par in enumerate(pares_ga):
        # El individuo GA está compuesto por [x_original, x_contrafactual]
        # Extraemos SOLO el original para dárselo a GS como punto de inicio puro
        x_orig = par[:d_dim] 
        
        # Growing spheres prefiere DataFrames si el modelo se entrenó con nombres de columnas
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        print(f" Refinando semilla {idx+1}/{len(pares_ga)}...\n", end=" ")
        try:
            # 3.1. Growing Spheres puro (busqueda hiper-esferal cercana a la semilla)
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=MUESTRAS_GS,
                ancho_banda=ANCHO_BANDA_GS,
                max_iters=MAX_ITERS_GS,
                random_state=42 + idx
            )
            
            # 3.2. Feature Selection (hacer que el vector de cambio modifique pocas variables)
            if USAR_FS:
                ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
                
            # Ensamblar el nuevo par optimizado [original, contraejemplo_gs]
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except Exception as e:
            # RuntimeError se lanza si el growing spheres no encuentra transiciones
            print(f"Fallo ({type(e).__name__}) -> Omitido.")
            continue

    # 4. Exportación
    print("\n=====================================================")
    if len(pares_memeticos) > 0:
        pares_memeticos_np = np.array(pares_memeticos)
        exportar_resultados(pares_memeticos_np, modelo_entrenado, nombres_dim, RUTA_SALIDA)
        print(f"Resultados guardados en '{RUTA_SALIDA}'.")
    else:
        print("Ningún individuo consiguió ser refinado por el operador local.")

if __name__ == "__main__":
    import warnings
    # Reducimos los warnings durante la ejecución para limpiar la salida
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
