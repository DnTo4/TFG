import numpy as np
import pandas as pd
import joblib

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

# ---- PARÁMETROS DE CONFIGURACIÓN Y RUTAS ----
RUTA_DATASET_TRAIN = "datos/originales/diabetes.csv"
RUTA_DATASET_TEST  = "datos/originales/diabetes.csv"
RUTA_MODELO  = "modelos/modelo.joblib"
RUTA_SALIDA  = "datos/procesados/contraejemplos_memeticos.csv"

TIPO_MODELO = "mlp" # Opciones: "svm", "mlp", "perceptron"

# ---- HIPERPARÁMETROS AG ----
TAMANO_POBLACION_GA = 100
GENERACIONES_GA      = 60
TASA_MUTACION_GA    = 0.20
NUM_PARES_GA        = 25 # Número de zonas prometedoras a extraer para refinamiento local

# ---- HIPERPARÁMETROS GS ----
MUESTRAS_GS = 500
ANCHO_BANDA_GS = 0.5
MAX_ITERS_GS = 50
USAR_FS = True # Aplicar feature selection

def main():
    """
    Función principal que coordina el Algoritmo Memético.
    
    Flujo de trabajo:
    1. Entrenar/Cargar el modelo.
    2. Ejecutar búsqueda global (AG) para identificar áreas con transiciones de clase.
    3. Para cada resultado del AG, ejecutar búsqueda local (GS) para ajustar el contraejemplo.
    4. Aplicar feature selection para simplificar la explicación del contraejemplo.
    5. Exportar los pares a un archivo CSV.
    """
    print("=====================================================")
    print("  ALGORITMO MEMETICO: AG + GS  ")
    print("=====================================================\n")
    
    # Entrenamientodel modelo
    print("Entrenando modelo...")
    modelo_entrenado, limites_datos, nombres_dim, scaler_genetico = entrenar_clasificador(
        RUTA_DATASET_TRAIN, RUTA_DATASET_TEST, TIPO_MODELO
    )
    
    # Persistencia del modelo para análisis posteriores
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, RUTA_MODELO)
    print(f"Modelo guardado en '{RUTA_MODELO}'.\n")
    
    # Búsqueda global (AG)
    print("--- Busqueda Global ---")
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
    
    # Verificación de resultados del AG
    if len(pares_ga) == 0:
        print("El AG no encontro ninguna transicion/individuo valido en las zonas exploradas.")
        return
        
    print(f"\nExploracion completada. Extraidos {len(pares_ga)} individuos semilla.")
    
    # 3. Búsqueda Local
    print("\n--- Busqueda Local (Minimizando distancia con GS) ---")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    
    # Función de predicción del modelo
    predict_fn = modelo_entrenado.predict
    
    for idx, par in enumerate(pares_ga):
        # Tomamos solo el punto original como centro para la búsqueda local de GS.
        x_orig = par[:d_dim] 
        
        # Convertimos a DataFrame para mantener compatibilidad con nombres de columnas
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        print(f" Refinando semilla {idx+1}/{len(pares_ga)}...\n", end=" ")
        try:
            # Growing Spheres
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=MUESTRAS_GS,
                ancho_banda=ANCHO_BANDA_GS,
                max_iters=MAX_ITERS_GS,
                random_state=42 + idx
            )
            
            # Feature selection
            if USAR_FS:
                ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
                
            # Ensamblar el nuevo par optimizado
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except Exception as e:
            # Captura fallos si GS no encuentra una transición en el número de iteraciones dado
            print(f"Fallo ({type(e).__name__}) -> Omitido.")
            continue

    # Exportación de Resultados
    print("\n=====================================================")
    if len(pares_memeticos) > 0:
        pares_memeticos_np = np.array(pares_memeticos)
        # Se reutiliza la función de exportación del módulo genético
        exportar_resultados(pares_memeticos_np, modelo_entrenado, nombres_dim, RUTA_SALIDA)
        print(f"Resultados guardados en '{RUTA_SALIDA}'.")
    else:
        print("Ningún individuo consiguio ser refinado")

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
