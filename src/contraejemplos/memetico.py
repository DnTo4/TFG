import numpy as np
import pandas as pd
import joblib

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection

"""Algoritmo memético para la generación de contraejemplos.

Combina la exploración global del algoritmo genético con la explotación local 
de Growing Spheres.
"""

def main():
    """Ejecutar el flujo completo del Algoritmo Memético.

    Configura argumentos, entrena el clasificador base, ejecuta la exploración
    global, optimiza localmente cada semilla frontera y exporta los resultados.
    """
    import argparse
    import os
    from src.utils.hiperparametros import obtener_hiperparametros

    parser = argparse.ArgumentParser(description="Algoritmo Memético (AG + GS) para contraejemplos.")
    parser.add_argument("--train", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de entrenamiento (default: datos/originales/diabetes.csv)")
    parser.add_argument("--test", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset de prueba (default: datos/originales/diabetes.csv)")
    parser.add_argument("--modelo-path", type=str, default="modelos/modelo.joblib", help="Ruta para guardar el modelo entrenado (default: modelos/modelo.joblib)")
    parser.add_argument("--salida", type=str, default="datos/procesados/contraejemplos_memeticos.csv", help="Ruta para exportar los resultados (default: datos/procesados/contraejemplos_memeticos.csv)")
    parser.add_argument("--modelo", type=str, default="mlp", choices=["svm", "mlp", "arbol_decision"], help="Tipo de clasificador base (default: mlp)")
    parser.add_argument("--num-pares-ga", type=int, default=25, help="Número de zonas prometedoras a extraer para refinamiento (default: 25)")
    parser.add_argument("--no-fs", action="store_true", help="Desactivar feature selection en Growing Spheres")
    
    args = parser.parse_args()

    print("============================")
    print("  ALGORITMO MEMETICO")
    print("============================\n")
    
    # Entrenar el clasificador base
    print("Entrenando modelo...")
    modelo_entrenado, limites_datos, nombres_dim, scaler_genetico = entrenar_clasificador(
        args.train, args.test, args.modelo
    )
    
    # Guardar modelo
    os.makedirs(os.path.dirname(args.modelo_path), exist_ok=True)
    joblib.dump({"modelo": modelo_entrenado, "nombres": nombres_dim}, args.modelo_path)
    print(f"Modelo guardado en '{args.modelo_path}'.\n")
    
    # Obtener hiperparámetros
    params_ga, params_gs = obtener_hiperparametros(args.train, args.modelo)
    
    # Búsqueda global
    print(f"--- Búsqueda Global (Población={params_ga['tamano_poblacion']}, Gen={params_ga['generaciones']}, Mutación={params_ga['tasa_mutacion']}) ---")
    pares_ga, historial = algoritmo_genetico(
        modelo=modelo_entrenado, 
        limites=limites_datos,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler_genetico,
        tamano_poblacion=params_ga["tamano_poblacion"], 
        generaciones=params_ga["generaciones"],
        tasa_mutacion=params_ga["tasa_mutacion"],
        num_pares=args.num_pares_ga
    )
    
    # Validar resultados del algoritmo genético
    if len(pares_ga) == 0:
        print("El AG no encontró ninguna transición/individuo válido en las zonas exploradas.")
        return
        
    print(f"\nExploración completada. Extraídos {len(pares_ga)} individuos semilla.")
    
    # Búsqueda local
    print(f"\n--- Búsqueda Local (Muestras={params_gs['muestras']}, AB={params_gs['ancho_banda']}, Max Iters={params_gs['max_iters']}) ---")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    
    predict_fn = modelo_entrenado.predict
    
    # Optimizar y refinar cada semilla
    for idx, par in enumerate(pares_ga):
        x_orig = par[:d_dim] 
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        print(f" Refinando semilla {idx+1}/{len(pares_ga)}...", end=" ")
        try:
            # Ejecutar Growing Spheres
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=params_gs["muestras"],
                ancho_banda=params_gs["ancho_banda"],
                max_iters=params_gs["max_iters"],
                random_state=42 + idx
            )
            
            # Reducir cambios mediante selección de variables
            if not args.no_fs:
                ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
                
            # Ensamblar y guardar el par
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
            print("Refinado con éxito.")
        except Exception as e:
            # Omitir semilla si falla
            print(f"Fallo ({type(e).__name__}) -> Omitido.")
            continue

    # Exportar resultados
    print("\n============================")
    if len(pares_memeticos) > 0:
        pares_memeticos_np = np.array(pares_memeticos)
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        exportar_resultados(pares_memeticos_np, modelo_entrenado, nombres_dim, args.salida)
        print(f"Resultados guardados en '{args.salida}'.")
    else:
        print("Ningún individuo consiguió ser refinado")

if __name__ == "__main__":
    import warnings
    # Lanzar suprimiendo advertencias externas
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
