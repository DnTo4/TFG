"""
Validación de algoritmos en conjuntos de datos sintéticos
"""
import os
import pandas as pd
import argparse
from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.visualizacion.analisis import plot_contraejemplos
from src.utils.hiperparametros import obtener_hiperparametros

def main(tipo_modelo="mlp", dataset="datos/originales/train_moons.csv"):
    print("=== Validación con Datos Sintéticos ===")
    
    # Rutas
    ruta_train = dataset
    ruta_test = dataset
    ruta_salida = "datos/procesados/contraejemplos_sinteticos.csv"
    
    # Determinar ruta dinámica para el gráfico
    nombre_dataset = os.path.splitext(os.path.basename(dataset))[0]
    ruta_grafico = f"resultados/comparacion_{nombre_dataset}.png"
    
    print(f"[*] Usando dataset: {ruta_train}")
    
    # Entrenar el modelo
    print(f"[*] Entrenando modelo {tipo_modelo.upper()}...")
    modelo, limites, nombres_dim, scaler = entrenar_clasificador(ruta_train, ruta_test, tipo_modelo=tipo_modelo)
    
    # Obtener hiperparámetros óptimos
    params_ga, _ = obtener_hiperparametros(dataset, tipo_modelo)
    
    # Ejecutar algoritmo genético
    print(f"[*] Generando contraejemplos con Algoritmo Genético (Población={params_ga['tamano_poblacion']}, Gen={params_ga['generaciones']}, Mutación={params_ga['tasa_mutacion']})...")
    resultados, historial = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=params_ga["tamano_poblacion"],
        generaciones=params_ga["generaciones"],
        tasa_mutacion=params_ga["tasa_mutacion"],
        num_pares=40
    )
    
    # Exportar CSV
    print(f"[*] Exportando pares de contraejemplos a {ruta_salida}...")
    exportar_resultados(resultados, modelo, nombres_dim, archivo_csv=ruta_salida)
    
    # Graficar
    print("[*] Generando visualización 2D de la frontera de decisión y los contraejemplos...")
    try:
        df_resultados = pd.read_csv(ruta_salida)
        plot_contraejemplos(df_resultados, var_x=nombres_dim[0], var_y=nombres_dim[1], 
                            modelo_entrenado=modelo, nombres_modelo_entrenado=nombres_dim, 
                            ruta_dataset=ruta_train, ruta_guardar=ruta_grafico)
        print("[+] Visualización completada.")
    except Exception as e:
        print(f"[-] Error al generar la visualización: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación en datos sintéticos.")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp", "arbol_decision"], help="Tipo de modelo a utilizar (svm, mlp, arbol_decision)")
    parser.add_argument("--dataset", type=str, default="datos/originales/train_moons.csv", help="Ruta al dataset CSV")
    args = parser.parse_args()
    main(tipo_modelo=args.modelo, dataset=args.dataset)
