"""
Fase 1: Validación de algoritmos en conjuntos de datos sintéticos (2D).
Este script demuestra que el algoritmo genético y los modelos
funcionan correctamente cruzando fronteras de decisión visualizables.
"""
import pandas as pd
import argparse
from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.visualizacion.analisis import plot_contraejemplos

def main(tipo_modelo="svm", dataset="datos/originales/train_moons.csv"):
    print("=== Fase 1: Validación con Datos Sintéticos ===")
    
    # 1. Definir rutas
    ruta_train = dataset
    ruta_test = dataset
    ruta_salida = "datos/procesados/contraejemplos_sinteticos.csv"
    
    print(f"[*] Usando dataset: {ruta_train}")
    
    # 2. Entrenar el modelo
    print(f"[*] Entrenando modelo {tipo_modelo.upper()}...")
    modelo, limites, nombres_dim, scaler = entrenar_clasificador(ruta_train, ruta_test, tipo_modelo=tipo_modelo)
    
    # 3. Ejecutar algoritmo genético
    print("[*] Generando contraejemplos con Algoritmo Genético...")
    # Para datasets sintéticos pequeños (2D), no necesitamos poblaciones enormes
    resultados, historial = algoritmo_genetico(
        modelo=modelo,
        limites=limites,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=150,
        generaciones=50,
        tasa_mutacion=0.1,
        num_pares=20
    )
    
    # 4. Exportar los resultados a CSV
    print(f"[*] Exportando pares de contraejemplos a {ruta_salida}...")
    exportar_resultados(resultados, modelo, nombres_dim, archivo_csv=ruta_salida)
    
    # 5. Visualizar gráficamente
    print("[*] Generando visualización 2D de la frontera de decisión y los contraejemplos...")
    try:
        df_resultados = pd.read_csv(ruta_salida)
        # Como estamos en 2D, pasamos directamente las dos dimensiones para graficar
        plot_contraejemplos(df_resultados, var_x=nombres_dim[0], var_y=nombres_dim[1])
        print("[+] Visualización completada.")
    except Exception as e:
        print(f"[-] Error al generar la visualización: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar la Fase 1 de validación en datos sintéticos.")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp", "perceptron"], help="Tipo de modelo a utilizar (svm, mlp, perceptron)")
    parser.add_argument("--dataset", type=str, default="datos/originales/train_moons.csv", help="Ruta al dataset CSV")
    args = parser.parse_args()
    main(tipo_modelo=args.modelo, dataset=args.dataset)
