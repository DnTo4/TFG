"""
Fase 3: Realismo y Factibilidad.
Este orquestador toma los contraejemplos generados por la Fase 2,
aplica restricciones de dominio biológicas/matemáticas (hard constraints)
y finalmente usa un Autoencoder para eliminar aquellos que caen 
fuera de la distribución real de los datos (soft constraints).
"""
import argparse
import os
from src.utils_datos.limpiar_datos import limpiar_y_validar
from src.modelos.autoencoder import load_and_prepare_data, train_autoencoder, filter_counterfactuals

def main(dataset_original, contraejemplos_in, modelo_path):
    print("=== Fase 3: Evaluación de Realismo y Factibilidad ===")
    print(f"[*] Dataset original: {dataset_original}")
    print(f"[*] Evaluando: {contraejemplos_in}")
    
    if not os.path.exists(contraejemplos_in):
        print(f"[-] Error: No se encuentra el archivo {contraejemplos_in}.")
        print("    Asegúrate de haber ejecutado la Fase 2 (comparar_generadores.py) primero.")
        return

    # 1. Filtro Duro (Reglas de Dominio)
    print("\n--- Paso 1: Restricciones de Dominio (Hard Constraints) ---")
    archivo_limpio = contraejemplos_in.replace(".csv", "_limpios.csv")
    limpiar_y_validar(
        file_path_in=contraejemplos_in,
        file_path_out=archivo_limpio,
        model_path=modelo_path
    )
    
    # 2. Filtro Suave (Autoencoder - Variedad de los datos)
    print("\n--- Paso 2: Filtrado por Autoencoder (Distribución) ---")
    print("[*] Entrenando Autoencoder para aprender la distribución original...")
    
    # Preparar datos originales para entrenar el autoencoder
    target_col = "Outcome" if "diabetes" in dataset_original.lower() else "class" if "iris" in dataset_original.lower() else None
    try:
        X_train_raw, _ = load_and_prepare_data(dataset_original, target_col=target_col)
        autoencoder, scaler, umbral_reconstruccion = train_autoencoder(X_train_raw)
        
        # Evaluar los contraejemplos que ya han pasado el filtro duro
        archivo_final_factible = contraejemplos_in.replace(".csv", "_factibles.csv")
        
        print(f"[*] Filtrando contraejemplos limpios con Umbral de Error L2 = {umbral_reconstruccion:.4f}...")
        filter_counterfactuals(
            autoencoder=autoencoder,
            scaler=scaler,
            umbral=umbral_reconstruccion,
            file_path_ce=archivo_limpio,
            output_path=archivo_final_factible
        )
        print("\n[+] Evaluación de Realismo completada con éxito.")
        
    except Exception as e:
        print(f"[-] Ocurrió un error al ejecutar el Autoencoder: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar la Fase 3: Realismo y Factibilidad.")
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset original de entrenamiento")
    parser.add_argument("--input", type=str, default="datos/procesados/contraejemplos_memetico.csv", help="CSV con los contraejemplos a evaluar")
    parser.add_argument("--modelo", type=str, default="modelos/modelo.joblib", help="Ruta al modelo entrenado (joblib)")
    args = parser.parse_args()
    
    main(dataset_original=args.dataset, contraejemplos_in=args.input, modelo_path=args.modelo)
