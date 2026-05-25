"""
Realismo y Factibilidad.
Aplica restricciones de dominio biológicas/matemáticas
y usa un Autoencoder para eliminar aquellos que caen 
fuera de la distribución real de los datos.
"""
import argparse
import os
import pandas as pd
import joblib
from src.utils_datos.limpiar_datos import limpiar_y_validar
from src.modelos.autoencoder import load_and_prepare_data, train_autoencoder, filter_counterfactuals

def main(dataset_original, contraejemplos_in, modelo_path):
    print("=== Evaluación de Realismo y Factibilidad ===")
    print(f"Dataset original: {dataset_original}")
    print(f"Evaluando: {contraejemplos_in}")
    
    if not os.path.exists(contraejemplos_in):
        print(f"Error: No se encuentra el archivo {contraejemplos_in}.")
        return

    # Autodetección Inteligente del Dataset y Alineación Automática
    try:
        df_temp = pd.read_csv(contraejemplos_in)
        cols_ce = [col for col in df_temp.columns if col.startswith('ce_')]
        cols_orig = [col.replace('ce_', '') for col in cols_ce]
        
        detected_dataset = None
        detected_dataset_path = None
        if 'sepal_length' in cols_orig:
            detected_dataset = 'iris'
            detected_dataset_path = "datos/originales/iris.data"
        elif 'Pregnancies' in cols_orig:
            detected_dataset = 'diabetes'
            detected_dataset_path = "datos/originales/diabetes.csv"
        elif 'x1' in cols_orig and 'x2' in cols_orig:
            detected_dataset = 'moons'
            detected_dataset_path = "datos/originales/train_moons.csv"

        if detected_dataset is not None:
            model_compatible = False
            if os.path.exists(modelo_path):
                try:
                    bundle = joblib.load(modelo_path)
                    modelo_cols = bundle.get("nombres", [])
                    if set(modelo_cols) == set(cols_orig):
                        model_compatible = True
                except:
                    pass
            
            if not model_compatible or dataset_original != detected_dataset_path:
                print(f"\n[Autocorrección] Detectado dataset '{detected_dataset.upper()}' en los contraejemplos.")
                print(f"               Alineando dataset original a: '{detected_dataset_path}'")
                dataset_original = detected_dataset_path
                
                print(f"               Reentrenando clasificador de referencia para '{detected_dataset.upper()}'...")
                try:
                    from src.modelos.mlp import train_model as train_mlp
                    m, _, _, n = train_mlp(detected_dataset_path, detected_dataset_path)
                    joblib.dump({"modelo": m, "nombres": n}, modelo_path)
                    print(f"               [+] Clasificador entrenado y guardado en '{modelo_path}'")
                except Exception as e:
                    print(f"               [!] No se pudo reentrenar el clasificador automáticamente: {e}")
    except Exception as e:
        print(f"[!] Advertencia al leer características de entrada para autodetección: {e}")

    # Reglas de Dominio
    print("\nRestricciones de Dominio")
    archivo_limpio = contraejemplos_in.replace(".csv", "_limpios.csv")
    limpiar_y_validar(
        file_path_in=contraejemplos_in,
        file_path_out=archivo_limpio,
        model_path=modelo_path
    )
    
    # Autoencoder
    print("\nFiltrado por Autoencoder")
    print("Entrenando Autoencoder...")
    
    # Preparar datos originales para entrenar el autoencoder
    target_col = "Outcome" if "diabetes" in dataset_original.lower() else "class" if "iris" in dataset_original.lower() else None
    try:
        X_train_raw = load_and_prepare_data(dataset_original, target_col=target_col)
        autoencoder, scaler, umbral_reconstruccion = train_autoencoder(X_train_raw)
        
        # Evaluar los contraejemplos que ya han pasado el filtro
        archivo_final_factible = contraejemplos_in.replace(".csv", "_factibles.csv")
        
        print(f"Filtrando contraejemplos con Umbral de Error L2 = {umbral_reconstruccion:.4f}...")
        filter_counterfactuals(
            autoencoder=autoencoder,
            scaler=scaler,
            umbral=umbral_reconstruccion,
            file_path_ce=archivo_limpio,
            output_path=archivo_final_factible
        )
        print("\nEvaluación de Realismo completada con éxito.")
        
    except Exception as e:
        print(f"Ocurrió un error al ejecutar el Autoencoder: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realismo y Factibilidad.")
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset original de entrenamiento")
    parser.add_argument("--input", type=str, default="datos/procesados/contraejemplos_memetico.csv", help="CSV con los contraejemplos a evaluar")
    parser.add_argument("--modelo", type=str, default="modelos/modelo.joblib", help="Ruta al modelo entrenado (joblib)")
    args = parser.parse_args()
    
    main(dataset_original=args.dataset, contraejemplos_in=args.input, modelo_path=args.modelo)
