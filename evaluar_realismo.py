import argparse
import os
import pandas as pd
import joblib
from src.utils_datos.limpiar_datos import limpiar_y_validar
from src.modelos.autoencoder import cargar_y_preparar_datos, entrenar_autoencoder, filtrar_contraejemplos

"""Evaluar y filtrar el realismo de contraejemplos generados.

Combina la verificación de restricciones de dominio y la evaluación
del error de reconstrucción mediante autoencoders para descartar contraejemplos inverosímiles.
"""

def main(dataset_original, contraejemplos_in, modelo_path):
    """Ejecutar el flujo completo de evaluación de realismo.

    Ejecuta filtros de dominio y utiliza un autoencoder para 
    evaluar si los contraejemplos siguen la distribución de los datos.
    """
    print("=== Evaluación de Realismo ===")
    print(f"Dataset original: {dataset_original}")
    print(f"Evaluando: {contraejemplos_in}")
    
    # Comprobar la existencia del archivo de contraejemplos
    if not os.path.exists(contraejemplos_in):
        print(f"Error: No se encuentra el archivo {contraejemplos_in}.")
        return

    # Validar compatibilidad de conjuntos de datos
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
            
            # Reentrenar clasificador si hay discrepancia de características
            if not model_compatible or dataset_original != detected_dataset_path:
                print(f"\nDetectado dataset '{detected_dataset.upper()}' en los contraejemplos.")
                print(f"Alineando dataset original a: '{detected_dataset_path}'")
                dataset_original = detected_dataset_path
                
                print(f"Reentrenando clasificador para '{detected_dataset.upper()}'...")
                try:
                    from src.modelos.mlp import train_model as train_mlp
                    m, _, _, n = train_mlp(detected_dataset_path, detected_dataset_path)
                    joblib.dump({"modelo": m, "nombres": n}, modelo_path)
                    print(f"Clasificador entrenado y guardado en '{modelo_path}'")
                except Exception as e:
                    print(f"No se pudo reentrenar el clasificador automáticamente: {e}")
    except Exception as e:
        print(e)

    # Aplicar restricciones de dominio
    print("\nRestricciones de Dominio")
    archivo_limpio = contraejemplos_in.replace(".csv", "_limpios.csv")
    limpiar_y_validar(
        file_path_in=contraejemplos_in,
        file_path_out=archivo_limpio,
        model_path=modelo_path
    )
    
    # Iniciar validación mediante autoencoder
    print("\nFiltrado por Autoencoder")
    print("Entrenando Autoencoder...")
    
    # Cargar y preparar datos originales para ajustar el autoencoder
    target_col = "Outcome" if "diabetes" in dataset_original.lower() else "class" if "iris" in dataset_original.lower() else None
    try:
        X_train_raw = cargar_y_preparar_datos(dataset_original, target_col=target_col)
        autoencoder, scaler, umbral_reconstruccion = entrenar_autoencoder(X_train_raw)
        
        # Filtrar contraejemplos utilizando el error de reconstrucción
        archivo_final_factible = contraejemplos_in.replace(".csv", "_factibles.csv")
        
        print(f"Filtrando contraejemplos con Umbral de Error L2 = {umbral_reconstruccion:.4f}...")
        filtrar_contraejemplos(
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
