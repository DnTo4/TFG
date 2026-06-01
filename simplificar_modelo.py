import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.contraejemplos.genetico import entrenar_clasificador, algoritmo_genetico, exportar_resultados
from src.contraejemplos.growing_spheres import growing_spheres_generacion, feature_selection
from src.modelos.arbol_decision import train_model as train_arbol_decision
from src.utils.hiperparametros import obtener_hiperparametros
from src.utils_datos.limpiar_datos import limpiar_y_validar
from src.modelos.autoencoder import cargar_y_preparar_datos, entrenar_autoencoder, filtrar_contraejemplos

"""Simplificación de modelos complejos mediante subrogados globales.

Entrena un clasificador de tipo caja negra, genera contraejemplos representativos en su frontera,
los valida y los incorpora al entrenamiento de un árbol de decisión para aproximar el modelo complejo.
"""

warnings.simplefilter("ignore")

def main():
    """Ejecutar el flujo de simplificación de modelos.

    Carga los datos, entrena el modelo de caja negra y el subrogado base,
    genera y filtra contraejemplos, entrena el subrogado aumentado y
    visualiza los resultados de la comparación en un gráfico.
    """
    parser = argparse.ArgumentParser(description="Simplificación de Modelo Complejo usando un Subrogado Global con Contraejemplos.")
    parser.add_argument("--dataset", type=str, default="datos/originales/diabetes.csv", help="Ruta al dataset original (CSV)")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp"], help="Tipo de modelo complejo (caja negra) a simplificar")
    parser.add_argument("--generaciones", type=str, default="120", help="Número de generaciones para el Algoritmo Genético")
    parser.add_argument("--tamano_poblacion", type=str, default="300", help="Tamaño de la población del Algoritmo Genético")
    args = parser.parse_args()

    # Convertir los parámetros numéricos
    n_generaciones = int(args.generaciones)
    n_poblacion = int(args.tamano_poblacion)

    print("============================")
    print("  MODELO SUBROGADO GLOBAL")
    print("============================")
    print(f"Dataset original: {args.dataset}")
    print(f"Modelo Caja Negra: {args.modelo.upper()}")
    print(f"Configuración Memético: Población={n_poblacion}, Gen={n_generaciones}")
    print("---------------------------------------------------------")

    if not os.path.exists(args.dataset):
        print(f"Error: No se encontró el dataset en {args.dataset}")
        return

    # Cargar y dividir en conjuntos de entrenamiento y prueba (80/20)
    print("Cargando y dividiendo dataset original...")
    df = pd.read_csv(args.dataset)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Crear rutas temporales de entrenamiento y prueba
    temp_train_path = "datos/procesados/temp_train.csv"
    temp_test_path = "datos/procesados/temp_test.csv"
    os.makedirs("datos/procesados", exist_ok=True)
    train_df.to_csv(temp_train_path, index=False)
    test_df.to_csv(temp_test_path, index=False)

    target_col = df.columns[-1]
    X_test_raw = test_df.drop(columns=[target_col])
    y_test_true = test_df[target_col].values

    # Entrenar el clasificador de caja negra
    print(f"Entrenando modelo de caja negra ({args.modelo.upper()})...")
    caja_negra, limites_scaled, nombres_dim, scaler = entrenar_clasificador(
        temp_train_path, temp_test_path, tipo_modelo=args.modelo
    )
    
    # Estimar la precisión de la caja negra en el conjunto de prueba
    y_pred_caja_negra = caja_negra.predict(X_test_raw)
    acc_caja_negra = accuracy_score(y_test_true, y_pred_caja_negra)
    print(f"Precisión Caja Negra ({args.modelo.upper()}): {acc_caja_negra*100:.2f}%")

    # Entrenar un árbol de decisión
    print("Entrenando Árbol de Decisión base...")
    arbol_base, _, acc_arbol_base, _ = train_arbol_decision(temp_train_path, temp_test_path)
    y_pred_arbol_base = arbol_base.predict(X_test_raw)
    
    print(f"Precisión Árbol Base: {acc_arbol_base*100:.2f}%")

    # Ajustar un subrogado global base
    print("Preparando datos para el Subrogado Global Base...")
    X_train_raw = train_df.drop(columns=[target_col])
    y_train_surr = caja_negra.predict(X_train_raw)
    
    train_surr_df = train_df.copy()
    train_surr_df[target_col] = y_train_surr
    
    temp_train_surr_path = "datos/procesados/temp_train_surr.csv"
    train_surr_df.to_csv(temp_train_surr_path, index=False)
    
    arbol_surr_base, _, _, _ = train_arbol_decision(temp_train_surr_path, temp_test_path)
    y_pred_surr_base = arbol_surr_base.predict(X_test_raw)
    acc_surr_base = accuracy_score(y_test_true, y_pred_surr_base)
    print(f"Precisión Subrogado Base: {acc_surr_base*100:.2f}%")

    # Generar contraejemplos
    print("Generando contraejemplos con Algoritmo Memético...")
    print(f"Búsqueda Global AG (Población={n_poblacion}, Gen={n_generaciones})")
    
    # Obtener hiperparámetros
    _, params_gs = obtener_hiperparametros(args.dataset, args.modelo)
    
    pares_ga, _ = algoritmo_genetico(
        modelo=caja_negra,
        limites=limites_scaled,
        nombres_caracteristicas=nombres_dim,
        scaler=scaler,
        tamano_poblacion=n_poblacion,
        generaciones=n_generaciones,
        tasa_mutacion=0.25,
        num_pares=20
    )
    
    if len(pares_ga) == 0:
        print("El Algoritmo Genético no encontró semillas viables.")
        # Eliminar archivos temporales
        for path in [temp_train_path, temp_test_path, temp_train_surr_path]:
            if os.path.exists(path):
                os.remove(path)
        return

    # Iniciar la optimización local
    print("Búsqueda Local GS...")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    predict_fn = caja_negra.predict
    
    # Optimizar localmente cada contraejemplo semilla
    for idx, par in enumerate(pares_ga):
        x_orig = par[:d_dim]
        df_orig = pd.DataFrame([x_orig], columns=nombres_dim)
        
        try:
            ce_gs = growing_spheres_generacion(
                predict_fn=predict_fn,
                x=df_orig,
                muestras=params_gs["muestras"],
                ancho_banda=params_gs["ancho_banda"],
                max_iters=params_gs["max_iters"],
                random_state=42 + idx
            )
            ce_gs = feature_selection(predict_fn, df_orig, ce_gs)
            pares_memeticos.append(np.concatenate([x_orig, ce_gs.flatten()]))
        except RuntimeError:
            pares_memeticos.append(par)
            
    pares = np.array(pares_memeticos)
    print(f"Refinados {len(pares)} contraejemplos con éxito.")

    # Exportar los contraejemplos a un fichero CSV
    ruta_ce_raw = "datos/procesados/contraejemplos_simplificacion_raw.csv"
    exportar_resultados(pares, caja_negra, nombres_dim, archivo_csv=ruta_ce_raw)

    # Guardar la caja negra entrenada
    joblib.dump({"modelo": caja_negra, "nombres": nombres_dim}, "modelos/modelo.joblib")

    # Aplicar restricciones de dominio
    print("Aplicando restricciones de dominio...")
    temp_limpios_path = "datos/procesados/temp_limpios.csv"
    limpiar_y_validar(
        file_path_in=ruta_ce_raw,
        file_path_out=temp_limpios_path,
        model_path="modelos/modelo.joblib"
    )

    # Filtrar contraejemplos inverosímiles
    print("Entrenando Autoencoder sobre datos de entrenamiento...")
    target_col_ae = "Outcome" if "diabetes" in args.dataset.lower() else "class" if "iris" in args.dataset.lower() else target_col
    X_train_raw_ae = cargar_y_preparar_datos(temp_train_path, target_col=target_col_ae)
    autoencoder, scaler_ae, umbral_reconstruccion = entrenar_autoencoder(X_train_raw_ae)
    
    print(f"Filtrando contraejemplos con Autoencoder (Umbral = {umbral_reconstruccion:.4f})...")
    temp_factibles_path = "datos/procesados/temp_factibles.csv"
    filtrar_contraejemplos(
        autoencoder=autoencoder,
        scaler=scaler_ae,
        umbral=umbral_reconstruccion,
        file_path_ce=temp_limpios_path,
        output_path=temp_factibles_path
    )

    # Decidir qué conjunto utilizar
    if os.path.exists(temp_factibles_path):
        df_factibles = pd.read_csv(temp_factibles_path)
        if len(df_factibles) > 0:
            print(f"{len(df_factibles)} contraejemplos LIMPIOS Y VERIFICADOS.")
            ruta_entrenamiento = temp_factibles_path
        else:
            print("Ningún contraejemplo sobrevivió al filtro de realismo.")
            ruta_entrenamiento = ruta_ce_raw
    else:
        ruta_entrenamiento = ruta_ce_raw

    # Entrenar el subrogado aumentado
    print("Entrenando Subrogado Global Aumentado...")
    
    from src.modelos.arbol_decision import parse_df
    
    # Cargar y extraer características
    df_ce = pd.read_csv(ruta_entrenamiento)
    X_ce_parsed, y_ce_parsed = parse_df(df_ce, None)
    
    # Formatear nombres y etiquetas
    y_train_surr_series = pd.Series(y_train_surr).rename(target_col)
    y_ce_parsed = pd.Series(y_ce_parsed).rename(target_col)
    
    # Fusionar instancias originales y sintéticas
    X_combined = pd.concat([X_train_raw, X_ce_parsed], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train_surr_series, y_ce_parsed], axis=0).reset_index(drop=True)
    
    # Generar el dataset aumentado
    df_combined = X_combined.copy()
    df_combined[target_col] = y_combined
    
    temp_combined_path = "datos/procesados/temp_combined_train.csv"
    df_combined.to_csv(temp_combined_path, index=False)
    
    # Entrenar el árbol aumentado
    arbol_surr_aug, _, _, _ = train_arbol_decision(temp_combined_path, temp_test_path)
    y_pred_surr_aug = arbol_surr_aug.predict(X_test_raw)
    acc_surr_aug = accuracy_score(y_test_true, y_pred_surr_aug)
    print(f"Precisión Subrogado Aumentado: {acc_surr_aug*100:.2f}%")
    
    # Guardar el modelo simplificado (árbol de decisión subrogado aumentado)
    joblib.dump({"modelo": arbol_surr_aug, "nombres": nombres_dim}, "modelos/modelo_simplificado.joblib")


    # Calcular la fidelidad de cada modelo
    fid_arbol_base = accuracy_score(y_pred_caja_negra, y_pred_arbol_base)
    fid_surr_base = accuracy_score(y_pred_caja_negra, y_pred_surr_base)
    fid_surr_aug = accuracy_score(y_pred_caja_negra, y_pred_surr_aug)

    # Informe comparativo
    print("\n=============")
    print("RESULTADOS")
    print("===============")
    print(f"Referencia Caja Negra ({args.modelo.upper()}) - Accuracy (vs Real): {acc_caja_negra*100:.2f}%\n")
    
    reporte_data = {
        "Modelo / Clasificador": [
            "Árbol de Decisión Base",
            "Subrogado Global Base",
            "Subrogado Global Aumentado"
        ],
        "Accuracy (vs Real)": [
            f"{acc_arbol_base*100:.2f}%",
            f"{acc_surr_base*100:.2f}%",
            f"{acc_surr_aug*100:.2f}%"
        ],
        "Fidelidad (vs Caja Negra)": [
            f"{fid_arbol_base*100:.2f}%",
            f"{fid_surr_base*100:.2f}%",
            f"{fid_surr_aug*100:.2f}%"
        ]
    }
    df_reporte = pd.DataFrame(reporte_data)
    print(df_reporte.to_string(index=False))
    print("=========================================================================")

    # Crear gráfico
    print("\nGenerando gráfico comparativo...")
    fig, ax = plt.subplots(figsize=(11, 6))
    
    labels = ['Árbol Base', 'Subrogado Base', 'Subrogado Aumentado']
    accuracies = [acc_arbol_base * 100, acc_surr_base * 100, acc_surr_aug * 100]
    fidelities = [fid_arbol_base * 100, fid_surr_base * 100, fid_surr_aug * 100]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (vs Real)', color='#FF7F50', edgecolor='#E0E0E0', linewidth=0.7)
    rects2 = ax.bar(x + width/2, fidelities, width, label='Fidelidad (vs Caja Negra)', color='#4682B4', edgecolor='#E0E0E0', linewidth=0.7)
    
    # Trazar línea de referencia para la precisión de la caja negra
    ax.axhline(y=acc_caja_negra*100, color='#800080', linestyle='--', linewidth=1.5, label=f'Caja Negra ({args.modelo.upper()})')
    
    ax.set_ylabel('Porcentaje (%)', fontweight='bold')
    ax.set_title(f'Comparativa del Modelo Subrogado Global (Dataset: {os.path.basename(args.dataset)})', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', color='#E0E0E0', alpha=0.7)
    ax.legend(frameon=True, edgecolor='#E0E0E0', loc='lower right')
    
    # Incorporar valores sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    # Guardar gráfico comparativo
    os.makedirs("resultados", exist_ok=True)
    grafico_path = "resultados/comparativa_subrogado.png"
    plt.savefig(grafico_path, dpi=150)
    plt.close()
    print(f"Gráfico comparativo guardado en '{grafico_path}'")

    # Limpiar todos los archivos temporales
    print("Limpiando archivos temporales...")
    for path in [temp_train_path, temp_test_path, temp_train_surr_path, ruta_ce_raw, temp_limpios_path, temp_factibles_path, temp_combined_path]:
        if os.path.exists(path):
            os.remove(path)
    print("Finalizado con éxito.\n")

if __name__ == "__main__":
    main()
