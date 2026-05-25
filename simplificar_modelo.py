"""
Estudio Comparativo y Validación de Modelo Subrogado Global (Árbol de Decisión)
potenciado con Contraejemplos de Caja Negra.
"""
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
from src.modelos.autoencoder import load_and_prepare_data, train_autoencoder, filter_counterfactuals

# Desactivar warnings para una salida limpia
warnings.simplefilter("ignore")

def main():
    parser = argparse.ArgumentParser(description="Simplificación de Modelo Complejo usando un Subrogado Global con Contraejemplos.")
    parser.add_argument("--dataset", type=str, default="datos/originales/train_moons.csv", help="Ruta al dataset original (CSV)")
    parser.add_argument("--modelo", type=str, default="svm", choices=["svm", "mlp"], help="Tipo de modelo complejo (caja negra) a simplificar")
    parser.add_argument("--generaciones", type=str, default="120", help="Número de generaciones para el Algoritmo Genético")
    parser.add_argument("--tamano_poblacion", type=str, default="300", help="Tamaño de la población del Algoritmo Genético")
    args = parser.parse_args()

    # Convertir a enteros los parámetros numéricos (por seguridad del parser)
    n_generaciones = int(args.generaciones)
    n_poblacion = int(args.tamano_poblacion)

    print("============================")
    print("  MODELO SUBROGADO GLOBAL")
    print("============================")
    print(f"[*] Dataset original: {args.dataset}")
    print(f"[*] Modelo complejo (Caja Negra): {args.modelo.upper()}")
    print(f"[*] Configuración Memético: Población={n_poblacion}, Gen={n_generaciones}")
    print("---------------------------------------------------------")

    if not os.path.exists(args.dataset):
        print(f"[!] Error: No se encontró el dataset en {args.dataset}")
        return

    # 1. Cargar y dividir en Train y Test (80% / 20%)
    print("[1] Cargando y dividiendo dataset original...")
    df = pd.read_csv(args.dataset)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Rutas temporales de entrenamiento y prueba
    temp_train_path = "datos/procesados/temp_train.csv"
    temp_test_path = "datos/procesados/temp_test.csv"
    os.makedirs("datos/procesados", exist_ok=True)
    train_df.to_csv(temp_train_path, index=False)
    test_df.to_csv(temp_test_path, index=False)

    target_col = df.columns[-1]
    X_test_raw = test_df.drop(columns=[target_col])
    y_test_true = test_df[target_col].values

    # 2. Entrenar el Modelo Complejo (Caja Negra)
    print(f"[2] Entrenando modelo de caja negra ({args.modelo.upper()})...")
    caja_negra, limites_scaled, nombres_dim, scaler = entrenar_clasificador(
        temp_train_path, temp_test_path, tipo_modelo=args.modelo
    )
    
    # Evaluar precisión de Caja Negra en Test
    y_pred_caja_negra = caja_negra.predict(X_test_raw)
    acc_caja_negra = accuracy_score(y_test_true, y_pred_caja_negra)
    print(f"    -> Precisión Caja Negra ({args.modelo.upper()}): {acc_caja_negra*100:.2f}%")

    # 3. Entrenar el Árbol de Decisión de Línea Base (Datos y Etiquetas Reales)
    print("[3] Entrenando Árbol de Decisión de línea base (con datos reales originales)...")
    arbol_base, _, acc_arbol_base, _ = train_arbol_decision(temp_train_path, temp_test_path)
    y_pred_arbol_base = arbol_base.predict(X_test_raw)
    
    print(f"    -> Precisión Árbol Base: {acc_arbol_base*100:.2f}%")

    # 4. Entrenar el Subrogado Global Base (Sin Contraejemplos)
    # Reemplazamos la columna objetivo con las predicciones de la caja negra
    print("[4] Preparando datos para el Subrogado Global Base (predicciones de caja negra)...")
    X_train_raw = train_df.drop(columns=[target_col])
    y_train_surr = caja_negra.predict(X_train_raw)
    
    train_surr_df = train_df.copy()
    train_surr_df[target_col] = y_train_surr
    
    temp_train_surr_path = "datos/procesados/temp_train_surr.csv"
    train_surr_df.to_csv(temp_train_surr_path, index=False)
    
    arbol_surr_base, _, _, _ = train_arbol_decision(temp_train_surr_path, temp_test_path)
    y_pred_surr_base = arbol_surr_base.predict(X_test_raw)
    acc_surr_base = accuracy_score(y_test_true, y_pred_surr_base)
    print(f"    -> Precisión Subrogado Base: {acc_surr_base*100:.2f}%")

    # 5. Generar Contraejemplos con Algoritmo Memético (Búsqueda Global AG + Búsqueda Local GS)
    print(f"[5] Generando contraejemplos con Algoritmo Memético...")
    print(f"    [+] Búsqueda Global AG (Población={n_poblacion}, Gen={n_generaciones})...")
    
    # Obtener hiperparámetros óptimos de GS
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
        print("[!] El Algoritmo Genético no encontró semillas viables.")
        # Limpieza
        for path in [temp_train_path, temp_test_path, temp_train_surr_path]:
            if os.path.exists(path):
                os.remove(path)
        return

    print("    [+] Búsqueda Local GS (Refinamiento en la frontera)...")
    d_dim = len(nombres_dim)
    pares_memeticos = []
    predict_fn = caja_negra.predict
    
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
            # Fallback al par genético original
            pares_memeticos.append(par)
            
    pares = np.array(pares_memeticos)
    print(f"    [+] Refinados {len(pares)} contraejemplos frontera meméticos con éxito.")

    # Exportar contraejemplos generados (el archivo contendrá las etiquetas del original)
    ruta_ce_raw = "datos/procesados/contraejemplos_simplificacion_raw.csv"
    exportar_resultados(pares, caja_negra, nombres_dim, archivo_csv=ruta_ce_raw)

    # Guardamos la caja negra en modelo.joblib para que los filtros la lean
    joblib.dump({"modelo": caja_negra, "nombres": nombres_dim}, "modelos/modelo.joblib")

    # Aplicar Restricciones de Dominio (Limpieza)
    print("    [+] Aplicando restricciones de dominio y reglas biológicas/matemáticas...")
    temp_limpios_path = "datos/procesados/temp_limpios.csv"
    limpiar_y_validar(
        file_path_in=ruta_ce_raw,
        file_path_out=temp_limpios_path,
        model_path="modelos/modelo.joblib"
    )

    # Filtrado por Autoencoder (Realismo)
    print("    [+] Entrenando Autoencoder sobre datos de entrenamiento reales...")
    target_col_ae = "Outcome" if "diabetes" in args.dataset.lower() else "class" if "iris" in args.dataset.lower() else target_col
    X_train_raw_ae = load_and_prepare_data(temp_train_path, target_col=target_col_ae)
    autoencoder, scaler_ae, umbral_reconstruccion = train_autoencoder(X_train_raw_ae)
    
    print(f"    [+] Filtrando contraejemplos con Autoencoder (Umbral MSE = {umbral_reconstruccion:.4f})...")
    temp_factibles_path = "datos/procesados/temp_factibles.csv"
    filter_counterfactuals(
        autoencoder=autoencoder,
        scaler=scaler_ae,
        umbral=umbral_reconstruccion,
        file_path_ce=temp_limpios_path,
        output_path=temp_factibles_path
    )

    # Determinar ruta final de entrenamiento para el subrogado
    if os.path.exists(temp_factibles_path):
        df_factibles = pd.read_csv(temp_factibles_path)
        if len(df_factibles) > 0:
            print(f"    [+] {len(df_factibles)} contraejemplos LIMPIOS Y VERIFICADOS sobrevivieron al filtro.")
            ruta_entrenamiento = temp_factibles_path
        else:
            print("    [!] Advertencia: Ningún contraejemplo sobrevivió al filtro de realismo. Usando crudos.")
            ruta_entrenamiento = ruta_ce_raw
    else:
        ruta_entrenamiento = ruta_ce_raw

    # 6. Entrenar el Subrogado Global Aumentado (Con Contraejemplos Limpios y Verificados)
    print("[6] Entrenando Subrogado Global Aumentado con Contraejemplos verificados...")
    
    from src.modelos.arbol_decision import parse_df
    
    # Cargar y parsear contraejemplos
    df_ce = pd.read_csv(ruta_entrenamiento)
    X_ce_parsed, y_ce_parsed = parse_df(df_ce, None)
    
    # Estandarizar nombres para concatenar de forma segura
    y_train_surr_series = pd.Series(y_train_surr).rename(target_col)
    y_ce_parsed = pd.Series(y_ce_parsed).rename(target_col)
    
    # Combinar original training set con los contraejemplos frontera
    X_combined = pd.concat([X_train_raw, X_ce_parsed], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train_surr_series, y_ce_parsed], axis=0).reset_index(drop=True)
    
    # Crear un dataframe estandarizado unificado
    df_combined = X_combined.copy()
    df_combined[target_col] = y_combined
    
    temp_combined_path = "datos/procesados/temp_combined_train.csv"
    df_combined.to_csv(temp_combined_path, index=False)
    
    # Entrenar sobre el dataset verdaderamente AUMENTADO
    arbol_surr_aug, _, _, _ = train_arbol_decision(temp_combined_path, temp_test_path)
    y_pred_surr_aug = arbol_surr_aug.predict(X_test_raw)
    acc_surr_aug = accuracy_score(y_test_true, y_pred_surr_aug)
    print(f"    -> Precisión Subrogado Aumentado: {acc_surr_aug*100:.2f}%")

    # 7. Imprimir Reporte Final Comparativo
    print("\n=========================================================")
    print("                RESULTADOS DEL EXPERIMENTO")
    print("=========================================================")
    reporte_data = {
        "Modelo / Clasificador": [
            f"Caja Negra ({args.modelo.upper()})",
            "Árbol de Decisión Línea Base (Real)",
            "Subrogado Global Base (Árbol de Decisión)",
            "Subrogado Global Aumentado (+Contraejemplos)"
        ],
        "Precisión (Accuracy)": [
            f"{acc_caja_negra*100:.2f}%",
            f"{acc_arbol_base*100:.2f}%",
            f"{acc_surr_base*100:.2f}%",
            f"{acc_surr_aug*100:.2f}%"
        ]
    }
    df_reporte = pd.DataFrame(reporte_data)
    print(df_reporte.to_string(index=False))
    print("=========================================================")

    # Analizar si la simplificación es viable
    ganancia_acc = (acc_surr_aug - acc_surr_base) * 100
    print("\nAnálisis de Viabilidad:")
    if acc_surr_aug > acc_surr_base:
        print(f"   -> ¡VIABILIDAD CONFIRMADA! La adición de contraejemplos frontera incrementó")
        print(f"      la Precisión del modelo subrogado en un {ganancia_acc:.2f}%.")
        print(f"      El Árbol de Decisión ahora modela con mayor exactitud la frontera de decisión compleja.")
    else:
        print(f"   -> La adición de contraejemplos no mostró un incremento de precisión en este conjunto de prueba.")
        print(f"      Pruebe aumentando las generaciones o con un modelo de frontera más marcado.")

    # 8. Visualización y Gráfico de Barras Estético
    print("\n[8] Generando gráfico comparativo...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Datos para el gráfico
    labels = ['Árbol Base', 'Subrogado Base', 'Subrogado Aumentado']
    accuracies = [acc_arbol_base * 100, acc_surr_base * 100, acc_surr_aug * 100]
    
    x = np.arange(len(labels))
    width = 0.5
    
    rects1 = ax.bar(x, accuracies, width, label='Accuracy', color='#FF7F50', edgecolor='#E0E0E0', linewidth=0.7)
    
    # Línea horizontal para marcar el rendimiento de la Caja Negra
    ax.axhline(y=acc_caja_negra*100, color='#800080', linestyle='--', linewidth=1.5, label=f'Caja Negra ({args.modelo.upper()})')
    
    # Estilizado moderno
    ax.set_ylabel('Porcentaje (%)', fontweight='bold')
    ax.set_title(f'Comparativa del Modelo Subrogado Global (Dataset: {os.path.basename(args.dataset)})', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', color='#E0E0E0', alpha=0.7)
    ax.legend(frameon=True, edgecolor='#E0E0E0', loc='lower right')
    
    # Añadir valores sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    autolabel(rects1)
    
    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    grafico_path = "resultados/comparativa_subrogado.png"
    plt.savefig(grafico_path, dpi=150)
    plt.close()
    print(f"[+] Gráfico comparativo guardado en '{grafico_path}'")

    # Limpiar archivos temporales para mantener limpio el entorno
    print("[*] Limpiando archivos temporales...")
    for path in [temp_train_path, temp_test_path, temp_train_surr_path, ruta_ce_raw, temp_limpios_path, temp_factibles_path, temp_combined_path]:
        if os.path.exists(path):
            os.remove(path)
    print("[+] Finalizado con éxito.\n")

if __name__ == "__main__":
    main()
