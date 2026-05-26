import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

"""Análisis estadístico e interpretación visual de contraejemplos.

Proporciona herramientas para medir la dispersión, magnitudes de cambio,
frecuencia de variación por variable, ranking global de relevancia de características
y graficar los pares.
"""

def analizar_csv(path, model_path="modelos/modelo.joblib", frontera=True, flechas=False, usar_shap=False):
    """Ejecutar el análisis completo sobre los contraejemplos.

    Mide distancias medias, variabilidad de cambios en variables, 
    determina el ranking de importancia y delega la graficación bidimensional.
    """
    df = pd.read_csv(path)

    print("\n==============================")
    print("ANÁLISIS DE CONTRAEJEMPLOS")
    print("==============================")

    n = len(df)
    print(f"Numero de contraejemplos: {n}")

    # Calcular estadísticas de distancia
    if "dist_l2" in df.columns:
        print("\nDistancia al contraejemplo")
        print("---------------------------")
        print(f"  media  : {df['dist_l2'].mean():.4f}")
        print(f"  minima : {df['dist_l2'].min():.4f}")
        print(f"  maxima : {df['dist_l2'].max():.4f}")
        print(f"  mediana: {df['dist_l2'].median():.4f}")

    # Estimar nivel de dispersión de las características variadas
    if "num_features_changed" in df.columns:
        print("\nNumero de variables modificadas")
        print("--------------------------------")
        print(f"  media  : {df['num_features_changed'].mean():.2f}")
        print(f"  mediana: {df['num_features_changed'].median():.0f}")

        conteo = df["num_features_changed"].value_counts().sort_index()
        print("\n  Distribucion:")
        for k, v in conteo.items():
            print(f"  {k} variables -> {v} veces ({100*v/n:.1f}%)")

    # Evaluar la frecuencia de modificación individual por variable
    changed_cols = [c for c in df.columns if c.startswith("changed_")]
    if changed_cols:
        print("\nFrecuencia de cambio por variable")
        print("----------------------------------")
        freq = df[changed_cols].mean()
        freq.index = [c.replace("changed_", "") for c in freq.index]
        ranking_freq = freq.sort_values(ascending=False)
        for var, val in ranking_freq.items():
            print(f"  {var}: {val:.3f}")

    # Medir la magnitud media de las alteraciones
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if delta_cols:
        print("\nMagnitud media del cambio")
        print("--------------------------")
        magnitud = df[delta_cols].abs().mean()
        magnitud.index = [c.replace("delta_", "") for c in magnitud.index]
        ranking_mag = magnitud.sort_values(ascending=False)
        for var, val in ranking_mag.items():
            print(f"  {var}: {val:.4f}")

    # Estimar el ranking de importancia
    nombres, score = None, None
    if changed_cols and delta_cols:
        print("\nRanking global de importancia")
        print("----------------------------------")
        nombres = [c.replace("changed_", "") for c in changed_cols]
        freq    = df[changed_cols].mean().values
        mag     = df[delta_cols].abs().mean().values
        score   = freq * mag
        ranking = np.argsort(score)[::-1]

        for pos, i in enumerate(ranking, 1):
            print(
                f"  {pos:2}. {nombres[i]:<30} "
                f"frecuencia={freq[i]:.3f}  "
                f"cambio_medio={mag[i]:.4f}  "
                f"score={score[i]:.4f}"
            )

    # Graficado
    graficar_contraejemplos(df, nombres, score, model_path=model_path, frontera=frontera, flechas=flechas, usar_shap=usar_shap)

def graficar_contraejemplos(df, nombres=None, score=None, var_x=None, var_y=None, modelo_entrenado=None, nombres_modelo_entrenado=None, ruta_dataset=None, ruta_guardar=None, model_path="modelos/modelo.joblib", frontera=True, flechas=False, usar_shap=False):
    """Graficar los contraejemplos respecto a las variables más relevantes.

    Determina las dos dimensiones prioritarias (SHAP o frecuencia x magnitud),
    proyecta los puntos originales y contraejemplos.
    """
    print("\nPreparando gráfico...")
    modelo = modelo_entrenado
    nombres_modelo = nombres_modelo_entrenado

    # Cargar el modelo
    if modelo is None or nombres_modelo is None:
        try:
            bundle         = joblib.load(model_path)
            modelo         = bundle["modelo"]
            nombres_modelo = bundle["nombres"]
        except FileNotFoundError:
            print(f"Aviso: no se encontró '{model_path}', se omite la frontera y SHAP.")

    var_a, var_b = var_x, var_y

    # Estimar relevancia con SHAP
    if usar_shap and modelo is not None and nombres_modelo is not None:
        try:
            import shap
            X_orig = df[nombres_modelo]
            predict_fn = modelo.predict_proba if hasattr(modelo, "predict_proba") else modelo.predict
            n_samples = min(200, len(X_orig))
            X_sample = shap.sample(X_orig, n_samples)
            
            explainer = shap.Explainer(predict_fn, X_sample)
            shap_values = explainer(X_sample)
            
            vals = shap_values.values
            shap_importance = np.abs(vals).mean(axis=(0, 2)) if len(vals.shape) == 3 else np.abs(vals).mean(axis=0)

            ranking_shap = np.argsort(shap_importance)[::-1]
            var_a, var_b = nombres_modelo[ranking_shap[0]], nombres_modelo[ranking_shap[1]]
            print(f"Variables SHAP: '{var_a}' y '{var_b}'")
        except Exception as e:
            print(f"Error SHAP: {e}")

    # Elegir variables usando score empírico
    if var_a is None or var_b is None:
        if nombres is not None and score is not None:
            print("Usando frecuencia x magnitud para los ejes.")
            ranking = np.argsort(score)[::-1]
            var_a, var_b = nombres[ranking[0]], nombres[ranking[1]]
        else:
            print("No se pudo seleccionar variables para el gráfico.")
            return

    # Extraer variables para los ejes cartesianos
    orig_a, orig_b = df[var_a].values, df[var_b].values
    ce_a, ce_b = df[f"ce_{var_a}"].values, df[f"ce_{var_b}"].values
    clases_all = sorted(df["pred_orig"].unique())
    colores_base = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4"]
    colores_osc  = ["#0D47A1", "#1B5E20", "#E65100", "#4A148C", "#880E4F", "#006064"]

    plt.figure(figsize=(7, 6))

    df_fondo = None
    x_min_val, x_max_val = min(orig_a.min(), ce_a.min()), max(orig_a.max(), ce_a.max())
    y_min_val, y_max_val = min(orig_b.min(), ce_b.min()), max(orig_b.max(), ce_b.max())
    
    # Cargar conjunto original para ajustar los límites
    if ruta_dataset is not None:
        try:
            df_fondo = pd.read_csv(ruta_dataset)
            if var_a in df_fondo.columns and var_b in df_fondo.columns:
                x_min_val = min(x_min_val, df_fondo[var_a].min())
                x_max_val = max(x_max_val, df_fondo[var_a].max())
                y_min_val = min(y_min_val, df_fondo[var_b].min())
                y_max_val = max(y_max_val, df_fondo[var_b].max())
        except Exception as e:
            print(f"Aviso al cargar dataset de fondo para límites: {e}")

    # Estimar y dibujar la frontera de decisión
    if frontera and modelo is not None:
        from matplotlib.colors import ListedColormap
        pad = 0.8
        x_min, x_max = x_min_val - pad, x_max_val + pad
        y_min, y_max = y_min_val - pad, y_max_val + pad

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
        
        # Muestrear el clasificador
        grid_full = np.zeros((xx.size, len(nombres_modelo)))
        for j, col in enumerate(nombres_modelo):
            if col == var_a: grid_full[:, j] = xx.ravel()
            elif col == var_b: grid_full[:, j] = yy.ravel()
            else: grid_full[:, j] = df[col].mean() if col in df.columns else 0.0

        grid_df = pd.DataFrame(grid_full, columns=nombres_modelo)
        Z_labels = modelo.predict(grid_df)
        
        clases_grid = sorted(list(set(clases_all) | set(Z_labels)))
        dict_clases = {cls: i for i, cls in enumerate(clases_grid)}
        Z_idx = np.array([dict_clases[val] for val in Z_labels]).reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_idx, alpha=0.15, cmap=ListedColormap(colores_base[:len(clases_grid)]), 
                     levels=np.arange(len(clases_grid) + 1) - 0.5)
    else:
        clases_grid = clases_all

    # Graficar puntos originales
    if df_fondo is not None:
        try:
            target_col = df_fondo.columns[-1]
            clases_fondo = sorted(df_fondo[target_col].unique())
            for cls in clases_fondo:
                idx_color = clases_grid.index(cls) if cls in clases_grid else list(clases_fondo).index(cls)
                mask = df_fondo[target_col] == cls
                plt.scatter(df_fondo[var_a][mask], df_fondo[var_b][mask],
                            color=colores_base[idx_color % len(colores_base)], s=15, alpha=0.15, zorder=1)
        except Exception as e:
            print(f"No se pudo graficar el dataset de fondo: {e}")

    # Graficar instancias originales
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        plt.scatter(orig_a[df["pred_orig"] == cls], orig_b[df["pred_orig"] == cls], 
                    color=colores_base[idx_color % len(colores_base)], s=40, alpha=0.8, label=f"{cls}", zorder=3)

    # Pintar flechas de transición
    if flechas:
        for xa, ya, xb, yb in zip(orig_a, orig_b, ce_a, ce_b):
            plt.annotate("", xy=(xb, yb), xytext=(xa, ya),
                         arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, alpha=0.5))

    # Graficar contraejemplos finales
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        mask = df["pred_orig"] == cls
        if mask.any():
            plt.scatter(ce_a[mask], ce_b[mask], color=colores_osc[idx_color % len(colores_osc)], 
                        marker="X", s=80, linewidths=0.6, edgecolors="white", label=f"Contraej. ({cls})", zorder=5)

    plt.xlabel(var_a); plt.ylabel(var_b)
    plt.title(f"Originales vs Contraejemplos  —  '{var_a}' y '{var_b}'")
    plt.legend(); plt.tight_layout()
    
    # Exportar el resultado si se indica una ruta de guardado
    if ruta_guardar:
        import os
        directorio = os.path.dirname(ruta_guardar)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        plt.savefig(ruta_guardar, dpi=150)
        print(f"[+] Gráfico guardado en: {ruta_guardar}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analiza y grafica contraejemplos.")
    parser.add_argument("--csv", type=str, default="datos/procesados/contraejemplos_memeticos.csv", help="Ruta al archivo CSV con contraejemplos.")
    parser.add_argument("--modelo", type=str, default="modelos/modelo.joblib", help="Ruta al modelo oráculo.")
    parser.add_argument("--sin-frontera", action="store_true", help="No dibujar la frontera de decisión.")
    parser.add_argument("--flechas", action="store_true", help="Dibujar flechas desde el origen al contraejemplo.")
    parser.add_argument("--usar-shap", action="store_true", help="Priorizar importancia global de variables con SHAP en vez de empírica.")
    args = parser.parse_args()

    analizar_csv(args.csv, model_path=args.modelo, frontera=not args.sin_frontera, flechas=args.flechas, usar_shap=args.usar_shap)
