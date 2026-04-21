import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

CSV_PATH   = "contraejemplos_memeticos.csv"
MODEL_PATH = "modelo.joblib" # Ruta donde se guarda el modelo

FRONTERA = True    # visualizar frontera de decision (requiere modelo)
FLECHAS  = False    # visualizar flechas origen -> contraejemplo
USAR_SHAP = False   # True = Priorizar importancia global del modelo, False = Priorizar los ejes donde hubo más movimiento empírico
# ----------------------------


def analizar_csv(path):

    df = pd.read_csv(path)

    print("\n==============================")
    print("ANÁLISIS DE CONTRAEJEMPLOS")
    print("==============================\n")

    n = len(df)
    print(f"Numero de contraejemplos: {n}")

    # Distancias
    if "dist_l2" in df.columns:

        print("\nDistancia al contraejemplo")
        print("---------------------------")

        print(f"  media  : {df['dist_l2'].mean():.4f}")
        print(f"  minima : {df['dist_l2'].min():.4f}")
        print(f"  maxima : {df['dist_l2'].max():.4f}")
        print(f"  mediana: {df['dist_l2'].median():.4f}")

    # Variables modificadas
    if "num_features_changed" in df.columns:

        print("\nNumero de variables modificadas")
        print("--------------------------------")

        print(f"  media  : {df['num_features_changed'].mean():.2f}")
        print(f"  mediana: {df['num_features_changed'].median():.0f}")

        conteo = df["num_features_changed"].value_counts().sort_index()

        print("\n  Distribucion:")
        for k, v in conteo.items():
            print(f"  {k} variables -> {v} veces ({100*v/n:.1f}%)")

    # Frecuencia de cambio
    changed_cols = [c for c in df.columns if c.startswith("changed_")]

    if changed_cols:

        print("\nFrecuencia de cambio por variable")
        print("----------------------------------")

        freq = df[changed_cols].mean()
        freq.index = [c.replace("changed_", "") for c in freq.index]
        ranking_freq = freq.sort_values(ascending=False)

        for var, val in ranking_freq.items():
            print(f"  {var}: {val:.3f}")

    # Magnitud del cambio
    delta_cols = [c for c in df.columns if c.startswith("delta_")]

    if delta_cols:

        print("\nMagnitud media del cambio")
        print("--------------------------")

        magnitud = df[delta_cols].abs().mean()
        magnitud.index = [c.replace("delta_", "") for c in magnitud.index]
        ranking_mag = magnitud.sort_values(ascending=False)

        for var, val in ranking_mag.items():
            print(f"  {var}: {val:.4f}")

    # Ranking global
    nombres, score = None, None

    if changed_cols and delta_cols:

        print("\nRanking global de importancia (frecuencia x magnitud)")
        print("------------------------------------------------------")

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

    # Visualización
    plot_contraejemplos(df, nombres, score)


def plot_contraejemplos(df, nombres=None, score=None):

    print("\nPreparando gráfico...")
    modelo = None
    nombres_modelo = None

    # Intentar cargar el modelo (necesario para SHAP y frontera)
    try:
        bundle         = joblib.load(MODEL_PATH)
        modelo         = bundle["modelo"]
        nombres_modelo = bundle["nombres"]
    except FileNotFoundError:
        print(f"Aviso: no se encontró '{MODEL_PATH}', se omite la frontera y SHAP.")

    var_a, var_b = None, None

    # Intentar primero con SHAP
    if USAR_SHAP and modelo is not None and nombres_modelo is not None:
        try:
            import shap
            X_orig = df[nombres_modelo]
            
            # Pasar función de predicción
            predict_fn = modelo.predict_proba if hasattr(modelo, "predict_proba") else modelo.predict
            
            # Tomar una muestra
            n_samples = min(200, len(X_orig))
            X_sample = shap.sample(X_orig, n_samples)
            
            explainer = shap.Explainer(predict_fn, X_sample)
            shap_values = explainer(X_sample)
            
            vals = shap_values.values
            if len(vals.shape) == 3: # Multiclase
                shap_importance = np.abs(vals).mean(axis=(0, 2))
            else:
                shap_importance = np.abs(vals).mean(axis=0)

            ranking_shap = np.argsort(shap_importance)[::-1]
            var_a = nombres_modelo[ranking_shap[0]]
            var_b = nombres_modelo[ranking_shap[1]]
            print(f"Variables SHAP: '{var_a}' y '{var_b}'")
        except Exception as e:
            print(f"Error SHAP: {e}")

    # Si SHAP falla, vuelve al antiguo
    if var_a is None or var_b is None:
        if nombres is not None and score is not None:
            print("Usando frecuencia x magnitud")
            ranking = np.argsort(score)[::-1]
            var_a, var_b = nombres[ranking[0]], nombres[ranking[1]]
        else:
            print("No se pudo seleccionar variables para el gráfico.")
            return

    orig_a = df[var_a].values
    orig_b = df[var_b].values
    ce_a   = df[f"ce_{var_a}"].values
    ce_b   = df[f"ce_{var_b}"].values

    clases_all = sorted(df["pred_orig"].unique())
    # Definimos colores base usando una paleta extensa por si hay muchas clases
    colores_base  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4"]
    colores_osc   = ["#0D47A1", "#1B5E20", "#E65100", "#4A148C", "#880E4F", "#006064"]

    plt.figure(figsize=(7, 6))

    # Graficar frontera de decision (regiones coloreadas)
    if FRONTERA and modelo is not None:
        from matplotlib.colors import ListedColormap
        pad   = 0.8
        x_min = min(orig_a.min(), ce_a.min()) - pad
        x_max = max(orig_a.max(), ce_a.max()) + pad
        y_min = min(orig_b.min(), ce_b.min()) - pad
        y_max = max(orig_b.max(), ce_b.max()) + pad

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 400),
            np.linspace(y_min, y_max, 400)
        )

        # Grid con todas las columnas del modelo;
        # las no seleccionadas se fijan a su media en el CSV
        grid_full = np.zeros((xx.size, len(nombres_modelo)))
        for j, col in enumerate(nombres_modelo):
            if col == var_a:
                grid_full[:, j] = xx.ravel()
            elif col == var_b:
                grid_full[:, j] = yy.ravel()
            else:
                grid_full[:, j] = df[col].mean() if col in df.columns else 0.0

        grid_df = pd.DataFrame(grid_full, columns=nombres_modelo)

        # Predecimos la clase para cada punto del mapa 2D
        Z_labels = modelo.predict(grid_df)
        
        # Extendemos las clases por si el grid predice alguna clase que no está en el dataset origen
        clases_grid = sorted(list(set(clases_all) | set(Z_labels)))
        dict_clases = {cls: i for i, cls in enumerate(clases_grid)}
        
        Z_idx = np.array([dict_clases[val] for val in Z_labels]).reshape(xx.shape)
        
        cmap_bg = ListedColormap(colores_base[:len(clases_grid)])
        
        # Rellenar con colores tenues las áreas
        plt.contourf(xx, yy, Z_idx, alpha=0.15, cmap=cmap_bg, levels=np.arange(len(clases_grid) + 1) - 0.5)

    else:
        clases_grid = clases_all

    # Graficar puntos originales 
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        color = colores_base[idx_color % len(colores_base)]
        mask = df["pred_orig"] == cls
        plt.scatter(orig_a[mask], orig_b[mask], color=color, s=40,
                    alpha=0.8, label=f"{cls}", zorder=3)

    # Flechas origen -> contraejemplo
    if FLECHAS:
        for xa, ya, xb, yb in zip(orig_a, orig_b, ce_a, ce_b):
            plt.annotate("", xy=(xb, yb), xytext=(xa, ya),
                         arrowprops=dict(arrowstyle="->", color="#555555",
                                         lw=0.8, alpha=0.5))

    # Contraejemplos
    for cls in clases_all:
        idx_color = clases_grid.index(cls) if cls in clases_grid else clases_all.index(cls)
        color_ce = colores_osc[idx_color % len(colores_osc)]
        mask = df["pred_orig"] == cls
        if mask.any():
            plt.scatter(ce_a[mask], ce_b[mask], color=color_ce, marker="X", s=80, 
                        linewidths=0.6, edgecolors="white", alpha=1.0, 
                        label=f"Contraej. ({cls})", zorder=5)

    plt.xlabel(var_a)
    plt.ylabel(var_b)
    plt.title(f"Originales vs Contraejemplos  —  '{var_a}' y '{var_b}'")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analizar_csv(CSV_PATH)