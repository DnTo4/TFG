import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

CSV_PATH   = "contraejemplos.csv"
MODEL_PATH = "modelo.joblib" # Ruta donde se guarda el modelo

FRONTERA = True    # visualizar frontera de decision (requiere modelo)
FLECHAS  = False    # visualizar flechas origen -> contraejemplo
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
    if nombres is not None:
        plot_contraejemplos(df, nombres, score)


def plot_contraejemplos(df, nombres, score):

    # Seleccionar las dos variables más importantes por score
    ranking = np.argsort(score)[::-1]
    idx_a, idx_b = ranking[0], ranking[1]
    var_a, var_b = nombres[idx_a], nombres[idx_b]

    orig_a = df[var_a].values
    orig_b = df[var_b].values
    ce_a   = df[f"ce_{var_a}"].values
    ce_b   = df[f"ce_{var_b}"].values

    plt.figure(figsize=(7, 6))

    # Graficar frontera de decision
    if FRONTERA:
        try:
            bundle         = joblib.load(MODEL_PATH)
            modelo         = bundle["modelo"]
            nombres_modelo = bundle["nombres"]

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

            if hasattr(modelo, "predict_proba"):
                Z = modelo.predict_proba(grid_df)[:, 1].reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
            elif hasattr(modelo, "decision_function"):
                dec = modelo.decision_function(grid_df)
                Z = (dec[:, 0] if dec.ndim > 1 else dec).reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0.0], colors="k", linewidths=1.2)
            else:
                Z = modelo.predict(grid_df).reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)

        except FileNotFoundError:
            print(f"Aviso: no se encontró '{MODEL_PATH}', se omite la frontera.")

    # Grficar puntos originales 
    colores_orig = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    clases = sorted(df["pred_orig"].unique())
    for cls, color in zip(clases, colores_orig):
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
    colores_ce = ["#0D47A1", "#1B5E20", "#E65100", "#4A148C"] # Versiones oscuras de colores_orig
    for cls, color_ce in zip(clases, colores_ce):
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