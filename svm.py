import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_data():
    df_train = pd.read_csv("train_nolineal.csv")
    df_test = pd.read_csv("test_nolineal.csv")
    
    X_train = df_train[["x1", "x2"]].values
    y_train = df_train["y"].values
    
    X_test = df_test[["x1", "x2"]].values
    y_test = df_test["y"].values
    
    return X_train, y_train, X_test, y_test

def train_model():
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data()

    # Crear y entrenar el modelo
    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=0)
    )

    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return modelo, (X_train, y_train, X_test, y_test), acc

if __name__ == "__main__":
    modelo, datos, precision = train_model()
    print(f"Modelo SVM entrenado con éxito.")
    print(f"Accuracy en test: {precision:.4f}")