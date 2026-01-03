import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_data():
    df_train = pd.read_csv("train_lineal.csv")
    df_test  = pd.read_csv("test_lineal.csv")
    X_train = df_train[["x1", "x2"]].values
    y_train = df_train["y"].values
    X_test  = df_test[["x1", "x2"]].values
    y_test  = df_test["y"].values
    return X_train, y_train, X_test, y_test

def train_model():
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data()

    # Crear y entrenar el modelo
    model = make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=1000, tol=1e-3, random_state=0)
        )
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, (X_train, y_train, X_test, y_test), acc
