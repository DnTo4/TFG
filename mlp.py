import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_data(train_path, test_path, target_column=None):
    # Leer datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Si no se indica variable objetivo, usar la última columna
    if target_column is None:
        target_column = df_train.columns[-1]

    # Separar variables
    y_train = df_train[target_column]
    X_train = df_train.drop(columns=[target_column])
    
    y_test = df_test[target_column]
    X_test = df_test.drop(columns=[target_column])

    # Convertir variables categóricas
    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined = pd.get_dummies(X_combined)

    X_train = X_combined.iloc[:len(X_train)]
    X_test = X_combined.iloc[len(X_train):]

    return X_train, y_train, X_test, y_test

def train_model(train_path, test_path, target_column=None):
    # Cargar datos
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Crear pipeline MLP
    modelo = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=0)
    )

    # Entrenar
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar nombres de columnas
    nombres = X_train.columns.tolist()

    # Devolver DataFrames
    return modelo, (X_train, y_train, X_test, y_test), acc, nombres


