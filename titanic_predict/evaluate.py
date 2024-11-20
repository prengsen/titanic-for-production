import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

def evaluate_model():
    data_test = pd.read_csv('../data/processed/test_dataset.csv')

    with open('../artifacts/feature_eng_configs.pkl', 'rb') as f:
        feature_eng_configs = pickle.load(f)

    # aplicación de ingeniería de características 
    data_test['Age'] = data_test['Age'].fillna(feature_eng_configs['age_imputed_value'])
    data_test['Embarked'] = data_test['Embarked'].fillna(feature_eng_configs['embarked_imputed_value'])
    data_test['Sex'] = pd.get_dummies(data_test['Sex'], drop_first=True).astype(int)
    data_test['Embarked'] = data_test['Embarked'].map(feature_eng_configs['codificador_embarked'])

    # cargamos scaler
    with open('../artifacts/std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)

    # cargamos modelo
    with open('../models/logitic_regresion_v1.pkl', 'rb') as f:
        modelo = pickle.load(f)

    y_true = data_test['Survived']
    data_test.drop(['Survived'], axis=1, inplace=True)

    X_data_test_std = std_scaler.transform(data_test)

    data_test_predicts = modelo.predict(X_data_test_std)

    model_acc = accuracy_score(y_true, data_test_predicts)
    print(f"Accuracy del modelo: {model_acc}")


# llamada a evaluación del modelo.
evaluate_model()
