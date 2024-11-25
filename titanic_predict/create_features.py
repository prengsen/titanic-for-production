"""
    Este modulo contine las funciones para aplicar la ingeniería de características.
"""
import pickle
import pandas as pd


def create_model_fatures():
    """_summary_: esta función crear los features para el modelo.
    """
    dataset = pd.read_csv("../data/raw/train.csv")

    dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    # eliminamos columan cabin por cantidad de nulos.
    dataset.drop(['Cabin'], axis=1, inplace=True)

    # Imputamos la media en la columna Age
    media_age = dataset['Age'].mean()
    dataset['Age'] = dataset['Age'].fillna(media_age).astype(int)

    # Imputamos Embarked con valor mas frecuente.
    embarked_imputed_value = dataset['Embarked'].value_counts().index[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_imputed_value)

    # Codificación de variable Sex con One Hot Encoding
    dataset['Sex'] = pd.get_dummies(dataset['Sex'], drop_first=True).astype(int)

    # Codificación de variable Embarked con Frecuency Encoding
    codificador_embarked = dataset['Embarked'].value_counts()
    dataset['Embarked'] = dataset['Embarked'].map(codificador_embarked)

    # # 4. Guardamos el Dataset Procesado.
    dataset.to_csv('../data/processed/features_for_model.csv', index=False)

    feature_eng_configs = {
        'age_imputed_value': int(media_age),
        'embarked_imputed_value': embarked_imputed_value,
        'codificador_embarked': codificador_embarked
    }

    with open('../artifacts/feature_eng_configs.pkl', 'wb') as f:
        pickle.dump(feature_eng_configs, f)
