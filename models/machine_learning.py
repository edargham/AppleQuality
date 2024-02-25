from typing import List
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import keras

from models.neural_network import NNmodel

def build_svc() -> tuple[SVC, dict]:
    return SVC(), {
        'kernel': ['rbf'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': [0.1, 1.0, 10.0, 100.0],
        'random_state': [42]
    }


def build_rf() -> tuple[RandomForestClassifier, dict]:
    return RandomForestClassifier(), {
        'n_estimators': [16, 32, 64, 128],
        'warm_start': [True, False],
        'max_depth': [8, 16, 32, 64],
        'ccp_alpha': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'bootstrap': [True],
        'max_samples': [0.33, 0.5, 0.67, 1.0],
        'random_state': [42]
    }

def build_lr() -> tuple[LogisticRegression, dict]:
    return LogisticRegression(), {
        "penalty":['l1','l2','elasticnet'],
        "C":[1,10,100],
        "solver":['lbfgs','liblinear','newton-cg'],
        "max_iter":[50,100,200,500]
    }

def build_adb() -> tuple[AdaBoostClassifier, dict]:
    return AdaBoostClassifier(), {
        "n_estimators": [16, 32, 64, 128],
        "learning_rate":[0.05,0.1,0.25,0.5,1,10]
    }

def build_knn() -> tuple[KNeighborsClassifier, dict]:
    return KNeighborsClassifier(), {
        "n_neighbors": [3,5,7,9],
        "p":[1,2]
    }

def build_nn(
        num_features: int,
        num_hidden_layers: int,
        num_hidden_units: int,
        learning_rate: float,
        metrics: List[keras.metrics.Metric]
) -> keras.Model:
    model = NNmodel(
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units
    )

    model.build(input_shape=(None, num_features))
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics
    )
    return model

