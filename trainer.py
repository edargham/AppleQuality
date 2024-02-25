import tensorflow as tf
import keras
from keras_tuner import GridSearch, HyperParameters
import sklearn.base

from preprocessing import preprocess_data
import pandas as pd
from typing import Callable

from sklearn.model_selection import GridSearchCV
import datetime


def run_training_process_ml(
    builder: Callable[[], tuple[sklearn.base.BaseEstimator, dict]],
    data: pd.DataFrame,
    target_col: str,
    score: str,
):
    if score not in ('accuracy', 'precision', 'recall', 'f1'):
        raise ValueError('Score must be one of "accuracy", "precision", "recall", "f1".')

    x_train, x_test, y_train, y_test = preprocess_data(data, target_col)

    model, params = builder()

    grid = GridSearchCV(model, params, scoring=score, n_jobs=-1, verbose=1, cv=3, refit=True)
    grid.fit(x_train, y_train)
    score = grid.best_estimator_.score(x_test, y_test)

    print('Grid Search Completed.')
    print('Best Parameters Found:', grid.best_params_)
    print('Best Train Score:', grid.best_score_)
    print('Best Test Score:', score)

    return grid.best_estimator_


def run_training_process_nn(
    builder: Callable[..., keras.Model],
    data: pd.DataFrame,
    target_col: str,
):
    x_train, x_test, y_train, y_test = preprocess_data(data, target_col)
    print(x_train.shape[1])

    def build_model(hp: HyperParameters):
        num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=5, step=1)
        num_hidden_units = hp.Int('num_hidden_units', min_value=16, max_value=64, step=16)
        learning_rate = hp.Float('learning_rate', min_value=6e-4, max_value=1e-3, step=1e-4)

        metrics = [
            keras.metrics.Accuracy(),
            keras.metrics.Precision(0.5),
            keras.metrics.Recall(0.5)
        ]

        model = builder(
            num_features=x_train.shape[1],
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            learning_rate=learning_rate,
            metrics=metrics
        )

        print(model.summary())

        return model

    train_data = tf.data\
        .Dataset\
        .from_tensor_slices((x_train, y_train))\
        .shuffle(128)\
        .batch(8)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)\
        .repeat(10)

    val_data = tf.data\
        .Dataset\
        .from_tensor_slices((x_test, y_test))\
        .batch(8)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    tuner = GridSearch(
        build_model,
        objective='val_loss',
        executions_per_trial=1,
        directory='tuner_results',
        project_name=f'NN_Hyperparam_Tuning_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    tuner.search(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
          The hyperparameter search is complete.\n
          Optimal Hyper parameters:\n
          {best_hps.values}.
        """
    )

    # Build the model with the optimal hyperparameters and train it on the data
    tuned_model = tuner.hypermodel.build(best_hps)
    history = tuned_model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
    )

    return tuned_model, history
