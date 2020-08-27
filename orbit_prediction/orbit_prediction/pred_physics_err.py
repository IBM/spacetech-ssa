# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def eval_models(models, data):
    """Calculates the root mean squared error (RMSE) and the coefficient of
    determination (R^2) for each of the models.

    :param models: Dictionary of the error model for each state vector
        component
    :type models: {str: xgboost.XGBRegressor}

    :param data: Dictionary containing the training and test datasets
    :type data: {str: numpy.array}

    :return: Returns a DataFrame containing the evaluation metric results
    :rtype: pandas.DataFrame
    """
    evals = []
    for target_col, reg in models.items():
        y_hat = reg.predict(data['X_test'])
        y = data['y_test'][target_col]
        rmse = metrics.mean_squared_error(y, y_hat, squared=False)
        r2 = metrics.r2_score(y, y_hat)
        eval_dict = {'Error': target_col, 'RMSE': rmse, 'R^2': r2}
        evals.append(eval_dict)
    return pd.DataFrame(evals)


def plot_feat_impts(models, data):
    """Plots the feature importances for each of the error models.
    For use in an interactive jupyter session.

    :param models: Dictionary of the error model for each state vector
        component
    :type models: {str: xgboost.XGBRegressor}

    :param data: Dictionary containing the training and test datasets
    :type data: {str: numpy.array}
    """
    feat_names = data['X_train'].columns
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    for (target_col, model), ax in zip(models.items(), axs.flat):
        feat_imp = pd.Series(model.feature_importances_, index=feat_names)
        feat_imp.sort_values(ascending=False, inplace=True)
        feat_imp.plot(kind='bar', ax=ax, title=target_col)
    plt.ylabel('Feature Importance Score')
    plt.tight_layout()


def get_state_vect_cols(prefix):
    """Get the column names of the state vector components with the
    provided `prefix`.

    :param prefix: The prefix that is used in front of the state vector
        components in the column names, examples are `physics_pred` and
        `physics_err`
    :type prefix: str

    :return: A list of the 6 names of the prefixed state vector components
    :rtype: [str]
    """
    vectors = ['r', 'v']
    components = ['x', 'y', 'z']
    col_names = [f'{prefix}_{v}_{c}'
                 for v, c
                 in itertools.product(vectors, components)]
    return col_names


def load_models(models_dir):
    """Loads previously trained XGBoost models from the `models_dir`

    :param models_dir: The path to where the serialized XGBoost JSON files are
    :type models_dir: str

    :return: A list of the loaded XGBoost models
    :rtype: [xgboost.XGBRegressor]
    """
    ml_models = []
    model_names = get_state_vect_cols('physics_err')
    for mn in model_names:
        model = xgb.XGBRegressor()
        model_path = os.path.join(models_dir, f'{mn}.json')
        model.load_model(model_path)
        ml_models.append(model)
    return ml_models


def save_models(models, models_dir):
    """Saves the error estimations models as JSON representations.

    :param models: Dictionary of the error model for each state vector
        component
    :type models: {str: xgboost.XGBRegressor}

    :param models_dir: The path to save the serialized XGBoost JSON files to
    :type models_dir: str
    """
    for model_name, err_model in models.items():
        file_name = f'{model_name}.json'
        file_path = os.path.join(models_dir, file_name)
        err_model.save_model(file_path)


def predict_err(models, physics_preds):
    """Uses the provide ML models to predict the error in the physics
    model orbit prediction.

    :param ml_models: The ML models to use to estimate the error in each
        of the predicted state vector components.
    :type ml_models: [xgboost.XGBRegressor]

    :param physcis_preds: The elapsed time in seconds and the predicted
        state vectors to estimate the errors for
    :type physcis_preds: numpy.array

    :return: The estimated errors
    :rtype: numpy.array
    """
    # Each model predicts the error for its respective state vector component
    err_preds = [m.predict(physics_preds) for m in models]
    # Orient the error estimates as column vectors
    err_preds = np.stack(err_preds, axis=1)
    return err_preds


def build_train_test_sets(df, test_size=0.2):
    """Builds training and testing sets from the provided DataFrame.

    :param df: The DataFrame to use to build training and test sets from
    :type df: pandas.DataFrame

    :param test_size: The percentage size of the DataFrame that should be used
        to build the test set
    :type test_size: float

    :return: A dictionary containing the feature and target training/test sets
    :rtype: dict[str, pandas.DataFrame]
    """
    # Features are the physics predicted state vectors and the amount of
    # time in seconds into the future the prediction was made
    feature_cols = ['elapsed_seconds'] + get_state_vect_cols('physics_pred')
    # The target values are the errors between the physical model predictions
    # and the ground truth observations
    target_cols = get_state_vect_cols('physics_err')
    # Create feature and target matrices
    X = df[feature_cols]
    y = df[target_cols]
    # Split feature and target data into training and test sets
    data_keys = ['X_train', 'X_test', 'y_train', 'y_test']
    data_vals = train_test_split(X, y, test_size=test_size)
    train_test_data = dict(zip(data_keys, data_vals))
    return train_test_data


def train_models(data, params={}, eval_metric='rmse'):
    """Trains gradient boosted regression tree models to estimate the error in
    each of the six state vector components in the physical model prediction

    :param data: Dictionary containing the training and test datasets
    :type data: {str: numpy.array}

    :param params: A dictionary of parameters to pass to the XGBRegressor
        constructor
    :type params: dict

    :param eval_metric: The loss function to use in model training
    :type eval_metric: str

    :return: Dictionary containing the trained models for each state vector
        component
    :rtype: {str: xgboost.XGBRegressor}
    """
    default_params = {
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }
    default_params.update(params)
    X, ys = data['X_train'], data['y_train']
    models = {}
    for target_col in ys.columns:
        y = ys[target_col]
        reg = xgb.XGBRegressor(**default_params)
        reg.fit(X, y, eval_metric=eval_metric)
        models[target_col] = reg
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Train baseline XGBoost models to estimate physical '
                     'prediction error'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_path',
        help=('The path to the parquet file containing the physical model '
              'prediction training data'),
        type=str,
        required=True
    )
    parser.add_argument(
        '--use_gpu',
        help='Use a GPU in model training',
        action='store_true'
    )
    parser.add_argument(
        '--out_dir',
        help=('The directory to serialize the models to'),
        type=str,
        required=True
    )
    args = parser.parse_args()

    logger.info('Loading physical model orbit prediction training data...')
    physics_pred_df = pd.read_parquet(args.input_path)
    logger.info('Building training and test sets...')
    train_test_data = build_train_test_sets(physics_pred_df)
    if args.use_gpu:
        params = {}
    else:
        params = {'tree_method': 'hist'}

    logger.info('Training Error Models...')
    err_models = train_models(train_test_data, params=params)
    logger.info(eval_models(err_models, train_test_data))
    logger.info('Serializing Error Models...')
    save_models(err_models, args.out_dir)
