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

# Standard lib imports
import os
import logging
from collections import OrderedDict
from orbit_prediction import get_state_vect_cols
# Numeric/ML lib imports
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ErrorGBRT:
    """A multiple output regression model that uses GBRT models to estimate the
    prediction error of an astrodynamical model.

    :param *args **kwargs: All parameters are passed to the underlying XGBoost
        GBRT implementation.  See here for all available parameters
        https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """
    def __init__(self, *args, **kwargs):
        self.reg = xgb.XGBRegressor(*args, **kwargs)
        self._models = OrderedDict()

    def get_model(self, model_name):
        """Gets a regression model by the name of the error column that
        it estimates.

        :param model_name: The name of the error column to fetch the model for
        :type model_name: str

        :return: The model responsible for estimating the error column
        :rtype: xgboost.XGBRegressor
        """
        return self._models.get('model_name')

    def set_model(self, model_name, model):
        """Stores the regression model by the name of the error column that it
        estimates.

        :param model_name: The name of the error column to store the model as
        :type model_name: str

        :param model: The regression model to store
        :type model: xgboost.XGBRegressor
        """
        self._models[model_name] = model

    def get_models(self):
        """Gets all the stored models.

        :return: A list model name, regression model pairs
        :rtype: [(str, xgboost.XGBRegressor)]
        """
        return list(self._models.items())

    def fit(self, X, ys, eval_metric='rmse'):
        """Fits the underlying GBRT models on the provided training data.

        :param X: The feature matrix to use in training
        :type X: numpy.ndarray

        :param ys: The multiple target columns to use in training
        :type ys: numpy.ndarray

        :param eval_metric: The metric to use to evaluate each GBRT
            model's performance
        :type eval_metric: str

        :return: The fitted multi-output regression model
        :rtype: orbit_prediction.ml_model.ErrorGBRT
        """
        # Check that the feature matrix and target matrix are the correct sizes
        check_X_y(X, ys, multi_output=True)
        # Get the XGBoost parameters to use for each regressor
        xgb_params = self.reg.get_params()
        # Build and train a GBRT model for each target column
        for target_col in ys.columns:
            y = ys[target_col]
            reg = xgb.XGBRegressor(**xgb_params)
            reg.fit(X, y, eval_metric=eval_metric)
            self.set_model(target_col, reg)

        return self

    def predict(self, X):
        """Uses the underlying GBRT models to estimate each component of the
        physical model error.

        :param X: The feature matrix to make predictions for
        :type X: numpy.ndarray

        :return: The estimated physical model error for each component
        :rtype: numpy.ndarray
        """
        # Make sure the input matrix is the right shape
        X = check_array(X)
        # Each model predicts the error for its respective state
        # vector component
        err_preds = [m.predict(X) for (_, m) in self.get_models()]
        # Orient the error estimates as column vectors
        err_preds = np.stack(err_preds, axis=1)
        return err_preds

    def eval_models(self, X, ys):
        """Calculates the root mean squared error (RMSE) and the coefficient of
        determination (R^2) for each of the models.

        :param X: The feature matrix to use in evaluating the regression models
        :type X: numpy.ndarray

        :param y: The target columns to use in evaluating the regression models
        :type y: numpy.ndarray

        :return: Returns a DataFrame containing the evaluation metric results
        :rtype: pandas.DataFrame
        """
        evals = []
        for target_col, reg in self.get_models():
            y_hat = reg.predict(X)
            y = ys[target_col]
            rmse = metrics.mean_squared_error(y, y_hat, squared=False)
            r2 = metrics.r2_score(y, y_hat)
            eval_dict = {'Error': target_col, 'RMSE': rmse, 'R^2': r2}
            evals.append(eval_dict)
        return pd.DataFrame(evals)

    def plot_feat_impts(self, X):
        """Plots the feature importances for each of the error models.
        For use in an interactive jupyter session.

        :param X: The feature matrix to use to calculate the feature
            importances from
        :type X:
        """
        feat_names = X.columns
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        for (target_col, model), ax in zip(self.get_models(), axs.flat):
            feat_imp = pd.Series(model.feature_importances_, index=feat_names)
            feat_imp.sort_values(ascending=False, inplace=True)
            feat_imp.plot(kind='bar', ax=ax, title=target_col)
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()

    def save(self, models_dir):
        """Saves the error estimations models as JSON representations.

        :param models_dir: The path to save the XGBoost JSON files to
        :type models_dir: str
        """
        for model_name, err_model in self.get_models():
            file_name = f'{model_name}.json'
            file_path = os.path.join(models_dir, file_name)
            err_model.save_model(file_path)

    def load(self, models_dir):
        """Loads previously trained XGBoost models from the `models_dir`

        :param models_dir: The path to where the XGBoost JSON files are
        :type models_dir: str

        :return: The error estimation model with the underlying previously
            trained GBRT models loaded.
        :rtype: orbit_prediction.ml_model.ErrorGBRT
        """
        model_names = get_state_vect_cols('physics_err')
        for mn in model_names:
            model = xgb.XGBRegressor()
            model_path = os.path.join(models_dir, f'{mn}.json')
            model.load_model(model_path)
            self.set_model(mn, model)
        return self


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
    feature_cols = ['elapsed_seconds'] + get_state_vect_cols('physics_pred') \
        + get_state_vect_cols('start')
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
    err_model = ErrorGBRT(**default_params)
    err_model.fit(X, ys, eval_metric=eval_metric)
    return err_model


def run(args):
    """Trains baseline XGBoost models to estimate physical model error based
    on parameters supplied by the CLI.

    :param args: The command line arguments
    :type args: argparse.Namespace
    """
    logger.info('Loading physical model orbit prediction training data...')
    physics_pred_df = pd.read_parquet(args.input_path)
    logger.info('Building training and test sets...')
    train_test_data = build_train_test_sets(physics_pred_df)
    if args.use_gpu:
        params = {}
    else:
        params = {'tree_method': 'hist'}

    logger.info('Training Error Models...')
    err_model = train_models(train_test_data, params=params)
    err_model_eval = err_model.eval_models(train_test_data['X_test'],
                                           train_test_data['y_test'])
    logger.info(err_model_eval)
    logger.info('Serializing Error Models...')
    err_model.save(args.out_dir)
