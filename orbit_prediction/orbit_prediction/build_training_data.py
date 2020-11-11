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

# Standard libraries
import os
import logging
import itertools
import datetime as dt
# Data processing libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
# Physics model
from orbit_prediction.physics_model import PhysicsModel

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_state_vectors(row):
    """Gets the position and velocity vectors from the DataFrame `row`.

    :param row: The row to extract the state vectors from
    :type row: pandas.Series

    :return: The position and velocity vectors
    :rtype: (numpy.array, numpy.array)
    """
    comps = ['x', 'y', 'z']
    r = row[[f'r_{comp}' for comp in comps]].to_numpy()
    v = row[[f'v_{comp}' for comp in comps]].to_numpy()
    return r, v


def predict_orbit(window):
    """Predict the state vectors of each future timestep in the given `window`
    using a physics astrodynamics model.

    :param window: The window of timesteps to predict the orbit of the ASO for
    :type window: pandas.DataFrame

    :return: The original timestep rows with the predicted state vectors added
    :rtype: pandas.DataFrame
    """
    # The `window` DataFrame is reverse sorted by time so the starting position
    # is the last row
    start_row = window.iloc[-1]
    start_epoch = start_row.name
    start_r, start_v = get_state_vectors(start_row)
    start_state = np.concatenate((np.array([start_epoch]),
                                  start_r, start_v))
    # Build an orbit model
    orbit_model = PhysicsModel()
    orbit_model.fit([start_state])
    future_rows = window.iloc[:-1].reset_index()
    # We add the epoch and the state vector components of the starting row
    # to the rows we will use the physics model to make predictions for
    future_rows['start_epoch'] = start_epoch
    state_vect_comps = ['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z']
    for svc in state_vect_comps:
        future_rows[f'start_{svc}'] = start_row[svc]
    # Calculate the elapsed time from the starting epoch to the
    # the epoch of all the rows to make predictions for
    time_deltas = future_rows.epoch - future_rows.start_epoch
    elapsed_seconds = time_deltas.dt.total_seconds()
    future_rows['elapsed_seconds'] = elapsed_seconds
    physics_cols = [f'physics_pred_{svc}' for svc in state_vect_comps]
    # Predict the state vectors for each of the rows in the "future"
    predicted_orbits = orbit_model.predict([elapsed_seconds.to_numpy()])
    future_rows[physics_cols] = predicted_orbits[0]
    return future_rows


def predict_orbits(df, last_n_days, n_pred_days):
    """Use a physics astrodynamics model to predict the orbits of the ASOs
    in the provided DataFrame.

    :param df: The DataFrame containing the observed orbital state vectors
        to use to make predictions from
    :type df: pandas.DataFrame

    :param last_n_days: Filter the DataFrame to use rows from only the last
        `n` days.  Use all the rows if `None` is passed, but this may take a
        very long time to run
    :type last_n_days: int

    :param n_pred_days: The number of days in the rolling prediction window
    :type n_pred_days: int
    """
    if last_n_days:
        time_cutoff = df.epoch.max() - dt.timedelta(days=last_n_days)
        df = df[df.epoch >= time_cutoff]
    epoch_df = df.sort_values('epoch', ascending=False).set_index('epoch')
    pred_window_length = f'{n_pred_days}d'
    # For each row in `df` we create a window of all of the observations for
    # that ASO that are within `n_pred_days` of the given row
    window_cols = ['aso_id', pd.Grouper(freq=pred_window_length)]
    windows = [w[1] for w in epoch_df.groupby(window_cols)]
    # Predict the orbits in each window in parallel
    window_dfs = Parallel(n_jobs=-1)(delayed(predict_orbit)(w)
                                     for w in tqdm(windows))
    # Join all of the window prediction DataFrames into a single DataFrame
    physics_pred_df = pd.concat(window_dfs).reset_index(drop=True)
    return physics_pred_df


def calc_physics_error(df):
    """Calculates the error in the state vector components between the ground truth
    observations and the physics model predictions.

    :param df: The DataFrame containing the ground truth observations and the
        physics model predictions
    :type df: pandas.DataFrame

    :return: The input DataFrame with the physical model error column added
    :rtype: pandas.DataFrame
    """
    comps = ['x', 'y', 'z']
    vects = ['r', 'v']
    for vect, comp in itertools.product(vects, comps):
        comp_col = f'{vect}_{comp}'
        err_col = f'physics_err_{comp_col}'
        err_val = df[f'physics_pred_{comp_col}'] - df[comp_col]
        df[err_col] = err_val
    return df


def run(args):
    """Builds a training data set of physics model errors based on the
    parameters supplied by the CLI.

    :param args: The command line arguments
    :type args: argparse.Namespace
    """
    logger.info('Loading input DataFrame...')
    input_df = pd.read_parquet(args.input_path)
    logger.info('Predicting orbits...')
    physics_pred_df = predict_orbits(input_df,
                                     last_n_days=args.last_n_days,
                                     n_pred_days=args.n_pred_days)
    logger.info('Calculating physical model error...')
    physics_pred_df = calc_physics_error(physics_pred_df)
    logger.info('Serializing results...')
    physics_pred_df.to_parquet(args.output_path)
