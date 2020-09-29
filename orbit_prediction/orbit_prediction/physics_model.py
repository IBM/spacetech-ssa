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
import argparse
import itertools
import datetime as dt
# Data processing libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
# Astrodynamics libraries
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_state_vectors(row):
    """Gets the position and velocity vectors from the DataFrame `row` along
    with their respective measurement units.

    :param row: The row to extract the state vectors from
    :type row: pandas.Series

    :return: The position and velocity vectors with measurement units
    :rtype: (astropy.units.quantity.Quantity, astropy.units.quantity.Quantity)
    """
    comps = ['x', 'y', 'z']
    position = row[[f'r_{comp}' for comp in comps]].to_numpy()
    position_vect = position * u.m
    velocity = row[[f'v_{comp}' for comp in comps]].to_numpy()
    velocity_vect = velocity * (u.m / u.s)
    return position_vect, velocity_vect


def build_orbit(row, epoch=None):
    """Builds an Orbit instance from a DataFrame row.

    :param row: A row from the USSTRATCOM ETL DataFrame
    :type row: pandas.Series

    :param epoch: An optional epoch to instantiate the orbit at, if no epoch is
        passed then use the epoch in the row
    :type epoch: pandas.Timestamp

    :return: An orbit object built using the orbital data in the row
    :rtype: poliastro.twobody.Orbit
    """
    r, v = get_state_vectors(row)
    if not epoch:
        epoch = row.epoch
    epoch_time = Time(epoch.to_numpy(), scale='utc')
    orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch_time)
    return orbit


def build_orbit_propagator(orbit, return_orbit=False):
    """Builds a function that propagates the given `orbit` to a specified
    point in time.

    :param orbit: The orbit to build the propagator for
    :type orbit: from poliastro.twobody.Orbit

    :return: The orbit propagator function
    :rtype: function
    """
    def _propagate_orbit(elapsed_seconds):
        """Propagates the closed over orbit into the future by
        `elapsed_seconds` and returns the resulting state vector
        of the new orbit with the position measured in meters and the
        velocity measured in meters per second.

        :param elapsed_seconds: The number of seconds into the future to
            propagate the orbit to
        :type elapsed_seconds: float

        :return: The 6 element state vector of the propagated orbit
        :rtype: pd.Series
        """
        prop_orbit = orbit.propagate(elapsed_seconds*u.s,
                                     method=cowell,
                                     ad=J2_perturbation,
                                     J2=Earth.J2.value,
                                     R=Earth.R.to(u.km).value)
        # Get the propagated position and velocity vectors
        prop_r, prop_v = prop_orbit.rv()
        # Join the position and velocity vectors into a single vector
        # after converting them to their respective measurements
        prop_state_vect = np.concatenate([prop_r.to(u.m).to_value(),
                                          prop_v.to(u.m/u.s).to_value()])

        if return_orbit:
            return prop_orbit, prop_state_vect
        else:
            return pd.Series(prop_state_vect)

    return _propagate_orbit


def predict_orbit(window):
    """Predict the state vectors of each future timestep in the given `window`
    using a physics astrodynamics model.

    :param window: The window of timesteps to predict the orbit of the RSO for
    :type window: pandas.DataFrame

    :return: The original timestep rows with the predicted state vectors added
    :rtype: pandas.DataFrame
    """
    # The `window` DataFrame is reverse sorted by time so the starting position
    # is the last row
    start_row = window.iloc[-1]
    start_epoch = start_row.name
    # Build an orbit object for the starting point of the timestep window
    orbit = build_orbit(start_row, start_epoch)
    # Build a function that will propagate the orbit to the future timesteps
    orbit_propagator = build_orbit_propagator(orbit)
    # The rows that we will predict the orbit for are all the rows in the
    # `window` but the last row
    future_rows = window.iloc[:-1].reset_index()
    future_rows['start_epoch'] = start_epoch
    # Calculate the elapsed time from the starting epoch to the
    # the epoch of all the rows to make predictions for
    time_deltas = future_rows.epoch - future_rows.start_epoch
    future_rows['elapsed_seconds'] = time_deltas.dt.total_seconds()
    physics_cols = ['physics_pred_r_x',
                    'physics_pred_r_y',
                    'physics_pred_r_z',
                    'physics_pred_v_x',
                    'physics_pred_v_y',
                    'physics_pred_v_z']
    # Predict the state vectors for each of the rows in the "future"
    future_rows[physics_cols] = future_rows.elapsed_seconds.apply(orbit_propagator)
    return future_rows


DEFAULT_LAST_N_DAYS = 30
DEFAULT_N_PRED_DAYS = 7


def predict_orbits(df,
                   last_n_days=DEFAULT_LAST_N_DAYS,
                   n_pred_days=DEFAULT_N_PRED_DAYS):
    """Use a physics astrodynamics model to predict the orbits of the RSOs
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
    # that RSO that are within `n_pred_days` of the given row
    window_cols = ['rso_id', pd.Grouper(freq=pred_window_length)]
    windows = [w[1] for w in epoch_df.groupby(window_cols)]
    # Predict the orbits in each window in parallel
    window_dfs = Parallel(n_jobs=-1)(delayed(predict_orbit)(w) for w in tqdm(windows))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict orbits using physics model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_path',
        required=True,
        help=('The path to the parquet file to load the orbit observation'
              'DataFrame from')
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help=('The path to the parquet file to save the physics model'
              'predictions to')
    )
    parser.add_argument(
        '--last_n_days',
        help='Only use observations from the last `n` days',
        type=int,
        default=DEFAULT_LAST_N_DAYS
    )
    parser.add_argument(
        '--n_pred_days',
        help='The number days in the prediction window',
        type=int,
        default=DEFAULT_N_PRED_DAYS
    )
    args = parser.parse_args()

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
