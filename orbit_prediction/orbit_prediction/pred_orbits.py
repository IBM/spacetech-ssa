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
import numpy as np
import pandas as pd
import datetime as dt
import orbit_prediction.spacetrack_etl as etl
from orbit_prediction.ml_model import ErrorGBRT
from orbit_prediction import get_state_vect_cols
from orbit_prediction.physics_model import PhysicsModel


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_latest_orbit_data(space_track_user,
                          space_track_password,
                          norad_ids=None):
    """Fetches the latest TLE data from Space Track

    :param space_track_user: The user name for the Space Track account
    :type space_track_user: str

    :param space_track_password: The password for the Space Track account
    :type space_track_password:

    :param norad_ids: An optional list of NORAD IDs to fetch the TLEs
        for.  If NORAD IDs are not provided then data will be fetched
        for all ASOs in LEO.
    :type norad_ids: [str]

    :return: A DataFrame containing the latest TLE data for the requested ASOs
    :rtype: pandas.DataFrame
    """
    stc = etl.build_space_track_client(space_track_user, space_track_password)
    etl_client = etl.SpaceTrackETL(stc)
    latest_orbit_data = etl_client.build_leo_df(norad_ids=norad_ids,
                                                last_n_days=30,
                                                only_latest=True)
    return latest_orbit_data


def predict_orbits(df, physics_model, ml_model, n_days, timestep):
    """Use a physical model to predict the future orbits of all ASOs in the
    provided DataFrame, then use ML models to predict the error in the physics
    predictions, and finally adjust the physical predictions based on the error
    estimates.

    :param df: The latest TLE data for the ASOs to predict the orbits of
    :type df: pandas.DataFrame

    :param physics_model: The physics model to use for making predictions
    :type physics_model: orbit_prediction.physics_model.PhysicsModel

    :param ml_model: The ML model to use to estimate the error for each
        component of the predicted state vector
    :type ml_model: orbit_prediction.ml_model.ErrorGBRT

    :param n_days: The number of days into the future to predict orbits for
    :type n_days: int

    :param timestep: The frequency in seconds to make orbital predictions at
    :type timestep: float

    :return: The input DataFrame with the physical orbit predictions, the
        estimated errors, and the corrected orbit predictions added
        as columns
    :rtype: pandas.DataFrame
    """
    # The columns needed by the physics model to make orbit predictions
    physics_cols = ['epoch'] + get_state_vect_cols()
    start_states = df[physics_cols].to_numpy()
    physics_model.fit(start_states)

    # Use the latest epoch in the dataset as the start of the prediction window
    pred_start = df.epoch.max()
    pred_end = pred_start + dt.timedelta(days=n_days)
    df['pred_start_dt'] = pred_start
    df['pred_end_dt'] = pred_end
    # Get the total amount of seconds in the prediction window
    pred_window_seconds = (pred_end - pred_start).total_seconds()
    # Calculate how many predictions we will make based on the
    # the length of the prediction window and the timestep
    n_pred_intervals = int(pred_window_seconds / timestep) - 1
    # The amount, in seconds, we need to offset each timestep so
    # they line up with the prediction start time
    offsets = (pred_start - df.epoch).dt.total_seconds().to_numpy()
    # Transform the offsets array to be a column vector
    offsets = offsets[:, np.newaxis]
    # An array of ones for each timestep
    timesteps = np.ones((len(df), n_pred_intervals))
    # Each timestep contains its ordinal position in the array
    timesteps = timesteps * range(1, n_pred_intervals+1)
    # Multiply each ordinal position with the timestep plus the offset
    timesteps = timesteps * (offsets + timestep)
    # Prepend the offset to the timesteps array
    timesteps = np.hstack((offsets, timesteps))

    logger.info('Predicting Orbits...')
    physics_preds = physics_model.predict(timesteps)

    logger.info('Estimating physics errors...')
    Xs = np.concatenate((timesteps[:, :, np.newaxis], physics_preds),
                        axis=2)
    ml_preds = np.array([ml_model.predict(X) for X in Xs])
    # The orbit predictions are the physical predictions corrected by the
    # learned estimated error
    orbit_preds = physics_preds - ml_preds

    # Add everything as columns to the DataFrame
    df['physics_preds'] = pd.Series([pp for pp in physics_preds])
    df['ml_err_preds'] = pd.Series(mp for mp in ml_preds)
    df['orbit_preds'] = [op for op in orbit_preds]

    return df


def run(args):
    """Combine physic and ML models to predict future orbits based on parameters
    specified by the CLI.

    :param args: The command line arguments
    :type args: argparse.Namespace
    """
    latest_orbit_data = get_latest_orbit_data(args.st_user,
                                              args.st_password,
                                              norad_ids=args.norad_ids)
    physics_model = PhysicsModel(n_jobs=-1)

    logger.info('Loading ML Models...')
    ml_model = ErrorGBRT()
    ml_model.load(args.ml_model_dir)

    orbit_pred_df = predict_orbits(latest_orbit_data,
                                   physics_model,
                                   ml_model,
                                   n_days=args.n_days,
                                   timestep=args.timestep)
    logger.info('Serializing Results...')
    orbit_pred_df.to_pickle(args.output_path)
