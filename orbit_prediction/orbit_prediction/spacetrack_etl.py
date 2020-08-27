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
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tletools import TLE
from astropy import units as u
import spacetrack.operators as op
from spacetrack import SpaceTrackClient


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_leo_rso_catalog(stc, norad_ids=None):
    """Retrieves entries from the Space Track Satellite Catalog for all RSOs
    that are in low Earth orbit.

    :param stc: The Space Track API client
    :type stc: spacetrack.SpaceTrackClient

    :param norad_ids: An optional list of NORAD IDs to fetch the TLEs
        for.  If NORAD IDs are not provided then data will be fetched
        for all RSOs in LEO.
    :type norad_ids: [str]

    :return: The catalog entries for LEO RSOs
    :rtype: pandas.DataFrame
    """
    query_params = {
        'decay': None,
        'current': 'Y'
    }
    if norad_ids:
        query_params['norad_cat_id'] = norad_ids
    else:
        query_params['period']  = op.less_than(128),
    leo_rsos = stc.satcat(**query_params)
    return pd.DataFrame(leo_rsos)


def get_object_types(rso_df):
    """Normalizes the object type and NORAD ID columns from the Space Track
    Satellite Catalog DataFrame.  This will be used to add the object type
    to the TLE data.

    :param rso_df: The DataFrame composed of entries from the Space Track
        Satellite Catalog
    :type rso_df: pandas.DataFrame

    :return: A pandas DataFrame containing the RSO's ID and normalized
        object type
    :rtype: pandas.DataFrame
    """
    cols = ['NORAD_CAT_ID', 'OBJECT_TYPE']
    col_mapper = {'NORAD_CAT_ID': 'rso_id', 'OBJECT_TYPE': 'object_type'}
    # Standardize the column names
    object_types = rso_df[cols].rename(columns=col_mapper)
    # Lowercase the object type strings and replace spaces with underscores
    norm_func = lambda s: s.lower().replace(' ', '_')
    object_types['object_type'] = object_types.object_type.apply(norm_func)
    return object_types


def get_leo_tles_str(stc, norad_ids, past_n_days, only_latest):
    """Uses the Space Track TLE API to get all the TLEs for the RSOs specified
    by the `norad_ids` for the specified time period.

    :param stc: The Space Track API client
    :type stc: spacetrack.SpaceTrackClient

    :param norad_ids: The NORAD IDs of the RSOs to get the TLEs for
    :type norad_ids: [str]

    :param past_n_days: The number of past days to get TLEs for
    :type past_n_days: int

    :param only_latest: Whether or not to only fetch the latest TLE for
        each RSO
    :type only_latest: bool

    :return: The three line elements for each specified RSO
    :rtype: str
    """
    query_params = {
        'epoch': f'>now-{past_n_days}',
        'norad_cat_id': norad_ids,
        'format': '3le'
    }
    if only_latest:
        query_params['ordinal'] = 1
        leo_tles_str = stc.tle_latest(**query_params)
    else:
        leo_tles_str = stc.tle(**query_params)

    return leo_tles_str


def get_tles(raw_tle_str):
    """Parses the raw TLE string and converts it to TLE objects

    :param raw_tle_str: The raw string form of the TLEs
    :type raw_tle_str: str

    :return: The parsed object representations of the TLEs
    :rtype: [tletools.TLE]
    """
    all_tle_lines = raw_tle_str.strip().splitlines()
    tles = []
    for i in range(len(all_tle_lines)//3):
        # Calculate offset
        j = i*3
        tle_lines = all_tle_lines[j:j+3]
        # Strip line number from object name line
        tle_lines[0] = tle_lines[0][2:]
        tle = TLE.from_lines(*tle_lines)
        tles.append(tle)
    return tles


def get_rso_data(tles):
    """Extracts the necessary data from the TLE objects
    for doing orbital prediction.

    :param tles: The list of TLE objects to extract orbit information from
    :type tles: [tletools.TLE]

    :return: A DataFrame of the extracted TLE data
    :rtype: pandas.DataFrame
    """
    tles_data = []
    for tle in tles:
        rso_data = {}
        rso_data['rso_name'] = tle.name
        rso_data['rso_id'] = tle.norad
        rso_data['epoch'] = tle.epoch.to_datetime()
        # Convert the TLE object to a poliastro.twobody.Orbit instance
        orbit = tle.to_orbit()
        # Calculate the position and velocity vectors
        r, v = orbit.rv()
        # Convert position vector from kilometers to meters
        r_m = r.to(u.m).to_value()
        # Convert the velocity vector from km/s to m/s
        v_ms = v.to(u.m/u.s).to_value()
        # Extract the components of the state vectiors
        rso_data['r_x'], rso_data['r_y'], rso_data['r_z'] = r_m
        rso_data['v_x'], rso_data['v_y'], rso_data['v_z'] = v_ms
        tles_data.append(rso_data)
    return pd.DataFrame(tles_data)


DEFAULT_PAST_N_DAYS = 30

def build_leo_df(stc, norad_ids=None, past_n_days=DEFAULT_PAST_N_DAYS,
                 only_latest=False):
    """Builds a pandas DataFrame of LEO RSO orbit observations from data
    provided by USSTRATCOM via space-track.org

    :param stc: The Space Track API client
    :type stc: spacetrack.SpaceTrackClient

    :param norad_ids: An optional list of NORAD IDs to fetch the TLEs
        for.  If NORAD IDs are not provided then data will be fetched
        for all RSOs in LEO.
    :type norad_ids: [str]

    :param past_n_days: The number of past days to get TLEs for
    :type past_n_days: int

    :param only_latest: Whether or not to only fetch the latest TLE for
        each RSO
    :type only_latest: bool

    :return: The Space Track orbit data for LEO RSOs
    :rtype: pandas.DataFrame

    """
    logger.info('Fetching Satellite Catalog Data...')
    leo_rsos = get_leo_rso_catalog(stc, norad_ids)
    norad_ids = leo_rsos['NORAD_CAT_ID']
    # The space-track.org API is rate limited and the response size
    # of the data is capped.  Experimenting found that we can reliably
    # get successful responses for about 500 RSOs so we break the
    # NORAD IDs into chunks for processing.
    n_chunks = len(norad_ids) // 500
    if n_chunks > 1:
        norad_chunks = np.array_split(norad_ids, n_chunks)
    else:
        norad_chunks = [norad_ids]

    logger.info(f'Number of TLE Batch Requests: {len(norad_chunks)}')

    leo_tles = []
    logger.info('Starting to fetch TLEs from space-track.org')
    for idx, norad_chunk in enumerate(norad_chunks):
        logger.info(f'Processing batch {idx+1}/{len(norad_chunks)}')
        logger.info(f'Fetching TLEs for {len(norad_chunk)} RSOs...')
        rso_ids = norad_chunk.to_list()
        chunk_tle_str = get_leo_tles_str(stc,
                                         rso_ids,
                                         past_n_days,
                                         only_latest)
        logger.info('Parsing raw TLE data...')
        chunk_tles = get_tles(chunk_tle_str)
        leo_tles += chunk_tles
    logger.info('Finished fetching TLEs')
    logger.info(f'Calculating orbital state vectors for {len(leo_tles)} TLEs...')
    rso_data = get_rso_data(leo_tles)
    object_types = get_object_types(leo_rsos)
    rso_data = rso_data.merge(object_types, on='rso_id', how='left')
    return rso_data


def st_callback(unitl):
    """Log the number of seconds the program is sleeping to stay
    within Space Track's API rate limit

    :param until: The time the program will sleep til
    :type until: float
    """
    duration = int(round(until - time.time()))
    logger.info(f'Sleeping for {duration} seconds.')


def build_space_track_client(username, password, log_delay=True):
    stc = SpaceTrackClient(identity=username,
                           password=password)
    if log_delay:
        # Add a call back to the space track client that logs
        # when the client has to sleep to abide by the rate limits
        stc.callback = st_callback
    return stc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch orbit data from space-track.org',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--st_user',
        help='The username for space-track.org',
        type=str,
        required=True
    )
    parser.add_argument(
        '--st_password',
        help='The password for space-track.org',
        type=str,
        required=True
    )
    parser.add_argument(
        '--norad_id_file',
        help=('A text file containing a single NORAD ID on each row to fetch '
              'orbit data for. If no file are passed then orbit data for '
              'all LEO RSOs will be fetched'),
        type=str
    )
    parser.add_argument(
        '--past_n_days',
        help=('The number of days into the past to fetch orbit data for each '
              'RSO, defaults to 30 days'),
        default=DEFAULT_PAST_N_DAYS,
        type=int
    )
    parser.add_argument(
        '--only_latest',
        help='Only fetch the latest TLE for each RSO',
        action='store_true'
    )
    parser.add_argument(
        '--output_path',
        help='The path to save the orbit data parquet file to',
        required=True,
        type=str
    )
    args = parser.parse_args()

    space_track_client = build_space_track_client(args.st_user,
                                                  args.st_password)

    if args.norad_id_file:
        with open(args.norad_id_file) as norad_id_file:
            norad_ids = [l.strip() for l in norad_id_file.readlines()]
    else:
        norad_ids = []

    orbit_data_df = build_leo_df(space_track_client,
                                 norad_ids=norad_ids,
                                 past_n_days=args.past_n_days,
                                 only_latest=args.only_latest)
    logger.info('Serializing data...')
    orbit_data_df.to_parquet(args.output_path)
