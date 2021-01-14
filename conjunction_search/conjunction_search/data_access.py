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

import io
import os
import requests
import ibm_boto3
import numpy as np
import pandas as pd
from urllib.parse import urljoin
from ibm_botocore.client import Config


def get_orbit_data():
    """Gets the orbital data to drive the UI from various sources based on
    environment variables.

    :return: Predicted orbtial trajectory DataFrame
    :rtype: pandas.DataFrame
    """
    dev = os.environ.get('DEV')
    arcade_url = os.environ.get('ARCADE_URL')
    if dev:
        orbit_df = pd.read_pickle('/app/sample_data/orbit_preds.pickle')
    elif arcade_url:
        arcade_client = ArcadeAPIClient(arcade_url)
        orbit_df = arcade_client.get_orbit_df()
    else:
        orbit_df = get_orbit_data_from_cos()
    return orbit_df


def get_orbit_data_from_cos():
    """Fetches the orbital prediction pandas DataFrame
    from COS.

    :return: The DataFrame containing the orbital predictions for every
             ASO
    :rtype: pandas.DataFrame
    """
    cos_endpoint = os.environ.get('COS_ENDPOINT')
    cos_api_key_id = os.environ.get('COS_API_KEY_ID')
    cos_instance_crn = os.environ.get('COS_INSTANCE_CRN')
    cos_client = ibm_boto3.resource('s3',
                                    ibm_api_key_id=cos_api_key_id,
                                    ibm_service_instance_id=cos_instance_crn,
                                    config=Config(signature_version='oauth'),
                                    endpoint_url=cos_endpoint)
    cos_bucket = os.environ.get('COS_BUCKET')
    cos_filename = os.environ.get('COS_FILENAME')
    df_obj = cos_client.Object(cos_bucket, cos_filename).get()
    df = pd.read_pickle(io.BytesIO(df_obj['Body'].read()))
    return df


class ArcadeAPIClient:
    """A client for getting orbital ephemeris data from the ARCADE API

    :param base_url: The base HTTP endpoint for the ARCADE API
    :type base_url: str
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def get_aso_ids(self):
        """Gets the list of ASO IDs that the ARCADE API knows about.

        :return: The list of ASO IDs
        :rtype: [str]
        """
        aso_url = urljoin(self.base_url, 'asos')
        asos = requests.get(aso_url).json()
        aso_ids = [aso['aso_id'] for aso in asos]
        return aso_ids

    def get_ephemeris(self, aso_id):
        """Uses the interpolation ARCADE endpoint to get the ephemeris data for
        the given ASO at 10 minute time steps.

        :param aso_id: The ID of the ASO to get ephemeris data for
        :type aso_id: str

        :return: The ARCADE ephemeris data for the ASO
        :rtype: dict
        """
        ephemeris_url = urljoin(self.base_url, f'interpolate/{aso_id}')
        ephemeris = requests.get(ephemeris_url,
                                 params={'step_size': 600}).json()
        return ephemeris

    def get_orbit_df(self):
        """Builds a DataFrame of the predicted orbits of all of the ASOs from
        the ARCADE API.

        :return: The orbital prediction data
        :rtype: pandas.DataFrame
        """
        aso_ids = self.get_aso_ids()
        ephems = [self.get_ephemeris(aso_id) for aso_id in aso_ids]
        key_map = [('aso_name', 'object_name'),
                   ('aso_id', 'object_id'),
                   ('pred_start_dt', 'start_time'),
                   ('pred_end_dt', 'stop_time')]
        ephem_records = []
        for ephem in ephems:
            ephem_record = dict()
            for df_key, arcade_key in key_map:
                ephem_record[df_key] = ephem[arcade_key]
                ephem_lines = [el['state_vector'] for el in ephem['ephemeris']]
                # Convert kilometers to meters
                ephem_record['orbit_preds'] = np.array(ephem_lines) * 1000
            ephem_records.append(ephem_record)
        orbit_df = pd.DataFrame(ephem_records)
        # For now, skip rows that have too many observations
        obs_counts = orbit_df.orbit_preds.apply(len)
        obs_mode = obs_counts.mode()[0]
        orbit_df = orbit_df[obs_counts == obs_mode]
        # Convert timestamps
        time_cols = ['pred_start_dt', 'pred_end_dt']
        for time_col in time_cols:
            orbit_df[time_col] = pd.to_datetime(orbit_df[time_col].mode()[0])
        orbit_df.reset_index(inplace=True, drop=True)
        return orbit_df
