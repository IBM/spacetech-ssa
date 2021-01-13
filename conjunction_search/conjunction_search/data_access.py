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
import ibm_boto3
import pandas as pd
from ibm_botocore.client import Config


def get_dataframe_from_cos():
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
