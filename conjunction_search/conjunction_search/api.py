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
import pandas as pd
from flask import Flask, request, jsonify, render_template
from conjunction_search.czml import CZMLBuilder
from conjunction_search.data_access import get_orbit_data
from conjunction_search.conjunction_search import (get_nns_for_object,
                                                   build_kd_forest)

app = Flask(__name__)

cesium_api_key = os.environ.get('CESIUM_API_KEY', '')


# Get orbit data
orbit_df = get_orbit_data()
# Build the KD-trees for each prediction timestep
kd_forest = build_kd_forest(orbit_df)


def _make_json_resp(conj_results):
    """Converts the conjunction search results into a
    JSON response payload.

    :param conj_results: The conjunction search results
    :type conj_results: [(int, int, float)]

    :return: A JSON list of the search results where each dictionary
             contains the UUID of the matching ASO, the distance to
             the query ASO, and the prediction timestep where the query
             matches.
    """
    keys = ('aso_id', 'timestep', 'distance')
    json_results = [dict(zip(keys, r)) for r in conj_results]
    return jsonify(json_results)


def _make_czml_resp(aso_id, conj_results):
    """Converts the conjunction search results to a CZML
    document to be used to display the orbits of the
    ASOs in Cesium

    :param aso_id: The ID of the ASO that the conjunction search
                   query was run for.
    :type aso_id: str

    :param conj_results: The conjunction search results
    :type conj_results: [(int, int, float)]

    return: A JSON CZML document describing the orbits of the
            conjunction search ASOs.
    """
    # Build a pandas DataFrame out of the conjunction search results
    conj_df = pd.DataFrame(data=conj_results,
                           columns=['aso_id', 'conj_timesteps', 'conj_dists'])
    # Gather conjunction timesteps and distances by aso_id
    conj_df = conj_df.pivot_table(index=['aso_id'],
                                  aggfunc=lambda x: list(x))
    conj_df.reset_index(inplace=True)
    if conj_df.empty:
        return jsonify([])
    # Select the query ASO and the conjunction search matches
    # from the orbital prediction DataFrame
    aso_ids = [aso_id] + [r[0] for r in conj_results]
    asos_df = orbit_df[orbit_df.aso_id.isin(aso_ids)].reset_index()
    # Join the conjunction search and selected orbital prediction
    # DataFrames into one DataFrame
    czml_df = asos_df.merge(conj_df, how='left', on='aso_id')
    # Build and return the CZML document
    czml_builder = CZMLBuilder(czml_df)
    czml_doc = czml_builder.build_czml_doc()
    return jsonify(czml_doc)


@app.route('/', methods=['GET'])
def ui():
    asos = orbit_df[['aso_name', 'aso_id']].to_dict(orient='records')
    return render_template('index.html',
                           asos=asos,
                           cesium_api_key=cesium_api_key)


@app.route('/conjunction_search/<aso_id>', methods=['GET'])
def conjunction_search(aso_id):
    """Flask route for querying the KD-trees to find nearest
    ASOs

    :param aso_id: The UUID of the ASO to find nearest neighbors
                   for.  This value should be the UUID in the `aso_id`
                   column of the `orbit_df` DataFrame.
    :type aso_id: str

    *Query Parameters*
    :param k: The number of nearest neighbor results to return
    :type k: int

    :param radius: Instead of returning the k nearest neighbors,
                   return all ASOs that are within the given radius
                   for each prediction timestep.
    :type radius: float

    """
    k = request.args.get('k', 1, type=int)
    search_radius = request.args.get('radius', type=float)
    results = get_nns_for_object(orbit_df,
                                 kd_forest,
                                 aso_id,
                                 k=k,
                                 radius=search_radius)
    if request.content_type == 'application/json':
        resp = _make_json_resp(results)
    else:
        resp = _make_czml_resp(aso_id, results)
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
