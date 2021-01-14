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

import numpy as np
from sklearn.neighbors import KDTree


def get_orbit_preds(df):
    """Converts the orbit prediction column of the provided
    DataFrame to a numpy array

    :param df: The orbital prediction DataFrame
    :type df: pandas.DataFrame

    :return: The orbtial prediction arrays as a numpy array of shape
             (<num_asos>, <num_prediction_timesteps>, 3)
    :rtype: numpy.array
    """
    orbit_preds = np.stack(df.orbit_preds.values)
    return orbit_preds


def get_aso_idx_from_id(df, aso_id):
    """Converts an ASO UUID to the corresponding array index
    in the orbital prediction numpy array

    :param df: The orbital prediction DataFrame
    :type df: pandas.DataFrame

    :param aso_id: The UUID of the ASO found in the `aso_id`
                   column of the provided `df` DataFrame
    :type aso_id: str

    :return: The corresponding index in the orbital prediction
             numpy array constructed by `get_orbit_preds`
    :rtype: int
    """
    return df.index[df.aso_id == aso_id][0]


def get_aso_id_from_idx(df, aso_idx):
    """Converts an index in the orbital prediction numpy array into the
    UUID for the ASO used in the `aso_id` column of the provided
    `df` DataFrame

    :param df: The orbital prediction DataFrame
    :type df: pandas.DataFrame

    :param aso_idx: The index in the numpy array to lookup the UUID for
    :type aso_idx: int

    :return: The corresponing ASO UUID string for the given numpy index
    :rtype: str
    """
    return df.iloc[aso_idx].aso_id


def build_kd_forest(df, leaf_size=700):
    """Builds a KD-tree for each prediction timestep

    :param df: The orbital prediction DataFrame
    :type df: pandas.DataFrame

    :param leaf_size: Number of points at which to switch to brute-force
        search. The default of 700 was found to be the most optimal in the
        initial analysis, but more optimal values may arise as data
        changes/grows.
    :type leaf_size: int

    :return: A list of data sets and KD-trees for each prediction timestep
    :rtype: [(numpy.array, sklearn.neighbors.KDTree)]
    """
    orbit_preds = get_orbit_preds(df)
    timestep_preds = np.rollaxis(orbit_preds, 1)
    kd_forest = [(X, KDTree(X, metric='euclidean', leaf_size=leaf_size))
                 for X in timestep_preds]
    return kd_forest


def _get_nn_for_timestamp(kd_tree, X, timestep, aso_idx, k, radius):
    """Returns the nearest ASOs to the provided `aso_idx` ASO.  If a `radius`
    is provided then the results are all the ASOs within that given radius,
    otherwise the results are the `k` nearest ASOs.

    :param kd_tree: The KD-tree build for the prediction timestep
    :type kd_tree: sklearn.neighbors.KDTree

    :param X: The numpy array of orbital predictions for each ASO for the
              prediction timestep
    :type X: numpy.array

    :param timestep: The orbital prediction timestep that the `X` array
                     represents
    :type timestep: int

    :param aso_idx: The index in `X` of the ASO to find nearest ASOs for
    :type aso_idx: int

    :param k: The number of nearest ASOs to return. Not used if `radius` is
        passed
    :type k: int

    :param radius: The radius, in meters, to use in determining what is a near
        ASO
    :type radius: float

    :return: A list of tuples representing all ASOs that match the provided
        query where the first value is the index in `X` of the matching ASO,
        the second value is the timestep where this match occurred, and the
        third value is the distance from the query ASO to the matching ASO.
    :rtype: [(int, int, float)]
    """
    query_point = X[aso_idx].reshape(1, -1)
    if radius:
        result_idxs, dists = kd_tree.query_radius(query_point,
                                                  r=radius,
                                                  return_distance=True)
    else:
        dists, result_idxs = kd_tree.query(query_point, k=k+1)

    idx_dists = zip(result_idxs[0], dists[0])
    if radius:
        # Only return results that have non-zero distance
        result = [(int(i), int(timestep), float(d))
                  for i, d in idx_dists
                  if d > 0]
    else:
        # Remove query object from results
        result = [(int(i), int(timestep), float(d))
                  for i, d in idx_dists
                  if i != aso_idx]
    return result


def get_nns_for_object(df, kd_forest, aso_id, k=1, radius=None):
    """Returns the nearest ASOs over all prediction timesteps.

    :param df: The orbital prediction DataFrame
    :type df: pandas.DataFrame

    :param kd_forest: The data and KD-trees for each orbital prediction
        timestep to use to find nearest ASOs.
    :type kd_forest: [(numpy.array, sklearn.neighbors.KDTree)]

    :param aso_idx: The index in `X` of the ASO to find nearest ASOs for
    :type aso_idx: int

    :param k: The number of nearest ASOs to return. Not used if `radius` is
        passed
    :type k: int

    :param radius: The radius, in meters, to use in determining what is a near
        ASO
    :type radius: float

    :return: A list of tuples representing all ASOs that match the provided
        query where the first value is the index in `X` of the matching ASO,
        the second value is the timestep where this match occurred, and the
        third value is the distance from the query ASO to the matching ASO.
    :rtype: [(int, int, float)]
    """
    aso_idx = get_aso_idx_from_id(df, aso_id)
    results = []
    for timestep, (X, kd_tree) in enumerate(kd_forest):
        result = _get_nn_for_timestamp(kd_tree,
                                       X,
                                       timestep,
                                       aso_idx,
                                       k,
                                       radius)
        results += result
    results.sort(key=lambda x: x[2])
    if radius is None:
        results = results[:k]
    results = [(get_aso_id_from_idx(df, i), ts, d) for i, ts, d in results]
    return results
