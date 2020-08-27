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
import py2neo
import logging
import argparse
import pandas as pd


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def query_astria_graph(graph, query_path):
    """Runs the Cypher query against the provided Neo4j graph database.

    :param graph: The Neo4j graph database connection to query
    :type graph: py2neo.Graph

    :param query_path: The path to the file that contains the Cypher query
    :type query_path: str

    :return: The results of the Cypher query as a pandas DataFrame
    :rtype: pandas.DataFrame
    """
    with open(query_path) as query_file:
        astria_query = query_file.read().strip()
        results_df = graph.run(astria_query).to_data_frame()
    return results_df


def transfrom_astria_data(df):
    """Runs basic column transformations on the DataFrame extracted from
    Neo4j.

    :param df: The Neo4j query results DataFrame
    :type df: pandas.DataFrame

    :return: The transformed DataFrame
    :rtype: pandas.DataFrame
    """
    # Convert `epoch` from a neo4j date to a pandas timestamp
    df['epoch'] = pd.to_datetime(df.epoch.apply(lambda e: e.to_native()))
    # Expand the cartesian state vectors into their components
    state_vect_comps = ['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z']
    df[state_vect_comps] = df.state_vect.apply(lambda sv: pd.Series(sv))
    # Drop the state vector column now that all the components are their
    # own columns
    df.drop('state_vect', inplace=True, axis=1)
    return df


def mode(s):
    """Calculates the mode of a pandas Series.  Returns the first mode
    value if there are multiple modes for the Series.
    :param s: The Series to calculate the mode for
    :type s: pandas.Series
    """
    m = s.mode()
    if not m.empty:
        return m.iat[0]


def agg_epoch_obs(df):
    """Aggregate all of the observations for each RSO for each epoch into a
    single observation per RSO per epoch.

    :param df: The DataFrame to aggregate epoch observations for each RSO
    :type df: pandas.DataFrame

    :return: The DataFrame of the aggregated epoch observations
    :rtype: pandas.DataFrame
    """
    # Average observation state vectors per RSO per epoch.  This is the
    # most naive way to solve the "data fusion" problem.
    agg_funcs = {
        'catalog_id': mode,
        'rso_name': mode,
        'r_x': 'mean',
        'r_y': 'mean',
        'r_z': 'mean',
        'v_x': 'mean',
        'v_y': 'mean',
        'v_z': 'mean'
    }
    agg_values = list(agg_funcs.keys())
    agg_df = df.pivot_table(values=agg_values,
                            index=['rso_id', 'epoch'],
                            aggfunc=agg_funcs)

    agg_df.reset_index(inplace=True)
    return agg_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETL orbit data out of ASTRIAGraph.')
    parser.add_argument('--neo4j_host',
                        help='The host URL for th Neo4j database',
                        type=str,
                        required=True)
    parser.add_argument('--neo4j_scheme',
                        help='The transport protocol to connect to Neo4j with',
                        type=str,
                        default='bolt')
    parser.add_argument('--neo4j_port',
                        help='The port to connect to Neo4j on',
                        type=int,
                        default=7687)
    parser.add_argument('--neo4j_user',
                        help='The Neo4j username',
                        type=str,
                        default='neo4j')
    parser.add_argument('--neo4j_pass',
                        help='The Neo4j password',
                        type=str)
    parser.add_argument('--cypher_path',
                        help='The path to the Cypher query file',
                        type=str,
                        required=True)
    parser.add_argument('--out_path',
                        help='The path to save the DataFrame parquet file to',
                        type=str,
                        required=True)
    args = parser.parse_args()

    neo4j_auth = (args.neo4j_user, args.neo4j_pass)
    neo4j_graph = py2neo.Graph(host=args.neo4j_host,
                               scheme=args.neo4j_scheme,
                               port=args.neo4j_port,
                               auth=neo4j_auth)
    logger.info('Fetching data from Neo4j...')
    df = query_astria_graph(neo4j_graph, args.cypher_path)
    logger.info('Transforming Columns...')
    df = transfrom_astria_data(df)
    logger.info('Aggregating Measurements...')
    df = agg_epoch_obs(df)
    logger.info('Serializing data...')
    df.to_parquet(args.out_path)
