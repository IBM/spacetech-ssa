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

import sys
import argparse
import orbit_prediction.pred_orbits as po
import orbit_prediction.spacetrack_etl as etl
import orbit_prediction.ml_model as ml
import orbit_prediction.build_training_data as td


# Functions to run based on the CLI subcommand
COMMAND_FUNCS = {
    'etl': etl.run,
    'build_train_data': td.run,
    'train_models': ml.run,
    'pred_orbits': po.run
}


def add_norad_ids(args):
    """Parses the optional NORAD IDs from a file and adds them to the `args`
    as a list.

    :param args: Arguments parsed from the command line
    :type args: argparse.Namespace
    """
    if args.norad_id_file:
        with open(args.norad_id_file) as norad_id_file:
            args.norad_ids = [line.strip()
                              for line
                              in norad_id_file.readlines()]
    else:
        args.norad_ids = []


def add_etl_parser(subparsers):
    """Adds CLI arguments for running the ETL process"""
    parser = subparsers.add_parser(
        'etl',
        help='Fetch orbit data from space-track.org')
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
              'all LEO ASOs will be fetched'),
        type=str
    )
    parser.add_argument(
        '--last_n_days',
        help=('The number of days into the past to fetch orbit data for each '
              'ASO, defaults to 30 days'),
        default=30,
        type=int
    )
    parser.add_argument(
        '--only_latest',
        help='Only fetch the latest TLE for each ASO',
        action='store_true'
    )
    parser.add_argument(
        '--output_path',
        help='The path to save the orbit data parquet file to',
        required=True,
        type=str
    )


def add_training_data_parser(subparsers):
    """Adds CLI arguments for building a training data set."""
    parser = subparsers.add_parser(
        'build_train_data',
        help=('Uses the data from space-track.org to build a training set of '
              'physical model errors.'))
    parser.add_argument(
        '--input_path',
        required=True,
        help=('The path to the parquet file to load the orbit observation '
              'DataFrame from')
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help=('The path to the parquet file to save the physics model '
              'predictions to')
    )
    parser.add_argument(
        '--last_n_days',
        help='Only use observations from the last `n` days',
        type=int,
        default=30
    )
    parser.add_argument(
        '--n_pred_days',
        help='The number days in the prediction window',
        type=int,
        default=3
    )


def add_ml_parser(subparsers):
    """Adds CLI arguments for training GBRT models."""
    parser = subparsers.add_parser(
        'train_models',
        help=('Train baseline XGBoost models to estimate physical '
              'prediction error'))
    parser.add_argument(
        '--input_path',
        help=('The path to the parquet file containing the physical model '
              'prediction training data'),
        type=str,
        required=True
    )
    parser.add_argument(
        '--use_gpu',
        help='Use a GPU in model training',
        action='store_true'
    )
    parser.add_argument(
        '--out_dir',
        help=('The directory to serialize the models to'),
        type=str,
        required=True
    )


def add_pred_orbits_parser(subparsers):
    """Adds CLI arguments for predicting future orbits."""
    parser = subparsers.add_parser(
        'pred_orbits',
        help='Predict orbits using physical and ML models')
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
        '--ml_model_dir',
        help=('The path to the directory containing the error prediction'
              ' models searilized as JSON'),
        required=True
    )
    parser.add_argument(
        '--norad_id_file',
        help=('A text file containing a single NORAD ID on each row to fetch '
              'orbit data for. If no file are passed then orbit data for '
              'all LEO ASOs will be fetched'),
        type=str
    )
    parser.add_argument(
        '--n_days',
        help='The number of days in the future to make orbit predictions for',
        default=3,
        type=int
    )
    parser.add_argument(
        '--timestep',
        help='The frequency in seconds to make orbit predictions for',
        default=600,
        type=float
    )
    parser.add_argument(
        '--output_path',
        help='The path to save the orbit prediction pickle file to',
        required=True,
        type=str
    )


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description=('Pipeline tools for using machine learning methods to '
                     'estimate error in orbital mechanics models'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add sub-parsers for the various pipeline stages
    subparsers = parser.add_subparsers(help='Available commands',
                                       dest='command')
    add_etl_parser(subparsers)
    add_training_data_parser(subparsers)
    add_ml_parser(subparsers)
    add_pred_orbits_parser(subparsers)

    # Parse the command line arguments and then print the help docs and exit
    # if no sub-command is passed
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # If a NORAD ID file is passed as a parameter, parse the file and the NORAD
    # IDs to the arguments as a list
    if args.command in ['etl', 'pred_orbits']:
        add_norad_ids(args)

    # Get the command to run from the sub-parser and find the function to
    # dispatch the arguments to
    if args.command:
        command_func = COMMAND_FUNCS[args.command]
        command_func(args)


if __name__ == '__main__':
    main()
