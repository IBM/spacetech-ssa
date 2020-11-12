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

import itertools


def get_state_vect_cols(prefix=''):
    """Get the column names of the state vector components with the
    provided `prefix`.

    :param prefix: The prefix that is used in front of the state vector
        components in the column names, examples are `physics_pred` and
        `physics_err` or none
    :type prefix: str

    :return: A list of the 6 names of the prefixed state vector components
    :rtype: [str]
    """
    if prefix:
        prefix += '_'
    vectors = ['r', 'v']
    components = ['x', 'y', 'z']
    col_names = [f'{prefix}{v}_{c}'
                 for v, c
                 in itertools.product(vectors, components)]
    return col_names
