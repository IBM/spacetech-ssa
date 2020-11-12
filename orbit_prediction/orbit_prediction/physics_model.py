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

# Numeric/ML libraries
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
# Astrodynamics libraries
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation


class PhysicsModel(BaseEstimator):
    """A simple model for orbit propagation implementing the
    scikit-learn API

    :param n_jobs: The number of processors to use in predicting orbtis
    :type n_jobs: int
    """

    def __init__(self, n_jobs=1):
        self.n_jobs = 1

    def _fit_row(self, row):
        """Creates an Orbit object based on the state values in `row`.

        :param row: An array containing the epoch and state vectors to
            instantiate the orbit at
        :type row: np.array

        :return: An Orbit object instantiated at the given state
        :rtype: poliastro.twobody.Orbit
        """
        epoch = Time(row[0], scale='utc')
        r = row[1:4] * u.m
        v = row[4:7] * (u.m / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit

    def fit(self, time_state_vectors):
        """Constructs orbit objects for each row of the provided matrix.

        :param time_state_vectors: A numpy array of shape (n, 7) where each
            row has the following data in these indexes:
                0 - epoch at which to instantiate the orbit
                [1:4] - the position vector, `r`, for the orbit
                [4:7] - the velocity vector, `v`, for the orbit
            This data is then used to create `n` orbit objects.
        :type time_state_vectors: numpy.ndarray
        """
        orbits = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_row)(row)
                                              for row in time_state_vectors)
        self.orbits = orbits
        return self

    def _predict(self, orbit, time_array):
        """Predicts the orbital state vectors for the given orbit for each
        provided time.

        :param orbit: The orbit to predict future state vectors of
        :type orbit: poliastro.twobody.Orbit

        :param time_array: Array of seconds in the future to propagate the
            orbit to
        :type time_array: np.array

        :return: A list of predicted state vectors for each provided time
        :rtype: list
        """
        def propagate(s):
            """Propagates the closed over orbit object to `s` seconds into
            the future.

            :param s: Seconds into the future to propagate the orbit to
            :type s: float

            :return: The predicted state vectors of the orbit at `s` seconds
                in the future
            :rtype: np.array
            """
            prop_orbit = orbit.propagate(s*u.s,
                                         method=cowell,
                                         ad=J2_perturbation,
                                         J2=Earth.J2.value,
                                         R=Earth.R.to(u.km).value)
            prop_r, prop_v = prop_orbit.rv()
            prop_state_vect = np.concatenate(
                [prop_r.to(u.m).to_value(),
                 prop_v.to(u.m/u.s).to_value()]
            )
            return prop_state_vect

        state_vects = [propagate(s) for s in time_array]
        return state_vects

    def predict(self, times):
        """Propagates the fitted orbits to the provided times.

        :param times: A numpy array of shape (n, m) where each row is an array
            of seconds since the fitted epoch to propagate the corresponding
            orbit to
        :type times: numpy.ndarray

        :return: A numpy array of shape (n, m , 6) representing the state
            vectors (`r` and `v`) of the propagated orbits for each time
            in `times`
        :rtype: numpy.ndarray
        """
        orbit_times = zip(self.orbits, times)
        orbit_vects = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(o, t)
                                                   for (o, t) in orbit_times)
        return np.array(orbit_vects)
