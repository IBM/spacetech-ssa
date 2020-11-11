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
from sklearn.base import BaseEstimator
# Astrodynamics libraries
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.core.perturbations import J2_perturbation


class PhysicsModel(BaseEstimator):
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
        orbits = []
        for row in time_state_vectors:
            epoch = Time(row[0], scale='utc')
            r = row[1:4] * u.m
            v = row[4:7] * (u.m / u.s)
            orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
            orbits.append(orbit)
        self.orbits = orbits

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
        orbit_vects = []
        for orbit, time_array in zip(self.orbits, times):
            state_vects = []
            for elapsed_seconds in time_array:
                prop_orbit = orbit.propagate(elapsed_seconds*u.s,
                                             method=cowell,
                                             ad=J2_perturbation,
                                             J2=Earth.J2.value,
                                             R=Earth.R.to(u.km).value)
                prop_r, prop_v = prop_orbit.rv()
                prop_state_vect = np.concatenate(
                    [prop_r.to(u.m).to_value(),
                     prop_v.to(u.m/u.s).to_value()]
                )
                state_vects.append(prop_state_vect)
            orbit_vects.append(state_vects)
        return np.array(orbit_vects)
