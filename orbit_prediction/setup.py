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

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='orbit_prediction',
      version='0.0.1',
      author='Colin Alstad',
      author_emai='colin.alstad@ibm.com',
      description=('A pipeline for experimenting with ML methods to estimate '
                   'error in orbital mechanics models'),
      long_description=long_description,
      license='Apache 2.0',
      url='https://ibm.github.io/spacetech-ssa/',
      packages=find_packages(),
      python_requires='>=3.8, <4',
      install_requires=['astropy',
                        'joblib',
                        'matplotlib',
                        'numpy',
                        'pandas',
                        'poliastro',
                        'pyarrow',
                        'scikit-learn',
                        'spacetrack',
                        'TLE-tools',
                        'tqdm',
                        'xgboost'],
      entry_points={
          'console_scripts': [
              'orbit_pred=orbit_prediction.cli:main']}
      )
