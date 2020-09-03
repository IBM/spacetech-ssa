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


.PHONY : login build run run_dev push clean

build:
	docker build -t ssa-conj-search:latest .

run: build
	docker run --rm -i -t \
        -p 8080:8080 \
        --env-file=vars.env \
        --name ssa-conj-search \
        ssa-conj-search:latest

clean:
	docker rmi ssa-conj-search:latest
