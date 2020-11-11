.PHONY: all venv test install clean

VENV := venv
PYTHON := ${VENV}/bin/python3

all: venv

venv: $(VENV)/bin/activate
$(VENV)/bin/activate: setup.py
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PYTHON) -m pip install -U pip wheel
	$(PYTHON) -m pip install -e .

dev_venv: dev_requirements.txt venv
	$(PYTHON) -m pip install -r dev_requirements.txt

test: dev_venv
	${PYTHON} -m pytest --cache-clear --flake8

install:
	python3 -m pip install .

clean:
	rm -rf venv
	rm -rf orbit_prediction.egg-info
	find . -type f -name ‘*.pyc’ -delete
