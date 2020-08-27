#!/usr/bin/env bash

if [[ -z "$ST_USER" ]]; then
    echo "Must provide ST_USER in environment" 1>&2
    exit 1
fi

if [[ -z "$ST_PASSWORD" ]]; then
    echo "Must provide ST_PASSWORD in environment" 1>&2
    exit 1
fi

# Make temporary directory for storing files
rm -rf /tmp/ssa_test
mkdir /tmp/ssa_test
mkdir /tmp/ssa_test/err_models

echo "Downloading Data from USSTRATCOM.."
python orbit_prediction/spacetrack_etl.py --st_user $ST_USER \
       --st_password $ST_PASSWORD \
       --norad_id_file sample_data/test_norad_ids.txt \
       --output_path /tmp/ssa_test/usstratcom_data.parquet

echo "Creating Physics Model Prediction/Error Training Data..."
python orbit_prediction/physics_model.py --input_path /tmp/ssa_test/usstratcom_data.parquet \
       --output_path /tmp/ssa_test/physics_preds.parquet

echo "Training XGBoost Models..."
python orbit_prediction/pred_physics_err.py --input_path /tmp/ssa_test/physics_preds.parquet \
       --out_dir /tmp/ssa_test/err_models

echo "Predicting Orbits using Physics/ML Hybrid Model..."
python orbit_prediction/pred_orbits.py --st_user $ST_USER \
           --st_password $ST_PASSWORD \
           --ml_model_dir /tmp/ssa_test/err_models \
           --norad_id_file sample_data/test_norad_ids.txt \
           --output_path /tmp/ssa_test/orbit_preds.pickle

echo "Pipeline Complete!"
