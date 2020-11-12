import numpy as np
from datetime import datetime
from orbit_prediction.physics_model import PhysicsModel


def test_orbit_model_preds():
    """Tests that the physics model prediction data is the right shape"""
    epoch = datetime.utcnow()
    r = np.array([1, 2, 3])
    v = np.array([0.5, 0.6, 0.7])
    state_vect = np.concatenate((np.array([epoch]), r, v))
    orbit_model = PhysicsModel()
    orbit_model.fit([state_vect])
    pred_times = [np.array([10, 20, 30, 40])]
    orbit_preds = orbit_model.predict(pred_times)
    assert orbit_preds.shape == (1, 4, 6)
