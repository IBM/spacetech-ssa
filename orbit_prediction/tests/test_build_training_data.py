from orbit_prediction.build_training_data import predict_orbits


def test_build_traiing_data(etl_client):
    """Tests that the training data set is built correctly from the
    mock data"""
    leo_df = etl_client.build_leo_df(norad_ids=['25544'],
                                     last_n_days=30,
                                     only_latest=False)
    training_data = predict_orbits(leo_df, 30, 3)
    assert len(training_data) == 2
