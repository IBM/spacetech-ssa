def test_get_aso_catalog(etl_client):
    """Tests that the catalog only has the ISS in it."""
    catalog = etl_client.get_leo_aso_catalog()
    assert len(catalog == 1)


def test_build_leo_df(etl_client):
    """Tests that the ETL client builds DataFrame rows for each TLE in the
    full dataset."""
    leo_df = etl_client.build_leo_df(norad_ids=['25544'],
                                     last_n_days=30,
                                     only_latest=False)
    assert len(leo_df) == 3


def test_build_latest_leo_df(etl_client):
    """Tests that the ETL client can build a DataFrame consisting of only the
    most recent TLE data."""
    latest_leo_df = etl_client.build_leo_df(norad_ids=['25544'],
                                            last_n_days=30,
                                            only_latest=True)
    assert len(latest_leo_df) == 1
