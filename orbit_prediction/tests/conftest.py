import pytest
from orbit_prediction.spacetrack_etl import SpaceTrackETL

TLES = [
    '0 ISS (ZARYA)\n',
    '1 25544U 98067A   20287.09347615  .00000576  00000-0  18428-4 0  9999\n',
    '2 25544  51.6446 125.0696 0001433  27.0371 118.9673 15.49299288250298\n',
    '0 ISS (ZARYA)\n',
    '1 25544U 98067A   20287.40118223  .00000617  00000-0  19152-4 0  9996\n',
    '2 25544  51.6447 123.5470 0001406  28.2704  35.0912 15.49300072250340\n',
    '0 ISS (ZARYA)\n',
    '1 25544U 98067A   20287.53670015  .00000762  00000-0  21766-4 0  9997\n',
    '2 25544  51.6440 122.8761 0001413  26.5472  73.1662 15.49301208250368)'
]


class MockSpaceTrackClient:
    """A mock space-track.org API client."""

    def satcat(self, *args, **kwargs):
        """Returns the catalog data for the ISS."""
        catalog = [
            {
                'INTLDES': '1998-067A',
                'NORAD_CAT_ID': '25544',
                'OBJECT_TYPE': 'PAYLOAD',
                'SATNAME': 'ISS (ZARYA)',
                'COUNTRY': 'ISS',
                'LAUNCH': '1998-11-20',
                'SITE': 'TTMTR',
                'DECAY': None,
                'PERIOD': '92.94',
                'INCLINATION': '51.65',
                'APOGEE': '420',
                'PERIGEE': '417',
                'COMMENT': None,
                'COMMENTCODE': None,
                'RCSVALUE': '0',
                'RCS_SIZE': 'LARGE',
                'FILE': '7478',
                'LAUNCH_YEAR': '1998',
                'LAUNCH_NUM': '67',
                'LAUNCH_PIECE': 'A',
                'CURRENT': 'Y',
                'OBJECT_NAME': 'ISS (ZARYA)',
                'OBJECT_ID': '1998-067A',
                'OBJECT_NUMBER': '25544'
            }
        ]
        return catalog

    def tle_latest(self, *args, **kwargs):
        """Returns the most recent TLE for the ISS."""
        latest_tle = TLES[-3:]
        return ''.join(latest_tle)

    def tle(self, *args, **kwargs):
        """Returns all TLEs for the ISS"""
        tles = ''.join(TLES)
        return tles


@pytest.fixture
def etl_client():
    """Builds an ETL client using the mock API client."""
    stc = MockSpaceTrackClient()
    return SpaceTrackETL(stc)
