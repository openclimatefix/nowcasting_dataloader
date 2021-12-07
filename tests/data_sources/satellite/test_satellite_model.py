""" Test for SatelliteML"""
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES

from nowcasting_dataset.data_sources.fake import satellite_fake

from nowcasting_dataloader.data_sources.satellite.satellite_model import SatelliteML


def test_satellite_to_ml():
    """Test satellite ml can be made from Satellite"""
    sat = satellite_fake()

    _ = SatelliteML.from_xr_dataset(sat)


def test_satellite_normalization():
    sat = satellite_fake()

    batch = SatelliteML.from_xr_dataset(sat)
    batch.channels = SAT_VARIABLE_NAMES[1:7]
    batch.normalize()

