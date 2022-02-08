""" Test for SatelliteML"""
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.fake.batch import satellite_fake

from nowcasting_dataloader.data_sources.satellite.satellite_model import SatelliteML
from nowcasting_dataset.config.model import Configuration


def test_satellite_to_ml():
    """Test satellite ml can be made from Satellite"""
    sat = satellite_fake()

    _ = SatelliteML.from_xr_dataset(sat)


def test_satellite_normalization():
    configuration = Configuration()
    configuration.input_data = Configuration().input_data.set_all_to_defaults()
    configuration.input_data.satellite.satellite_channels \
        = configuration.input_data.satellite.satellite_channels[1:11]
    
    sat = satellite_fake(configuration=configuration)

    batch = SatelliteML.from_xr_dataset(sat)
    batch.channels = SAT_VARIABLE_NAMES[1:11]
    batch.normalize()
