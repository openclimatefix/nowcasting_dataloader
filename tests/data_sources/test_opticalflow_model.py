""" Test for OpticalFLow"""
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.fake.batch import optical_flow_fake

from nowcasting_dataloader.data_sources.opticalflow.opticalflow_model import OpticalFlowML


def test_satellite_to_ml():
    """Test satellite ml can be made from Satellite"""
    opticalflow = optical_flow_fake()

    _ = OpticalFlowML.from_xr_dataset(opticalflow)


def test_satellite_normalization():
    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.opticalflow.opticalflow_channels = SAT_VARIABLE_NAMES[5:11]

    opticalflow = optical_flow_fake(configuration=configuration)

    opticalflow = OpticalFlowML.from_xr_dataset(opticalflow)

    opticalflow.normalize()
