"""Test subselecting functions """
import pytest
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.subset import subselect_data


@pytest.fixture
def configuration():
    """Create Configuration for test"""
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_subselect_date(test_data_folder, configuration):
    """Test subselecting Data"""
    x = Batch.fake(configuration=configuration)

    batch = subselect_data(
        x,
        current_timestep_index=7,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (4, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (4, 5, 64, 64, 10)


#
def test_subselect_date_with_to_dt(test_data_folder, configuration):
    """Test subselecting Data with datetimes"""
    x = Batch.fake(configuration=configuration)

    batch = subselect_data(
        x,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (4, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (4, 5, 64, 64, 10)
