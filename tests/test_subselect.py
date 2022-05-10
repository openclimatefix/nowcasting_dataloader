"""Test subselecting functions """
import pytest
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.subset import subselect_data


def test_subselect_date(test_data_folder, configuration):
    """Test subselecting Data"""
    x = Batch.fake(configuration=configuration)

    batch = subselect_data(
        x,
        current_timestep_index=7,
        history_minutes=60,
        forecast_minutes=60,
    )

    assert batch.satellite.data.shape == (4, 19, 11, 21, 21)
    assert batch.nwp.data.shape == (4, 2, 64, 64, 10)
    assert batch.pv.x_osgb.shape == (4, 128)


@pytest.mark.skip("Broken test: bug #63")
def test_subselect_date_with_to_dt(test_data_folder, configuration):
    """Test subselecting Data with datetimes"""
    x = Batch.fake(configuration=configuration)

    batch = subselect_data(
        x,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (4, 5, 64, 64, 10)
    assert batch.nwp.data.shape == (4, 5, 64, 64, 17)
