from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataloader.subset import subselect_data


def test_subselect_date(test_data_folder):

    x = Batch.fake()

    batch = subselect_data(
        x,
        current_timestep_index=7,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (32, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (32, 5, 64, 64, 10)


#
def test_subselect_date_with_to_dt(test_data_folder):

    x = Batch.fake()

    batch = subselect_data(
        x,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (32, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (32, 5, 64, 64, 10)
