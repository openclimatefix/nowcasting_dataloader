from nowcasting_dataloader.datamodules import SatFlowDataModule
import pytest


@pytest.mark.skip(
    "Nowcasting-dataset is changing a lot with https://github.com/openclimatefix/nowcasting_dataset/issues/213, so skipping this failing test for now"
)
def test_datamodule_subsetting():
    dataset = SatFlowDataModule(fake_data=True, configuration_filename="tests/config/test.yaml")
    dataset.setup()
    train_dset = dataset.train_dataloader()
    sample, target = next(iter(train_dset))
    assert sample["sat_data"].shape == (32, 7, 16, 16, 12)
    assert target["sat_data"].shape == (32, 48, 16, 16, 12)
