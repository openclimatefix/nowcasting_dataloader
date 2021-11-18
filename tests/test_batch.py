"""Test BatchML creation"""
import pytest
import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.fake import FakeDataset


@pytest.fixture
def configuration():
    """Create Configuration object for tests"""
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_batch_to_batch_ml(configuration):
    """Test creating BatchML from Batch"""
    _ = BatchML.from_batch(batch=Batch.fake(configuration=configuration))


def test_fake_dataset(configuration):
    """Test creating fake dataset"""
    train = torch.utils.data.DataLoader(FakeDataset(configuration=configuration), batch_size=None)
    i = iter(train)
    x = next(i)

    x = BatchML(**x)
    # IT WORKS
    assert type(x.satellite.data) == torch.Tensor


def test_fake_dataset_position_encodings(configuration):
    """Test creating fake dataset"""
    train = torch.utils.data.DataLoader(
        FakeDataset(configuration=configuration, add_position_encoding=True), batch_size=None
    )
    i = iter(train)
    x = next(i)
    assert type(x["satellite_position_encoding"]) == torch.Tensor
    x = BatchML(**x)
    # IT WORKS
    assert type(x.satellite.data) == torch.Tensor
