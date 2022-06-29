"""Test BatchML creation"""
import pytest
import torch
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.fake import FakeDataset


def test_batch_to_batch_ml(configuration):
    """Test creating BatchML from Batch"""
    _ = BatchML.from_batch(batch=Batch.fake(configuration=configuration))


def test_batch_to_batch_ml_normalize(configuration):
    """Test creating BatchML from Batch and normalizing the data"""
    batch = BatchML.from_batch(batch=Batch.fake(configuration=configuration))
    batch.normalize()


def test_fake_dataset(configuration):
    """Test creating fake dataset"""
    train = torch.utils.data.DataLoader(FakeDataset(configuration=configuration), batch_size=None)
    i = iter(train)
    x = next(i)

    x = BatchML(**x)
    # IT WORKS
    assert type(x.satellite.data) == torch.Tensor
