"""Test position encoding"""
import pytest
import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.utils.optical_flow import compute_optical_flow_for_batch


@pytest.fixture
def configuration():
    """Create configuration object"""
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 32
    return con


def test_batch_encoding(configuration):
    """Test batch encoding"""
    batch: Batch = Batch.fake(configuration=configuration)
    optical_flow = compute_optical_flow_for_batch(batch, final_image_size_pixels=32)
    assert "optical_flow" in optical_flow.keys()
    predictions = optical_flow["optical_flow"]
    assert predictions.shape == (32, 12, 12, 32, 32)
    assert torch.isfinite(predictions).all()
