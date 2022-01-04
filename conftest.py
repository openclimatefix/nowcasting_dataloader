"""Configure PyTest"""
import os

import nowcasting_dataset
import pytest
from nowcasting_dataset.config.model import Configuration, InputData

import nowcasting_dataloader
from nowcasting_dataloader.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)

pytest.IMAGE_SIZE_PIXELS = 128

# need to run these to ensure that xarray DataArray and Dataset have torch functions
register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


def pytest_addoption(parser):
    """
    Setup pytest for cloud data

    Args:
        parser: Parser

    """
    parser.addoption(
        "--use_cloud_data",
        action="store_true",
        default=False,
        help="Use large datasets on GCP instead of local test datasets.",
    )


@pytest.fixture
def use_cloud_data(request):
    """Whether to use cloud data"""
    return request.config.getoption("--use_cloud_data")


@pytest.fixture
def configuration():
    
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.input_data.pv.n_pv_systems_per_example = 128
    c.process.batch_size = 4

    return c


@pytest.fixture
def test_data_folder():
    """Get the test data folder"""
    return os.path.join(os.path.dirname(nowcasting_dataloader.__file__), "../tests/data")
