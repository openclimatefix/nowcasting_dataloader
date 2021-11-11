"""Configure PyTest"""
import os

import nowcasting_dataset
import pytest
from nowcasting_dataset.config.load import load_yaml_configuration

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
    """Get the GCP configuration and return it"""
    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
    configuration = load_yaml_configuration(filename)

    return configuration


@pytest.fixture
def test_data_folder():
    """Get the test data folder"""
    return os.path.join(os.path.dirname(nowcasting_dataloader.__file__), "../tests/data")
