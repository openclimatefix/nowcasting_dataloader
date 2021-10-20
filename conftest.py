"""Configure PyTest"""
import os
from pathlib import Path

import pytest

import nowcasting_dataloader
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)

pytest.IMAGE_SIZE_PIXELS = 128

# need to run these to ensure that xarray DataArray and Dataset have torch functions
register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


def pytest_addoption(parser):
    parser.addoption(
        "--use_cloud_data",
        action="store_true",
        default=False,
        help="Use large datasets on GCP instead of local test datasets.",
    )


@pytest.fixture
def use_cloud_data(request):
    return request.config.getoption("--use_cloud_data")


@pytest.fixture
def configuration():
    filename = os.path.join(os.path.dirname(nowcasting_dataloader.__file__), "config", "gcp.yaml")
    configuration = load_yaml_configuration(filename)

    return configuration


@pytest.fixture
def test_data_folder():
    return os.path.join(os.path.dirname(nowcasting_dataloader.__file__), "../tests/data")
