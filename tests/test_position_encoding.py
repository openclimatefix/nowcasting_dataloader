"""Test position encoding"""
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.utils.position_encoding import (
    combine_space_and_time_features,
    create_datetime_features,
    determine_shape_of_encoding,
    encode_absolute_position,
    encode_modalities,
    encode_year,
    generate_position_encodings_for_batch,
    normalize_geospatial_coordinates,
)


@pytest.fixture
def configuration():
    """Create configuration object"""
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 32
    return con


@pytest.mark.parametrize(
    ["key", "expected_shape"],
    [
        ("nwp", [32, 17, 19, 64, 64]),
        ("satellite", [32, 10, 19, 64, 64]),
        ("topographic", [32, 1, 1, 64, 64]),
        ("pv", [32, 128, 19, 128]),
        ("gsp", [32, 32, 4, 32]),
    ],
)
def test_shape_encoding(key, expected_shape, configuration):
    """Test shape encoding creation"""
    batch: Batch = Batch.fake(configuration=configuration)
    xr_dataset = getattr(batch, key)
    shape = determine_shape_of_encoding(xr_dataset)
    assert shape == expected_shape


def test_batch_encoding(configuration):
    """Test batch encoding"""
    batch: Batch = Batch.fake(configuration=configuration)
    position_encodings = generate_position_encodings_for_batch(
        batch,
        num_bands=16,
    )
    for key in ["nwp", "satellite", "topographic", "gsp", "pv"]:
        position_encoding_key = key + "_position_encoding"
        assert position_encoding_key in position_encodings.keys()
        assert torch.isfinite(position_encodings[position_encoding_key]).all()
        assert torch.min(position_encodings[position_encoding_key]) >= -1.0
        assert torch.max(position_encodings[position_encoding_key]) <= 1.0


def get_data(batch_size: int = 12, interval="5min", spatial_size: int = 64):
    """Create data for tests"""
    datetimes = []
    for month in range(1, batch_size):
        datetimes.append(
            pd.date_range(
                start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 01:00", freq=interval
            )
        )
    datetimes.append(pd.date_range(start="2020-12-31 22:55", end="2020-12-31 23:55", freq=interval))
    geospatial_bounds = {"x_min": -2900.0, "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = np.sort(np.random.rand(batch_size, spatial_size) * 9)
    y = np.sort(np.random.rand(batch_size, spatial_size) * 120)
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    return datetimes, geospatial_bounds, geospatial_coordinates


def test_datetime_feature_creation():
    """Test datetime feature creation"""
    # Generate fake datetime data for the whole year
    datetimes = []
    for month in range(1, 12):
        datetimes.append(
            pd.date_range(start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 06:00", freq="5min")
        )
    datetimes.append(pd.date_range(start="2020-12-31 17:55", end="2020-12-31 23:55", freq="5min"))
    datetime_features = create_datetime_features(datetimes)
    assert len(datetime_features) == 2
    for feature in datetime_features:
        assert feature.size() == (12, 73, 13)
        assert torch.min(feature) >= -1.0
        assert torch.max(feature) <= 1.0


def test_encode_year_fourier():
    """Test year fourier encoding"""
    datetimes = []
    for year in range(2016, 2022):
        datetimes.append(
            pd.date_range(start=f"{year}-01-01 00:00", end=f"{year}-01-01 01:00", freq="5min")
        )
    datetimes.append(pd.date_range(start="2020-12-31 22:55", end="2020-12-31 23:55", freq="5min"))
    year_encoding = encode_year(
        datetimes,
        time_range=(
            datetime.datetime(year=2016, month=1, day=1),
            datetime.datetime(year=2022, month=12, day=31),
        ),
    )
    assert year_encoding.size() == (7, 1, 25)
    assert torch.min(year_encoding) >= -1.0
    assert torch.max(year_encoding) <= 1.0


def test_geospatial_normalization():
    """Test geospatial normalization"""
    geospatial_bounds = {"x_min": -2900.0, "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = np.sort(np.random.rand(32, 128) * 9)
    y = np.sort(np.random.rand(32, 128) * 120)
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    normalized_coordinates = normalize_geospatial_coordinates(
        geospatial_coordinates, geospatial_bounds=geospatial_bounds, max_freq=64, num_bands=128
    )
    assert normalized_coordinates.size() == (32, 128, 128, 514)
    assert torch.min(normalized_coordinates) >= -1.0
    assert torch.max(normalized_coordinates) <= 1.0


def test_encode_absolute_position():
    """Test encoding absolute position"""
    datetimes, geospatial_bounds, geospatial_coordinates = get_data()
    absolute_position_encoding = encode_absolute_position(
        shape=(12, 5, 13, 64, 64),
        geospatial_bounds=geospatial_bounds,
        geospatial_coordinates=geospatial_coordinates,
        datetimes=datetimes,
        time_range=(
            datetime.datetime(year=2016, month=1, day=1),
            datetime.datetime(year=2021, month=12, day=31),
        ),
        num_bands=32,
    )
    assert absolute_position_encoding.size() == (12, 181, 13, 64, 64)
    assert torch.min(absolute_position_encoding) >= -1.0
    assert torch.max(absolute_position_encoding) <= 1.0


def test_combine_space_and_time_features():
    """Test combining space and time features"""
    space_features = torch.randn(32, 66, 10, 64, 64)
    time_features = [torch.randn(32, 10, 1) for _ in range(4)]
    shape = [32, 5, 10, 64, 64]
    combined_encoding = combine_space_and_time_features(
        spatial_features=space_features, datetime_features=time_features, shape=shape
    )
    assert torch.isfinite(combined_encoding).all()
    assert combined_encoding.shape == (
        32,
        70,
        10,
        64,
        64,
    )  # 70 from the 66 spatial channels, and 4 temporal ones


def test_encode_modalities():
    """Test encoding of modalities"""
    datetimes, geospatial_bounds, geospatial_coordinates = get_data()
    encoded_position = encode_modalities(
        modalities_to_encode={"NWP": torch.randn(12, 10, 13, 64, 64)},
        datetimes={"NWP": datetimes},
        geospatial_coordinates={"NWP": geospatial_coordinates},
        geospatial_bounds=geospatial_bounds,
        time_range=(
            datetime.datetime(year=2016, month=1, day=1),
            datetime.datetime(year=2021, month=12, day=31),
        ),
        max_freq=128,
        num_bands=32,
    )
    assert "NWP" in encoded_position.keys()
    assert "NWP_position_encoding" in encoded_position.keys()
    assert encoded_position["NWP_position_encoding"].size() == (12, 181, 13, 64, 64)
    combined = torch.cat(
        [encoded_position["NWP"], encoded_position["NWP_position_encoding"]], dim=1
    )
    assert combined.size() == (12, 191, 13, 64, 64)


def test_encode_multiple_modalities():
    """Test encoding multiple modalities"""
    datetimes, geospatial_bounds, geospatial_coordinates = get_data()
    sat_datetimes, geospatial_bounds, _ = get_data(interval="30min", batch_size=12)
    pv_datetimes, geospatial_bounds, pv_geospatial_coordinates = get_data(
        batch_size=12, interval="15min", spatial_size=1
    )  # PV systems have x, y coord for their location, but not spatial extant
    encoded_position = encode_modalities(
        modalities_to_encode={
            "NWP": torch.randn(12, 10, 13, 64, 64),
            "Sat": torch.randn(12, 12, 3, 64, 64),
            "PV": torch.randn(12, 1, 5, 1, 1),
        },
        datetimes={"NWP": datetimes, "Sat": sat_datetimes, "PV": pv_datetimes},
        geospatial_coordinates={
            "NWP": geospatial_coordinates,
            "Sat": deepcopy(
                geospatial_coordinates
            ),  # Otherwise the coordinates get overwritten and fail the checks
            "PV": pv_geospatial_coordinates,
        },
        geospatial_bounds=geospatial_bounds,
        time_range=(
            datetime.datetime(year=2016, month=1, day=1),
            datetime.datetime(year=2021, month=12, day=31),
        ),
        max_freq=128,
        num_bands=32,
    )
    assert "NWP" in encoded_position.keys()
    assert "NWP_position_encoding" in encoded_position.keys()
    assert encoded_position["NWP_position_encoding"].size() == (12, 181, 13, 64, 64)
    assert "Sat" in encoded_position.keys()
    assert "Sat_position_encoding" in encoded_position.keys()
    assert encoded_position["Sat_position_encoding"].size() == (12, 181, 3, 64, 64)
    assert "PV" in encoded_position.keys()
    assert "PV_position_encoding" in encoded_position.keys()
    assert encoded_position["PV_position_encoding"].size() == (12, 181, 5, 1, 1)

    # Check that time and space features match for NWP and Sat when the times line up
    assert np.all(
        np.isclose(
            encoded_position["NWP_position_encoding"][:, :, 0, :, :],
            encoded_position["Sat_position_encoding"][:, :, 0, :, :],
        )
    )
    assert np.all(
        np.isclose(
            encoded_position["NWP_position_encoding"][:, :, -1, :, :],
            encoded_position["Sat_position_encoding"][:, :, -1, :, :],
        )
    )
    assert np.all(
        np.isclose(
            encoded_position["NWP_position_encoding"][:, :, 6, :, :],
            encoded_position["Sat_position_encoding"][:, :, 1, :, :],
        )
    )
    assert not np.all(
        np.isclose(
            encoded_position["NWP_position_encoding"][:, :, 7, :, :],
            encoded_position["Sat_position_encoding"][:, :, 1, :, :],
        )
    )
    # Check that the time and space features are the same for the start and end of them
    # Time channels are the last 4 in the encoding, so check those,
    # first and last time values should be the same
    # Intermediate ones should vary
    for i in range(130, 135):
        assert np.all(
            np.isclose(
                encoded_position["PV_position_encoding"][:, i, 0, 0, 0],
                encoded_position["NWP_position_encoding"][:, i, 0, 0, 0],
            )
        )
        assert np.all(
            np.isclose(
                encoded_position["NWP_position_encoding"][:, i, 0, 0, 0],
                encoded_position["Sat_position_encoding"][:, i, 0, 0, 0],
            )
        )
        assert np.all(
            np.isclose(
                encoded_position["PV_position_encoding"][:, i, -1, 0, 0],
                encoded_position["NWP_position_encoding"][:, i, -1, 0, 0],
            )
        )
        assert np.all(
            np.isclose(
                encoded_position["NWP_position_encoding"][:, i, -1, 0, 0],
                encoded_position["Sat_position_encoding"][:, i, -1, 0, 0],
            )
        )
        assert np.all(
            np.isclose(
                encoded_position["PV_position_encoding"][:, i, 2, 0, 0],
                encoded_position["NWP_position_encoding"][:, i, 6, 0, 0],
            )
        )
        assert np.all(
            np.isclose(
                encoded_position["NWP_position_encoding"][:, i, 6, 0, 0],
                encoded_position["Sat_position_encoding"][:, i, 1, 0, 0],
            )
        )
