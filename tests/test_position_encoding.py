import numpy as np

from nowcasting_dataloader.utils.position_encoding import (
    encode_modalities,
    encode_absolute_position,
    create_datetime_features,
    normalize_geospatial_coordinates,
    generate_position_encodings_for_batch,
)
from nowcasting_dataset.dataset.batch import Batch
import pytest
import pandas as pd
import torch
from copy import deepcopy


def test_batch_encoding():
    batch: Batch = Batch.fake()
    position_encodings = generate_position_encodings_for_batch(
        batch,
        max_freq=64,
        num_bands=16,
    )
    print(position_encodings.keys())


def get_data(batch_size: int = 12, interval="5min", spatial_size: int = 64):
    datetimes = []
    for month in range(1, batch_size):
        datetimes.append(
            pd.date_range(
                start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 01:00", freq=interval
            )
        )
    datetimes.append(
        pd.date_range(start=f"2020-12-31 22:55", end=f"2020-12-31 23:55", freq=interval)
    )
    geospatial_bounds = {"x_min": -2900.0, "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = torch.sort(torch.rand(batch_size, spatial_size) * 9)[0]
    y = torch.sort(torch.rand(batch_size, spatial_size) * 120, descending=True)[0]
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    return datetimes, geospatial_bounds, geospatial_coordinates


def test_datetime_feature_creation():
    # Generate fake datetime data for the whole year
    datetimes = []
    for month in range(1, 12):
        datetimes.append(
            pd.date_range(start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 06:00", freq="5min")
        )
    datetimes.append(pd.date_range(start=f"2020-12-31 17:55", end=f"2020-12-31 23:55", freq="5min"))
    datetime_features = create_datetime_features(datetimes)
    assert len(datetime_features) == 4
    for feature in datetime_features:
        assert feature.size() == (12, 73)
        assert torch.min(feature) >= -1.0
        assert torch.max(feature) <= 1.0


def test_geospatial_normalization():
    geospatial_bounds = {"x_min": -2900.0, "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = torch.sort(torch.rand(32, 128) * 9)[0]
    y = torch.sort(torch.rand(32, 128) * 120, descending=True)[0]
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    normalized_coordinates = normalize_geospatial_coordinates(
        geospatial_coordinates, geospatial_bounds=geospatial_bounds, max_freq=64, num_bands=128
    )
    assert normalized_coordinates.size() == (32, 128, 128, 514)
    assert torch.min(normalized_coordinates) >= -1.0
    assert torch.max(normalized_coordinates) <= 1.0


def test_encode_absolute_position():
    datetimes, geospatial_bounds, geospatial_coordinates = get_data()
    absolute_position_encoding = encode_absolute_position(
        shape=(12, 5, 13, 64, 64),
        geospatial_bounds=geospatial_bounds,
        geospatial_coordinates=geospatial_coordinates,
        datetimes=datetimes,
        max_freq=128,
        num_bands=32,
    )
    assert absolute_position_encoding.size() == (12, 134, 13, 64, 64)
    assert torch.min(absolute_position_encoding) >= -1.0
    assert torch.max(absolute_position_encoding) <= 1.0


def test_encode_modalities():
    datetimes, geospatial_bounds, geospatial_coordinates = get_data()
    encoded_position = encode_modalities(
        modalities_to_encode={"NWP": torch.randn(12, 10, 13, 64, 64)},
        datetimes={"NWP": datetimes},
        geospatial_coordinates={"NWP": geospatial_coordinates},
        geospatial_bounds=geospatial_bounds,
        max_freq=128,
        num_bands=32,
    )
    assert "NWP" in encoded_position.keys()
    assert "NWP_position_encoding" in encoded_position.keys()
    assert encoded_position["NWP_position_encoding"].size() == (12, 134, 13, 64, 64)
    combined = torch.cat(
        [encoded_position["NWP"], encoded_position["NWP_position_encoding"]], dim=1
    )
    assert combined.size() == (12, 144, 13, 64, 64)


def test_encode_multiple_modalities():
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
        max_freq=128,
        num_bands=32,
    )
    assert "NWP" in encoded_position.keys()
    assert "NWP_position_encoding" in encoded_position.keys()
    assert encoded_position["NWP_position_encoding"].size() == (12, 134, 13, 64, 64)
    assert "Sat" in encoded_position.keys()
    assert "Sat_position_encoding" in encoded_position.keys()
    assert encoded_position["Sat_position_encoding"].size() == (12, 134, 3, 64, 64)
    assert "PV" in encoded_position.keys()
    assert "PV_position_encoding" in encoded_position.keys()
    assert encoded_position["PV_position_encoding"].size() == (12, 134, 5, 1, 1)

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
    # Time channels are the last 4 in the encoding, so check those, first and last time values should be the same
    # Intermediate ones should vary
    for i in range(130, 134):
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
