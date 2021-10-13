from nowcasting_dataloader.utils.position_encoding import encode_position, encode_modalities, encode_absolute_position, \
    create_datetime_features, normalize_geospatial_coordinates
import pytest
import pandas as pd
import torch


def test_datetime_feature_creation():
    # Generate fake datetime data for the whole year
    datetimes = []
    for month in range(1,12):
        datetimes.append(pd.date_range(start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 06:00", freq="5min"))
    datetimes.append(pd.date_range(start=f"2020-12-31 17:55", end=f"2020-12-31 23:55", freq="5min"))
    datetime_features = create_datetime_features(datetimes)
    assert len(datetime_features) == 4
    for feature in datetime_features:
        assert feature.size() == (12, 73)
        assert torch.min(feature) >= -1.0
        assert torch.max(feature) <= 1.0


def test_geospatial_normalization():
    geospatial_bounds = {"x_min": -2900., "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = torch.sort(torch.rand(32, 128) * 9)[0]
    y = torch.sort(torch.rand(32, 128) * 120, descending=True)[0]
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    normalized_coordinates = normalize_geospatial_coordinates(geospatial_coordinates, geospatial_bounds=geospatial_bounds, max_freq=64,num_bands=128)
    assert normalized_coordinates.size() == (32, 128, 128, 514)
    assert torch.min(normalized_coordinates) >= -1.0
    assert torch.max(normalized_coordinates) <= 1.0


def test_encode_absolute_position():
    # Generate fake datetime data for the whole year
    datetimes = []
    for month in range(1,12):
        datetimes.append(pd.date_range(start=f"2020-{month}-01 00:00", end=f"2020-{month}-01 01:00", freq="5min"))
    datetimes.append(pd.date_range(start=f"2020-12-31 22:55", end=f"2020-12-31 23:55", freq="5min"))
    geospatial_bounds = {"x_min": -2900., "y_min": -20, "x_max": 230000, "y_max": 670430}
    geospatial_coordinates = []
    x = torch.sort(torch.rand(12, 64) * 9)[0]
    y = torch.sort(torch.rand(12, 64) * 120, descending=True)[0]
    geospatial_coordinates.append(x)
    geospatial_coordinates.append(y)
    absolute_position_encoding = encode_absolute_position(shape=(12, 13, 64, 64), geospatial_bounds=geospatial_bounds, geospatial_coordinates=geospatial_coordinates, datetimes=datetimes, max_freq=128,num_bands=32)
    assert absolute_position_encoding.size() == (12, 650, 13, 64, 64)
    assert torch.min(absolute_position_encoding) >= -1.0
    assert torch.max(absolute_position_encoding) <= 1.0


def test_encode_modalities():
    pass


def test_fourier_encoding():
    pass


def test_absolute_encoding():
    pass


def test_fake_method_option():
    with pytest.raises(AssertionError):
        encode_position(shape=[1, 1, 1, 1], positioning="relative", method="test_method")


def test_fake_positioning_option():
    with pytest.raises(AssertionError):
        encode_position(shape=[1, 1, 1, 1], positioning="fake positioning scheme")


def test_multi_modality_encoding():
    pass


def test_5min_30min_encoding():
    pass
