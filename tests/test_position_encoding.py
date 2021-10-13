from nowcasting_dataloader.utils.position_encoding import encode_position, encode_modalities, encode_absolute_position, \
    create_datetime_features, normalize_geospatial_coordinates
import pytest
import datetime
import pandas as pd
import torch
import numpy as np


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
    pass


def test_encode_absolute_position():
    pass


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
