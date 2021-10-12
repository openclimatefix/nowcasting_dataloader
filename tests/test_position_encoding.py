from nowcasting_dataloader.utils.position_encoding import encode_position
import pytest


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


def test_satellite_pv_encoding():
    pass
