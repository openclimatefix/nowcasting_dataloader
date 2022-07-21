""" Test for SatelliteML"""
import pytest
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.fake.batch import gsp_fake
from pydantic import validate_model

from nowcasting_dataloader.data_sources.gsp.gsp_model import GSPML


def test_gsp_to_ml():
    """Test gsp ml can be made from GSP"""
    gsp = gsp_fake()

    _ = GSPML.from_xr_dataset(gsp)


def test_gsp_normalization():
    """Test GSP normalization works"""
    gsp = gsp_fake()

    gsp = GSPML.from_xr_dataset(gsp)
    gsp.normalize()

    validate_model(gsp.__class__, gsp.__dict__)

    # check for max value
    gsp.gsp_x_coords[0, 0] = 2
    gsp.gsp_y_coords[0, 0] = 0.5
    with pytest.raises(Exception):
        validate_model(gsp.__class__, gsp.__dict__)

    # check for max value
    gsp.gsp_x_coords[0, 0] = 0.5
    gsp.gsp_y_coords[0, 0] = 2
    # Onl raises an warning now
    # with pytest.raises(Exception):
    #     validate_model(gsp.__class__, gsp.__dict__)

    gsp.gsp_x_coords[0, 0] = -1
    gsp.gsp_y_coords[0, 0] = 0.5
    with pytest.raises(Exception):
        validate_model(gsp.__class__, gsp.__dict__)

    gsp.gsp_x_coords[0, 0] = 0.5
    gsp.gsp_y_coords[0, 0] = -1
    with pytest.raises(Exception):
        validate_model(gsp.__class__, gsp.__dict__)


def test_gsp_normalization_zero_capacity():
    
    gsp = gsp_fake()

    gsp = GSPML.from_xr_dataset(gsp)

    gsp.gsp_capacity[0,:,0] = 0
    gsp.gsp_yield[0, 0, 0] = 0
    gsp.gsp_yield[0, 1, 0] = 0

    gsp.normalize()
    
    assert gsp.gsp_yield[0, 0, 0] == 0
    assert gsp.gsp_yield[0, 1, 0] == 0
    
    
    