""" Test for SatelliteML"""
import pytest

from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.fake.batch import pv_fake

from nowcasting_dataloader.data_sources.pv.pv_model import PVML

from pydantic import validate_model


def test_pv_to_ml():
    """Test pv ml can be made from PV"""
    pv = pv_fake()

    _ = PVML.from_xr_dataset(pv)


def test_pv_normalization():
    """ Test PV normalization works """
    pv = pv_fake()

    pv = PVML.from_xr_dataset(pv)
    pv.normalize()

    validate_model(pv.__class__, pv.__dict__)

    # check for max x value
    pv.pv_system_x_coords[0,0] = 2
    pv.pv_system_y_coords[0, 0] = 0.5
    with pytest.raises(Exception):
        validate_model(pv.__class__, gsp.__dict__)
        
    # check for max y value
    pv.pv_system_x_coords[0, 0] = 0.5
    pv.pv_system_y_coords[0, 0] = 2
    with pytest.raises(Exception):
        validate_model(pv.__class__, gsp.__dict__)

    pv.pv_system_x_coords[0, 0] = -1
    pv.pv_system_y_coords[0, 0] = 0.5
    with pytest.raises(Exception):
        validate_model(pv.__class__, gsp.__dict__)
        
    pv.pv_system_x_coords[0, 0] = 0.5
    pv.pv_system_y_coords[0, 0] = -1
    with pytest.raises(Exception):
        validate_model(pv.__class__, gsp.__dict__)

    
    
    
