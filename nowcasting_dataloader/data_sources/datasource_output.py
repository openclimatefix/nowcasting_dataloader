""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
import xarray as xr
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Array = Union[xr.DataArray, np.ndarray, torch.Tensor, list]

OSGB_X_MAX = 654665
OSGB_Y_MAX = 1151577


class DataSourceOutputML(BaseModel):
    """General Data Source output pydantic class.

    Data source output classes should inherit from this class
    """

    class Config:
        """Allowed classes e.g. tensor.Tensor"""

        # TODO maybe there is a better way to do this
        arbitrary_types_allowed = True

    batch_size: int = Field(
        0,
        ge=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item i.e Example",
    )

    normalized: bool = Field(
        False,
        description="If the data field has been normalized or not",
    )

    def get_name(self) -> str:
        """Get the name of the class"""
        return self.__class__.__name__.lower()

    def get_datetime_index(self):
        """Datetime index for the data"""
        pass

    def normalize(self):
        """Normalize the data"""
        pass
