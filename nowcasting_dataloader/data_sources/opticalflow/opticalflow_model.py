""" Model for output of satellite data """
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr
from pydantic import Field

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML
from nowcasting_dataloader.xr_utils import re_order_dims

logger = logging.getLogger(__name__)

OPTICALFLOW_MEAN = {
    "HRV": 236.13257536395903,
    "IR_016": 291.61620182554185,
    "IR_039": 858.8040610176552,
    "IR_087": 738.3103442750336,
    "IR_097": 773.0910794778366,
    "IR_108": 607.5318145165666,
    "IR_120": 860.6716261423857,
    "IR_134": 925.0477987594331,
    "VIS006": 228.02134593063957,
    "VIS008": 257.56333202381205,
    "WV_062": 633.5975770915588,
    "WV_073": 543.4963868823854,
}

OPTICALFLOW_STD = {
    "HRV": 935.9717382401759,
    "IR_016": 172.01044433112992,
    "IR_039": 96.53756504807913,
    "IR_087": 96.21369354283686,
    "IR_097": 86.72892737648276,
    "IR_108": 156.20651744208888,
    "IR_120": 104.35287930753246,
    "IR_134": 104.36462050405994,
    "VIS006": 150.2399269307514,
    "VIS008": 152.16086321818398,
    "WV_062": 111.8514878214775,
    "WV_073": 106.8855172848904,
}


class OpticalFlowML(DataSourceOutputML):
    """Model for output of satellite data"""

    # Shape: [batch_size,] seq_length, width, height, channel
    data: Array = Field(
        ...,
        description="Optical flow images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    x: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the optical flow images. "
        "Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the optical flow images. "
        "Shape: [batch_size,] height",
    )

    time: Array = Field(
        ...,
        description="Time index of optical flow data at 5 minutes past the hour {0, 5, ..., 55}. "
        "*not* the {4, 9, ..., 59} timings of the optical flow imagery. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.",
    )

    channels: Optional[Array] = Field(
        list(OPTICALFLOW_MEAN.keys()), description="List of the optical flow channels"
    )

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model."""

        # make sure the dims are in the correct order
        expected_dims_order = (
            "example",
            "channels_index",
            "time_index",
            "y_geostationary_index",
            "x_geostationary_index",
        )
        xr_dataset = re_order_dims(xr_dataset, expected_dims_order=expected_dims_order)

        # convert to torch dictionary
        opticalflow_batch_ml = xr_dataset.torch.to_tensor(
            ["data", "time", "y_geostationary", "x_geostationary"]
        )

        # set channels, just take the first example
        opticalflow_batch_ml["channels"] = xr_dataset.channels[0].values

        # rename x and y channel
        opticalflow_batch_ml["x"] = opticalflow_batch_ml.pop("x_geostationary")
        opticalflow_batch_ml["y"] = opticalflow_batch_ml.pop("y_geostationary")

        # move to Model
        return OpticalFlowML(**opticalflow_batch_ml)

    def normalize(self):
        """Normalize the satellite data"""
        if not self.normalized:
            mean = np.array([OPTICALFLOW_MEAN[b] for b in self.channels])
            std = np.array([OPTICALFLOW_STD[b] for b in self.channels])
            # Need to get to the same shape, so add 3 1-dimensions
            mean = np.expand_dims(mean, axis=[1, 2, 3])
            std = np.expand_dims(std, axis=[1, 2, 3])
            self.data = self.data - mean
            self.data = self.data / std
            self.normalized = True
