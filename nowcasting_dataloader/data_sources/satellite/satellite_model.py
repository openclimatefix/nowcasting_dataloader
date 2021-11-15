""" Model for output of satellite data """
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr
from nowcasting_dataset.time import make_random_time_vectors
from pydantic import Field

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML
from nowcasting_dataloader.xr_utils import re_order_dims

logger = logging.getLogger(__name__)

SAT_MEAN = {
    'HRV': 115,
    'IR_016': 139,
    'IR_039': 36,
    'IR_087': 57,
    'IR_097': 30,
    'IR_108': 149,
    'IR_120': 51,
    'IR_134': 35,
    'VIS006': 115,
    'VIS008': 120,
    'WV_062': 98,
    'WV_073': 99
    }


SAT_STD = {
    'HRV': 115,
    'IR_016': 139,
    'IR_039': 36,
    'IR_087': 57,
    'IR_097': 30,
    'IR_108': 149,
    'IR_120': 51,
    'IR_134': 35,
    'VIS006': 115,
    'VIS008': 120,
    'WV_062': 98,
    'WV_073': 99
    }


class SatelliteML(DataSourceOutputML):
    """Model for output of satellite data"""

    # Shape: [batch_size,] seq_length, width, height, channel
    data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    x: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the satellite images. "
        "Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the satellite images. "
        "Shape: [batch_size,] height",
    )

    time: Array = Field(
        ...,
        description="Time index of satellite data at 5 minutes past the hour {0, 5, ..., 55}. "
        "*not* the {4, 9, ..., 59} timings of the satellite imagery. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.",
    )

    channels: Optional[Array] = Field(None, description="List of the satellite channels")

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        satellite_image_size_pixels=64,
        number_sat_channels=7,
        time_5=None,
    ):
        """Create fake data"""
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )

        s = SatelliteML(
            batch_size=batch_size,
            data=np.random.randn(
                batch_size,
                number_sat_channels,
                seq_length_5,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
            ).astype(np.float32),
            x=np.sort(np.random.randn(batch_size, satellite_image_size_pixels)),
            y=np.sort(np.random.randn(batch_size, satellite_image_size_pixels))[:, ::-1].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            time=time_5,
            channels=np.array([list(range(number_sat_channels)) for _ in range(batch_size)]),
        )

        return s

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model."""

        # make sure the dims are in the correct order
        xr_dataset = re_order_dims(xr_dataset)

        # convert to torch dictionary
        satellite_batch_ml = xr_dataset.torch.to_tensor(["data", "time", "x", "y"])

        # move to Modle
        return SatelliteML(**satellite_batch_ml)

    def normalize(self):
        """Normalize the satellite data"""
        if not self.normalized:
            mean = np.array([SAT_MEAN[b] for b in self.channels])
            std = np.array([SAT_STD[b] for b in self.channels])
            self.data = self.data - mean
            self.data = self.data / std
            self.normalized = True
