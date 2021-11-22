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
    "HRV": 105.83116724,
    "IR_016": 141.92817573,
    "IR_039": 800.73679222,
    "IR_087": 701.5911239,
    "IR_097": 733.78406155,
    "IR_108": 573.94072497,
    "IR_120": 819.39910065,
    "IR_134": 880.35105517,
    "VIS006": 97.78966381,
    "VIS008": 116.07948311,
    "WV_062": 601.81221011,
    "WV_073": 517.77309103,
}


SAT_STD = {
    "HRV": 128.32319421,
    "IR_016": 157.25982331,
    "IR_039": 200.56013175,
    "IR_087": 181.3212391,
    "IR_097": 183.25528804,
    "IR_108": 199.49770784,
    "IR_120": 207.71619773,
    "IR_134": 219.66516666,
    "VIS006": 129.34155682,
    "VIS008": 135.47021997,
    "WV_062": 177.58304628,
    "WV_073": 159.41967647,
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

    channels: Optional[Array] = Field(
        list(SAT_MEAN.keys()), description="List of the satellite channels"
    )

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
            time_5 = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )["time_5"]

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

        # move to Model
        return SatelliteML(**satellite_batch_ml)

    def normalize(self):
        """Normalize the satellite data"""
        if not self.normalized:
            mean = np.array([SAT_MEAN[b] for b in self.channels])
            std = np.array([SAT_STD[b] for b in self.channels])
            # Need to get to the same shape, so add 3 1-dimensions
            mean = np.expand_dims(mean, axis=[1, 2, 3])
            std = np.expand_dims(std, axis=[1, 2, 3])
            self.data = self.data - mean
            self.data = self.data / std
            self.normalized = True
