""" Model for Optical Flow data"""
from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from nowcasting_dataset.consts import Array
from nowcasting_dataset.time import make_random_time_vectors
from pydantic import Field

from nowcasting_dataloader.data_sources.datasource_output import DataSourceOutputML

logger = logging.getLogger(__name__)


class OpticalFlowML(DataSourceOutputML):
    """Model for output of Optical Flow"""

    data: Array = Field(
        ...,
        description="Optical Flow data. Shape: [batch_size,] seq_length, width, height, channel",
    )
    x: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the optical flow. "
        "Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the optical flow. "
        "Shape: [batch_size,] height",
    )

    time: Array = Field(
        ...,
        description="Time index of optical flow at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.",
    )

    channels: Array = Field(..., description="Optical Flow Channels")

    @staticmethod
    def fake(batch_size=32, seq_length_5=19, satellite_image_size_pixels=64, time_5=None):
        """Create fake data"""
        # TODO Make it only for the future
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )

        s = OpticalFlowML(
            batch_size=batch_size,
            data=np.random.randn(
                batch_size,
                seq_length_5,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
                2,
            ),
            x=np.sort(np.random.randn(batch_size, satellite_image_size_pixels)),
            y=np.sort(np.random.randn(batch_size, satellite_image_size_pixels))[:, ::-1].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            time=time_5,
            channels=np.array([list(range(2)) for _ in range(batch_size)]),
        )

        return s

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model."""
        satellite_batch_ml = xr_dataset.torch.to_tensor(["data", "time", "x", "y", "channels"])

        return OpticalFlowML(**satellite_batch_ml)
