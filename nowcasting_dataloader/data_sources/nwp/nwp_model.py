""" Model for output of NWP data """
from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from pydantic import Field

from nowcasting_dataset.consts import Array
from nowcasting_dataloader.data_sources.datasource_output import (
    DataSourceOutputML,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class NWPML(DataSourceOutputML):
    """ Model for output of NWP data """

    # Shape: [batch_size,] seq_length, width, height, channel
    data: Array = Field(
        ...,
        description=" Numerical weather predictions (NWPs) \
    : Shape: [batch_size,] channel, seq_length, width, height",
    )

    x: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] height",
    )

    target_time: Array = Field(
        ...,
        description="Time index of nwp data at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.  The 'target time' is the time the NWP is _about_.",
    )

    init_time: Array = Field(..., description="The time when the nwp forecast was made")

    channels: Array = Field(..., description="List of the nwp channels")

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        image_size_pixels=64,
        number_nwp_channels=7,
        time_5=None,
    ):
        """Create fake data"""
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )

        s = NWPML(
            batch_size=batch_size,
            data=np.random.randn(
                batch_size,
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
                number_nwp_channels,
            ),
            x=np.sort(np.random.randn(batch_size, image_size_pixels)),
            y=np.sort(np.random.randn(batch_size, image_size_pixels))[:, ::-1].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            target_time=time_5,
            init_time=time_5[0],
            channels=np.array([list(range(number_nwp_channels)) for _ in range(batch_size)]),
        )

        return s

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.target_time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model with tensors"""
        nwp_batch_ml = xr_dataset.torch.to_tensor(
            ["data", "target_time", "init_time", "x", "y", "channels"]
        )

        return NWPML(**nwp_batch_ml)
