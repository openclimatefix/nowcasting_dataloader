""" Model for output of NWP data """
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

# Means computed with
# nwp_ds = NWPDataSource(...)
# nwp_ds.open()
# mean = nwp_ds.data.isel(init_time=slice(0, 10)).mean(
#     dim=['step', 'x', 'init_time', 'y']).compute()
NWP_MEAN = [
    2.8041010e02,
    1.6854691e01,
    6.7529683e-05,
    8.1832832e01,
    7.1233767e-03,
    8.8566933e00,
    4.3474598e04,
    4.9820110e01,
    4.8095409e01,
    4.2833260e01,
]

NWP_STD = [
    2.5812180e00,
    4.1278820e01,
    2.7507244e-04,
    9.0967312e00,
    1.4110464e-01,
    4.3616886e00,
    2.3853148e04,
    3.8900299e01,
    4.2830105e01,
    4.2778091e01,
]


class NWPML(DataSourceOutputML):
    """Model for output of NWP data"""

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

    time: Array = Field(
        ...,
        description="Time index of nwp data at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.  The 'target time' is the time the NWP is _about_.",
    )

    init_time: Array = Field(..., description="The time when the nwp forecast was made")

    channels: Optional[Array] = Field(None, description="List of the nwp channels")

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
                number_nwp_channels,
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
            ).astype(np.float32),
            x=np.sort(np.random.randn(batch_size, image_size_pixels).astype(np.float32)),
            y=np.sort(np.random.randn(batch_size, image_size_pixels).astype(np.float32))[
                :, ::-1
            ].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            time=time_5,
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

        # make sure dims are the in the correct order
        xr_dataset = re_order_dims(xr_dataset)

        # convert to torch dict
        nwp_batch_ml = xr_dataset.torch.to_tensor(["data", "time", "init_time", "x", "y"])

        # make into Model
        return NWPML(**nwp_batch_ml)

    def normalize(self):
        """Normalize the nwp data"""
        if not self.normalized:
            self.data = self.data - NWP_MEAN
            self.data = self.data / NWP_STD
            self.normalized = True
