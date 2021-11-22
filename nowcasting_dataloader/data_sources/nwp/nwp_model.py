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
NWP_MEAN = {
    "cdcb": 742.401,
    "lcc": 66.756538,
    "mcc": 157.74133,
    "hcc": 15542.797,
    "sde": 3.2133293,
    "hcct": -5909.7021,
    "dswrf": 49265.621,
    "dlwrf": 167.16484,
    "h": 1707.0344,
    "t": 171.22368,
    "r": 63.139393,
    "dpt": 158.05571,
    "vis": 16480.631,
    "si10": -5607.0537,
    "wdir10": 151.85562,
    "prmsl": 52296.281,
    "prate": 975.62311,
}


NWP_STD = {
    "cdcb": 1437.9733,
    "lcc": 32.845642,
    "mcc": 122.90843,
    "hcc": 22072.18,
    "sde": 4.2809649,
    "hcct": 13865.189,
    "dswrf": 50599.918,
    "dlwrf": 165.04289,
    "h": 1566.2006,
    "t": 118.57211,
    "r": 35.996311,
    "dpt": 128.49652,
    "vis": 22340.301,
    "si10": 13388.914,
    "wdir10": 153.62816,
    "prmsl": 50509.055,
    "prate": 1263.6071,
}


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
        seq_length_60=2,
        image_size_pixels=64,
        number_nwp_channels=7,
        time_60=None,
    ):
        """Create fake data"""
        if time_60 is None:
            time_60 = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=0, seq_length_60_minutes=seq_length_60
            )["time_60"]

        s = NWPML(
            batch_size=batch_size,
            data=np.random.randn(
                batch_size,
                number_nwp_channels,
                seq_length_60,
                image_size_pixels,
                image_size_pixels,
            ).astype(np.float32),
            x=np.sort(np.random.randn(batch_size, image_size_pixels).astype(np.float32)),
            y=np.sort(np.random.randn(batch_size, image_size_pixels).astype(np.float32))[
                :, ::-1
            ].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            time=time_60,
            init_time=time_60[0],
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
            # Only take the channels that are used
            mean = np.array([NWP_MEAN[b] for b in self.channels])
            std = np.array([NWP_STD[b] for b in self.channels])
            # Expand for normaliation
            mean = np.expand_dims(mean, axis=[1, 2, 3])
            std = np.expand_dims(std, axis=[1, 2, 3])
            self.data = self.data - mean
            self.data = self.data / std
            self.normalized = True
