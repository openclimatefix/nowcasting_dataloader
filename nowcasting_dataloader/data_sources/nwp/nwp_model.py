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

# Means and std computed with
# nowcasting_dataset/scripts/compute_stats_from_batches.py
# using v15 training batches on 2021-11-24.
NWP_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "r": 75.57106712435926,
    "sde": 0.0024915961594965614,
    "si10": 4.931356852411006,
    "vis": 22321.762918384553,
    "lcc": 47.90454236572895,
    "mcc": 44.22781694449808,
    "hcc": 32.87577371914454,
}

NWP_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "r": 15.705370079694358,
    "sde": 0.07560040052148084,
    "si10": 2.664583614352396,
    "vis": 12963.802514945439,
    "lcc": 40.06675870700349,
    "mcc": 41.927221148316384,
    "hcc": 39.05157559763763,
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
