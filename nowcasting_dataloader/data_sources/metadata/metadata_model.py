""" Model for output of general/metadata data, useful for a batch """
from typing import Union

import numpy as np
import torch
import xarray as xr
from nowcasting_dataset.time import make_random_time_vectors
from pydantic import Field

from nowcasting_dataloader.data_sources.datasource_output import DataSourceOutputML


class MetadataML(DataSourceOutputML):
    """Model for output of general/metadata data"""

    # TODO add descriptions
    t0_dt: Union[xr.DataArray, np.ndarray, torch.Tensor, int]  #: Shape: [batch_size,]
    x_meters_center: Union[xr.DataArray, np.ndarray, torch.Tensor, int]
    y_meters_center: Union[xr.DataArray, np.ndarray, torch.Tensor, int]
    object_at_center_label: Union[xr.DataArray, np.ndarray, torch.Tensor, int] = Field(
        ...,
        description="What object is at the center of the batch data "
        "0: Nothing at the center, "
        "1: GSP system, "
        "2: PV system",
    )

    @staticmethod
    def fake(batch_size, t0_dt=None):
        """Make a xr dataset"""
        if t0_dt is None:
            t0_dt, _, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=0, seq_length_30_minutes=0
            )

        return MetadataML(
            batch_size=batch_size,
            t0_dt=t0_dt,
            x_meters_center=np.random.randn(
                batch_size,
            ),
            y_meters_center=np.random.randn(
                batch_size,
            ),
            object_at_center_label=np.array([1] * batch_size),
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""
        return MetadataML(
            batch_size=xr_dataset.t0_dt.shape[0],
            t0_dt=xr_dataset.t0_dt.values,
            x_meters_center=xr_dataset.x_meters_center.values,
            y_meters_center=xr_dataset.y_meters_center.values,
            object_at_center_label=xr_dataset.object_at_center_label.values,
        )
