""" Model for Sun features """
import logging

import numpy as np
from pydantic import Field, validator

from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataloader.data_sources.datasource_output import (
    DataSourceOutputML,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class SunML(DataSourceOutputML):
    """Model for Sun features"""

    sun_azimuth_angle: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )

    sun_elevation_angle: Array = Field(
        ...,
        description="PV elevation angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )
    sun_datetime_index: Array

    @validator("sun_elevation_angle")
    def elevation_shape(cls, v, values):
        """
        Validate 'sun_elevation_angle'.

        This is done by change shape is the same as the "sun_azimuth_angle"
        """
        assert v.shape == values["sun_azimuth_angle"].shape
        return v

    @validator("sun_datetime_index")
    def sun_datetime_index_shape(cls, v, values):
        """
        Validate 'sun_datetime_index'.

        This is done by checking last dimension is the same as the last dim of 'sun_azimuth_angle'
        i.e the time dimension
        """
        assert v.shape[-1] == values["sun_azimuth_angle"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, time_5=None):
        """Create fake data"""
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )

        return SunML(
            batch_size=batch_size,
            sun_azimuth_angle=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            sun_elevation_angle=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            sun_datetime_index=time_5,
        )

    def get_datetime_index(self):
        """Get the datetime index of this data"""
        return self.sun_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""
        if SUN_AZIMUTH_ANGLE in xr_dataset.keys():
            return SunML(
                batch_size=xr_dataset[SUN_AZIMUTH_ANGLE].shape[0],
                sun_azimuth_angle=xr_dataset[SUN_AZIMUTH_ANGLE],
                sun_elevation_angle=xr_dataset[SUN_ELEVATION_ANGLE],
                sun_datetime_index=xr_dataset["sun_datetime_index"],
            )
        else:
            return None
