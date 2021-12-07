""" Model for Sun features """
import logging

from nowcasting_dataset.consts import SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from pydantic import Field, validator

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML

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

    def get_datetime_index(self):
        """Get the datetime index of this data"""
        return self.sun_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        sun_batch_ml = xr_dataset.torch.to_tensor(["azimuth", "time", "elevation"])

        sun_batch_ml["sun_datetime_index"] = sun_batch_ml.pop("time")
        sun_batch_ml[SUN_AZIMUTH_ANGLE] = sun_batch_ml.pop("azimuth")
        sun_batch_ml[SUN_ELEVATION_ANGLE] = sun_batch_ml.pop("elevation")

        return SunML(**sun_batch_ml)
