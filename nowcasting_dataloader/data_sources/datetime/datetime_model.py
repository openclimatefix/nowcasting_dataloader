""" Model for output of datetime data """
import numpy as np
import xarray as xr
from pydantic import validator

from nowcasting_dataset.consts import Array, DATETIME_FEATURE_NAMES
from nowcasting_dataloader.data_sources.datasource_output import (
    DataSourceOutputML,
)

from nowcasting_dataset.utils import coord_to_range


class DatetimeML(DataSourceOutputML):
    """Model for output of datetime data"""

    hour_of_day_sin: Array  #: Shape: [batch_size,] seq_length
    hour_of_day_cos: Array
    day_of_year_sin: Array
    day_of_year_cos: Array
    datetime_index: Array

    @property
    def sequence_length(self):
        """The sequence length of the pv data"""
        return self.hour_of_day_sin.shape[-1]

    @validator("hour_of_day_cos")
    def v_hour_of_day_cos(cls, v, values):
        """Validate 'hour_of_day_cos'"""
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @validator("day_of_year_sin")
    def v_day_of_year_sin(cls, v, values):
        """Validate 'day_of_year_sin'"""
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @validator("day_of_year_cos")
    def v_day_of_year_cos(cls, v, values):
        """Validate 'day_of_year_cos'"""
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5):
        """Make a fake Datetime object"""
        return DatetimeML(
            batch_size=batch_size,
            hour_of_day_sin=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            hour_of_day_cos=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            day_of_year_sin=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            day_of_year_cos=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            datetime_index=np.sort(np.random.randn(batch_size, seq_length_5))[:, ::-1].copy(),
            # copy is needed as torch doesnt not support negative strides
        )

    def to_xr_dataset(self, _):
        """Make a xr dataset"""
        individual_datasets = []
        for name in DATETIME_FEATURE_NAMES:

            var = self.__getattribute__(name)

            data = xr.DataArray(
                var,
                dims=["time"],
                coords={"time": self.datetime_index},
                name=name,
            )

            ds = data.to_dataset()
            ds = coord_to_range(ds, "time", prefix=None)
            individual_datasets.append(ds)

        return xr.merge(individual_datasets)

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""
        if "hour_of_day_sin" in xr_dataset.keys():
            return DatetimeML(
                batch_size=xr_dataset["hour_of_day_sin"].shape[0],
                hour_of_day_sin=xr_dataset["hour_of_day_sin"],
                hour_of_day_cos=xr_dataset["hour_of_day_cos"],
                day_of_year_sin=xr_dataset["day_of_year_sin"],
                day_of_year_cos=xr_dataset["day_of_year_cos"],
                datetime_index=xr_dataset["hour_of_day_sin"].time,
            )
        else:
            return None
