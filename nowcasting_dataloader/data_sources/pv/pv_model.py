""" Model for output of PV data """
import logging

import numpy as np
from pydantic import Field, validator

from nowcasting_dataset.consts import (
    Array,
    PV_YIELD,
    PV_DATETIME_INDEX,
    PV_SYSTEM_Y_COORDS,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_ID,
)
from nowcasting_dataloader.data_sources.datasource_output import (
    DataSourceOutputML,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class PVML(DataSourceOutputML):
    """ Model for output of PV data """

    # Shape: [batch_size,] seq_length, width, height, channel
    pv_yield: Array = Field(
        ...,
        description=" PV yield from all PV systems in the region of interest (ROI). \
    : Includes central PV system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_pv_systems_per_example",
    )

    #: PV identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_id: Array = Field(..., description="PV system ID, e.g. from PVoutput.org")
    pv_system_row_number: Array = Field(..., description="pv row number, made by OCF  TODO")

    pv_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the pv system data. shape = [batch_size, ] sequence length,",
    )

    pv_system_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )
    pv_system_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )

    @property
    def number_of_pv_systems(self):
        """The number of pv systems"""
        return self.pv_yield.shape[-1]

    @property
    def sequence_length(self):
        """The sequence length of the pv data"""
        return self.pv_yield.shape[-2]

    @validator("pv_system_x_coords")
    def x_coordinates_shape(cls, v, values):
        """ Validate 'pv_system_x_coords' """
        assert v.shape[-1] == values["pv_yield"].shape[-1]
        return v

    @validator("pv_system_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'pv_system_y_coords' """
        assert v.shape[-1] == values["pv_yield"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, n_pv_systems_per_batch, time_5=None):
        """ Create fake data """
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_length_5_minutes=seq_length_5, seq_length_30_minutes=0
            )

        return PVML(
            batch_size=batch_size,
            pv_yield=np.random.randn(
                batch_size,
                seq_length_5,
                n_pv_systems_per_batch,
            ),
            pv_system_id=np.sort(np.random.randint(0, 10000, (batch_size, n_pv_systems_per_batch))),
            pv_system_row_number=np.sort(
                np.random.randint(0, 1000, (batch_size, n_pv_systems_per_batch))
            ),
            pv_datetime_index=time_5,
            pv_system_x_coords=np.sort(np.random.randn(batch_size, n_pv_systems_per_batch)),
            pv_system_y_coords=np.sort(np.random.randn(batch_size, n_pv_systems_per_batch))[
                :, ::-1
            ].copy(),  # copy is needed as torch doesnt not support negative strides
        )

    def get_datetime_index(self) -> Array:
        """ Get the datetime index of this data """
        return self.pv_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if PV_YIELD in xr_dataset.keys():
            return PVML(
                batch_size=xr_dataset[PV_YIELD].shape[0],
                pv_yield=xr_dataset[PV_YIELD],
                pv_system_id=xr_dataset[PV_SYSTEM_ID],
                pv_system_row_number=xr_dataset[PV_SYSTEM_ROW_NUMBER],
                pv_datetime_index=xr_dataset[PV_DATETIME_INDEX],
                pv_system_x_coords=xr_dataset[PV_SYSTEM_X_COORDS],
                pv_system_y_coords=xr_dataset[PV_SYSTEM_Y_COORDS],
            )
        else:
            return None
