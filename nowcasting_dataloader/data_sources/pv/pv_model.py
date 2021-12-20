""" Model for output of PV data """
import logging

import numpy as np
from nowcasting_dataset.consts import (
    PV_DATETIME_INDEX,
    PV_SYSTEM_ID,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    PV_YIELD,
)
from pydantic import Field, validator

from nowcasting_dataloader.data_sources.datasource_output import (
    OSGB_X_MAX,
    OSGB_Y_MAX,
    Array,
    DataSourceOutputML,
)

logger = logging.getLogger(__name__)


class PVML(DataSourceOutputML):
    """Model for output of PV data"""

    # Shape: [batch_size,] seq_length, n_pv_systems_per_example
    pv_yield: Array = Field(
        ...,
        description=" PV yield from all PV systems in the region of interest (ROI). \
    : Includes central PV system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_pv_systems_per_example",
    )

    # Shape: [batch_size,], n_pv_systems_per_example
    pv_capacity: Array = Field(
        ...,
        description=" PV capacity from all PV systems in the region of interest (ROI). \
        : Includes central PV system, which will always be the first entry. "
        "PV capacity is assume constant: \
        : shape = [batch_size, ], n_pv_systems_per_example",
    )

    #: PV identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_id: Array = Field(..., description="PV system ID, e.g. from PVoutput.org")
    pv_system_row_number: Array = Field(..., description="pv row number, made by OCF  TODO")

    pv_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the pv system data. "
        "shape = [batch_size, ] sequence length,",
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
        """Validate 'pv_system_x_coords'"""
        assert v.shape[-1] == values["pv_yield"].shape[-1]

        if values["normalized"]:
            if v.max() > 1:
                raise Exception(
                    f"PV data is normalized, "
                    f"but PV X coordinates maximum value is above 1, {v.max()}"
                )
            if v.min() < 0:
                raise Exception(
                    f"PV data is normalized, "
                    f"but PV X coordinates minimum value is below 0, {v.min()}"
                )

        return v

    @validator("pv_system_y_coords")
    def y_coordinates(cls, v, values):
        """Validate 'pv_system_y_coords'"""
        assert v.shape[-1] == values["pv_yield"].shape[-1]

        if values["normalized"]:
            if v.max() > 1:
                raise Exception(
                    f"PV data is normalized, "
                    f"but PV Y coordinates maximum value is above 1, {v.max()}"
                )
            if v.min() < 0:
                raise Exception(
                    f"PV data is normalized, "
                    f"but PV Y coordinates minimum value is below 0, {v.min()}"
                )

        return v

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.pv_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        pv_batch_ml = xr_dataset.torch.to_tensor(
            ["power_mw", "capacity_mwp", "time", "x_osgb", "y_osgb", "id"]
        )

        pv_batch_ml[PV_YIELD] = pv_batch_ml.pop("power_mw")
        pv_batch_ml["pv_capacity"] = pv_batch_ml.pop("capacity_mwp")
        pv_batch_ml[PV_SYSTEM_ID] = pv_batch_ml["id"]
        pv_batch_ml[PV_SYSTEM_ROW_NUMBER] = pv_batch_ml.pop("id")
        pv_batch_ml[PV_DATETIME_INDEX] = pv_batch_ml.pop("time")
        pv_batch_ml[PV_SYSTEM_X_COORDS] = pv_batch_ml.pop("x_osgb")
        pv_batch_ml[PV_SYSTEM_Y_COORDS] = pv_batch_ml.pop("y_osgb")

        return PVML(**pv_batch_ml)

    def normalize(self):
        """Normalize the gsp data"""
        if not self.normalized:
            # Expand capacity to the same timesteps for broadcasting
            capacity = np.expand_dims(self.pv_capacity, axis=1)
            self.pv_yield = self.pv_yield / capacity

            self.pv_system_x_coords = self.pv_system_x_coords / OSGB_X_MAX
            self.pv_system_y_coords = self.pv_system_y_coords / OSGB_Y_MAX

            self.normalized = True
