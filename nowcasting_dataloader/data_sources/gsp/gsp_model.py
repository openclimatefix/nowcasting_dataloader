""" Model for output of GSP data """
import logging

from nowcasting_dataset.consts import (
    GSP_DATETIME_INDEX,
    GSP_ID,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_YIELD,
)
from pydantic import Field, validator

from nowcasting_dataloader.data_sources.datasource_output import (
    OSGB_X_MAX,
    OSGB_Y_MAX,
    Array,
    DataSourceOutputML,
)

logger = logging.getLogger(__name__)


class GSPML(DataSourceOutputML):
    """Model for output of GSP data"""

    # Shape: [batch_size,] seq_length, n_pv_systems_per_example
    gsp_yield: Array = Field(
        ...,
        description=" GSP yield from all GSP in the region of interest (ROI). \
    : Includes central GSP system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_gsp_per_example",
    )

    # Shape: [batch_size,] seq_length, n_pv_systems_per_example
    gsp_capacity: Array = Field(
        ...,
        description=" GSP capacity from all GSP in the region of interest (ROI). \
        : Includes central GSP system, which will always be the first entry. \
        : shape = [batch_size, ] seq_length, n_gsp_per_example",
    )

    #: GSP identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    gsp_id: Array = Field(..., description="gsp id from NG")

    gsp_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the gsp data. "
        "shape = [batch_size, ] sequence length,",
    )

    gsp_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the gsp. "
        "This is in fact the x centroid of the GSP region"
        "Shape: [batch_size,] n_gsp_per_example",
    )
    gsp_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the gsp. "
        "This are in fact the y centroid of the GSP region"
        "Shape: [batch_size,] n_gsp_per_example",
    )

    @property
    def number_of_gsp(self):
        """The number of Grid Supply Points in this example"""
        return self.gsp_yield.shape[-1]

    @property
    def sequence_length(self):
        """The sequence length of the GSP PV power timeseries data"""
        return self.gsp_yield.shape[-2]

    @validator("gsp_yield")
    def gsp_yield_shape(cls, v, values):
        """Validate 'gsp_yield'"""
        assert len(v.shape) == 3

        return v

    @validator("gsp_x_coords")
    def x_coordinates(cls, v, values):
        """Validate 'gsp_x_coords'"""
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        if values["normalized"]:
            if v.max() > 1:
                raise Exception(
                    f"GSP data is normalized, "
                    f"but GSP X coordinates maximum value is above 1, {v.max()}"
                )
            if v.min() < 0:
                raise Exception(
                    f"GSP data is normalized, "
                    f"but GSP X coordinates minimum value is below 0, {v.min()}"
                )
        return v

    @validator("gsp_y_coords")
    def y_coordinates_shape(cls, v, values):
        """Validate 'gsp_y_coords'"""
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        if values["normalized"]:
            if v.max() > 1:
                raise Exception(
                    f"GSP data is normalized, "
                    f"but GSP Y coordinates maximum value is above 1, {v.max()}"
                )
            if v.min() < 0:
                raise Exception(
                    f"GSP data is normalized, "
                    f"but GSP Y coordinates minimum value is below 0, {v.min()}"
                )
        return v

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.gsp_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        gsp_batch_ml = xr_dataset.torch.to_tensor(
            ["power_mw", "capacity_mwp", "time", "x_coords", "y_coords", "id"]
        )

        gsp_batch_ml[GSP_YIELD] = gsp_batch_ml.pop("power_mw")
        gsp_batch_ml["gsp_capacity"] = gsp_batch_ml.pop("capacity_mwp")
        gsp_batch_ml[GSP_ID] = gsp_batch_ml.pop("id")
        gsp_batch_ml[GSP_DATETIME_INDEX] = gsp_batch_ml.pop("time")
        gsp_batch_ml[GSP_X_COORDS] = gsp_batch_ml.pop("x_coords")
        gsp_batch_ml[GSP_Y_COORDS] = gsp_batch_ml.pop("y_coords")
        gsp_batch_ml["batch_size"] = gsp_batch_ml[GSP_YIELD].shape[0]

        return GSPML(**gsp_batch_ml)

    def normalize(self):
        """Normalize the gsp data"""
        if not self.normalized:
            self.gsp_yield = self.gsp_yield / self.gsp_capacity

            self.gsp_x_coords = self.gsp_x_coords / OSGB_X_MAX
            self.gsp_y_coords = self.gsp_y_coords / OSGB_Y_MAX

            self.normalized = True
