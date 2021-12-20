""" Model for Topogrpahic features """
import logging

from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from pydantic import Field, validator

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML

logger = logging.getLogger(__name__)

# Computed using nowcasting_dataset/notebooks/computs_stats_from_batches.ipynb
# on 2021-11-24 using the training batch of v15 of the dataset.
TOPO_MEAN = 135.88652096070271
TOPO_STD = 145.65013699767726867


class TopographicML(DataSourceOutputML):
    """
    Topographic/elevation map features.
    """

    # Shape: [batch_size,] width, height
    topo_data: Array = Field(
        ...,
        description="Elevation map of the area covered by the satellite data. "
        "Shape: [batch_size], width, height",
    )
    topo_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the topographic images. "
        "Shape: [batch_size,] width",
    )
    topo_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the topographic images. "
        "Shape: [batch_size,] height",
    )

    @property
    def height(self):
        """The height of the topographic image"""
        return self.topo_data.shape[-1]

    @property
    def width(self):
        """The width of the topographic image"""
        return self.topo_data.shape[-2]

    @validator("topo_x_coords")
    def x_coordinates_shape(cls, v, values):
        """Validate 'topo_x_coords'"""
        assert v.shape[-1] == values["topo_data"].shape[-2]
        return v

    @validator("topo_y_coords")
    def y_coordinates_shape(cls, v, values):
        """Validate 'topo_y_coords'"""
        assert v.shape[-1] == values["topo_data"].shape[-1]
        return v

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        topographic_batch_ml = xr_dataset.torch.to_tensor(["data", "x_osgb", "y_osgb"])

        topographic_batch_ml[TOPOGRAPHIC_DATA] = topographic_batch_ml.pop("data")
        topographic_batch_ml["topo_x_coords"] = topographic_batch_ml.pop("x_osgb")
        topographic_batch_ml["topo_y_coords"] = topographic_batch_ml.pop("y_osgb")

        return TopographicML(**topographic_batch_ml)

    def normalize(self):
        """Normalize the topological data"""
        if not self.normalized:
            self.topo_data = self.topo_data - TOPO_MEAN
            self.topo_data = self.topo_data / TOPO_STD
            self.normalized = True
