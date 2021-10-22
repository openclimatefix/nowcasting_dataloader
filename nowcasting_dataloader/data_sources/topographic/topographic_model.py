""" Model for Topogrpahic features """
import logging

import numpy as np
from pydantic import Field, validator

from nowcasting_dataset.consts import Array
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from nowcasting_dataloader.data_sources.datasource_output import DataSourceOutputML

logger = logging.getLogger(__name__)

TOPO_MEAN = 365.486887
TOPO_STD = 478.841369


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
        description="The x (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )
    topo_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] height",
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
    def fake(batch_size, image_size_pixels):
        """Create fake data"""
        return TopographicML(
            batch_size=batch_size,
            topo_data=np.random.randn(
                batch_size,
                image_size_pixels,
                image_size_pixels,
            ),
            topo_x_coords=np.sort(np.random.randn(batch_size, image_size_pixels)),
            topo_y_coords=np.sort(np.random.randn(batch_size, image_size_pixels))[:, ::-1].copy(),
            # copy is needed as torch doesnt not support negative strides
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""
        if TOPOGRAPHIC_DATA in xr_dataset.keys():
            return TopographicML(
                batch_size=xr_dataset.data.shape[0],
                topo_data=xr_dataset.data,
                topo_x_coords=xr_dataset.x,
                topo_y_coords=xr_dataset.y,
            )
        else:
            return None

    def normalize(self):
        """Normalize the topological data"""
        if not self.normalized:
            self.topo_data = self.topo_data - TOPO_MEAN
            self.topo_data = self.topo_data / TOPO_STD
            self.normalized = True
