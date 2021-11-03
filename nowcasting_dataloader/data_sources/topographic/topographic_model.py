""" Model for Topogrpahic features """
import logging

import numpy as np
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from pydantic import Field, validator

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML

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
    def fake(batch_size, image_size_pixels):
        """Create fake data"""
        return TopographicML(
            batch_size=batch_size,
            topo_data=np.random.randn(
                batch_size,
                image_size_pixels,
                image_size_pixels,
            ).astype(np.float32),
            topo_x_coords=np.sort(
                np.random.randn(batch_size, image_size_pixels).astype(np.float32)
            ),
            topo_y_coords=np.sort(
                np.random.randn(batch_size, image_size_pixels).astype(np.float32)
            )[:, ::-1].copy(),
            # copy is needed as torch doesnt not support negative strides
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        topographic_batch_ml = xr_dataset.torch.to_tensor(["data", "x", "y"])

        topographic_batch_ml[TOPOGRAPHIC_DATA] = topographic_batch_ml.pop("data")
        topographic_batch_ml["topo_x_coords"] = topographic_batch_ml.pop("x")
        topographic_batch_ml["topo_y_coords"] = topographic_batch_ml.pop("y")

        return TopographicML(**topographic_batch_ml)

    def normalize(self):
        """Normalize the topological data"""
        if not self.normalized:
            self.topo_data = self.topo_data - TOPO_MEAN
            self.topo_data = self.topo_data / TOPO_STD
            self.normalized = True
