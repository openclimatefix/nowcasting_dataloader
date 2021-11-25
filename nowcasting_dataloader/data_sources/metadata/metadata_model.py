""" Model for output of GSP data """
import logging

from nowcasting_dataset.data_sources.fake import metadata_fake
from pydantic import Field

from nowcasting_dataloader.data_sources.datasource_output import Array, DataSourceOutputML

logger = logging.getLogger(__name__)


class MetadataML(DataSourceOutputML):
    """Model for output of GSP data"""

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    t0_datetime_utc: Array = Field(
        ...,
        description="The t0s of each example ",
    )

    x_center_osgb: Array = Field(
        ...,
        description="The x centers of each example in OSGB coordinates",
    )

    y_center_osgb: Array = Field(
        ...,
        description="The y centers of each example in OSGB coordinates",
    )

    @staticmethod
    def fake(batch_size):
        """Make a fake GSP object"""
        metadata = metadata_fake(batch_size=batch_size)

        return MetadataML(**metadata.dict())
