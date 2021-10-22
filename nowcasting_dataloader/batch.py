""" batch functions """
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataloader.data_sources import (
    TopographicML,
    SatelliteML,
    MetadataML,
    PVML,
    SunML,
    GSPML,
    NWPML,
    DatetimeML,
)

_LOG = logging.getLogger(__name__)

register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


class Example(BaseModel):
    """
    Single Data item

    Note that this is currently not really used
    """

    metadata: Optional[MetadataML]
    satellite: Optional[SatelliteML]
    topographic: Optional[TopographicML]
    pv: Optional[PVML]
    sun: Optional[SunML]
    gsp: Optional[GSPML]
    nwp: Optional[NWPML]
    datetime: Optional[DatetimeML]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.topographic,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
            self.datetime,
            self.metadata,
        ]


class BatchML(Example):
    """
    Batch data object.

    Contains the following data sources
    - gsp, satellite, topogrpahic, sun, pv, nwp and datetime.
    Also contains metadata of the class

    """

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    @staticmethod
    def fake(configuration: Configuration = Configuration()):
        """Create fake batch"""
        process = configuration.process
        input_data = configuration.input_data

        t0_dt, time_5, time_30 = make_random_time_vectors(
            batch_size=process.batch_size,
            seq_length_5_minutes=input_data.default_seq_length_5_minutes,
            seq_length_30_minutes=input_data.default_seq_length_5_minutes // 6,
        )

        return BatchML(
            batch_size=process.batch_size,
            metadata=MetadataML.fake(batch_size=process.batch_size, t0_dt=t0_dt),
            satellite=SatelliteML.fake(
                process.batch_size,
                input_data.default_seq_length_5_minutes,
                input_data.satellite.satellite_image_size_pixels,
                len(input_data.satellite.sat_channels),
                time_5=time_5,
            ),
            topographic=TopographicML.fake(
                batch_size=process.batch_size,
                image_size_pixels=input_data.satellite.satellite_image_size_pixels,
            ),
            pv=PVML.fake(
                batch_size=process.batch_size,
                seq_length_5=input_data.default_seq_length_5_minutes,
                n_pv_systems_per_batch=128,
                time_5=time_5,
            ),
            sun=SunML.fake(
                batch_size=process.batch_size, seq_length_5=input_data.default_seq_length_5_minutes
            ),
            nwp=NWPML.fake(
                batch_size=process.batch_size,
                seq_length_5=input_data.default_seq_length_5_minutes,
                image_size_pixels=input_data.nwp.nwp_image_size_pixels,
                number_nwp_channels=len(input_data.nwp.nwp_channels),
                time_5=time_5,
            ),
            datetime=DatetimeML.fake(
                batch_size=process.batch_size, seq_length_5=input_data.default_seq_length_5_minutes
            ),
        )

    @staticmethod
    def from_batch(batch: Batch) -> BatchML:
        """Change batch to ML batch"""
        data_sources_names = Example.__fields__.keys()

        data_sources_dict = {}
        for data_source_name in data_sources_names:

            data_source = BatchML.__fields__[data_source_name].type_

            xr_dataset = getattr(batch, data_source_name)
            if xr_dataset is not None:

                data_sources_dict[data_source_name] = data_source.from_xr_dataset(xr_dataset)

        data_sources_dict["batch_size"] = data_sources_dict["satellite"].batch_size

        return BatchML(**data_sources_dict)

    def normalize(self):
        """Normalize the batch"""

        # loop over all data sources and normalize
        for data_sources in self.data_sources:
            data_sources.normalize()
