""" batch functions """
from __future__ import annotations

import logging
from typing import Optional

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.time import make_random_time_vectors
from pydantic import BaseModel, Field

from nowcasting_dataloader.data_sources import GSPML, NWPML, PVML, SatelliteML, SunML, TopographicML
from nowcasting_dataloader.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)

_LOG = logging.getLogger(__name__)

register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


class Example(BaseModel):
    """
    Single Data item

    Note that this is currently not really used
    """

    satellite: Optional[SatelliteML]
    hrvsatellite: Optional[SatelliteML]
    topographic: Optional[TopographicML]
    pv: Optional[PVML]
    sun: Optional[SunML]
    gsp: Optional[GSPML]
    nwp: Optional[NWPML]

    @property
    def data_sources(self):
        """The different data sources"""
        return [
            self.satellite,
            self.hrvsatellite,
            self.topographic,
            self.pv,
            self.sun,
            self.gsp,
            self.nwp,
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

        seq_length_30 = int(
            (input_data.default_history_minutes + input_data.default_forecast_minutes) / 30 + 1
        )

        seq_length_60 = int(
            (input_data.default_history_minutes + input_data.default_forecast_minutes) / 60 + 1
        )

        time_vectors = make_random_time_vectors(
            batch_size=process.batch_size,
            seq_length_5_minutes=input_data.default_seq_length_5_minutes,
            seq_length_30_minutes=seq_length_30,
            seq_length_60_minutes=seq_length_60,
        )

        return BatchML(
            batch_size=process.batch_size,
            satellite=SatelliteML.fake(
                process.batch_size,
                input_data.default_seq_length_5_minutes,
                input_data.satellite.satellite_image_size_pixels,
                len(input_data.satellite.satellite_channels),
                time_5=time_vectors["time_5"],
            ),
            hrvsatellite=SatelliteML.fake(
                process.batch_size,
                input_data.default_seq_length_5_minutes,
                input_data.satellite.satellite_image_size_pixels,
                len(input_data.satellite.satellite_channels),
                time_5=time_vectors["time_5"],
            ),
            topographic=TopographicML.fake(
                batch_size=process.batch_size,
                image_size_pixels=input_data.satellite.satellite_image_size_pixels,
            ),
            pv=PVML.fake(
                batch_size=process.batch_size,
                seq_length_5=input_data.default_seq_length_5_minutes,
                n_pv_systems_per_batch=128,
                time_5=time_vectors["time_5"],
            ),
            gsp=GSPML.fake(
                process.batch_size,
                seq_length_30=seq_length_30,
                n_gsp_per_batch=32,
                time_30=time_vectors["time_30"],
            ),
            sun=SunML.fake(
                batch_size=process.batch_size, seq_length_5=input_data.default_seq_length_5_minutes
            ),
            nwp=NWPML.fake(
                batch_size=process.batch_size,
                seq_length_60=seq_length_60,
                image_size_pixels=input_data.nwp.nwp_image_size_pixels,
                number_nwp_channels=len(input_data.nwp.nwp_channels),
                time_60=time_vectors["time_60"],
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
                if "satellite" in data_source_name or "nwp" in data_source_name:
                    # Add in the channels being used
                    # Only need it from the first example
                    data_sources_dict[data_source_name].channels = xr_dataset["channels"][0].values
        data_sources_dict["batch_size"] = data_sources_dict["satellite"].batch_size

        return BatchML(**data_sources_dict)

    def normalize(self):
        """Normalize the batch"""

        # loop over all data sources and normalize
        for data_sources in self.data_sources:
            data_sources.normalize()
