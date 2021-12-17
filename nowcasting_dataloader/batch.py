""" batch functions """
from __future__ import annotations

import logging
from typing import Optional

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch
from pydantic import BaseModel

from nowcasting_dataloader.data_sources import (
    GSPML,
    NWPML,
    PVML,
    MetadataML,
    OpticalFlowML,
    SatelliteML,
    SunML,
    TopographicML,
)
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
    opticalflow: Optional[OpticalFlowML]
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
            self.opticalflow,
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

    metadata: MetadataML

    @staticmethod
    def fake(configuration: Configuration = Configuration()):
        """Create fake batch"""

        batch: Batch = Batch.fake(configuration=configuration)
        batch: BatchML = BatchML.from_batch(batch=batch)

        return batch

    @staticmethod
    def from_batch(batch: Batch) -> BatchML:
        """Change batch to ML batch"""
        data_sources_names = Example.__fields__.keys()

        data_sources_dict = {}
        for data_source_name in data_sources_names:

            data_source = BatchML.__fields__[data_source_name].type_
            if not hasattr(batch, data_source_name):
                continue
            xr_dataset = getattr(batch, data_source_name)
            if xr_dataset is not None:
                try:
                    data_sources_dict[data_source_name] = data_source.from_xr_dataset(xr_dataset)
                except Exception as e:
                    _LOG.error(
                        f"Could not change xr dataset to " f"pydantic model for {data_source_name}"
                    )
                    raise e

                if (
                    "satellite" in data_source_name
                    or "nwp" in data_source_name
                    or "opticalflow" in data_source_name
                ):
                    # Add in the channels being used
                    # Only need it from the first example
                    data_sources_dict[data_source_name].channels = xr_dataset["channels"][0].values

        data_sources_dict["metadata"] = batch.metadata.dict()

        return BatchML(**data_sources_dict)

    def normalize(self):
        """Normalize the batch"""

        # loop over all data sources and normalize
        for data_sources in self.data_sources:
            if data_sources is not None:
                data_sources.normalize()
