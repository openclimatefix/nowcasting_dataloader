""" batch functions """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.data_sources.datetime.datetime_model import DatetimeML, Datetime
from nowcasting_dataset.data_sources.gsp.gsp_model import GSPML, GSP
from nowcasting_dataset.data_sources.metadata.metadata_model import MetadataML, Metadata
from nowcasting_dataset.data_sources.nwp.nwp_model import (
    NWPML,
    NWP,
)
from nowcasting_dataset.data_sources.pv.pv_model import PVML, PV
from nowcasting_dataset.data_sources.satellite.satellite_model import SatelliteML, Satellite
from nowcasting_dataset.data_sources.sun.sun_model import SunML, Sun
from nowcasting_dataset.data_sources.topographic.topographic_model import TopographicML, Topographic
from nowcasting_dataset.dataset.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.data_sources.fake import (
    datetime_fake,
    metadata_fake,
    gsp_fake,
    pv_fake,
    satellite_fake,
    sun_fake,
    topographic_fake,
    nwp_fake,
)

_LOG = logging.getLogger(__name__)

register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()

data_sources = [Metadata, Satellite, Topographic, PV, Sun, GSP, NWP, Datetime]


class Batch(BaseModel):
    """
    Batch data object

    Contains the following data sources
    - gsp, satellite, topogrpahic, sun, pv, nwp and datetime.
    Also contains metadata of the class.

    All data sources are xr.Datasets

    """

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    metadata: Optional[Metadata]
    satellite: Optional[Satellite]
    topographic: Optional[Topographic]
    pv: Optional[PV]
    sun: Optional[Sun]
    gsp: Optional[GSP]
    nwp: Optional[NWP]
    datetime: Optional[Datetime]

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

    @staticmethod
    def fake(configuration: Configuration = Configuration()):
        """ Make fake batch object """
        batch_size = configuration.process.batch_size
        seq_length_5 = configuration.process.seq_length_5_minutes
        seq_length_30 = configuration.process.seq_length_30_minutes
        image_size_pixels = configuration.process.satellite_image_size_pixels

        return Batch(
            batch_size=batch_size,
            satellite=satellite_fake(
                batch_size=batch_size,
                seq_length_5=seq_length_5,
                satellite_image_size_pixels=image_size_pixels,
                number_sat_channels=len(configuration.process.sat_channels),
            ),
            nwp=nwp_fake(
                batch_size=batch_size,
                seq_length_5=seq_length_5,
                image_size_pixels=image_size_pixels,
                number_nwp_channels=len(configuration.process.nwp_channels),
            ),
            metadata=metadata_fake(batch_size=batch_size),
            pv=pv_fake(
                batch_size=batch_size, seq_length_5=seq_length_5, n_pv_systems_per_batch=128
            ),
            gsp=gsp_fake(batch_size=batch_size, seq_length_30=seq_length_30, n_gsp_per_batch=32),
            sun=sun_fake(batch_size=batch_size, seq_length_5=seq_length_5),
            topographic=topographic_fake(
                batch_size=batch_size, image_size_pixels=image_size_pixels
            ),
            datetime=datetime_fake(batch_size=batch_size, seq_length_5=seq_length_5),
        )

    def save_netcdf(self, batch_i: int, path: Path):
        """
        Save batch to netcdf file

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.

        """
        for data_source in self.data_sources:
            if data_source is not None:
                data_source.save_netcdf(batch_i=batch_i, path=path)

    @staticmethod
    def load_netcdf(local_netcdf_path: Union[Path, str], batch_idx: int):
        """Load batch from netcdf file"""
        data_sources_names = Example.__fields__.keys()

        # collect data sources
        batch_dict = {}
        for data_source_name in data_sources_names:

            local_netcdf_filename = os.path.join(
                local_netcdf_path, data_source_name, f"{batch_idx}.nc"
            )
            if os.path.exists(local_netcdf_filename):
                xr_dataset = xr.load_dataset(local_netcdf_filename)
            else:
                xr_dataset = None

            batch_dict[data_source_name] = xr_dataset

        batch_dict["batch_size"] = len(batch_dict["metadata"].example)

        return Batch(**batch_dict)


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

        t0_dt, time_5, time_30 = make_random_time_vectors(
            batch_size=process.batch_size,
            seq_length_5_minutes=process.seq_length_5_minutes,
            seq_length_30_minutes=process.seq_length_30_minutes,
        )

        return BatchML(
            batch_size=process.batch_size,
            metadata=MetadataML.fake(batch_size=process.batch_size, t0_dt=t0_dt),
            satellite=SatelliteML.fake(
                process.batch_size,
                process.seq_length_5_minutes,
                process.satellite_image_size_pixels,
                len(process.sat_channels),
                time_5=time_5,
            ),
            topographic=TopographicML.fake(
                batch_size=process.batch_size,
                image_size_pixels=process.satellite_image_size_pixels,
            ),
            pv=PVML.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_length_5_minutes,
                n_pv_systems_per_batch=128,
                time_5=time_5,
            ),
            sun=SunML.fake(
                batch_size=process.batch_size, seq_length_5=process.seq_length_5_minutes
            ),
            nwp=NWPML.fake(
                batch_size=process.batch_size,
                seq_length_5=process.seq_length_5_minutes,
                image_size_pixels=process.nwp_image_size_pixels,
                number_nwp_channels=len(process.nwp_channels),
                time_5=time_5,
            ),
            datetime=DatetimeML.fake(
                batch_size=process.batch_size, seq_length_5=process.seq_length_5_minutes
            ),
        )

    @staticmethod
    def from_batch(batch: Batch) -> BatchML:
        """ Change batch to ML batch """
        data_sources_names = Example.__fields__.keys()

        data_sources_dict = {}
        for data_source_name in data_sources_names:

            data_source = BatchML.__fields__[data_source_name].type_

            xr_dataset = getattr(batch, data_source_name)
            if xr_dataset is not None:

                data_sources_dict[data_source_name] = data_source.from_xr_dataset(xr_dataset)

        data_sources_dict["batch_size"] = data_sources_dict["satellite"].batch_size

        return BatchML(**data_sources_dict)
