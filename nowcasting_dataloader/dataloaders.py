""" Dataset and functions"""
import logging
import os
from typing import List, Tuple, Union, Optional

import boto3
import gcsfs
import numpy as np
import pandas as pd
import torch
import xarray as xr

from nowcasting_dataset import utils as nd_utils
from nowcasting_dataset.filesystem.utils import download_to_local, delete_all_files_in_temp_path
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    GSP_YIELD,
    GSP_DATETIME_INDEX,
    SATELLITE_DATA,
    NWP_DATA,
    PV_YIELD,
    SUN_ELEVATION_ANGLE,
    SUN_AZIMUTH_ANGLE,
    SATELLITE_DATETIME_INDEX,
    NWP_TARGET_TIME,
    PV_DATETIME_INDEX,
    DEFAULT_REQUIRED_KEYS,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_Y_COORDS,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_Y_COORDS,
    SATELLITE_X_COORDS,
    DATETIME_FEATURE_NAMES,
)
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SAT_VARIABLE_NAMES

from nowcasting_dataset.dataset.batch import Batch

logger = logging.getLogger(__name__)

"""
This file contains the following classes
NetCDFDataset- torch.utils.data.Dataset: Use for loading pre-made batches
NowcastingDataset - torch.utils.data.IterableDataset: Dataset for making batches
"""

SAT_MEAN = xr.DataArray(
    data=[
        93.23458,
        131.71373,
        843.7779,
        736.6148,
        771.1189,
        589.66034,
        862.29816,
        927.69586,
        90.70885,
        107.58985,
        618.4583,
        532.47394,
    ],
    dims=["sat_variable"],
    coords={"sat_variable": list(SAT_VARIABLE_NAMES)},
).astype(np.float32)

SAT_STD = xr.DataArray(
    data=[
        115.34247,
        139.92636,
        36.99538,
        57.366386,
        30.346825,
        149.68007,
        51.70631,
        35.872967,
        115.77212,
        120.997154,
        98.57828,
        99.76469,
    ],
    dims=["sat_variable"],
    coords={"sat_variable": list(SAT_VARIABLE_NAMES)},
).astype(np.float32)

_LOG = logging.getLogger(__name__)


class NetCDFDataset(torch.utils.data.Dataset):
    """
    Loads data saved by the `prepare_ml_training_data.py` script.

    Moved from predict_pv_yield
    """

    def __init__(
        self,
        n_batches: int,
        src_path: str,
        tmp_path: str,
        configuration: Configuration,
        cloud: str = "gcp",
        required_keys: Union[Tuple[str], List[str]] = None,
        history_minutes: Optional[int] = None,
        forecast_minutes: Optional[int] = None,
    ):
        """
        Netcdf Dataset

        Args:
            n_batches: Number of batches available on disk.
            src_path: The full path (including 'gs://') to the data on
                Google Cloud storage.
            tmp_path: The full path to the local temporary directory
                (on a local filesystem).
            cloud:
            required_keys: Tuple or list of keys required in the example for it to be considered usable
            history_minutes: How many past minutes of data to use, if subsetting the batch
            forecast_minutes: How many future minutes of data to use, if reducing the amount of forecast time
            configuration: configuration object
            cloud: which cloud is used, can be "gcp", "aws" or "local".
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.cloud = cloud
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.configuration = configuration

        logger.info(f"Setting up NetCDFDataset for {src_path}")

        if self.forecast_minutes is None:
            self.forecast_minutes = configuration.process.forecast_minutes
        if self.history_minutes is None:
            self.history_minutes = configuration.process.history_minutes

        # see if we need to select the subset of data. If turned on -
        # only history_minutes + current time + forecast_minutes data is used.
        self.select_subset_data = False
        if self.forecast_minutes != configuration.process.forecast_minutes:
            self.select_subset_data = True
        if self.history_minutes != configuration.process.history_minutes:
            self.select_subset_data = True

        # Index into either sat_datetime_index or nwp_target_time indicating the current time,
        self.current_timestep_5_index = int(configuration.process.history_minutes // 5) + 1

        if required_keys is None:
            required_keys = DEFAULT_REQUIRED_KEYS
        self.required_keys = list(required_keys)

        # setup cloud connections as None
        self.gcs = None
        self.s3_resource = None

        assert cloud in ["gcp", "aws", "local"]

        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)

    def per_worker_init(self, worker_id: int):
        """ Function called by a worker """
        if self.cloud == "gcp":
            self.gcs = gcsfs.GCSFileSystem()
        elif self.cloud == "aws":
            self.s3_resource = boto3.resource("s3")

    def __len__(self):
        """ Length of dataset """
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> Batch:
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        logger.debug(f"Getting batch {batch_idx}")
        if not 0 <= batch_idx < self.n_batches:
            raise IndexError(
                "batch_idx must be in the range" f" [0, {self.n_batches}), not {batch_idx}!"
            )
        netcdf_filename = nd_utils.get_netcdf_filename(batch_idx)
        # remote_netcdf_folder = os.path.join(self.src_path, netcdf_filename)
        # local_netcdf_filename = os.path.join(self.tmp_path, netcdf_filename)

        if self.cloud in ["gcp", "aws"]:
            # TODO check this works for mulitple files
            download_to_local(
                remote_filename=self.src_path,
                local_filename=self.tmp_path,
            )
            local_netcdf_folder = self.tmp_path
        else:
            local_netcdf_folder = self.src_path

        batch = Batch.load_netcdf(local_netcdf_folder, batch_idx=batch_idx)
        # netcdf_batch = xr.load_dataset(local_netcdf_filename)
        if self.cloud != "local":
            # remove files in a folder, but not the folder itself
            delete_all_files_in_temp_path(self.src_path)

        # batch = example.xr_to_example(batch_xr=netcdf_batch, required_keys=self.required_keys)

        # Todo this may should be done when the data is created
        if SATELLITE_DATA in self.required_keys:
            sat_data = batch.satellite.sat_data
            if sat_data.dtype == np.int16:
                sat_data = sat_data.astype(np.float32)
                sat_data = sat_data - SAT_MEAN
                sat_data = sat_data / SAT_STD
                batch.satellite.sat_data = sat_data

        if self.select_subset_data:
            batch = subselect_data(
                batch=batch,
                required_keys=self.required_keys,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_5_index,
            )

        batch.change_type_to_numpy()

        return batch


class SatFlowDataset(NetCDFDataset):
    """Loads data saved by the `prepare_ml_training_data.py` script.
    Adapted from predict_pv_yield
    """

    def __init__(
        self,
        n_batches: int,
        src_path: str,
        tmp_path: str,
        configuration: Configuration,
        cloud: str = "gcp",
        required_keys: Union[Tuple[str], List[str]] = [
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
            SATELLITE_DATETIME_INDEX,
            TOPOGRAPHIC_DATA,
        ]
        + list(DATETIME_FEATURE_NAMES),
        history_minutes: int = 30,
        forecast_minutes: int = 60,
        combine_inputs: bool = False,
    ):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        batch_size: Batch size, if requested, will subset data along batch dimension
        """
        super().__init__(
            n_batches,
            src_path,
            tmp_path,
            configuration,
            cloud,
            required_keys,
            history_minutes,
            forecast_minutes,
        )
        # SatFlow specific changes, i.e. which timestep to split on
        self.required_keys = list(required_keys)
        self.combine_inputs = combine_inputs
        self.current_timestep_index = (history_minutes // 5) + 1
        self.current_timestep_index_30 = (history_minutes // 30) + 1

    def __getitem__(self, batch_idx: int) -> Tuple[dict, dict]:
        """
        SatFlow extension for the dataloader, splitting up the past and future images/data to give to the model

        Args:
            batch_idx: Batch ID to load

        Returns:
            Tuple of dicts of torch.Tensors holding the data
        """
        batch = super().__getitem__(batch_idx)

        # Need to partition out past and future sat images here, along with the rest of the data
        past_satellite_data = batch[SATELLITE_DATA][:, : self.current_timestep_index]
        past_gsp_data = batch[GSP_YIELD][:, : self.current_timestep_index_30]
        future_gsp_data = batch[GSP_YIELD][:, self.current_timestep_index_30 :]

        future_sat_data = batch[SATELLITE_DATA][:, self.current_timestep_index :]
        x = {
            SATELLITE_DATA: past_satellite_data,
            SATELLITE_X_COORDS: batch.get(SATELLITE_X_COORDS, None),
            SATELLITE_Y_COORDS: batch.get(SATELLITE_Y_COORDS, None),
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][
                :, : self.current_timestep_index
            ],
            GSP_YIELD: past_gsp_data,
        }
        y = {
            SATELLITE_DATA: future_sat_data,
            SATELLITE_DATETIME_INDEX: batch[SATELLITE_DATETIME_INDEX][
                :, self.current_timestep_index :
            ],
            GSP_YIELD: future_gsp_data,
        }

        for k in list(DATETIME_FEATURE_NAMES):
            if k in self.required_keys:
                x[k] = batch[k][:, : self.current_timestep_index]

        if NWP_DATA in self.required_keys:
            past_nwp_data = batch[NWP_DATA][:, :, : self.current_timestep_index]
            x[NWP_DATA] = past_nwp_data
            x[NWP_X_COORDS] = batch.get(NWP_X_COORDS, None)
            x[NWP_Y_COORDS] = batch.get(NWP_Y_COORDS, None)

        if TOPOGRAPHIC_DATA in self.required_keys:
            # Need to expand dims to get a single channel one
            # Results in topographic maps with [Batch, Channel, H, W]
            x[TOPOGRAPHIC_DATA] = np.expand_dims(batch[TOPOGRAPHIC_DATA], axis=1)

        return x, y


def subselect_data(
    batch: Batch,
    required_keys: Union[Tuple[str], List[str]],
    history_minutes: int,
    forecast_minutes: int,
    current_timestep_index: Optional[int] = None,
) -> Batch:
    """
    Subselects the data temporally. This function selects all data within the time range [t0 - history_minutes, t0 + forecast_minutes]

    Args:
        batch: Example dictionary containing at least the required_keys
        required_keys: The required keys present in the dictionary to use
        current_timestep_index: The index into either SATELLITE_DATETIME_INDEX or NWP_TARGET_TIME giving the current timestep
        history_minutes: How many minutes of history to use
        forecast_minutes: How many minutes of future data to use for forecasting

    Returns:
        Example with only data between [t0 - history_minutes, t0 + forecast_minutes] remaining
    """
    _LOG.debug(
        f"Select sub data with new historic minutes of {history_minutes} "
        f"and forecast minutes if {forecast_minutes}"
    )

    # We are subsetting the data, so we need to select the t0_dt, i.e the time now for eahc Example.
    # We infact only need this from the first example in each batch
    if current_timestep_index is None:
        # t0_dt or if not available use a different datetime index
        t0_dt_of_first_example = batch.metadata.t0_dt[0].values
    else:
        if SATELLITE_DATA in required_keys:
            t0_dt_of_first_example = batch.satellite.sat_datetime_index[
                0, current_timestep_index
            ].values
        else:
            t0_dt_of_first_example = batch.satellite.sat_datetime_index[
                0, current_timestep_index
            ].values

    # make this a datetime object
    t0_dt_of_first_example = pd.to_datetime(t0_dt_of_first_example)

    if batch.satellite is not None:
        batch.satellite.select_time_period(
            keys=[SATELLITE_DATA, SATELLITE_DATETIME_INDEX],
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for NWP, if used
    if batch.nwp is not None:
        batch.nwp.select_time_period(
            keys=[NWP_DATA, NWP_TARGET_TIME],
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )
    #
    # Now for GSP, if used
    if batch.gsp is not None:
        batch.gsp.select_time_period(
            keys=[GSP_DATETIME_INDEX, GSP_YIELD],
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for PV, if used
    if batch.pv is not None:
        batch.pv.select_time_period(
            keys=[PV_DATETIME_INDEX, PV_YIELD],
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for SUN, if used
    if batch.sun is not None:
        batch.sun.select_time_period(
            keys=[SUN_ELEVATION_ANGLE, SUN_AZIMUTH_ANGLE],
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # DATETIME TODO

    return batch
