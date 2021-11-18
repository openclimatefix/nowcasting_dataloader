""" Dataset and functions"""
import logging
import os
from typing import List, Optional, Tuple, Union

import boto3
import gcsfs
import numpy as np
import torch
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    DEFAULT_REQUIRED_KEYS,
    GSP_ID,
    GSP_YIELD,
    NWP_DATA,
    PV_SYSTEM_ID,
    PV_YIELD,
    SATELLITE_DATA,
    TOPOGRAPHIC_DATA,
)
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.filesystem.utils import delete_all_files_in_temp_path, download_to_local
from nowcasting_dataset.utils import set_fsspec_for_multiprocess

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.subset import subselect_data
from nowcasting_dataloader.utils.position_encoding import generate_position_encodings_for_batch

logger = logging.getLogger(__name__)

"""
This file contains the following classes
NetCDFDataset- torch.utils.data.Dataset: Use for loading pre-made batches
NowcastingDataset - torch.utils.data.IterableDataset: Dataset for making batches
"""


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
        normalize: bool = False,
        add_position_encoding: bool = False,
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
            required_keys: Tuple or list of keys required in the example for
                it to be considered usable
            history_minutes: How many past minutes of data to use, if subsetting the batch
            forecast_minutes: How many future minutes of data to use, if reducing the amount
                of forecast time
            configuration: configuration object
            cloud: which cloud is used, can be "gcp", "aws" or "local".
            normalize: normalize the batch data
            add_position_encoding: Whether to add position encoding or not
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.cloud = cloud
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.configuration = configuration
        self.normalize = normalize
        self.add_position_encoding = add_position_encoding

        logger.info(f"Setting up NetCDFDataset for {src_path}")

        if self.forecast_minutes is None:
            self.forecast_minutes = configuration.input_data.default_forecast_minutes
        if self.history_minutes is None:
            self.history_minutes = configuration.input_data.default_history_minutes

        # see if we need to select the subset of data. If turned on -
        # only history_minutes + current time + forecast_minutes data is used.
        self.select_subset_data = False
        if self.forecast_minutes != configuration.input_data.default_forecast_minutes:
            self.select_subset_data = True
        if self.history_minutes != configuration.input_data.default_history_minutes:
            self.select_subset_data = True

        # Index into either sat_datetime_index or nwp_target_time indicating the current time,
        self.current_timestep_5_index = (
            int(configuration.input_data.default_history_minutes // 5) + 1
        )

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
        """Function called by a worker"""
        if self.cloud == "gcp":
            self.gcs = gcsfs.GCSFileSystem()
        elif self.cloud == "aws":
            self.s3_resource = boto3.resource("s3")

    def __len__(self):
        """Length of dataset"""
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> dict:
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

        if self.cloud in ["gcp", "aws"]:
            # TODO check this works for multiple files
            download_to_local(
                remote_filename=self.src_path,
                local_filename=self.tmp_path,
            )
            local_netcdf_folder = self.tmp_path
        else:
            local_netcdf_folder = self.src_path

        batch: Batch = Batch.load_netcdf(local_netcdf_folder, batch_idx=batch_idx)

        if self.select_subset_data:
            batch = subselect_data(
                batch=batch,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_5_index,
            )
        if self.add_position_encoding:
            position_encodings = generate_position_encodings_for_batch(batch)
        # change batch into ML learning batch ready for training
        batch: BatchML = BatchML.from_batch(batch=batch)

        if self.cloud != "local":
            # remove files in a folder, but not the folder itself
            delete_all_files_in_temp_path(self.src_path)

        # normalize the data
        if self.normalize:
            batch.normalize()

        batch: dict = batch.dict()
        if self.add_position_encoding:
            # Add position encodings
            batch.update(position_encodings)
        return batch


class SatFlowDataset(NetCDFDataset):
    """
    SatFlow dataset for training the joint model

    Dataset for training the joint model, ensures that the target contains the future GSP data, and
    satellite imagery. It also ensures that no past GSP data is given to the model as it wont be
    available at inference time. See issue #59 for more details

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
        normalize: bool = False,
        add_position_encoding: bool = False,
        add_satellite_target: bool = False,
        add_hrv_satellite_target: bool = False,
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
            required_keys: Tuple or list of keys required in the example for
                it to be considered usable
            history_minutes: How many past minutes of data to use, if subsetting the batch
            forecast_minutes: How many future minutes of data to use, if reducing the amount
                of forecast time
            configuration: configuration object
            cloud: which cloud is used, can be "gcp", "aws" or "local".
            normalize: normalize the batch data
            add_position_encoding: Whether to add position encoding or not
            add_satellite_target: Whether to add future satellite imagery to the target or not
            add_hrv_satellite_target: Whether to add future HRV satellite imagery to the target or not
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
            normalize,
            add_position_encoding,
        )

        self.add_satellite_target = add_satellite_target
        self.add_hrv_satellite_target = add_hrv_satellite_target

        # SatFlow specific changes, i.e. which timestep to split on
        self.current_timestep_index = (self.history_minutes // 5) + 1
        self.current_timestep_index_30 = (self.history_minutes // 30) + 1

    def __getitem__(self, batch_idx: int) -> Tuple[dict, dict]:
        """
        Satflow extension for the dataloader

        Args:
            batch_idx: Batch ID to load

        Returns:
            Tuple of dicts of torch.Tensors holding the data
        """
        batch = super().__getitem__(batch_idx)
        print(batch.keys())
        x = {}
        target = {}
        # Need to partition out past and future sat images here, along with the rest of the data
        if batch["satellite"].get(SATELLITE_DATA, False):
            past_satellite_data = batch["satellite"][SATELLITE_DATA][
                :, : self.current_timestep_index
            ]
            x[SATELLITE_DATA] = past_satellite_data
        if batch["hrvsatellite"].get(SATELLITE_DATA, False):
            past_hrv_satellite_data = batch["hrvsatellite"][SATELLITE_DATA][
                :, :, self.current_timestep_index :
            ]
            x["hrv_" + SATELLITE_DATA] = past_hrv_satellite_data
        if batch["pv"].get(PV_YIELD, False):
            past_pv_data = batch["pv"][PV_YIELD][:, :, self.current_timestep_index :]
            x[PV_YIELD] = past_pv_data
            x[PV_SYSTEM_ID] = batch["pv"][PV_SYSTEM_ID]
        if batch["nwp"].get(NWP_DATA, False):
            # We can give future NWP too, as that will be available
            x[NWP_DATA] = batch["nwp"][NWP_DATA]
        if batch["topographic"].get(TOPOGRAPHIC_DATA, False):
            # Need to expand dims to get a single channel one
            # Results in topographic maps with [Batch, Channel, H, W]
            x[TOPOGRAPHIC_DATA] = batch["topographic"][TOPOGRAPHIC_DATA]

        # Only GSP information we give to the model to train on is the IDs and physical locations
        x[GSP_ID] = GSP_ID

        # Now creating the target data
        target[GSP_YIELD] = batch["gsp"][GSP_YIELD][:, :, : self.current_timestep_index_30]
        target[GSP_ID] = GSP_ID

        if self.add_satellite_target:
            future_sat_data = batch["satellite"][SATELLITE_DATA][
                :, :, : self.current_timestep_index
            ]
            target[SATELLITE_DATA] = future_sat_data
        if self.add_hrv_satellite_target:
            future_hrv_sat_data = batch["hrvsatellite"][SATELLITE_DATA][
                :, :, : self.current_timestep_index
            ]
            target["hrv_" + SATELLITE_DATA] = future_hrv_sat_data

        # Add position encodings
        if self.add_position_encoding:
            if x.get(x[SATELLITE_DATA], False):
                x = self.add_encodings(
                    x, SATELLITE_DATA, batch, self.current_timestep_index, self.add_satellite_target
                )
            if x.get("hrv_" + SATELLITE_DATA, False):
                x = self.add_encodings(
                    x,
                    "hrv_" + SATELLITE_DATA,
                    batch,
                    self.current_timestep_index,
                    self.add_hrv_satellite_target,
                )
            if x.get(TOPOGRAPHIC_DATA, False):
                x = self.add_encodings(x, TOPOGRAPHIC_DATA, batch, 0, False)
            if x.get(NWP_DATA, False):
                x = self.add_encodings(x, NWP_DATA, batch, len(x[NWP_DATA]), False)
            if x.get(PV_YIELD, False):
                x = self.add_encodings(x, PV_YIELD, batch, self.current_timestep_index, False)
            # Add the future GSP position encoding for querying
            x[GSP_YIELD + "_query"] = batch[GSP_YIELD + "_position_encoding"][
                :, :, : self.current_timestep_index_30
            ]

        return x, target

    def add_encodings(
        self,
        x: dict,
        key: str,
        batch: dict,
        current_timestep_index: int,
        add_future_encodings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Adds encodings to the targets and inputs

        Args:
            x: Dictionary containing model inputs
            key: Key to check and insert
            batch: Batch of data to use
            current_timestep_index: The index of the t0 time in the data, used to split into past
                and future encodings
            add_future_encodings: Whether to add the future position encodings to the target

        Returns:
            x dictionary with the added/updated keys
        """
        if key + "_position_encoding" in batch:
            past_encoding = batch[key + "_position_encoding"][:, :, current_timestep_index:]
            x[key] = torch.cat([x[key], past_encoding], dim=1)
            if add_future_encodings:
                future_encoding = batch[key + "_position_encoding"][:, :, :current_timestep_index]
                x[key + "_query"] = future_encoding
        return x


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    1. Get fsspec ready for multi process
    2. To call NowcastingDataset.per_worker_init().
    """
    # fix for fsspec when using multprocess
    set_fsspec_for_multiprocess()

    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print("worker_info is None!")
    else:
        # The NowcastingDataset copy in this worker process.
        dataset_obj = worker_info.dataset
        dataset_obj.per_worker_init(worker_info.id)
