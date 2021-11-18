""" Dataset and functions"""
import logging
import os
from typing import List, Optional, Tuple, Union

import boto3
import gcsfs
import torch
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import DEFAULT_REQUIRED_KEYS
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
            position_encodings = generate_position_encodings_for_batch(batch, num_bands=16)
        # change batch into ML learning batch ready for training
        batch: BatchML = BatchML.from_batch(batch=batch)

        # netcdf_batch = xr.load_dataset(local_netcdf_filename)
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
