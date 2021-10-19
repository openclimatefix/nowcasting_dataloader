""" Dataset and functions"""
import logging
import os
from concurrent import futures
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple, Callable, Union, Optional

import boto3
import gcsfs
import numpy as np
import pandas as pd
import torch
import xarray as xr

from nowcasting_dataset import data_sources
from nowcasting_dataset import utils as nd_utils
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    SATELLITE_DATA,
    DEFAULT_REQUIRED_KEYS,
)
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.dataset.batch import BatchML, Batch
from nowcasting_dataset.dataset.subset import subselect_data
from nowcasting_dataset.filesystem.utils import download_to_local, delete_all_files_in_temp_path
from nowcasting_dataset.utils import set_fsspec_for_multiprocess

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

        batch = Batch.load_netcdf(local_netcdf_folder, batch_idx=batch_idx)

        if self.select_subset_data:
            batch = subselect_data(
                batch=batch,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_5_index,
            )

        # change batch into ML learning batch ready for training
        batch = BatchML.from_batch(batch=batch)

        # netcdf_batch = xr.load_dataset(local_netcdf_filename)
        if self.cloud != "local":
            # remove files in a folder, but not the folder itself
            delete_all_files_in_temp_path(self.src_path)

        # Todo issue - https://github.com/openclimatefix/nowcasting_dataset/issues/231
        if SATELLITE_DATA in self.required_keys:
            sat_data = batch.satellite.data
            if sat_data.dtype == np.int16:
                sat_data = sat_data.astype(np.float32)
                sat_data = sat_data - SAT_MEAN
                sat_data = sat_data / SAT_STD
                batch.satellite.data = sat_data

        return batch.dict()


@dataclass
class NowcastingDataset(torch.utils.data.IterableDataset):
    """
    The first data_source will be used to select the geo locations each batch.
    """

    batch_size: int
    n_batches_per_epoch_per_worker: int
    #: Number of times to re-use each timestep. Must exactly divide batch_size.
    n_samples_per_timestep: int
    data_sources: List[data_sources.DataSource]
    t0_datetimes: pd.DatetimeIndex  #: Valid t0 datetimes.
    collate_fn: Callable = torch.utils.data._utils.collate.default_collate

    # useful way to skip batches if creating dataset fails halfway through.
    # This might not be that useful, as re-running creation of datasets may cause off issues like duplicate data.
    skip_batch_index: int = 0
    batch_index: int = 0

    def __post_init__(self):
        """ Post Init """
        super().__init__()
        self._per_worker_init_has_run = False
        self._n_timesteps_per_batch = self.batch_size // self.n_samples_per_timestep

        # Sanity checks.
        if self.batch_size % self.n_samples_per_timestep != 0:
            raise ValueError("n_crops_per_timestep must exactly divide batch_size!")
        if len(self.t0_datetimes) < self._n_timesteps_per_batch:
            raise ValueError(
                f"start_dt_index only has {len(self.start_dt_index)}"
                " timestamps."
                f"  Must have at least {self._n_timesteps_per_batch}!"
            )

        if self.skip_batch_index > 0:
            _LOG.warning(f"Will be skipping {self.skip_batch_index}, is this correct?")

    def per_worker_init(self, worker_id: int) -> None:
        """
        Called by worker_init_fn on each copy of NowcastingDataset

        This happens after the worker process has been spawned.
        """
        # Each worker must have a different seed for its random number gen.
        # Otherwise all the workers will output exactly the same data!
        self.worker_id = worker_id
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)

        # Initialise each data_source.
        for data_source in self.data_sources:
            _LOG.debug(f"Opening {type(data_source).__name__}")
            data_source.open()

        # fix for fsspecs
        set_fsspec_for_multiprocess()

        self._per_worker_init_has_run = True

    def __iter__(self):
        """Yields a complete batch at a time."""
        if not self._per_worker_init_has_run:
            raise RuntimeError("per_worker_init() must be run!")
        for _ in range(self.n_batches_per_epoch_per_worker):
            yield self._get_batch()

    def _get_batch(self) -> Batch:

        _LOG.debug(f"Getting batch {self.batch_index}")

        self.batch_index += 1
        if self.batch_index < self.skip_batch_index:
            _LOG.debug(f"Skipping batch {self.batch_index}")
            return []

        t0_datetimes = self._get_t0_datetimes_for_batch()
        x_locations, y_locations = self._get_locations_for_batch(t0_datetimes)

        examples = {}
        n_threads = len(self.data_sources)
        with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit tasks to the executor.
            future_examples_per_source = []
            for data_source in self.data_sources:

                future_examples = executor.submit(
                    data_source.get_batch,
                    t0_datetimes=t0_datetimes,
                    x_locations=x_locations,
                    y_locations=y_locations,
                )
                future_examples_per_source.append(future_examples)

            # Collect results from each thread.
            for future_examples in future_examples_per_source:
                examples_from_source = future_examples.result()

                # print(type(examples_from_source))
                name = type(examples_from_source).__name__.lower()
                examples[name] = examples_from_source

        examples["batch_size"] = len(t0_datetimes)

        return Batch(**examples)

    def _get_t0_datetimes_for_batch(self) -> pd.DatetimeIndex:
        # Pick random datetimes.
        t0_datetimes = self.rng.choice(
            self.t0_datetimes, size=self._n_timesteps_per_batch, replace=False
        )
        # Duplicate these random datetimes.
        t0_datetimes = np.tile(t0_datetimes, reps=self.n_samples_per_timestep)
        return pd.DatetimeIndex(t0_datetimes)

    def _get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        return self.data_sources[0].get_locations_for_batch(t0_datetimes)


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
