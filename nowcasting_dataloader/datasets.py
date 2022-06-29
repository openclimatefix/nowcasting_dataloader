""" Dataset and functions"""
import logging
import os
from typing import List, Optional, Tuple, Union

import einops
import fsspec
import numpy as np
import torch
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    DEFAULT_REQUIRED_KEYS,
    GSP_DATETIME_INDEX,
    GSP_ID,
    GSP_YIELD,
    NWP_DATA,
    PV_SYSTEM_ID,
    PV_YIELD,
    SATELLITE_DATA,
    TOPOGRAPHIC_DATA,
)
from nowcasting_dataset.dataset.batch import Batch, Example, join_two_batches
from nowcasting_dataset.filesystem.utils import delete_all_files_in_temp_path
from nowcasting_dataset.utils import set_fsspec_for_multiprocess

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.subset import subselect_data

logger = logging.getLogger(__name__)


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
        required_keys: Union[Tuple[str], List[str]] = None,
        history_minutes: Optional[int] = None,
        forecast_minutes: Optional[int] = None,
        normalize: bool = True,
        data_sources_names: Optional[list[str]] = None,
        nwp_channels: Optional[list[str]] = None,
        num_bands: int = 4,
        mix_two_batches: bool = True,
        save_first_batch: Optional[str] = None,
        seed: bool = 234,
    ):
        """
        Netcdf Dataset

        Args:
            n_batches: Number of batches available on disk.
            src_path: The full path (including 'gs://') to the data on
                Google Cloud storage.
            tmp_path: The full path to the local temporary directory
                (on a local filesystem).
            required_keys: Tuple or list of keys required in the example for
                it to be considered usable
            history_minutes: How many past minutes of data to use, if subsetting the batch
            forecast_minutes: How many future minutes of data to use, if reducing the amount
                of forecast time
            configuration: configuration object
            normalize: normalize the batch data
            add_position_encoding: Whether to add position encoding or not
            data_sources_names: Names of data sources to load, if not using all of them
            num_bands: Number of bands for the Fourier features for the position encoding
            mix_two_batches: option to mix two batches together
            save_first_batch: Option to save the first generated batch to disk
            seed: random seed for peaking second batch when mixing two batches
            nwp_channels: Useful for training to be able to reduce the number of channels
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.configuration = configuration
        self.normalize = normalize
        self.seed = seed
        self.mix_two_batches = mix_two_batches
        self.save_first_batch = save_first_batch

        self.num_bands = num_bands
        if data_sources_names is None:
            data_sources_names = list(Example.__fields__.keys())
            data_sources_names = [
                data_source_name
                for data_source_name in data_sources_names
                if getattr(self.configuration.input_data, data_source_name) is not None
            ]
        self.data_sources_names = data_sources_names

        self.nwp_channels = nwp_channels

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

        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path, exist_ok=True)

        # set seed
        np.random.seed(seed)

        if len(self) == 1:
            logger.warning("Wanted to mix batches but there is only one")
            self.mix_two_batches = False

    def per_worker_init(self, worker_id: int):
        """Function called by a worker"""

        # adjust temp path for each worker
        if self.src_path != self.tmp_path:
            self.tmp_path = f"{self.tmp_path}/{worker_id}"
            os.mkdir(f"{self.tmp_path}")

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

        # get batches indexes
        batch_indexes = [batch_idx]
        if self.mix_two_batches:
            second_batch_idx = np.random.randint(0, len(self) - 1)

            # make sure second index is not the same as the first one
            if second_batch_idx == batch_idx:
                second_batch_idx = (second_batch_idx + 1) % len(self)

            batch_indexes.append(second_batch_idx)

        # download batches
        batches = []
        for batch_idx in batch_indexes:

            if self.src_path != self.tmp_path:

                batch = Batch.download_batch_and_load_batch(
                    batch_idx=batch_idx,
                    data_sources_names=self.data_sources_names,
                    tmp_path=self.tmp_path,
                    src_path=self.src_path,
                )
            else:
                batch: Batch = Batch.load_netcdf(
                    self.src_path,
                    batch_idx=batch_idx,
                    data_sources_names=self.data_sources_names,
                )
            batches.append(batch)

        # join batches
        batch = join_two_batches(batches=batches, data_sources_names=self.data_sources_names)

        if self.nwp_channels is not None:
            # get the channel index we need
            # assume the all examples have the same channel names
            channels = batch.nwp.channels[0]
            channels_index = [
                i for i, channel in enumerate(channels) if channel in self.nwp_channels
            ]

            # reduce nwp data to only the channels we want
            batch.nwp = batch.nwp.sel(channels_index=channels_index)

        if self.select_subset_data:
            batch = subselect_data(
                batch=batch,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_5_index,
            )

        # change batch into ML learning batch ready for training
        try:
            batch: BatchML = BatchML.from_batch(batch=batch)
        except Exception as e:
            logger.error(
                f"Could not change Batch to BatchML " f"for batch index {batch_idx}, {batch}"
            )
            raise e

        if self.src_path != self.tmp_path:
            # remove files in a folder, but not the folder itself
            delete_all_files_in_temp_path(self.tmp_path)

        # normalize the data
        if self.normalize:
            batch.normalize()

        batch: dict = batch.dict()

        if self.save_first_batch is not None and batch_idx == 0:
            # Save out the dictionary to disk
            np.save("tmp.npy", batch)
            fs = fsspec.open(self.save_first_batch).fs
            fs.put("tmp.npy", self.save_first_batch)
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
