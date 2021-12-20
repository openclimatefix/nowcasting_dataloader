""" Dataset and functions"""
import logging
import os
from typing import List, Optional, Tuple, Union

import einops
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
from nowcasting_dataloader.utils.position_encoding import generate_position_encodings_for_batch

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
        add_position_encoding: bool = False,
        data_sources_names: Optional[list[str]] = None,
        num_bands: int = 4,
        mix_two_batches: bool = True,
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
            mix_two_batches: option to mix tow batches together
            seed: random seed for peaking second batch when mixing two batches
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.configuration = configuration
        self.normalize = normalize
        self.add_position_encoding = add_position_encoding
        self.seed = seed
        self.mix_two_batches = mix_two_batches

        self.num_bands = num_bands
        if data_sources_names is None:
            data_sources_names = list(Example.__fields__.keys())
        self.data_sources_names = data_sources_names

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
            os.mkdir(self.tmp_path)

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

        if self.select_subset_data:
            batch = subselect_data(
                batch=batch,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_5_index,
            )
        if self.add_position_encoding:
            position_encodings = generate_position_encodings_for_batch(
                batch, num_bands=self.num_bands
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
        if self.add_position_encoding:
            # Add position encodings
            batch.update(position_encodings)
        return batch


class SatFlowDataset(NetCDFDataset):
    """SatFlow dataset for filtering and splitting up output from NetCDFDataset properly"""

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
        add_position_encoding: bool = False,
        add_satellite_target: bool = False,
        add_hrv_satellite_target: bool = False,
        data_sources_names: Optional[list[str]] = None,
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
            add_satellite_target: Whether to add future satellite imagery to the target or not
            add_hrv_satellite_target: Whether to add future HRV satellite imagery to the target
            data_sources_names: Names of data sources to load, if not using all of them
        """
        super().__init__(
            n_batches=n_batches,
            src_path=src_path,
            tmp_path=tmp_path,
            configuration=configuration,
            required_keys=required_keys,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            normalize=normalize,
            add_position_encoding=add_position_encoding,
            data_sources_names=data_sources_names,
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
            Tuple of dicts of torch.Tensors holding the data, with the first one being the inputs
            and the second being the target
        """
        batch = super().__getitem__(batch_idx)
        x = {}
        target = {}
        # Need to partition out past and future sat images here, along with the rest of the data
        if "satellite" in self.data_sources_names and len(batch["satellite"].get("data", [])) > 0:
            past_satellite_data = batch["satellite"]["data"][:, :, : self.current_timestep_index]
            x["satellite"] = past_satellite_data
        if (
            "hrvsatellite" in self.data_sources_names
            and len(batch["hrvsatellite"].get("data", [])) > 0
        ):
            past_hrv_satellite_data = batch["hrvsatellite"]["data"][
                :, :, : self.current_timestep_index
            ]
            x["hrvsatellite"] = past_hrv_satellite_data
        if "pv" in self.data_sources_names and len(batch["pv"].get(PV_YIELD, [])) > 0:
            past_pv_data = torch.unsqueeze(
                batch["pv"][PV_YIELD][:, : self.current_timestep_index], dim=1
            )
            x[PV_YIELD] = past_pv_data
            x[PV_SYSTEM_ID] = torch.nan_to_num(batch["pv"][PV_SYSTEM_ID])
        if "nwp" in self.data_sources_names and len(batch["nwp"].get("data", [])) > 0:
            # We can give future NWP too, as that will be available
            x[NWP_DATA] = batch["nwp"]["data"]
        if (
            "topographic" in self.data_sources_names
            and len(batch["topographic"].get(TOPOGRAPHIC_DATA, [])) > 0
        ):
            # Need to expand dims to get a single channel one
            # Results in topographic maps with [Batch, Channel, H, W]
            x[TOPOGRAPHIC_DATA] = torch.unsqueeze(
                torch.unsqueeze(batch["topographic"][TOPOGRAPHIC_DATA], dim=1), dim=1
            )

        if "gsp" in self.data_sources_names:
            # Only GSP information we give to the model to train on is the IDs & physical locations
            x[GSP_ID] = torch.nan_to_num(batch["gsp"][GSP_ID])

            # Now creating the target data, only want the first GSP as the target
            target[GSP_YIELD] = batch["gsp"][GSP_YIELD][:, self.current_timestep_index_30 :, 0]
            target[GSP_ID] = batch["gsp"][GSP_ID][:, 0]
            # Add timestep, so we can compare results better
            target[GSP_DATETIME_INDEX] = batch["gsp"][GSP_DATETIME_INDEX][
                :, self.current_timestep_index_30 - 1 :
            ]
            target["gsp_capacity"] = batch["gsp"]["gsp_capacity"][
                :, self.current_timestep_index_30 :, 0
            ]

        if self.add_satellite_target:
            future_sat_data = batch["satellite"]["data"][:, :, self.current_timestep_index :]
            target[SATELLITE_DATA] = future_sat_data
        if self.add_hrv_satellite_target:
            future_hrv_sat_data = batch["hrvsatellite"]["data"][:, :, self.current_timestep_index :]
            target["hrv_" + SATELLITE_DATA] = future_hrv_sat_data

        # Add position encodings
        if self.add_position_encoding:
            if len(x.get("satellite", [])) > 0:
                x = self.add_encodings(
                    x, "satellite", batch, self.current_timestep_index, self.add_satellite_target
                )
                if self.add_hrv_satellite_target:
                    x[SATELLITE_DATA + "_query"] = x.pop("satellite_query")
            if len(x.get("hrvsatellite", [])) > 0:
                x = self.add_encodings(
                    x,
                    "hrvsatellite",
                    batch,
                    self.current_timestep_index,
                    self.add_hrv_satellite_target,
                )
                if self.add_hrv_satellite_target:
                    x["hrv_" + SATELLITE_DATA + "_query"] = x.pop("hrvsatellite_query")
            if len(x.get(TOPOGRAPHIC_DATA, [])) > 0:
                x[TOPOGRAPHIC_DATA] = torch.cat(
                    [x[TOPOGRAPHIC_DATA], batch["topographic_position_encoding"]], dim=1
                )
            if len(x.get(NWP_DATA, [])) > 0:
                x[NWP_DATA] = torch.cat(
                    [x[NWP_DATA], batch[NWP_DATA + "_position_encoding"]], dim=1
                )
            if len(x.get(PV_YIELD, [])) > 0:
                past_encoding = batch["pv_position_encoding"][:, :, : self.current_timestep_index]
                x[PV_YIELD] = torch.cat([x[PV_YIELD], past_encoding], dim=1)

            if "gsp" in self.data_sources_names:
                # Add the future GSP position encoding for querying
                x[GSP_YIELD + "_query"] = batch["gsp_position_encoding"][
                    :, :, self.current_timestep_index_30 :
                ]

        # Rename to match other ones better
        if len(x.get("satellite", [])) > 0:
            x[SATELLITE_DATA] = x.pop("satellite")
        if len(x.get("hrvsatellite", [])) > 0:
            x["hrv_" + SATELLITE_DATA] = x.pop("hrvsatellite")

        # Zero out NaN PV and GSP Yield
        x = self.zero_out_nan_pv_systems(x)

        # Convert all to float32 if not already done so
        for k in x.keys():
            if x[k].dtype != torch.float32:
                x[k] = x[k].float()
        for k in target.keys():
            if target[k].dtype != torch.float32:
                target[k] = target[k].float()

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
            past_encoding = batch[key + "_position_encoding"][:, :, :current_timestep_index]
            x[key] = torch.cat([x[key], past_encoding], dim=1)
            if add_future_encodings:
                future_encoding = batch[key + "_position_encoding"][:, :, current_timestep_index:]
                x[key + "_query"] = future_encoding
        return x

    def zero_out_nan_pv_systems(self, x: dict) -> dict:
        """
        Zeros out NaN PV systems and their position encodings

        This takes advantage of the fact that NaN values for PV and GSP systems are always at the
        end of the example, so just need to find the first one and can zero out everything after.

        This should be used after the position encodings are appended, so that it zeros out all the
        channels as well

        Args:
            x: dictionary of inputs

        Returns:
            The dictionary x with the PV systems zeroed out and position encodings zeroed as well
        """

        for key in [PV_YIELD, GSP_YIELD]:
            if key in x:
                mask = torch.isnan(x[key][:, 0, 0, :])  # Only looking at the yield
                mask = einops.repeat(mask, "b id -> b c t id", c=x[key].shape[1], t=x[key].shape[2])
                # Zero out for all entries related to
                x[key][mask] = 0.0

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
