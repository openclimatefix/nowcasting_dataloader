""" Dataset and functions"""
import logging
from typing import List, Tuple, Union

import numpy as np
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    DATETIME_FEATURE_NAMES,
    GSP_YIELD,
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_DATA,
    SATELLITE_DATETIME_INDEX,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    TOPOGRAPHIC_DATA,
)

from nowcasting_dataloader.datasets import NetCDFDataset

logger = logging.getLogger(__name__)


class SatFlowDataset(NetCDFDataset):
    """Loads data saved by the `prepare_ml_training_data.py` script."""

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
    ):
        """
        Extension to NetCDFDataset for specific Satflow model training

        Args:
            n_batches: Number of batches
            src_path: The source path for the training files
            tmp_path: The temporary path to use if streaming from a remote filesystem
            configuration: Nowcasting Configuration to use
            cloud: Which cloud is being used, either 'gcp', 'aws', or 'local' for local filesystem
            required_keys: What keys are required in the final batch
            history_minutes: Number of history minutes to use
            forecast_minutes: Number of forecast minutes to use
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
        self.current_timestep_index = (history_minutes // 5) + 1
        self.current_timestep_index_30 = (history_minutes // 30) + 1

    def __getitem__(self, batch_idx: int) -> Tuple[dict, dict]:
        """
        Satflow extension for the dataloader

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
