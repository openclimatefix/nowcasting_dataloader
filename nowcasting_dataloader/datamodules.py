"""Datamodules to use for training"""
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from nowcasting_dataset.config import load_yaml_configuration
from nowcasting_dataset.config.model import Configuration
from pytorch_lightning import LightningDataModule

from nowcasting_dataloader.datasets import NetCDFDataset, SatFlowDataset, worker_init_fn
from nowcasting_dataloader.fake import FakeDataset

_LOG = logging.getLogger(__name__)


class SatFlowDataModule(LightningDataModule):
    """
    Example of LightningDataModule for SatFlow dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        temp_path: str,
        configuration: Union[Configuration, str],
        required_keys: Union[Tuple[str], List[str]] = None,
        history_minutes: Optional[int] = None,
        forecast_minutes: Optional[int] = None,
        normalize: bool = True,
        add_position_encoding: bool = False,
        add_satellite_target: bool = False,
        add_hrv_satellite_target: bool = False,
        data_sources_names: Optional[list[str]] = None,
        pin_memory: bool = True,
        num_workers: int = 1,
    ):
        """
        Datamodule for the SatFlow dataset

        Args:
            temp_path: temp path of data
            configuration: Configuration to use, or path to configuration file
            required_keys: Required keys for the dataset
            history_minutes: Number of history minutes to use
            forecast_minutes: Number of forecast minutes to use
            normalize: Whether to normalize the data
            add_position_encoding: Whether to add position encoding
            add_satellite_target: Whether to add satellite imagery to the targets
            add_hrv_satellite_target: Whether to add HRV satellite target
            data_sources_names: Names of data sources to load. Leave as None to load all.
            pin_memory: Whether to pin memory in the dataloader
            num_workers: Number of workers for each dataloader
        """
        super().__init__()
        self.temp_path = temp_path
        if type(configuration) == str:
            configuration = load_yaml_configuration(configuration)
        self.configuration: Configuration = configuration
        self.n_train_data = self.configuration.process.n_train_batches
        self.n_val_data = self.configuration.process.n_validation_batches
        self.n_test_data = self.configuration.process.n_test_batches
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.required_keys = required_keys
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.normalize = normalize
        self.add_position_encoding = add_position_encoding
        self.add_satellite_target = add_satellite_target
        self.add_hrv_satellite_target = add_hrv_satellite_target
        self.data_sources_names = data_sources_names

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        """Train dataloader"""
        train_dataset = SatFlowDataset(
            self.n_train_data,
            os.path.join(self.configuration.output_data.filepath, "train"),
            os.path.join(self.temp_path, "train"),
            configuration=self.configuration,
            required_keys=self.required_keys,
            history_minutes=self.history_minutes,
            forecast_minutes=self.forecast_minutes,
            normalize=self.normalize,
            add_position_encoding=self.add_position_encoding,
            add_satellite_target=self.add_satellite_target,
            add_hrv_satellite_target=self.add_hrv_satellite_target,
            data_sources_names=self.data_sources_names,
        )

        return torch.utils.data.DataLoader(train_dataset, shuffle=True, **self.dataloader_config)

    def val_dataloader(self):
        """Validation dataloader"""
        val_dataset = SatFlowDataset(
            self.n_val_data,
            os.path.join(self.configuration.output_data.filepath, "validation"),
            os.path.join(self.temp_path, "validation"),
            configuration=self.configuration,
            required_keys=self.required_keys,
            history_minutes=self.history_minutes,
            forecast_minutes=self.forecast_minutes,
            normalize=self.normalize,
            add_position_encoding=self.add_position_encoding,
            add_satellite_target=self.add_satellite_target,
            add_hrv_satellite_target=self.add_hrv_satellite_target,
            data_sources_names=self.data_sources_names,
        )

        return torch.utils.data.DataLoader(val_dataset, shuffle=False, **self.dataloader_config)

    def test_dataloader(self):
        """Test dataloader"""
        test_dataset = SatFlowDataset(
            self.n_test_data,
            os.path.join(self.configuration.output_data.filepath, "test"),
            os.path.join(self.temp_path, "test"),
            configuration=self.configuration,
            required_keys=self.required_keys,
            history_minutes=self.history_minutes,
            forecast_minutes=self.forecast_minutes,
            normalize=self.normalize,
            add_position_encoding=self.add_position_encoding,
            add_satellite_target=self.add_satellite_target,
            add_hrv_satellite_target=self.add_hrv_satellite_target,
            data_sources_names=self.data_sources_names,
        )

        return torch.utils.data.DataLoader(test_dataset, shuffle=False, **self.dataloader_config)


class NetCDFDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        temp_path: str = ".",
        n_train_data: int = 24900,
        n_val_data: int = 1000,
        n_test_data: int = 1000,
        num_workers: int = 8,
        pin_memory: bool = True,
        data_path="prepared_ML_training_data/v4/",
        fake_data: bool = False,
        shuffle_train: bool = False,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.temp_path = temp_path
        self.data_path = data_path
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.n_test_data = n_test_data
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fake_data = fake_data
        self.shuffle_train = shuffle_train

        filename = os.path.join(data_path, "configuration.yaml")
        _LOG.debug(f"Will be loading the configuration file {filename}")
        self.configuration = load_yaml_configuration(filename=filename)

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        """Get the train dataloader"""
        if self.fake_data:
            train_dataset = FakeDataset(configuration=self.configuration)
        else:
            train_dataset = NetCDFDataset(
                self.n_train_data,
                os.path.join(self.data_path, "train"),
                os.path.join(self.temp_path, "train"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(
            train_dataset, shuffle=self.shuffle_train, **self.dataloader_config
        )

    def val_dataloader(self):
        """Get the validation dataloader"""
        if self.fake_data:
            val_dataset = FakeDataset(configuration=self.configuration)
        else:
            val_dataset = NetCDFDataset(
                self.n_val_data,
                os.path.join(self.data_path, "validation"),
                os.path.join(self.temp_path, "validation"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(val_dataset, shuffle=False, **self.dataloader_config)

    def test_dataloader(self):
        """Get the test dataloader"""
        if self.fake_data:
            test_dataset = FakeDataset(configuration=self.configuration)
        else:
            test_dataset = NetCDFDataset(
                self.n_test_data,
                os.path.join(self.data_path, "test"),
                os.path.join(self.temp_path, "test"),
                configuration=self.configuration,
            )

        return torch.utils.data.DataLoader(test_dataset, shuffle=False, **self.dataloader_config)
