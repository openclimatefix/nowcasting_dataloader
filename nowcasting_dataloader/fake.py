""" A class to create a fake dataset """
import torch
from nowcasting_dataset.config.model import Configuration

from nowcasting_dataloader.batch import BatchML


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, configuration: Configuration, length: int = 10):
        """
        Init

        Args:
            configuration: configuration object
            length: length of dataset
        """
        self.number_nwp_channels = len(configuration.input_data.nwp.nwp_channels)
        self.length = length
        self.configuration = configuration

    def __len__(self):
        """Number of pieces of data"""
        return self.length

    def per_worker_init(self, worker_id: int):
        """Nothing to do for FakeDataset"""
        pass

    def __getitem__(self, idx):
        """
        Get item, use for iter and next method

        Args:
            idx: batch index

        Returns: Dictionary of random data

        """
        x: BatchML = BatchML.fake(configuration=self.configuration)

        return x.dict()
