""" A class to create a fake dataset """
import torch
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.utils.position_encoding import generate_position_encodings_for_batch


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(
        self,
        configuration: Configuration,
        length: int = 10,
        add_position_encoding: bool = False,
    ):
        """
        Init

        Args:
            configuration: configuration object
            length: length of dataset
            add_position_encoding: Whether to add position encoding or not
        """
        self.add_position_encoding = add_position_encoding
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
        if self.add_position_encoding:
            batch: Batch = Batch.fake(configuration=self.configuration)
            position_encodings = generate_position_encodings_for_batch(batch, num_bands=12)
            batch: BatchML = BatchML.from_batch(batch=batch)
            batch: dict = batch.dict()
            batch.update(position_encodings)
            return batch
        else:
            x: BatchML = BatchML.fake(configuration=self.configuration)
            return x.dict()
