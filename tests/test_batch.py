import torch
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.model import Configuration



def test_batch_to_batch_ml():

    _ = BatchML.from_batch(batch=Batch.fake())


def test_fake_dataset():
    train = torch.utils.data.DataLoader(FakeDataset(configuration=Configuration()), batch_size=None)
    i = iter(train)
    x = next(i)

    x = BatchML(**x)
    # IT WORKS
    assert type(x.satellite.data) == torch.Tensor