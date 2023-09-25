import os
import tempfile
from pathlib import Path

import torch
from nowcasting_dataset.config.save import save_yaml_configuration
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.datamodules import NetCDFDataModule

torch.set_default_dtype(torch.float32)


def test_netcdf_datamodule_init(configuration):
    configuration.input_data.satellite.satellite_image_size_pixels_height = 24
    configuration.input_data.satellite.satellite_image_size_pixels_width = 24
    configuration.process.n_test_batches = 0
    configuration.process.n_validation_batches = 0
    configuration.process.n_train_batches = 1

    with tempfile.TemporaryDirectory() as tmpdirname:
        f = Batch.fake(configuration=configuration)
        train_tmp = os.path.join(tmpdirname, "train")
        f.save_netcdf(batch_i=0, path=Path(train_tmp))
        f.save_netcdf(batch_i=1, path=Path(train_tmp))

        configuration.output_data.filepath = tmpdirname
        save_yaml_configuration(configuration, filename=f"{tmpdirname}/configuration.yaml")

        datamodule = NetCDFDataModule(temp_path=tmpdirname, data_path=tmpdirname, n_train_data=2)

        t = iter(datamodule.train_dataloader())
        x = next(t)

        for k in [
            "pv",
            "gsp",
            "nwp",
            "satellite",
        ]:
            assert k in x.keys()
            assert type(x[k]) == dict

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(tmpdirname, "train", "nwp/000000.nc"))
