import os
import tempfile
from pathlib import Path

import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.datamodules import SatFlowDataModule
import glob
torch.set_default_dtype(torch.float32)


def test_satflow_datamodule_init():
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.satellite.satellite_image_size_pixels = 24
    configuration = c

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=c)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))
        print(list(glob.glob(tmpdirname+"/*")))
        DATA_PATH = tmpdirname
        configuration.output_data.filepath = DATA_PATH
        TEMP_PATH = tmpdirname

        datamodule = SatFlowDataModule(
            temp_path=TEMP_PATH,
            configuration=configuration,
            n_train_data=1,
            n_val_data=0,
            n_test_data=0,
            cloud="local",
            add_position_encoding=True,
            add_satellite_target=True,
            add_hrv_satellite_target=True,
            history_minutes=10,
            forecast_minutes=10,
        )

        t = iter(datamodule.train_dataloader())
        x, y = next(t)

        for k in [
            "pv_yield",
            "pv_system_id",
            "nwp",
            "topo_data",
            "gsp_id",
            "sat_data_query",
            "hrv_sat_data_query",
            "gsp_yield_query",
            "sat_data",
            "hrv_sat_data",
        ]:
            assert k in x.keys()
            assert type(x[k]) == torch.Tensor

        for k in ["gsp_yield", "gsp_id", "sat_data", "hrv_sat_data"]:
            assert k in y.keys()
            assert type(y[k]) == torch.Tensor
        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))
