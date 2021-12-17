"""Test SatFlow Dataset"""
import os
import tempfile
from pathlib import Path

import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.datasets import SatFlowDataset, worker_init_fn

torch.set_default_dtype(torch.float32)


def test_satflow_dataset_local_using_configuration():
    """Test satflow locally"""
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.nwp.nwp_channels = c.input_data.nwp.nwp_channels[0:1]
    c.input_data.satellite.satellite_channels = c.input_data.satellite.satellite_channels[0:2]
    configuration = c

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=c)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = SatFlowDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=10,
            forecast_minutes=10,
            configuration=configuration,
        )

        dataloader_config = dict(
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

        _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

        train_dataset.per_worker_init(1)
        t = iter(train_dataset)
        x, y = next(t)

        for k in [
            "pv_yield",
            "pv_system_id",
            "nwp",
            "topo_data",
            "gsp_id",
            "sat_data",
            "hrv_sat_data",
        ]:
            assert k in x.keys()
            assert type(x[k]) == torch.Tensor
            assert x[k].dtype == torch.float32

        for k in [
            "gsp_yield",
            "gsp_id",
        ]:
            assert k in y.keys()
            assert type(y[k]) == torch.Tensor
            assert y[k].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_satflow_dataset_local_using_configuration_with_position_encoding():
    """Test satflow locally"""
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.satellite.satellite_image_size_pixels = 24
    configuration = c

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=c)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = SatFlowDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=10,
            forecast_minutes=10,
            configuration=configuration,
            add_position_encoding=True,
            add_hrv_satellite_target=True,
            add_satellite_target=True,
        )

        dataloader_config = dict(
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

        _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

        train_dataset.per_worker_init(1)
        t = iter(train_dataset)
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
            assert x[k].dtype == torch.float32

        for k in ["gsp_yield", "gsp_id", "sat_data", "hrv_sat_data"]:
            assert k in y.keys()
            assert type(y[k]) == torch.Tensor
            assert y[k].dtype == torch.float32
        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_satflow_dataset_local_using_configuration_with_position_encoding_subset_of_sources():
    """Test satflow locally"""
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.satellite.satellite_image_size_pixels = 24
    configuration = c

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=c)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = SatFlowDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=10,
            forecast_minutes=10,
            configuration=configuration,
            add_position_encoding=True,
            add_hrv_satellite_target=True,
            add_satellite_target=False,
            data_sources_names=["gsp", "pv", "hrvsatellite"],
        )

        dataloader_config = dict(
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

        _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

        train_dataset.per_worker_init(1)
        t = iter(train_dataset)
        x, y = next(t)

        for k in [
            "pv_yield",
            "pv_system_id",
            "gsp_id",
            "hrv_sat_data_query",
            "gsp_yield_query",
            "hrv_sat_data",
        ]:
            assert k in x.keys()
            assert type(x[k]) == torch.Tensor
            assert x[k].dtype == torch.float32

        assert x["pv_yield"].shape == (4, 37, 3, 128)

        for k in ["gsp_yield", "gsp_id", "hrv_sat_data"]:
            assert k in y.keys()
            assert type(y[k]) == torch.Tensor
            assert y[k].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_zero_pv_systems():
    c = Configuration()
    c.input_data = InputData.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.satellite.satellite_image_size_pixels = 24
    configuration = c
    batch = Batch.fake(configuration=c)
    x = BatchML.from_batch(batch)
    x = x.dict()
    x["pv"]["pv_yield"][0, :, 100:] = float("nan")
    x["pv"]["pv_yield"][2, :, 64:] = float("nan")
    x["gsp"]["gsp_yield"][0, :, 30:] = float("nan")
    x["gsp"]["gsp_yield"][3, :, 15:] = float("nan")
    x["pv_yield"] = torch.unsqueeze(x["pv"]["pv_yield"], dim=1)
    x["gsp_yield"] = torch.unsqueeze(x["gsp"]["gsp_yield"], dim=1)
    dset = SatFlowDataset(
        1,
        ".",
        ".",
        history_minutes=10,
        forecast_minutes=10,
        configuration=configuration,
        add_position_encoding=True,
        add_hrv_satellite_target=True,
        add_satellite_target=True,
    )
    cleaned = dset.zero_out_nan_pv_systems(x)
    assert torch.isnan(cleaned["pv_yield"]).sum() == 0
    assert torch.isnan(cleaned["gsp_yield"]).sum() == 0
    assert torch.isclose(torch.sum(cleaned["gsp_yield"][0, :, :, 30:]), torch.zeros(1))
    assert not torch.isclose(torch.sum(cleaned["gsp_yield"][1, :, :, 30:]), torch.zeros(1))
