"""Test NetCDF Dataset"""
import os
import tempfile
from pathlib import Path

import torch
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.datasets import NetCDFDataset, worker_init_fn

torch.set_default_dtype(torch.float32)


def test_netcdf_dataset_local_using_configuration_on_one_batch(configuration):
    """Test netcdf locally, just loading one batch"""
    configuration.input_data.nwp.nwp_channels = configuration.input_data.nwp.nwp_channels[0:2]
    configuration.input_data.satellite.satellite_channels = (
        configuration.input_data.satellite.satellite_channels[0:2]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = NetCDFDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=30,
            forecast_minutes=60,
            configuration=configuration,
            normalize=False,
            mix_two_batches=False,
            nwp_channels=configuration.input_data.nwp.nwp_channels[0:1],
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
        data = next(t)

        batch_ml = BatchML(**data)

        sat_data = batch_ml.satellite.data
        # TODO
        # Sat is in 5min increments, so should have 2 history + current + 2 future
        assert sat_data.shape == (4, 2, 19, 21, 21)
        assert batch_ml.nwp.data.shape == (4, 1, 3, 64, 64)
        assert batch_ml.topographic.topo_data.shape == (4, 64, 64)

        assert len(batch_ml.metadata.t0_datetime_utc) == 4
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_netcdf_dataset_local_using_configuration(configuration):
    """Test netcdf locally, mix two batches"""

    configuration.input_data.nwp.nwp_channels = configuration.input_data.nwp.nwp_channels[0:1]
    configuration.input_data.satellite.satellite_channels = (
        configuration.input_data.satellite.satellite_channels[0:2]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=1, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = NetCDFDataset(
            2,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=30,
            forecast_minutes=60,
            configuration=configuration,
            normalize=False,
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
        data = next(t)

        batch_ml = BatchML(**data)

        sat_data = batch_ml.satellite.data

        # TODO
        # Sat is in 5min increments, so should have 2 history + current + 2 future
        assert sat_data.shape == (4, 2, 19, 21, 21)
        assert batch_ml.nwp.data.shape == (4, 1, 3, 64, 64)
        assert batch_ml.topographic.topo_data.shape == (4, 64, 64)

        assert len(batch_ml.metadata.t0_datetime_utc) == 4
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_netcdf_dataset_local_using_configuration_with_saving_locally(configuration):
    """Test netcdf locally, mix two batches"""

    configuration.input_data.nwp.nwp_channels = configuration.input_data.nwp.nwp_channels[0:1]
    configuration.input_data.satellite.satellite_channels = (
        configuration.input_data.satellite.satellite_channels[0:2]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=1, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = NetCDFDataset(
            2,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=30,
            forecast_minutes=60,
            configuration=configuration,
            normalize=False,
            mix_two_batches=False,
            save_first_batch=os.path.join(tmpdirname, "saved_batch.npy"),
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
        data = next(t)

        batch_ml = BatchML(**data)

        sat_data = batch_ml.satellite.data

        # TODO
        # Sat is in 5min increments, so should have 2 history + current + 2 future
        assert sat_data.shape == (4, 2, 19, 21, 21)
        assert batch_ml.nwp.data.shape == (4, 1, 3, 64, 64)
        assert batch_ml.topographic.topo_data.shape == (4, 64, 64)

        assert len(batch_ml.metadata.t0_datetime_utc) == 4
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))

        # Make sure that the batch is written out!
        assert os.path.exists(os.path.join(tmpdirname, "saved_batch.npy"))


def test_netcdf_dataset_local_using_configuration_subset_of_data_sources(configuration):
    """Test netcdf locally"""

    configuration.input_data.nwp.nwp_channels = configuration.input_data.nwp.nwp_channels[0:1]
    configuration.input_data.satellite.satellite_channels = (
        configuration.input_data.satellite.satellite_channels[0:2]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=0, path=Path(tmpdirname))

        DATA_PATH = tmpdirname
        TEMP_PATH = tmpdirname

        train_dataset = NetCDFDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=30,
            forecast_minutes=60,
            configuration=configuration,
            normalize=False,
            data_sources_names=["pv", "gsp", "hrvsatellite"],
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
        data = next(t)

        batch_ml = BatchML(**data)

        assert batch_ml.nwp is None
        assert batch_ml.topographic is None
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun is None
        assert batch_ml.satellite is None
        assert batch_ml.hrvsatellite is not None
        assert batch_ml.hrvsatellite.data.shape == (4, 1, 19, 64, 64)

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_netcdf_dataset_copy_from_data_path(configuration):
    """Test netcdf locally"""
    configuration.input_data.nwp.nwp_channels = configuration.input_data.nwp.nwp_channels[0:1]
    configuration.input_data.satellite.satellite_channels = (
        configuration.input_data.satellite.satellite_channels[0:2]
    )

    with tempfile.TemporaryDirectory() as tmpdirname, tempfile.TemporaryDirectory() as data_path:

        f = Batch.fake(configuration=configuration)
        f.save_netcdf(batch_i=0, path=Path(data_path))
        assert os.path.exists(os.path.join(data_path, "satellite/000000.nc"))

        DATA_PATH = data_path
        TEMP_PATH = tmpdirname

        train_dataset = NetCDFDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            history_minutes=30,
            forecast_minutes=60,
            configuration=configuration,
            normalize=False,
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
        assert os.path.exists(os.path.join(data_path, "satellite/000000.nc"))
        _ = next(t)

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))
