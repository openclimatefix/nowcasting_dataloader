"""Test NetCDF Dataset"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
import pytest
import torch
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.consts import GSP_DATETIME_INDEX, NWP_DATA, PV_YIELD, SATELLITE_DATA
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.datasets import NetCDFDataset, worker_init_fn

torch.set_default_dtype(torch.float32)


def test_netcdf_dataset_local_using_configuration():
    """Test netcdf locally"""
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

        train_dataset = NetCDFDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            cloud="local",
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
        assert sat_data.shape == (4, 1, 19, 64, 64)
        assert batch_ml.nwp.data.shape == (4, 1, 2, 64, 64)
        assert batch_ml.topographic.topo_data.shape == (4, 64, 64)
        assert batch_ml.pv.pv_yield.shape == (4, 19, 128)
        assert batch_ml.gsp.gsp_yield.shape == (4, 4, 32)
        assert batch_ml.sun.sun_azimuth_angle.shape == (4, 19)
        assert batch_ml.sun.sun_elevation_angle.shape == (4, 19)

        assert type(batch_ml.nwp.data) == torch.Tensor
        assert batch_ml.nwp.data[0, 0, 0, 0, 0].dtype == torch.float32

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


def test_netcdf_dataset_local_using_configuration_subset_of_data_sources():
    """Test netcdf locally"""
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

        train_dataset = NetCDFDataset(
            1,
            DATA_PATH,
            TEMP_PATH,
            cloud="local",
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

        # Make sure file isn't deleted!
        assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))


@pytest.mark.skip("CD does not have access to GCS")
def test_get_dataloaders_gcp(configuration: Configuration):
    """Test dataloader on GCP"""
    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v6/"
    TEMP_PATH = "../nowcasting_dataset"

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, "train"),
        os.path.join(TEMP_PATH, "train"),
        configuration=configuration,
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    train_dataset.per_worker_init(1)
    t = iter(train_dataset)
    data: BatchML = next(t)

    # image
    z = data.satellite.sat_data[0][0][:, :, 0]
    _ = data.gsp.gsp_yield[0][:, 0]

    _ = pd.to_datetime(data.satellite.sat_datetime_index[0, 0], unit="s")

    fig = go.Figure(data=go.Contour(z=z))

    plotly.offline.plot(fig, filename="../filename.html", auto_open=True)


@pytest.mark.skip("CD does not have access to AWS")
def test_get_dataloaders_aws(configuration: Configuration):
    """Test dataloader on AWS"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        TEMP_PATH = Path(tmpdirname)
        DATA_PATH = "prepared_ML_training_data/v4/"

        os.mkdir(os.path.join(TEMP_PATH, "train"))

        train_dataset = NetCDFDataset(
            24_900,
            os.path.join(DATA_PATH, "train"),
            os.path.join(TEMP_PATH, "train"),
            cloud="aws",
            configuration=configuration,
        )

        dataloader_config = dict(
            pin_memory=True,
            num_workers=2,
            prefetch_factor=8,
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

        assert SATELLITE_DATA in data.keys()


@pytest.mark.skip("CD does not have access to GCP")
def test_required_keys_gcp(configuration: Configuration):
    """More test on GCP dataloader"""
    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v6/"
    TEMP_PATH = "../nowcasting_dataset"
    if os.path.isdir(os.path.join(TEMP_PATH, "train")):
        os.removedirs(os.path.join(TEMP_PATH, "train"))
    os.mkdir(os.path.join(TEMP_PATH, "train"))

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, "train"),
        os.path.join(TEMP_PATH, "train"),
        cloud="gcp",
        configuration=configuration,
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
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

    assert SATELLITE_DATA in data.keys()
    assert PV_YIELD not in data.keys()
    assert GSP_DATETIME_INDEX in data.keys()


@pytest.mark.skip("CD does not have access to GCP")
def test_subsetting_gcp(configuration: Configuration):
    """Test subsetting GCP dataloader"""
    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/"
    TEMP_PATH = "../nowcasting_dataset"
    if os.path.isdir(os.path.join(TEMP_PATH, "train")):
        os.removedirs(os.path.join(TEMP_PATH, "train"))
    os.mkdir(os.path.join(TEMP_PATH, "train"))

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, "train"),
        os.path.join(TEMP_PATH, "train"),
        cloud="gcp",
        history_minutes=10,
        forecast_minutes=10,
        configuration=configuration,
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
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

    sat_data = data[SATELLITE_DATA]

    # Sat is in 5min increments, so should have 2 history + current + 2 future
    assert sat_data.shape[1] == 5
    assert data[NWP_DATA].shape[2] == 5
