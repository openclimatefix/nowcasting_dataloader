"""Test NetCDF Dataset"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
import pytest
import torch
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import GSP_DATETIME_INDEX, NWP_DATA, PV_YIELD, SATELLITE_DATA

import nowcasting_dataloader
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataloader.datasets import NetCDFDataset, worker_init_fn


def test_netcdf_dataset_local_using_configuration(configuration: Configuration):
    """Test netcdf locally"""
    DATA_PATH = os.path.join(
        os.path.dirname(nowcasting_dataloader.__file__), "../tests", "data", "batch"
    )
    TEMP_PATH = os.path.join(
        os.path.dirname(nowcasting_dataloader.__file__), "../tests", "data", "batch", "temp"
    )

    train_dataset = NetCDFDataset(
        1,
        DATA_PATH,
        TEMP_PATH,
        cloud="local",
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
    data = next(t)

    batch_ml = BatchML(**data)

    sat_data = batch_ml.satellite.data
    # TODO
    # Sat is in 5min increments, so should have 2 history + current + 2 future
    assert sat_data.shape[1] == 5
    assert batch_ml.nwp.data.shape == (4, 5, 64, 64, 1)
    assert batch_ml.topographic.topo_data.shape == (4, 64, 64)
    assert batch_ml.metadata.t0_dt.shape == (4,)
    assert batch_ml.pv.pv_yield.shape == (4, 5, 128)
    assert batch_ml.gsp.gsp_yield.shape == (4, 5, 32)
    assert batch_ml.sun.sun_azimuth_angle.shape == (4, 5)
    assert batch_ml.sun.sun_elevation_angle.shape == (4, 5)

    # Make sure file isn't deleted!
    assert os.path.exists(os.path.join(DATA_PATH, "metadata/0.nc"))


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
