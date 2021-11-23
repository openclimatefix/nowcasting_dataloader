""" Test for xr utils """
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataloader.batch import BatchML


def test_xr_utils_time():
    """Test xr utils for time"""

    # setup config
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4

    # make batches
    batch = Batch.fake(configuration=con)
    batch_ml = BatchML.from_batch(batch=batch)

    # get time out
    x = batch_ml.gsp.gsp_datetime_index[0, :].numpy()
    time_out = pd.to_datetime(x, unit="ns").values

    # get time in
    time_in = batch.gsp.time[0].values

    # check
    assert time_in[0] == time_out[0]
    assert (time_out <= pd.Timestamp(batch.gsp.time.max().values)).all()
    assert (time_out >= pd.Timestamp(batch.gsp.time.min().values)).all()
