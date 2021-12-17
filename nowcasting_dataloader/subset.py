""" Take subsets of xr.datasets """
import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
from nowcasting_dataset.dataset.batch import Batch

logger = logging.getLogger(__name__)


def subselect_data(
    batch: Batch,
    history_minutes: int,
    forecast_minutes: int,
    current_timestep_index: Optional[int] = None,
) -> Batch:
    """
    Subselects the data temporally.

    This function selects all data within the time range [t0-history_minutes, t0+forecast_minutes]

    Args: batch: Example dictionary containing at least the required_keys required_keys: The
    required keys present in the dictionary to use current_timestep_index: The index into either
    SATELLITE_DATETIME_INDEX or NWP_TARGET_TIME giving the current timestep history_minutes: How
    many minutes of history to use forecast_minutes: How many minutes of future data to use for
    forecasting

    Returns:
        Example with only data between [t0 - history_minutes, t0 + forecast_minutes] remaining
    """
    logger.debug(
        f"Select sub data with new historic minutes of {history_minutes} "
        f"and forecast minutes if {forecast_minutes}"
    )

    # We are subsetting the data, so we need to select the t0_dt, i.e the time now for each Example.
    # We in fact only need this from the first example in each batch
    if current_timestep_index is None:
        # t0_dt or if not available use a different datetime index
        t0_dt_of_first_example = pd.Timestamp(batch.metadata.t0_dt[0].values[0])
    else:
        if batch.satellite is not None:
            t0_dt_of_first_example = batch.satellite.time[0, current_timestep_index].values
        elif batch.hrvsatellite is not None:
            t0_dt_of_first_example = batch.hrvsatellite.time[0, current_timestep_index].values
        elif batch.pv is not None:
            t0_dt_of_first_example = batch.pv.time[0, current_timestep_index].values
        else:
            t0_dt_of_first_example = batch.nwp.time[0, current_timestep_index].values

    if batch.satellite is not None:
        batch.satellite = select_time_period(
            x=batch.satellite,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for NWP, if used
    if batch.nwp is not None:
        batch.nwp = select_time_period(
            x=batch.nwp,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for GSP, if used
    if batch.gsp is not None:
        batch.gsp = select_time_period(
            x=batch.gsp,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for PV, if used
    if batch.pv is not None:
        batch.pv = select_time_period(
            x=batch.pv,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    # Now for SUN, if used
    if batch.sun is not None:
        batch.sun = select_time_period(
            x=batch.sun,
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            t0_dt_of_first_example=t0_dt_of_first_example,
        )

    return batch


def select_time_period(
    x,
    history_minutes: int,
    forecast_minutes: int,
    t0_dt_of_first_example: Union[datetime, pd.Timestamp],
):
    """
    Selects a subset of data between the indicies of [start, end] for each key in keys

    Note that class is edited so nothing is returned.

    Args:
        x: dataset that is ot be reduced
        t0_dt_of_first_example: datetime of the current time (t0) in the first example of the batch
        history_minutes: How many minutes of history to use
        forecast_minutes: How many minutes of future data to use for forecasting

    """
    logger.debug(
        f"Taking a sub-selection of the batch data based on a history minutes of {history_minutes} "
        f"and forecast minutes of {forecast_minutes}"
    )

    start_time_of_first_example = t0_dt_of_first_example - pd.to_timedelta(
        f"{history_minutes} minute 30 second"
    )
    end_time_of_first_example = t0_dt_of_first_example + pd.to_timedelta(
        f"{forecast_minutes} minute 30 second"
    )

    logger.debug(f"New start time for first example is {start_time_of_first_example}")
    logger.debug(f"New end time for first example is {end_time_of_first_example}")

    if hasattr(x, "time"):
        time_of_first_example = pd.to_datetime(x.time[0].values)

    else:
        # for nwp, maybe rename
        time_of_first_example = pd.to_datetime(x.target_time[0].values)

    # find the start and end index, that we will then use to slice the data
    start_i, end_i = np.searchsorted(
        time_of_first_example, [start_time_of_first_example, end_time_of_first_example]
    )

    # slice all the data
    y = x.where(((x.time_index >= start_i) & (x.time_index < end_i)), drop=True)

    # the `where` statement seems to add extra dimension to data variables,
    # if they don't originally have `time_index` dimension
    for key in y.keys():
        if "time_index" not in getattr(x, key).dims:
            y.__setitem__(key, getattr(x, key))

    return y
