"""Functions for computing the optical flow on the fly for satellite images"""
import logging

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_dataset.dataset.batch import Batch

_LOG = logging.getLogger("nowcasting_dataset")

IMAGE_BUFFER_SIZE = 16


def compute_optical_flow_for_batch(
    batch: Batch, previous_timestep_for_flow: int, final_image_size_pixels: int
) -> Batch:
    """
    Computes the optical flow for satellite images in the batch

    Assumes metadata is also in Batch, for getting t0

    Args:
        batch: Batch containing at least metadata and satellite data

    Returns:
        Batch with the optical flow added
    """

    assert batch.satellite is not None, (
        "Satellite data does not exist in batch, required for " "optical flow"
    )
    assert batch.metadata is not None, "Metadata does not exist in batch, required for optical flow"

    # Only do optical flow for satellite data
    for i in range(batch.batch_size):
        satellite_data = batch.satellite[i]
        t0_dt = batch.metadata.t0_dt.values[i]
        batch.optical_flow = _compute_and_return_optical_flow(satellite_data, t0_dt)

    return batch


def _compute_previous_timestep(
    satellite_data: xr.DataArray, t0_dt: pd.Timestamp, previous_timestep_for_flow: int
) -> pd.Timestamp:
    """
    Get timestamp of previous

    Args:
        satellite_data: Satellite data to use
        t0_dt: Timestamp

    Returns:
        The previous timesteps
    """
    satellite_data = satellite_data.where(satellite_data.time <= t0_dt, drop=True)
    return satellite_data.isel(
        time=len(satellite_data.time) - previous_timestep_for_flow
    ).time.values


def _get_number_future_timesteps(satellite_data: xr.DataArray, t0_dt: pd.Timestamp) -> int:
    """
    Get number of future timestamps

    Args:
        satellite_data: Satellite data to use
        t0_dt: The timestamp of the t0 image

    Returns:
        The number of future timesteps
    """
    satellite_data = satellite_data.where(satellite_data.time > t0_dt, drop=True)
    return len(satellite_data.coords["time"])


def _compute_and_return_optical_flow(
    satellite_data: xr.DataArray,
    t0_dt: pd.Timestamp,
    previous_timestep_for_flow: int,
    final_image_size_pixels: int,
) -> xr.DataArray:
    """
    Compute and return optical flow predictions for the example

    Args:
        satellite_data: Satellite DataArray
        t0_dt: t0 timestamp

    Returns:
        The xr.DataArray with the optical flow predictions for t0 to forecast horizon
    """

    prediction_dictionary = {}
    # Get the previous timestamp
    previous_timestamp = _compute_previous_timestep(
        satellite_data, t0_dt=t0_dt, previous_timestep_for_flow=previous_timestep_for_flow
    )
    for prediction_timestep in range(_get_number_future_timesteps(satellite_data, t0_dt)):
        predictions = []
        for channel in satellite_data.coords["channels"]:
            channel_images = satellite_data.sel(channels=channel)
            t0_image = channel_images.sel(time=t0_dt).values
            previous_image = channel_images.sel(time=previous_timestamp).values
            optical_flow = _compute_optical_flow(t0_image, previous_image)
            # Do predictions now
            flow = optical_flow * prediction_timestep
            warped_image = _remap_image(t0_image, flow)
            warped_image = crop_center(
                warped_image,
                final_image_size_pixels,
                final_image_size_pixels,
            )
            predictions.append(warped_image)
        # Add the block of predictions for all channels
        prediction_dictionary[prediction_timestep] = np.concatenate(predictions, axis=-1)
    # Make a block of T, H, W, C ordering
    prediction = np.stack([prediction_dictionary[k] for k in prediction_dictionary.keys()], axis=0)
    if len(satellite_data.coords["channels"]) == 1:
        # Only case where another channel needs to be added
        prediction = np.expand_dims(prediction, axis=-1)
    # Swap out data for the future part of the dataarray
    dataarray = _update_dataarray_with_predictions(
        satellite_data, predictions=prediction, t0_dt=t0_dt
    )
    return dataarray


def _update_dataarray_with_predictions(
    satellite_data: xr.DataArray,
    predictions: np.ndarray,
    t0_dt: pd.Timestamp,
    final_image_size_pixels: int,
) -> xr.DataArray:
    """
    Updates the dataarray with predictions

     Additionally, changes the temporal size to t0+1 to forecast horizon

    Args:
        satellite_data: Satellite data
        predictions: Predictions from the optical flow

    Returns:
        The Xarray DataArray with the optical flow predictions
    """

    # Combine all channels for a single timestep
    satellite_data = satellite_data.where(satellite_data.time > t0_dt, drop=True)
    # Make sure its the correct size
    # buffer is divisble by 2
    buffer_size = (len(satellite_data.x) - final_image_size_pixels) / 2
    satellite_data = satellite_data.isel(
        x=slice(buffer_size, len(satellite_data.x) - buffer_size),
        y=slice(buffer_size, len(satellite_data.y) - buffer_size),
    )
    dataarray = xr.DataArray(
        data=predictions,
        dims=satellite_data.dims,
        coords=satellite_data.coords,
    )

    return dataarray


def _compute_optical_flow(t0_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray:
    """
    Compute the optical flow for a set of images

    Args:
        t0_image: t0 image
        previous_image: previous image to compute optical flow with

    Returns:
        Optical Flow field
    """
    return cv2.calcOpticalFlowFarneback(
        prev=previous_image,
        next=t0_image,
        flow=None,
        pyr_scale=0.5,
        levels=2,
        winsize=40,
        iterations=3,
        poly_n=5,
        poly_sigma=0.7,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def _remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Takes an image and warps it forwards in time according to the flow field.

    Args:
        image: The grayscale image to warp.
        flow: A 3D array.  The first two dimensions must be the same size as the first two
            dimensions of the image.  The third dimension represented the x and y displacement.

    Returns:  Warped image.  The border has values np.NaN.
    """
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # map_x
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # map_y
    return cv2.remap(
        src=image,
        map1=remap,
        map2=None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.NaN,
    )


def crop_center(img, cropx, cropy):
    """
    Crop center of numpy image

    Args:
        img: Image to crop
        cropx: Size in x direction
        cropy: Size in y direction

    Returns:
        The cropped image
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]
