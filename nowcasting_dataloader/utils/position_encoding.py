"""
This file contains various ways of performing positional encoding.

These encodings can be:
- Absolute positioning (i.e. this pixel is at this latitude/longitude, and is at 16:00)

These encodings can also be performed with:
- Fourier Features, based off what is done in PerceiverIO
"""
import numpy as np
import torch
import einops
from math import pi
from typing import Union, Optional, Dict, List, Tuple, Any
import datetime
import pandas as pd
from nowcasting_dataset.dataset.batch import Batch
import xarray as xr
from nowcasting_dataset.geospatial import lat_lon_to_osgb

TIME_DIM = 2
HEIGHT_DIM = 3
WIDTH_DIM = 4
# For GSP and PV, have an ID dimension
ID_DIM = 3


def get_seviri_rss_bounds() -> Dict[str, float]:
    """
    Computes the SEVIRI RSS bounds in OSGB coordinates

    SEVIRI RSS is the imager that takes all the satellite imagery currently used by the nowcasting dataset and models

    Returns:
        Dictionary containing the geographic bounds of the RSS images in OSGB coordinates
    """
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for lat in [15, 70]:
        for lon in [-45, 65]:
            x, y = lat_lon_to_osgb(lat, lon)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def generate_position_encodings_for_batch(batch: Batch, **kwargs) -> dict[str, torch.Tensor]:
    """
    Generates positional encodings and returns them as a dictionary

    This is not returned with the Batch, as that would require more keys, etc. in Batch

    Args:
        batch: Batch object holding the data

    Returns:
        Dictionary containing the keys of the modalities in the Batch + '_position_encoding'
    """

    position_encodings = {}
    # Go for each modality where a position encoding makes sense
    for k in batch.__fields__.keys():
        if k in [
            "nwp",
            "satellite",
            "topographic",
            "gsp",
            "pv",
        ]:
            xr_dataset = getattr(batch, k)
            if xr_dataset is not None:
                datetimes = None
                if hasattr(xr_dataset, "time"):
                    datetimes = xr_dataset.time.values
                geospatial_coordinates = (
                    [xr_dataset.x.values, xr_dataset.y.values]
                    if "x_index" in xr_dataset.sizes
                    else [xr_dataset.x_coords.values, xr_dataset.y_coords.values]
                )
                position_encodings[k + "_position_encoding"] = encode_absolute_position(
                    shape=determine_shape_of_encoding(xr_dataset),
                    geospatial_coordinates=geospatial_coordinates,
                    datetimes=datetimes,
                    geospatial_bounds=get_seviri_rss_bounds(),
                    **kwargs,
                )

    return position_encodings


def determine_shape_of_encoding(xr_dataset: xr.Dataset) -> List[int]:
    """
    Determine the shape of the encoding needed for the batch example

    Args:
        xr_dataset: Xarray dataset containing the data

    Returns:
        The determined shape that the encoding needs to be, either a 5-element list for image modalities,
         or a 4-element list for point modalities (GSP, PV systems)
    """
    channel_key = "channels_index" if "channels_index" in xr_dataset.sizes else "id_index"
    shape = [xr_dataset.sizes["example"]]
    shape.append(
        xr_dataset.sizes.get(channel_key, 1)
    )  # If no channels, count as single channel image)
    shape.append(xr_dataset.sizes.get("time_index", 1))  # If no time dimension, just a single one)

    # Now for the main issue, either 4 or 5D here
    if "x_index" in xr_dataset.sizes:
        shape.append(xr_dataset.sizes["x_index"])
        shape.append(xr_dataset.sizes["y_index"])
    else:
        # No spatial extant i.e. GSP, or PV
        # Then the output should be for each ID, so would then be the same as the channels ID
        shape.append(xr_dataset.sizes["id_index"])
    return shape


def encode_modalities(
    modalities_to_encode: Dict[str, torch.Tensor],
    datetimes: Dict[str, List[datetime.datetime]],
    geospatial_coordinates: Dict[str, Tuple[np.ndarray, np.ndarray]],
    geospatial_bounds: Dict[str, float],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Create a consistent position encoding and encode the positions of the different modalities in time and space

    This position encoding is added as new keys to the dictionary containing the modalities to encode. This is done
    instead of appending the position encoding in case the position encoding needs to be used for the query to the
    Perceiver IO model

    This code assumes that there is at least 2 timesteps of at least one modality to be encoded

    Args:
        modalities_to_encode: Dict of input modalities, i.e. NWP, Satellite, PV, GSP, etc as torch.Tensors in [B, C, T, H, W] ordering
        datetimes: Dict of datetimes for each modality, giving the actual date for each timestep in the modality
        geospatial_coordinates: Dict of x, y coordinates for each modality with pixels, used to determine smallest spatial step needed, in OSGB coordinates
        geospatial_bounds: Max extant of the area where examples could be drawn from, used for normalizing coordinates within an area of interest
            in the format of a dictionary with the keys {'x_min', 'x_max', 'y_min', 'y_max'}
        kwargs: Passed to fourier_encode

    Returns:
        Input modality dictionary where for every 'key' in modalities_to_encode, a new key called 'key+'_position_encoding' will be added
        containing the absolute position encoding of the examples
    """
    position_encodings = {}
    for key in modalities_to_encode.keys():
        position_encodings[key + "_position_encoding"] = encode_absolute_position(
            shape=modalities_to_encode[key].shape,
            geospatial_coordinates=geospatial_coordinates[key],
            datetimes=datetimes[key],
            geospatial_bounds=geospatial_bounds,
            **kwargs,
        )
    # Update original dictionary
    modalities_to_encode.update(position_encodings)
    return modalities_to_encode


def encode_absolute_position(
    shape: List[int],
    geospatial_coordinates: List[np.ndarray],
    geospatial_bounds: Dict[str, float],
    datetimes: Optional[List[datetime.datetime]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Encodes the absolute position of the pixels/voxels in time and space

    This should be done per-modality and can be thought of as the relative position of the input modalities across a
    given year and the area of the Earth covered by all the examples.

    Args:
        shape: Shape to encode positions for
        geospatial_coordinates: The geospatial coordinates, in OSGB format
        datetimes: Time of day and date as a list of datetimes, one for each timestep
        geospatial_bounds: The geospatial bounds of the area where the examples come from, e.g. the coordinates of the area covered by the SEVIRI RSS image
        **kwargs:

    Returns:
        The absolute position encoding for the given shape
    """
    # Fourier Features of absolute position
    encoded_geo_position = normalize_geospatial_coordinates(
        geospatial_coordinates, geospatial_bounds, **kwargs
    )
    absolute_position_encoding = einops.repeat(
        encoded_geo_position, "b h w c -> b c t h w", t=shape[TIME_DIM]
    )

    if len(shape) == 4:  # Point systems
        # Probably GSP or PV, just need the diagonal of the values, so have B, C, T, GSP/PV ID
        absolute_position_encoding = torch.diagonal(absolute_position_encoding, dim1=-2, dim2=-1)

    if datetimes is not None:
        datetime_features = create_datetime_features(datetimes)

        absolute_position_encoding = combine_space_and_time_features(
            absolute_position_encoding, datetime_features=datetime_features, shape=shape
        )

    return absolute_position_encoding


def combine_space_and_time_features(
    spatial_features: torch.Tensor, datetime_features: List[torch.Tensor], shape: List[int]
) -> torch.Tensor:
    """
    Combine spatial and temporal features a list of Tensors to be concatenated

    Args:
        spatial_features: Spatial features
        datetime_features: List of datetime features
        shape: The desired shape of the encoding

    Returns:
        Tensor containing the combined space and time features
    """
    to_concat = [spatial_features]
    # Combine time and space features
    for date_feature in datetime_features:
        if len(shape) == 5:
            date_feature = einops.repeat(
                date_feature, "b t -> b c t h w", h=shape[HEIGHT_DIM], w=shape[WIDTH_DIM], c=1
            )
        else:
            date_feature = einops.repeat(date_feature, "b t -> b c t id", id=shape[ID_DIM], c=1)
        to_concat.append(date_feature)
    space_and_time_encoding = torch.cat(to_concat, dim=1)
    return space_and_time_encoding


def normalize_geospatial_coordinates(
    geospatial_coordinates: List[np.ndarray], geospatial_bounds: Dict[str, float], **kwargs
) -> torch.Tensor:
    """
    Normalize the geospatial coordinates by the max extant to keep everything between -1 and 1, in sin and cos

    This normalization should be against a set geospatial area, so that the same place has the same spatial encoding
    every time.

    Args:
        geospatial_coordinates: The coordinates for the pixels in the image
        geospatial_bounds: The maximum extant

    Returns:
        The normalized geospatial coordinates, rescaled to between -1 and 1 for the whole extant of the training area

    """
    # Normalize the X first
    geospatial_coordinates[0] = (geospatial_coordinates[0] - geospatial_bounds["x_min"]) / (
        geospatial_bounds["x_max"] - geospatial_bounds["x_min"]
    )
    # Normalize the Y second
    geospatial_coordinates[1] = (geospatial_coordinates[1] - geospatial_bounds["y_min"]) / (
        geospatial_bounds["y_max"] - geospatial_bounds["y_min"]
    )

    # Now those are between 0 and 1, want between -1 and 1
    geospatial_coordinates[0] = geospatial_coordinates[0] * 2 - 1
    geospatial_coordinates[1] = geospatial_coordinates[1] * 2 - 1
    # Now create a grid of the coordinates
    # Have to do it for each individual example in the batch, and zip together x and y for it
    to_concat = []
    for idx in range(len(geospatial_coordinates[0])):
        x = torch.from_numpy(geospatial_coordinates[0][idx])
        y = torch.from_numpy(geospatial_coordinates[1][idx])
        grid = torch.meshgrid(x, y)
        pos = torch.stack(grid, dim=-1)
        encoded_position = fourier_encode(pos, **kwargs)
        encoded_position = einops.rearrange(encoded_position, "... n d -> ... (n d)")
        to_concat.append(encoded_position)

    encoded_position = torch.stack(to_concat, dim=0)
    return encoded_position


def create_datetime_features(
    datetimes: List[List[datetime.datetime]],
) -> List[torch.Tensor]:
    """
    Converts a list of datetimes to day of year, hour of day sin and cos representation

    Args:
        datetimes: List of list of datetimes for the examples in a batch

    Returns:
        Tuple of torch Tensors containing the hour of day sin,cos, and day of year sin,cos
    """
    hour_of_day = []
    day_of_year = []
    for batch_idx in range(len(datetimes)):
        hours = []
        days = []
        for index in datetimes[batch_idx]:
            time_index = pd.Timestamp(index)
            hours.append((time_index.hour + (time_index.minute / 60) / 24))
            days.append((time_index.timetuple().tm_yday / 365))
        hour_of_day.append(hours)
        day_of_year.append(days)

    outputs = []
    for index in [hour_of_day, day_of_year]:
        index = torch.as_tensor(index)
        radians = index * 2 * np.pi
        index_sin = torch.sin(radians)
        index_cos = torch.cos(radians)
        outputs.append(index_sin)
        outputs.append(index_cos)

    return outputs


def fourier_encode(
    x: torch.Tensor,
    max_freq: float,
    num_bands: int = 4,
    sine_only: bool = False,
) -> torch.Tensor:
    """
    Create Fourier Encoding

    Args:
        x: Input Torch Tensor
        max_freq: Maximum frequency for the Fourier features
        num_bands: Number of frequency bands
        sine_only: Whether to only use sine or both sine and cosine features

    Returns:
        Torch Tensor with the fourier position encoded concatenated
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(
        1.0,
        max_freq / 2,
        num_bands,
        device=device,
        dtype=dtype,
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = x.sin() if sine_only else torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x
