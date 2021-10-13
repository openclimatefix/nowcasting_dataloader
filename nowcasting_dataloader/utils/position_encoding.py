"""
This file contains various ways of performing positional encoding.

These encodings can be:
- Relative positioning (i.e. this pixel is this far from the top left, and this many timesteps in the future)
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


def encode_modalities(
    positioning: str,
    modalities_to_encode: Dict[str, torch.Tensor],
    datetimes: Dict[str, List[datetime.datetime]],
    geospatial_coordinates: Dict[str, Tuple[np.ndarray, np.ndarray]],
    geospatial_bounds: Dict[str, float],
) -> dict:
    """
    Create a consistent position encoding and encode the positions of the different modalities in time and space

    This position encoding is added as new keys to the dictionary containing the modalities to encode. This is done
    instead of appending the position encoding in case the position encoding needs to be used for the query to the
    Perceiver IO model

    This code assumes that there is at least 2 timesteps of at least one modality to be encoded

    Args:
        positioning: The type of positioning used, either 'relative' for relative positioning, or 'absolute', or 'both'
        modalities_to_encode: Dict of input modalities, i.e. NWP, Satellite, PV, GSP, etc as torch.Tensors in [B, C, T, H, W] ordering
        datetimes: Dict of datetimes for each modality, giving the actual date for each timestep in the modality
        geospatial_coordinates: Dict of lat/lon coordinates for each modality with pixels, optional, used to determine smallest spatial step needed, in OSGB coordinates
        geospatial_bounds: Max extant of the area where examples could be drawn from, used for normalizing coordinates within an area of interest

    Returns:
        Input modality dictionary with extra keys added containing the absolute position encoding of the examples
    """
    assert positioning in ["relative", "absolute", "both"], AssertionError(
        f"positioning must be one of 'relative', 'absolute' or 'both', not '{positioning}'"
    )
    # Build Absolute position encoding for each modality
    if positioning == "absolute":
        # If absolute, can just skip all the computing for relative encoding
        for key in modalities_to_encode.keys():
            modalities_to_encode[key + "_position_encoding"] = encode_position(
                modalities_to_encode[key].shape,
                positioning=positioning,
                geospatial_coordinates=geospatial_coordinates[key],
                datetimes=datetimes[key],
                geospatial_bounds=geospatial_bounds,
            )
        return modalities_to_encode
    else:
        raise NotImplementedError(f"Position encodings {positioning} is not implemented yet")


def encode_position(
    shape: List[int],
    positioning: str,
    geospatial_coordinates: Optional[List[np.ndarray]] = None,
    datetimes: Optional[List[datetime.datetime]] = None,
    geospatial_bounds: Optional[Dict[str, float]] = None,
    method: str = "fourier",
) -> torch.Tensor:
    """
    This function wraps a variety of different methods for generating position features for given inputs.

    Args:
        shape: The shape of the input to be encoded, should be the largest or finest-grained input
            For example, if the inputs are shapes (12, 6, 128, 128) and (1, 6), (12, 6, 128, 128) should be passed in as
            shape, as it has the most elements and the input (1, 6) can just subselect the position encoding
        geospatial_coordinates: The latitude/longitude of the inputs for shape, in OSGB coordinates, unused if using relative positioning only
        datetimes: time of day and date for each of the timesteps in the shape, unused if using relative positioning only
        method: Method of the encoding, either 'fourier' for Fourier Features
        positioning: The type of positioning used, either 'relative' for relative positioning, or 'absolute', or 'both'
        geospatial_bounds: The bounds of the geospatial area covered, in a dict with the keys 'x_min', 'y_min', 'x_max', 'y_max'

    Returns:
        The position encodings for all items in the batch
    """
    assert method in [
        "fourier",
    ], AssertionError(f"method must be one of 'fourier', not '{method}'")
    assert positioning in ["absolute"], AssertionError(
        f"positioning must be one of 'relative', 'absolute' or 'both', not '{positioning}'"
    )

    position_encoding = encode_absolute_position(
        shape, geospatial_coordinates, geospatial_bounds, datetimes
    )
    return position_encoding


def encode_absolute_position(
    shape: List[int],
    geospatial_coordinates: List[np.ndarray],
    geospatial_bounds: Dict[str, float],
    datetimes,
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
    datetime_features = create_datetime_features(datetimes)

    # Fourier Features of absolute position
    encoded_latlon = normalize_geospatial_coordinates(
        geospatial_coordinates, geospatial_bounds, **kwargs
    )

    # Combine time and space features
    to_concat = [einops.repeat(encoded_latlon, "b c h w -> b c t h w", t=shape[1])]
    for date_feature in datetime_features:
        to_concat.append(
            einops.repeat(date_feature, "b c t -> b c t h w", h=shape[-2], w=shape[-1])
        )

    # Now combined into one large encoding
    absolute_position_encoding = torch.cat(to_concat, dim=1)

    return absolute_position_encoding


def normalize_geospatial_coordinates(
    geospatial_coordinates, geospatial_bounds: Dict[str, float], **kwargs
) -> torch.Tensor:
    """
    Normalize the geospatial coordinates by the max extant to keep everything between -1 and 1, in sin and cos

    This should work on a batch level, as the geospatial bounds should be the same for every example in the batch

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
        x = geospatial_coordinates[0][idx]
        y = geospatial_coordinates[1][idx]
        grid = torch.meshgrid(x, y)
        pos = torch.stack(grid, dim=-1)
        encoded_position = fourier_encode(pos, **kwargs)
        encoded_position = einops.rearrange(encoded_position, "... n d -> ... (n d)")
        to_concat.append(encoded_position)

    # And now convert to Fourier features, based off the absolute positions of the coordinates
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
            hours.append((index.hour + (index.minute / 60) / 24))
            days.append((index.timetuple().tm_yday / 365))
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


def encode_fouier_position(
    batch_size: int,
    axis: list,
    max_frequency: float,
    num_frequency_bands: int,
    sine_only: bool = False,
) -> torch.Tensor:
    """
    Encode the Fourier Features and return them

    Args:
        batch_size: Batch size
        axis: List containing the size of each axis
        max_frequency: Max frequency
        num_frequency_bands: Number of frequency bands to use
        sine_only: (bool) Whether to only use Sine features or both Sine and Cosine, defaults to both

    Returns:
        Torch tensor containing the Fourier Features of shape [Batch, *axis]
    """
    axis_pos = list(
        map(
            lambda size: torch.linspace(-1.0, 1.0, steps=size),
            axis,
        )
    )
    pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
    enc_pos = fourier_encode(
        pos,
        max_frequency,
        num_frequency_bands,
        sine_only=sine_only,
    )
    enc_pos = einops.rearrange(enc_pos, "... n d -> ... (n d)")
    enc_pos = einops.repeat(enc_pos, "... -> b ...", b=batch_size)
    return enc_pos


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
