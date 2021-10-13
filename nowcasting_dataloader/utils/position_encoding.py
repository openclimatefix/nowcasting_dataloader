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
    resolutions: Dict[str, float] = None,
    **kwargs,
) -> dict:
    """
    Create a consistent position encoding and encode the positions of the different modalities in time and space

    This position encoding is added as new keys to the dictionary containing the modalities to encode. This is done
    instead of appending the position encoding in case the position encoding needs to be used for the query to the
    Perceiver IO model

    Args:
        positioning: The type of positioning used, either 'relative' for relative positioning, or 'absolute', or 'both'
        modalities_to_encode: Dict of input modalities, i.e. NWP, Satellite, PV, GSP, etc as torch.Tensors in [B, C, T, H, W] ordering
        datetimes: Dict of datetimes for each modality, giving the actual date for each timestep in the modality
        resolutions: Dict of physical resolutions for each pixel in the input, optional, used to determine smallest spatial step needed, if None, assumes all pixels cover the same physical size
        **kwargs:

    Returns:
        Input modali
    """
    assert positioning in ["relative", "absolute", "both"], AssertionError(
        f"positioning must be one of 'relative', 'absolute' or 'both', not '{positioning}'"
    )
    # Build Absolute position encoding for each modality
    if positioning == "absolute":
        pass
    # Step 1: Find the total time range covered

    # Step 1.5: Find smallest temporal interval
    # Step 2: Find the total spatial extant covered
    # Step 2.5: Find the smallest spatial interval
    # Step 3: Build relative position encoding
    # Step 4: Return position encodings as new keys

    return modalities_to_encode


def encode_position(
    shape: List[int],
    positioning: str,
    geospatial_coordinates: Optional[Tuple[List[int], List[int]]] = None,
    datetimes: Optional[List[datetime.datetime]] = None,
    geospatial_bounds: Optional[Dict[str, float]] = None,
    method: str = "fourier",
    **kwargs,
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
        geospatial_bounds: The bounds of the geospatial area covered, in x_min, y_min, x_max, y_max ordering, only used for absolute coordinates

    Returns:
        The position encodings for all items in the batch
    """
    assert method in [
        "fourier",
    ], AssertionError(f"method must be one of 'fourier', not '{method}'")
    assert positioning in ["relative", "absolute", "both"], AssertionError(
        f"positioning must be one of 'relative', 'absolute' or 'both', not '{positioning}'"
    )

    if positioning == "relative":
        position_encoding = encode_relative_position(shape, **kwargs)
    elif positioning == "absolute":
        position_encoding = encode_absolute_position(
            shape, geospatial_coordinates, geospatial_bounds, datetimes
        )
    else:
        # Both position encodings
        position_encoding = torch.cat(
            [
                encode_relative_position(shape),
                encode_absolute_position(
                    shape, geospatial_coordinates, geospatial_bounds, datetimes
                ),
            ],
            dim=-1,
        )
    return position_encoding


def encode_relative_position(shape: List[int], **kwargs) -> torch.Tensor:
    """
    Encode the relative position of the pixels/voxels

    This relative position is in relation the the union of all the input modalities. This means, for example, if
    the inputs are the last 4 hourly NWPs, and the last 6 5-minutely satellite imagery, the relative positioning
    will be generated for the last 4 hours at 5 minutely intervals, so that it can capture both the NWP and satellite
    imagery in the same position encoding.

    Args:
        shape:

    Returns:
        The relative position encoding as a torch Tensor

    """
    position_encoding = encode_fouier_position(1, shape, **kwargs)
    return position_encoding


def encode_absolute_position(
    shape: List[int], geospatial_coordinates, geospatial_bounds, datetimes, **kwargs
) -> torch.Tensor:
    """
    Encodes the absolute position of the pixels/voxels in time and space

    This should be done per-modality and can be thought of as the relative position of the input modalities across a
    given year and the area of the Earth covered by all the examples.

    Args:
        shape: Shape to encode positions for
        geospatial_coordinates: The geospatial coordinates, in OSGB format
        datetimes: Time of day and date as a list of datetimes, one for each timestep
        **kwargs:

    Returns:
        The absolute position encoding for the given shape
    """
    hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos = create_datetime_features(
        datetimes
    )

    # Fourier Features of absolute position
    encoded_latlon = normalize_geospatial_coordinates(
        geospatial_coordinates, geospatial_bounds, **kwargs
    )

    # Combine time and space features
    # Time features should be in shape [Channels,Timestep]
    # Space features should be in [Channels, Height, Width]
    # So can just concat along channels, after expanding time features to Height, Width, and Space along Time
    hour_of_day_sin = einops.repeat(hour_of_day_sin, "b c t -> b c t h w", h=shape[-2], w=shape[-1])
    hour_of_day_cos = einops.repeat(hour_of_day_cos, "b c t -> b c t h w", h=shape[-2], w=shape[-1])
    day_of_year_sin = einops.repeat(day_of_year_sin, "b c t -> b c t h w", h=shape[-2], w=shape[-1])
    day_of_year_cos = einops.repeat(day_of_year_cos, "b c t -> b c t h w", h=shape[-2], w=shape[-1])
    # Now do for latlon encoding
    encoded_latlon = einops.repeat(encoded_latlon, "b c h w -> b c t h w", t=shape[1])

    # Now combined into one large encoding
    absolute_position_encoding = torch.cat(
        [encoded_latlon, hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos], dim=1
    )

    return absolute_position_encoding


def normalize_geospatial_coordinates(
    geospatial_coordinates, geospatial_bounds, **kwargs
) -> torch.Tensor:
    """
    Normalize the geospatial coordinates by the max extant to keep everything between -1 and 1, in sin and cos

    Args:
        geospatial_coordinates: The coordinates for the pixels in the image
        geospatial_bounds: The maximum extant

    Returns:
        The normalized geospatial coordinates, rescaled to between -1 and 1

    """
    # Normalize the X first
    geospatial_coordinates[0] = (geospatial_coordinates[0] - geospatial_bounds[0]) / (
        geospatial_bounds[2] - geospatial_bounds[0]
    )
    # Normalize the Y second
    geospatial_coordinates[1] = (geospatial_coordinates[1] - geospatial_bounds[1]) / (
        geospatial_bounds[3] - geospatial_bounds[1]
    )

    # Now those are between 0 and 1, want between -1 and 1
    geospatial_coordinates[0] = geospatial_coordinates[0] * 2 - 1
    geospatial_coordinates[1] = geospatial_coordinates[1] * 2 - 1

    # Now create a grid of the coordinates
    pos = torch.stack(torch.meshgrid(*geospatial_coordinates), dim=-1)

    # And now convert to Fourier features, based off the absolute positions of the coordinates
    encoded_position = fourier_encode(pos, **kwargs)
    return encoded_position


def create_datetime_features(
    datetimes: List[datetime.datetime],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of datetimes to day of year, hour of day sin and cos representation

    Args:
        datetimes: List of datetimes

    Returns:
        Tuple of torch Tensors containing the hour of day sin,cos, and day of year sin,cos
    """
    hour_of_day = []
    day_of_year = []
    for index in datetimes:
        hour_of_day.append((index.hour + (index.minute / 60) / 24))
        day_of_year.append((index.timetuple().tm_yday / 365))
    hour_of_day = torch.as_tensor(hour_of_day)
    day_of_year = torch.as_tensor(day_of_year)
    hour_radians = hour_of_day * 2 * np.pi
    day_radians = day_of_year * 2 * np.pi
    hour_of_day_sin = torch.sin(hour_radians)
    hour_of_day_cos = torch.cos(hour_radians)
    day_of_year_sin = torch.sin(day_radians)
    day_of_year_cos = torch.cos(day_radians)

    return hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos


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
