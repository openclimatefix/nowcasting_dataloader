""" Util functions to adjust xr dims (currently only one)"""
import xarray as xr


def re_order_dims(xr_dataset: xr.Dataset):
    """
    Re order dims to B,C,T,H,W
    """
    expected_dims_order = ("example", "channels_index", "time_index", "y_index", "x_index")

    if xr_dataset.data.dims != expected_dims_order:
        xr_dataset.__setitem__("data", xr_dataset.data.transpose(*expected_dims_order))

    return xr_dataset
