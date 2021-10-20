#!/usr/bin/env python3
from nowcasting_dataset import consts
import pandas as pd
from pathlib import Path
import numcodecs
import xarray as xr


START = pd.Timestamp("2019-01-01T12:00")
END = pd.Timestamp("2019-01-01T18:00")
OUTPUT_PATH = Path(__file__).parent.absolute() / ".." / "nowcasting_dataset" / "tests" / "data"


def generate_satellite_test_data():
    output_filename = OUTPUT_PATH / "sat_data.zarr"
    print("Writing satellite tests data to", output_filename)
    sat_data = xr.open_zarr(consts.SAT_DATA_ZARR, consolidated=True)
    sat_data = sat_data.sel(variable=["HRV"], time=slice(START, END))
    encoding = {"stacked_eumetsat_data": {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)}}
    sat_data = sat_data.chunk({"time": 1, "y": 704, "x": 548, "variable": 1})
    sat_data.to_zarr(output_filename, mode="w", consolidated=False, encoding=encoding)


if __name__ == "__main__":
    generate_satellite_test_data()
