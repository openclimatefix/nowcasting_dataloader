import io
import os
import time
from datetime import datetime
from pathlib import Path

import gcsfs
import numcodecs
import pandas as pd
import xarray as xr

import nowcasting_dataloader
from nowcasting_dataset.data_sources.nwp.nwp_data_source import open_nwp, NWP_VARIABLE_NAMES
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

# set up
BUCKET = Path("solar-pv-nowcasting-data")
local_path = os.path.dirname(nowcasting_dataloader.__file__) + "/.."
PV_PATH = BUCKET / "PV/PVOutput.org"
PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

# set up variables
filename = PV_PATH / "UK_PV_timeseries_batch.nc"
metadata_filename = f"gs://{PV_METADATA_FILENAME}"
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-01-02 00:00:00.000+00:00")

# link to gcs
gcs = gcsfs.GCSFileSystem(access="read_only")

# get metadata, reduce, and save to test data
pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)
pv_metadata = pv_metadata.iloc[500:600]  # just take a few sites
pv_metadata.to_csv(f"{local_path}/tests/data/pv_metadata/UK_PV_metadata.csv")

# get pv_data
t = time.time()
with gcs.open(filename, mode="rb") as file:
    file_bytes = file.read()


with io.BytesIO(file_bytes) as file:
    pv_power = xr.open_dataset(file, engine="h5netcdf")
    pv_power = pv_power.sel(datetime=slice(start_dt, end_dt))
    pv_power_df = pv_power.to_dataframe()

# process data
system_ids_xarray = [int(i) for i in pv_power.data_vars]
system_ids = [
    str(system_id) for system_id in pv_metadata.index.to_list() if system_id in system_ids_xarray
]

# only take the system ids we need
pv_power_df = pv_power_df[system_ids]
pv_power_df = pv_power_df.dropna(axis="columns", how="all")
pv_power_df = pv_power_df.clip(lower=0, upper=5e7)
pv_power_new = pv_power_df.to_xarray()

# save to test data
pv_power_new.to_netcdf(f"{local_path}/tests/data/pv_data/test.nc")

############################
# NWP, this makes a file that is 9.5MW big
###########################

# Numerical weather predictions
NWP_BASE_PATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/"
    "UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr"
)

nwp_data_raw = open_nwp(filename=NWP_BASE_PATH, consolidated=True)
nwp_data = nwp_data_raw.sel(variable=list(NWP_VARIABLE_NAMES))
nwp_data = nwp_data.sel(init_time=slice(start_dt, end_dt))
nwp_data = nwp_data.sel(variable=["t"])
nwp_data = nwp_data.sel(step=slice(nwp_data.step[0], nwp_data.step[4]))  # take 4 hours periods
# nwp_data = nwp_data.sel(x=slice(nwp_data.x[50], nwp_data.x[100]))
# nwp_data = nwp_data.sel(y=slice(nwp_data.y[50], nwp_data.y[100]))

nwp_data.to_zarr(f"{local_path}/tests/data/nwp_data/test.zarr")

# ### GSP data
filename = "gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr"

gsp_power = xr.open_dataset(filename, engine="zarr")
gsp_power = gsp_power.sel(datetime_gmt=slice(start_dt, end_dt))
gsp_power = gsp_power.sel(gsp_id=slice(gsp_power.gsp_id[0], gsp_power.gsp_id[20]))

gsp_power["gsp_id"] = gsp_power.gsp_id.astype("str")

encoding = {
    var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=5)} for var in gsp_power.data_vars
}

gsp_power.to_zarr(f"{local_path}/tests/data/gsp/test.zarr", mode="w", encoding=encoding)

# TODO reduce and save


#####################
# SUN
#####################

filename = "gs://solar-pv-nowcasting-data/Sun/v0/sun.zarr/"
# filename = "./scripts/sun.zarr"

# open file
sun_xr = xr.open_dataset(filename, engine="zarr", mode="r", consolidated=True, chunks=None)

# just select one date
sun_xr = sun_xr.sel(time_5=slice(start_dt, end_dt))
sun_xr["locations"] = sun_xr.locations.astype("str")

# save to file
sun_xr.to_zarr(f"{local_path}/tests/data/sun/test.zarr", mode="w")


########
# batch0.nc
########

c = Configuration()
c.process.batch_size = 4
c.process.nwp_channels = c.process.nwp_channels[0:1]
c.process.sat_channels = c.process.sat_channels[0:1]

f = Batch.fake(configuration=c)
f.save_netcdf(batch_i=0, path=Path(f"{local_path}/tests/data/batch"))
