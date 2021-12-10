""" See issue https://github.com/openclimatefix/nowcasting_dataloader/issues/106 """
import io

import fsspec
import xarray as xr
import pandas as pd

from nowcasting_dataset.geospatial import lat_lon_to_osgb
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso


PV_file = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0/system_metadata.csv"


pv = pd.read_csv(PV_file)

lat = pv.latitude
lon = pv.longitude

x, y = lat_lon_to_osgb(lat=lat, lon=lon)


x_max = max(x)
x_min = min(x)

y_max = max(y)
y_min = min(y)

print("PV")
print(f"{x_max=}")
print(f"{x_min=}")
print(f"{y_max=}")
print(f"{y_min=}")

gsp = get_gsp_metadata_from_eso()

x = gsp.centroid_x
y = gsp.centroid_y

x_max = max(x)
x_min = min(x)

y_max = max(y)
y_min = min(y)

print("GSP")
print(f"{x_max=}")
print(f"{x_min=}")
print(f"{y_max=}")
print(f"{y_min=}")


