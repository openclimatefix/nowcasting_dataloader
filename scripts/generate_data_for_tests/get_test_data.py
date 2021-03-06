"""Generate test data for tests"""
import os
from pathlib import Path

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

import nowcasting_dataloader

# set up
local_path = os.path.dirname(nowcasting_dataloader.__file__) + "/.."

########
# batch0.nc
########

c = Configuration()
c.input_data = InputData.set_all_to_defaults()
c.process.batch_size = 4
c.input_data.nwp.nwp_channels = c.input_data.nwp.nwp_channels[0:1]
c.input_data.satellite.satellite_channels = c.input_data.satellite.satellite_channels[0:1]

f = Batch.fake(configuration=c)
f.save_netcdf(batch_i=0, path=Path(f"{local_path}/tests/data/batch"))
