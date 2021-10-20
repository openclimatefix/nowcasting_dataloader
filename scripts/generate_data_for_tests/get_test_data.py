import os
from pathlib import Path
import nowcasting_dataloader
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch

# set up
local_path = os.path.dirname(nowcasting_dataloader.__file__) + "/.."

########
# batch0.nc
########

c = Configuration()
c.process.batch_size = 4
c.process.nwp_channels = c.process.nwp_channels[0:1]
c.process.sat_channels = c.process.sat_channels[0:1]

f = Batch.fake(configuration=c)
f.save_netcdf(batch_i=0, path=Path(f"{local_path}/tests/data/batch"))
