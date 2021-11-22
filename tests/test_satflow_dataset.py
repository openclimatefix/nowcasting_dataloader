# """Test SatFlow Dataset"""
# import os
# import tempfile
# from pathlib import Path
#
# import torch
# from nowcasting_dataset.config.model import Configuration, InputData
# from nowcasting_dataset.dataset.batch import Batch
#
# from nowcasting_dataloader.datasets import SatFlowDataset, worker_init_fn
#
# torch.set_default_dtype(torch.float32)
#
#
# @pytest.mark.skip("Temp skipping")
# def test_satflow_dataset_local_using_configuration():
#     """Test satflow locally"""
#     c = Configuration()
#     c.input_data = InputData.set_all_to_defaults()
#     c.process.batch_size = 4
#     c.input_data.nwp.nwp_channels = c.input_data.nwp.nwp_channels[0:1]
#     c.input_data.satellite.satellite_channels = c.input_data.satellite.satellite_channels[0:2]
#     configuration = c
#
#     with tempfile.TemporaryDirectory() as tmpdirname:
#
#         f = Batch.fake(configuration=c)
#         f.save_netcdf(batch_i=0, path=Path(tmpdirname))
#
#         DATA_PATH = tmpdirname
#         TEMP_PATH = tmpdirname
#
#         train_dataset = SatFlowDataset(
#             1,
#             DATA_PATH,
#             TEMP_PATH,
#             cloud="local",
#             history_minutes=10,
#             forecast_minutes=10,
#             configuration=configuration,
#         )
#
#         dataloader_config = dict(
#             pin_memory=True,
#             num_workers=1,
#             prefetch_factor=1,
#             worker_init_fn=worker_init_fn,
#             persistent_workers=True,
#             # Disable automatic batching because dataset
#             # returns complete batches.
#             batch_size=None,
#         )
#
#         _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)
#
#         train_dataset.per_worker_init(1)
#         t = iter(train_dataset)
#         x, y = next(t)
#
#         for k in [
#             "pv_yield",
#             "pv_system_id",
#             "nwp",
#             "topo_data",
#             "gsp_id",
#             "sat_data",
#             "hrv_sat_data",
#         ]:
#             assert k in x.keys()
#             assert type(x[k]) == torch.Tensor
#
#         for k in [
#             "gsp_yield",
#             "gsp_id",
#         ]:
#             assert k in y.keys()
#             assert type(y[k]) == torch.Tensor
#
#         # Make sure file isn't deleted!
#         assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))
#
#
# @pytest.mark.skip("Temp skipping")
# def test_satflow_dataset_local_using_configuration_with_position_encoding():
#     """Test satflow locally"""
#     c = Configuration()
#     c.input_data = InputData.set_all_to_defaults()
#     c.process.batch_size = 4
#     c.input_data.satellite.satellite_image_size_pixels = 24
#     configuration = c
#
#     with tempfile.TemporaryDirectory() as tmpdirname:
#
#         f = Batch.fake(configuration=c)
#         f.save_netcdf(batch_i=0, path=Path(tmpdirname))
#
#         DATA_PATH = tmpdirname
#         TEMP_PATH = tmpdirname
#
#         train_dataset = SatFlowDataset(
#             1,
#             DATA_PATH,
#             TEMP_PATH,
#             cloud="local",
#             history_minutes=10,
#             forecast_minutes=10,
#             configuration=configuration,
#             add_position_encoding=True,
#             add_hrv_satellite_target=True,
#             add_satellite_target=True,
#         )
#
#         dataloader_config = dict(
#             pin_memory=True,
#             num_workers=1,
#             prefetch_factor=1,
#             worker_init_fn=worker_init_fn,
#             persistent_workers=True,
#             # Disable automatic batching because dataset
#             # returns complete batches.
#             batch_size=None,
#         )
#
#         _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)
#
#         train_dataset.per_worker_init(1)
#         t = iter(train_dataset)
#         x, y = next(t)
#
#         for k in [
#             "pv_yield",
#             "pv_system_id",
#             "nwp",
#             "topo_data",
#             "gsp_id",
#             "sat_data_query",
#             "hrv_sat_data_query",
#             "gsp_yield_query",
#             "sat_data",
#             "hrv_sat_data",
#         ]:
#             assert k in x.keys()
#             assert type(x[k]) == torch.Tensor
#
#         for k in ["gsp_yield", "gsp_id", "sat_data", "hrv_sat_data"]:
#             assert k in y.keys()
#             assert type(y[k]) == torch.Tensor
#         # Make sure file isn't deleted!
#         assert os.path.exists(os.path.join(DATA_PATH, "nwp/000000.nc"))
