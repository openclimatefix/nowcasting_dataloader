from nowcasting_dataloader.data_sources.satellite.satellite_model import SatelliteML

from nowcasting_dataset.data_sources.fake import satellite_fake


def test_satellite_to_ml():
    sat = satellite_fake()

    _ = SatelliteML.from_xr_dataset(sat)

