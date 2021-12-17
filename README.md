# nowcasting-dataloader

[![codecov](https://codecov.io/gh/openclimatefix/nowcasting_dataloader/branch/main/graph/badge.svg?token=ABTGF6GYHN)](https://codecov.io/gh/openclimatefix/nowcasting_dataloader)


PyTorch Dataloader for working with multi-modal data for nowcasting applications.  In particular, this code loads the pre-prepared batches saved to disk by [`nowcasting_dataset`](https://github.com/openclimatefix/nowcasting_dataset).  This code also computes some optional, additional input features.

# Usage

## Installation

Run `pip install nowcasting-dataloader`

# Conventions

The dataloader assumes that data is generally in B, C, T, H, W ordering, where B is Batch size, C is number of channels, T is timestep, H is height, and W is width.
