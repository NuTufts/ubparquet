# larmatch2d

Scripts to produce LArMatch 2D data from Triplet Truth files.

Triplet truth files made from MicroBooNE LArCV and larlite files and processed by [larflow/larflow/PrepMatchTriplets/test/run_prepmatchtriplets_wfulltruth.py](https://github.com/NuTufts/larflow/blob/dlgen2/larflow/PrepFlowMatchData/test/run_prepmatchtriplets_wfulltruth.py).

The triplet truth files are in a ROOT format that stores instances of a custom class.

We extract the data we need to produce larmatch training data.
These are parquet files that store 2D numpy arrays.

## The Schema


## To run

* You need to build and setup the `ubdl` repository to be able to use these files. *

## To view the output data

* Should be able to use this jupyter notebook after importing python packages via pip (no ROOT or UB code needed). *

## Slurm scripts to run on Tufts

We run the code in a container. You can get the container here (dockerhub link to follow).






