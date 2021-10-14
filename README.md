# ubparquet
Conversion of ROOT-based files into parquet-based files for ease of sharing datasets

Provided in file:
* spacepoints with true or ghost labels
* ssnet labels for spacepoints
* keypoint labels for spacepoints
* voxels (1 cm) with true or ghost labels
* ssnet labels for voxels

To view the data, two notebooks are provided:

* `view_spacepoint_data.ipynb` for data represented by 3D points. These are the result of looking at possible wire intersections. The number of spacepoints per event is O(100K).
* `view_voxel_data.ipynb` for data represented as 1 cm voxels. These are made by binning the space points. The number of non-zero voxels per event is O(10K).

To do:
* [done] instance labels for spacepoints and voxels (requires instance numpy array maker)
* [done] nu versus cosmic labels for spacepoints and voxels (requires numpy array maker)
* [done] instance and nu voxels -- requires storing that info into voxel in voxelizer and adding function to make arrays
* [done] mc truth meta-data -- requires parser of larlite objects into python dict
* [done] test multi-event file
* [done] test multi-file parquet loading