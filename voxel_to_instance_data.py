from __future__ import print_function
import os,sys
import numpy as np
from readparquet import UBParquetReader
from ctypes import c_int
import pyarrow as pa
import pyarrow.parquet


# DATA FILES
# specify location where parquet file(s) live
datafolder="./numu_file00.parquet"

# Read data
reader = UBParquetReader(datafolder)
NENTRIES = reader.nentries
columns = ['instvoxcoord','instvoxcoord_shape','label']
entry_dict = {}
for col in columns:
    entry_dict[col] = []

    
for ENTRY in range(NENTRIES):
    print("Entry",ENTRY+1,"/",NENTRIES)
    data = reader.get_entry(ENTRY)
    
    # Retrieve the 3d positions
    pos3d = data["voxcoord"].astype(np.float)*1.0
    pos3d[:,1] -= 117.0 

    no_ghost_points = True
    if no_ghost_points:
        pos = pos3d[data['voxlabel']==1]
    else:
        pos = pos3d
        
        
    # Retreive Labels
    labels = data["voxinstance"]
    unique_labels = np.unique(labels)
    nonzero_labels = labels[data['voxlabel']==1]
    ssnetlabels = data["voxssnet"][data['voxlabel']==1]

    
    # Iterate over data and get matching labels
    for ulab in unique_labels.tolist():
        
        # List for voxel data points
        templist = []
        
        # List for sslabels of each data point
        sslist = []

        for i2, lab in enumerate(nonzero_labels):
            if lab == ulab:
                if len(templist) == 0:
                    templist = np.array([pos[i2]])
                else:
                    templist = np.vstack((templist,pos[i2]))
                sslist.append(ssnetlabels[i2])
        if len(sslist) != 0:
            s = np.array( templist.shape, dtype=np.int )
            entry_dict['label'].append(int(np.argmax(np.bincount(sslist))))
            entry_dict['instvoxcoord_shape'].append(s)
            entry_dict['instvoxcoord'].append(templist.flatten())

print("write table")
pa_table = pa.table( entry_dict )
pyarrow.parquet.write_table(pa_table, "temp.parquet",compression="GZIP")
    

    



