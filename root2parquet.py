import os,sys
from ctypes import c_int

import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import numpy as np
import ROOT as rt
from larcv import larcv
from larflow import larflow

input_rootfile_v = ["test.root"]
f_v = rt.std.vector("std::string")()
for f in input_rootfile_v:
    f_v.push_back( f )

kploader = larflow.keypoints.LoaderKeypointData( f_v )
kploader.set_verbosity( larcv.msg.kDEBUG )
kploader.exclude_false_triplets( False )
    
nentries = kploader.GetEntries()
print("Entries in file: ",nentries)

columns = ['matchtriplet','ssnet_label','kplabel','spacepoint_t']
entry_dict = {}
for col in columns:
    entry_dict[col] = []
    entry_dict[col+"_shape"] = []

for ientry in range(nentries):

    kploader.load_entry(ientry)

    # turn shuffle off
    tripdata = kploader.triplet_v.at(0).setShuffleWhenSampling( False )
    
    # get 3d spacepoints
    tripdata = kploader.triplet_v.at(0).get_all_triplet_data( True )
    spacepoints = kploader.triplet_v.at(0).make_spacepoint_charge_array()
    print("data: ",tripdata.shape," spacepoints: ",spacepoints.keys())
    print("triplet index: ",tripdata[:10,:])
    print("spacepoints: ",spacepoints["spacepoint_t"].shape)

    ntriplets = tripdata.shape[0]
    nfilled = c_int(0)    
    data = kploader.sample_data( ntriplets, nfilled, True )

    print("data: ",data.keys())
    print("triplet index: ",data["matchtriplet"][:10,:])
    for k,arr in data.items():
        print(k,": shape=",arr.shape)

    data.update(spacepoints)

    for col in columns:
        d = data[col].flatten()
        print("col: ",d.shape)
        da = pa.array(d)
        entry_dict[col].append( d )
        s = np.array( data[col].shape, dtype=np.int )
        sa = pa.array( s )
        print("col-s: ",s)
        entry_dict[col+"_shape"].append( s )


print("write table")
pa_table = pa.table( entry_dict )
pyarrow.parquet.write_table(pa_table, "temp.parquet",compression="GZIP")
    

    



