import os,sys
from ctypes import c_int

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

entry_dict = {}

for ientry in range(nentries):

    kploader.load_entry(ientry)

    # turn shuffle off
    tripdata = kploader.triplet_v.at(0).setShuffleWhenSampling( False )
    
    # get 3d spacepoints
    tripdata = kploader.triplet_v.at(0).get_all_triplet_data( True )
    spacepoints = kploader.triplet_v.at(0).make_spacepoint_charge_array()
    print("data: ",tripdata.shape," spacepoints: ",spacepoints.keys())
    print("triplet index: ",tripdata[:10,:])

    ntriplets = tripdata.shape[0]
    nfilled = c_int(0)    
    data = kploader.sample_data( ntriplets, nfilled, True )

    print("data: ",data.keys())
    print("triplet index: ",data["matchtriplet"][:10,:])
    for k,arr in data.items():
        print(k,": shape=",arr.shape)

    data["spacepoints"] = spacepoints
    entry_dict["ientry"] = data


np.savez_compressed( "temp", entry_dict )
    

    



