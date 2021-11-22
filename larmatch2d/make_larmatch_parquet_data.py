import os,sys
from ctypes import c_int
import pyarrow as pa
import pyarrow.parquet
import numpy as np
import ROOT as rt
from larcv import larcv
from larflow import larflow


input_rootfile_v = ["larmatchtriplet_bnbnu_0000.root"]
f_v = rt.std.vector("std::string")()
for f in input_rootfile_v:
    f_v.push_back( f )

# c++ extension that provides spacepoint labels
kploader = larflow.keypoints.LoaderKeypointData( f_v )
kploader.set_verbosity( larcv.msg.kDEBUG )
kploader.exclude_false_triplets( False )

# Get the number of entries in the tree
nentries = kploader.GetEntries()

# output container for data
columns = ['matchtriplet','ssnet_label','kplabel',"spacepoint_t"]
entry_dict = {}
for col in columns:
    entry_dict[col] = []
    entry_dict[col+"_shape"] = []
    
# 2d wire images
for p in range(3):
    entry_dict["wireimg_feat%d"%(p)] = []
    entry_dict["wireimg_feat%d_shape"%(p)] = []
    entry_dict["wireimg_coord%d"%(p)] = []
    entry_dict["wireimg_coord%d_shape"%(p)] = []
    columns += ["wireimg_feat%d"%(p)]
    columns += ["wireimg_coord%d"%(p)]    

# add run,subrun,event to be able to collate back to original truth
entry_dict["run"]    = []
entry_dict["subrun"] = []
entry_dict["event"]  = []

for ientry in range(nentries):

    # Get the first entry (or row) in the tree (i.e. table)
    kploader.load_entry(ientry)

    entry_dict["run"].append( kploader.run() )
    entry_dict["subrun"].append( kploader.subrun() )
    entry_dict["event"].append( kploader.event() )    

    # turn shuffle off (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).setShuffleWhenSampling( False )

    # 2d images
    wireimg_dict = {}
    for p in range(3):
        wireimg = kploader.triplet_v.at(0).make_sparse_image( p )
        wireimg_coord = wireimg[:,:2].astype(np.long)
        wireimg_feat  = wireimg[:,2]
        wireimg_dict["wireimg_coord%d"%(p)] = wireimg_coord
        wireimg_dict["wireimg_feat%d"%(p)] = wireimg_feat        

    # get 3d spacepoints (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).get_all_triplet_data( True )
    spacepoints = kploader.triplet_v.at(0).make_spacepoint_charge_array()    
    nfilled = c_int(0)
    ntriplets = tripdata.shape[0]    
    
    data = kploader.sample_data( ntriplets, nfilled, True )
    data.update(spacepoints)
    data.update( wireimg_dict )

    # to do: the commands here are still awfully wonky
    
    print("numpy arrays in tripdata: ",tripdata.shape)
    print("numpy arrays from kploader: ",data.keys())

    # append to entry dict container
    for col in columns:
        print("col: ",col)
        d = data[col].flatten()
        print("  data: ",d.shape)
        da = pa.array(d)
        entry_dict[col].append( d )
        s = np.array( data[col].shape, dtype=np.int )
        sa = pa.array( s )
        print("  shape: ",s)
        entry_dict[col+"_shape"].append( s )

    max_u = np.max( data["matchtriplet"][:,0] )
    max_v = np.max( data["matchtriplet"][:,1] )
    max_y = np.max( data["matchtriplet"][:,2] )
    print("sanity check, max indices: ",max_u,max_v,max_y)

print(entry_dict["run"])
print(entry_dict["subrun"])
print(entry_dict["event"])
print("write table")
pa_table = pa.table( entry_dict )
pyarrow.parquet.write_table(pa_table, "temp.parquet",compression="GZIP")

    

    
