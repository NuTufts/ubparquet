from __future__ import print_function
import os,sys,argparse
from ctypes import c_int

parser = argparse.ArgumentParser("Test PrepKeypointData")
parser.add_argument("-dl", "--input-dlmerged",required=True,type=str,help="Input dlmerged file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()

import torch
import pyarrow as pa
import pyarrow.parquet
import numpy as np

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

input_rootfile_v = ["test.root"]
f_v = rt.std.vector("std::string")()
for f in input_rootfile_v:
    f_v.push_back( f )

# load larcv and larlite IO managers
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  args.input_dlmerged )
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
ioll.open()

if args.tick_backward:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
iolcv.add_in_file( args.input_dlmerged )
for treename in [args.adc,"larflow","segment","instance","ancestor"]:
    iolcv.specify_data_read( larcv.kProductImage2D,  treename )
iolcv.specify_data_read( larcv.kProductChStatus, args.adc )
iolcv.reverse_all_products()
iolcv.initialize()

nentries = iolcv.get_n_entries()
print("Number of entries: ",nentries)

# output files
outfile = rt.TFile(args.output,"recreate")

# data creation algorithms
# ------------------------

# triplet maker (we put it into a tree)
ev_triplet = std.vector("larflow::prep::PrepMatchTriplets")(1)
triptree = rt.TTree("larmatchtriplet","LArMatch triplets")
triptree.Branch("triplet_v",ev_triplet)

# make spacepoint candidates
kpana = larflow.keypoints.PrepKeypointData()
kpana.setADCimageTreeName( args.adc )
kpana.set_verbosity( larcv.msg.kDEBUG )

# voxel data generator (conversion from spacepoints)
_voxelsize_cm = 1.0
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_voxel_size_cm( _voxelsize_cm )
origin_x = voxelizer.get_origin()[0]
origin_y = voxelizer.get_origin()[1]
origin_z = voxelizer.get_origin()[2]

print("Entries in file: ",nentries)

columns = ['spacepoint_t', 'imgcoord_t', 'instance_t', 'segment_t', 'truetriplet_t',"kplabel_t",'origin_t',
           'voxcoord', 'voxfeat', 'voxlabel','voxkplabel','voxssnet','voxinstance','voxorigin','voxoriginweight']

entry_dict = {}
for col in columns:
    entry_dict[col] = []
    entry_dict[col+"_shape"] = []

# Extra columns
if False:
    # Indexing    
    entry_dict["run"] = []
    entry_dict['subrun'] = []
    entry_dict['event'] = []

    # MC truth columns
    entry_dict['truepos'] = []
    entry_dict['nu_energy'] = []
    entry_dict['nu_pid'] = []
    entry_dict['primary_mom'] = []
    entry_dict['primary_pid'] = []

for ientry in range(nentries):

    # load the entry data in the ROOT files
    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)
    
    tripmaker = ev_triplet[0]
    
    ev_adc = iolcv.get_data( larcv.kProductImage2D, args.adc )
    print("number of images: ",ev_adc.Image2DArray().size())
    adc_v = ev_adc.Image2DArray()
    for p in range(adc_v.size()):
        print(" image[",p,"] ",adc_v[p].meta().dump())

    ev_larflow = iolcv.get_data( larcv.kProductImage2D, "larflow" )
    larflow_v  = ev_larflow.Image2DArray()
    
    # spacepoints and true/ghostpoint labels
    tripmaker.process( iolcv, args.adc, args.adc, 10.0, True )
    tripmaker.process_truth_labels( iolcv, ioll, args.adc )

    # keypoint labels
    kpana.process( iolcv, ioll )
    kpana.make_proposal_labels( tripmaker )
    
    # make voxels
    voxelizer.make_voxeldata( tripmaker )

    # Get NUMPY arrays
    
    # get 3d spacepoints
    tripdata = tripmaker.make_triplet_ndarray( False )

    # keypoint labels
    kplabel = kpana.get_triplet_score_array( 10.0 ) # sigma in cm
    tripdata["kplabel_t"] = kplabel

    print("tripdata: ",tripdata.keys())    

    # get voxel label dictionary
    voxdata = voxelizer.make_voxeldata_dict( tripmaker )
    tripdata.update(voxdata)

    # voxel keypoints
    voxkpdict = voxelizer.make_kplabel_dict_fromprep( kpana, voxdata["voxlabel"] )
    tripdata.update(voxkpdict)

    # voxel ssnet
    voxssnetdict = voxelizer.make_ssnet_dict_labels( tripmaker )
    tripdata.update(voxssnetdict)

    # voxel instance
    voxinstancedict = voxelizer.make_instance_dict_labels( tripmaker );
    tripdata.update(voxinstancedict)

    # voxel origin
    voxorigindict = voxelizer.make_origin_dict_labels( tripmaker );
    tripdata.update(voxorigindict)
    
    # mc truth

    print("tripdata+voxels: ",tripdata.keys())        
    
    for col in columns:
        d = tripdata[col].flatten()
        print("col: ",d.shape)
        da = pa.array(d)
        entry_dict[col].append( d )
        s = np.array( tripdata[col].shape, dtype=np.int )
        sa = pa.array( s )
        print("col-s: ",s)
        entry_dict[col+"_shape"].append( s )

    if True:
        break

print("Entry loop finished")
print("write root output file")
#outfile.Write()
outfile.Close()

print("write table")
pa_table = pa.table( entry_dict )
pyarrow.parquet.write_table(pa_table, "temp.parquet",compression="GZIP")
    

    



