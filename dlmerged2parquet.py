from __future__ import print_function
import os,sys,argparse
from math import sqrt
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

# mc truth
save_mctruth = True
larbysmc = ublarcvapp.mctools.LArbysMC()

# track conversion
track_convertor = ublarcvapp.mctools.TruthTrackSCE()
shower_convertor = ublarcvapp.mctools.TruthShowerTrunkSCE()

print("Entries in file: ",nentries)

columns = ['spacepoint_t', 'imgcoord_t', 'instance_t', 'segment_t', 'truetriplet_t',"kplabel_t",'origin_t',
           'voxcoord', 'voxfeat', 'voxlabel','voxkplabel','voxssnet','voxinstance','voxorigin','voxoriginweight']

entry_dict = {}
for col in columns:
    entry_dict[col] = []
    entry_dict[col+"_shape"] = []

# Extra columns
if save_mctruth:
    # Indexing    
    entry_dict["run"] = []
    entry_dict['subrun'] = []
    entry_dict['event'] = []

    # MC truth columns
    entry_dict['truepos'] = []
    entry_dict['nu_energy'] = []
    entry_dict['nu_pid'] = []
    entry_dict['nu_ccnc'] = []    
    entry_dict['nu_interaction'] = []
    entry_dict['nu_geniemode'] = []    
    entry_dict['primary_mom'] = []
    entry_dict['primary_pid'] = []
    entry_dict['primary_start'] = []
    entry_dict['primary_trackid'] = []

    # voxel instance map
    entry_dict['voxinstancelist'] = []
    entry_dict['voxidlist'] = []    

for ientry in range(2,nentries):

    # load the entry data in the ROOT files
    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    # extract truth info
    if save_mctruth:
        larbysmc.process( ioll )
        
        entry_dict['run'].append( larbysmc._run )
        entry_dict['subrun'].append( larbysmc._subrun )
        entry_dict['event'].append( larbysmc._event )

        entry_dict['truepos'].append( [larbysmc._vtx_sce_x,larbysmc._vtx_sce_y,larbysmc._vtx_sce_z] )
        entry_dict['nu_energy'].append( larbysmc._Enu_true )
        entry_dict['nu_pid'].append( larbysmc._nu_pdg )
        entry_dict['nu_ccnc'].append( larbysmc._current_type )
        entry_dict['nu_interaction'].append( larbysmc._interaction_type )
        entry_dict['nu_geniemode'].append( larbysmc._genie_mode )

        ev_mctrack  = ioll.get_data(larlite.data.kMCTrack,"mcreco")
        ev_mcshower = ioll.get_data(larlite.data.kMCShower,"mcreco")

        # store mctrack, store mcshower
        prim_mom = []
        prim_start = []
        prim_pid = []
        prim_trackid = []

        for itrack in range(ev_mctrack.size()):
            track = ev_mctrack.at(itrack)
            if track.Origin()!=1:
                continue
            sce_track = track_convertor.applySCE( track )
            # we record the initial direction and start point
            trackE = track.Start().E()
            trackP = sqrt( track.Start().Px()*track.Start().Px() + track.Start().Py()*track.Start().Py() + track.Start().Pz()*track.Start().Pz() )
            if sce_track.NumberTrajectoryPoints()<2:
                continue
            start = sce_track.LocationAtPoint(0)
            sdir  = sce_track.DirectionAtPoint(0)
            mom = [ trackP*sdir[i] for i in range(3) ]
            prim_mom.append( [trackE] + mom )
            prim_pid.append( track.PdgCode() )
            prim_start.append( [ start[i] for i in range(3) ] )
            prim_trackid.append( track.TrackID() )

        for ishower in range(ev_mcshower.size()):
            shower = ev_mcshower.at(ishower)
            if shower.Origin()!=1:
                continue
            sce_track = shower_convertor.applySCE( shower )
            if sce_track.NumberTrajectoryPoints()<2:
                continue
                        
            # we record the initial direction and start point
            trackE = shower.Start().E()
            trackP = sqrt( shower.Start().Px()*shower.Start().Px() + shower.Start().Py()*shower.Start().Py() + shower.Start().Pz()*shower.Start().Pz() )
            start = sce_track.LocationAtPoint(0)
            sdir  = sce_track.DirectionAtPoint(0)
            mom = [ trackP*sdir[i] for i in range(3) ]
            prim_mom.append( [trackE] + mom )
            prim_pid.append( shower.PdgCode() )
            prim_start.append( [ start[i] for i in range(3) ] )
            prim_trackid.append( shower.TrackID() )            
            
        entry_dict['primary_mom'].append( prim_mom )
        entry_dict['primary_start'].append( prim_start )
        entry_dict['primary_pid'].append( prim_pid )
        entry_dict['primary_trackid'].append( prim_trackid )

    
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
    # convert instance2id into a numpy array
    instancelist = []
    idlist = []
    for instance,iid in voxinstancedict['voxinstance2id'].items():
        instancelist.append(int(instance))
        idlist.append(int(iid))
    print("instancelist: ",instancelist)
    print("idlist      : ",idlist)
    entry_dict['voxinstancelist'].append( instancelist )
    entry_dict['voxidlist'].append( idlist )    

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

    if True and ientry>=4:
        break

print("Entry loop finished")
print("write root output file")
#outfile.Write()
outfile.Close()

print("write table")
pa_table = pa.table( entry_dict )
pyarrow.parquet.write_table(pa_table, "temp2.parquet",compression="GZIP")
    

    



