import os,sys,time
import numpy as np
import torch
import torch.optim
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
from larennet_dataset import larennetDataset
from larennet import LArEnnet


device = torch.device("cuda")
#device = torch.device("cpu")

# model
model = LArEnnet(device).to(device)
smax = torch.nn.Softmax( dim=1 )
focal_loss_gamma = 2
LR=1.0e-3
if False:
    # dump model and quit
    print(model)
    for n,p in model.named_parameters():
        print(n,": ",p.shape)    
    sys.exit(0)

batch_size = 1
test = larennetDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"], voxelize=True )
print("NENTRIES: ",len(test))

loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larennetDataset.collate_fn)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LR,
                             weight_decay=1.0e-4)

NITERS = 5000

data = next(iter(loader))
print(data[0].keys())
    
print("spacepoints: ",data[0]["spacepoint_t"].shape)
spt = data[0]["spacepoint_t"]
truth = data[0]["truetriplet_t"]
# reduce
truth = truth[ spt[:,2]<256 ]
spt = spt[ spt[:,2]<256 ]

truth = truth[ spt[:,0]>0 ]
spt = spt[ spt[:,0]>0 ]

truth = truth[ spt[:,0]<256 ]
spt = spt[ spt[:,0]<256 ]

pos = torch.from_numpy( spt[:,0:3] )
pix = torch.from_numpy( spt[:,3:] )
mean_pix = pix.mean()
pix -= mean_pix
tru = torch.from_numpy( truth ).to(device)
print("pos: ",pos.shape," ",pos[:10])
print("pix: ",pix.shape," ",pix[:10])
print("tru: ",tru.shape)
print("[enter] to continue")
if True:
    input()


for iiter in range(NITERS):
    optimizer.zero_grad()    

    out = model(pos,pix)
    #print("out: ",out.shape,out.requires_grad)
    #print("out raw: ",out[:10,:])
    out = smax(out)
    #print("softmax(out): ",out[:10,:])

    # focal loss
    fmatchlabel = tru.type(torch.float).requires_grad_(False)
    #print("tru: ",tru[:10])

    p_t = fmatchlabel*(out[:,1]+1.0e-6) + (1-fmatchlabel)*(out[:,0]+1.0e-6) # p if y==1; 1-p if y==0            
    #print("p_t: ",p_t[:10]," ",p_t.requires_grad)

    loss = (-torch.log( p_t )*torch.pow( 1-p_t, focal_loss_gamma )).mean()
    #loss = (-torch.log( p_t )).mean()    
    print("loss: ",loss," ",loss.requires_grad)

    loss.backward()
    optimizer.step()
    
    #for name,p in model.named_parameters():
    #    print(name,": ",p.shape,p.grad)

    with torch.no_grad():
        acc = ((out[:,1]>0.5).type(torch.long).eq( tru )).type(torch.float).mean()
        print("acc: ",acc.item())

    if False:
        print("[enter] to quit.")
        input()

print("DONE")
