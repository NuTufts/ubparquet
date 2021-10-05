from __future__ import print_function
import time
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace

class LArEnnet(torch.nn.Module):

    def __init__(self,device):
        super(LArEnnet,self).__init__()

        self.irreps_input  = o3.Irreps("3x0e")
        self.irreps_mid    = o3.Irreps("4x0e + 4x1e")
        self.irreps_output = o3.Irreps("16x0e")
        self.irreps_sh     = o3.Irreps.spherical_harmonics(lmax=2)
        self.num_basis     = 3
        self.max_radius    = 3.0
        self.num_core_layers = 3
        self.device = device

        # define the tensor product
        self.tp1 = o3.FullyConnectedTensorProduct(self.irreps_input,
                                                  self.irreps_sh,
                                                  self.irreps_mid,
                                                  shared_weights=False)
        self.fc1 = nn.FullyConnectedNet([self.num_basis, 16, self.tp1.weight_numel], torch.relu)
        
        self.core = torch.nn.ModuleList()
        self.tp2 = o3.FullyConnectedTensorProduct(self.irreps_mid,
                                                  self.irreps_sh,
                                                  self.irreps_mid,
                                                  shared_weights=False)        
        for n in range(self.num_core_layers):
            fc = nn.FullyConnectedNet([self.num_basis, 16, self.tp2.weight_numel], torch.relu)
            self.core.append(fc)
            
        self.tp3 = o3.FullyConnectedTensorProduct(self.irreps_mid,
                                                  self.irreps_sh,
                                                  self.irreps_output,
                                                  shared_weights=False)
        self.fc3 = nn.FullyConnectedNet([self.num_basis, 16, self.tp3.weight_numel], torch.relu)

        # classifier
        # 1d convolution going from 16 scalar features to binary classification
        self.fc = torch.nn.Conv1d(16,2,1)
        
    def init_conv( self, pos, f_in ):

        num_nodes = pos.shape[0]
        
        # make the graph edges
        dt_graph = time.time()        
        self.edge_src, self.edge_dst = radius_graph(pos, self.max_radius, max_num_neighbors=num_nodes - 1)
        self.edge_vec = pos[self.edge_dst] - pos[self.edge_src]
        dt_graph = time.time()-dt_graph
        print("time to make graph: %.2f"%(dt_graph))
    
        # compute z
        self.num_neighbors = len(self.edge_src) / num_nodes
        print("z, the average num neighbors: ",self.num_neighbors)

        # embedding for the edge lengths
        dt_embed = time.time()
        num_basis = 3
        edge_length_embedding = soft_one_hot_linspace(
            self.edge_vec.norm(dim=1),
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis='smooth_finite',
            cutoff=True,
        )
        self.edge_length_embedding = edge_length_embedding.mul(self.num_basis**0.5).to(self.device)
        dt_embed = time.time() - dt_embed
        print("dt_embed: %.2f sec"%(dt_embed))

        self.edge_src = self.edge_src.to(self.device)
        self.edge_dst = self.edge_dst.to(self.device)        
        self.edge_vec = self.edge_vec.to(self.device)
        f_in = f_in.to(self.device)

        # apply spherical harmonic function on edge vectors
        self.sh = o3.spherical_harmonics(self.irreps_sh, self.edge_vec, normalize=True, normalization='component').to(self.device)

        # define fully connected layer
        weight = self.fc1(self.edge_length_embedding)

        edge_features = self.tp1(f_in[self.edge_src], self.sh, weight)
        f_out = scatter(edge_features,self.edge_dst,dim=0).div(self.num_neighbors**0.5)
        print("f_out[input]: ",f_out.shape," ",f_out.requires_grad)
        
        return f_out

    def core_conv( self, f_in, fc ):
        weight = fc(self.edge_length_embedding)
        edge_features = self.tp2(f_in[self.edge_src], self.sh, weight)
        f_out = scatter(edge_features,self.edge_dst,dim=0).div(self.num_neighbors**0.5)
        print("f_out[core]: ",f_out.shape," ",f_out.requires_grad)
        return f_out
        
    def feat_conv( self, f_in ):
        weight = self.fc3(self.edge_length_embedding)
        edge_features = self.tp3(f_in[self.edge_src], self.sh, weight)
        f_out = scatter(edge_features,self.edge_dst,dim=0).div(self.num_neighbors**0.5)
        print("f_out[feat]: ",f_out.shape," ",f_out.requires_grad)
        return f_out


    def forward(self, pos_t, feat_t ):
        """
        all pass, from input to head outputs for select match indices
        """
        nnodes = pos_t.shape[0]
        x = self.init_conv( pos_t, feat_t )
        #print("x: ",x.shape," grad_fn=",x.grad_fn," ",x[:10])
        for fc in self.core:
            x = self.core_conv( x, fc )
            #print("x: ",x.shape," grad_fn=",x.grad_fn," ",x[:10])     
        x = self.feat_conv( x )    # (N,16)
        #print("x: ",x.shape," grad_fn=",x.grad_fn," ",x[:10])
        nfeats = x.shape[1]
        x = torch.transpose(x,1,0).reshape(1,nfeats,nnodes) # (1,16,N)
        #print("x: ",x.shape," grad_fn=",x.grad_fn," ",x[:10])
        x = self.fc(x).reshape(2,nnodes).transpose(1,0)  # (N,2)
        #print("x: ",x.shape," grad_fn=",x.grad_fn," ",x[:10])
        return x
            

        

