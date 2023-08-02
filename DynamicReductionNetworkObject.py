
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv, MessagePassing
from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops,add_self_loops, degree
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus,
                                max_pool, max_pool_x, global_max_pool,
                                avg_pool, avg_pool_x, global_mean_pool,
                                global_add_pool)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class GraphMessagePassing(MessagePassing):
    # 3 view message passing in three_d_message mode 
    def __init__(self, input_dim, hidden_dim):
        super(GraphMessagePassing, self).__init__(aggr='add')
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class DynamicReductionNetworkObject(nn.Module):
    '''
     This class implements the Dynamic Reduction Network to 2D or 3D protoDUNE data images. 
     There are 3 training modes:
         two_d: utilize 2D (wire and time ticks + collected charge) images from collection wire plane
         three_d: utilize 3D (reco 3D space points + collected charge) images
         three_d_message: 3 independent 2D networks (2 induction planes and collection plane)
             where message passing and is performed on 2D hits between views that are connected by 3D space point-
             the output of each network is passed through a linear layer to regress the output energy
         Other architecures have been investigated (including 3 independent 2D networks and pooling to a single value)-
         however, these were found less useful and have been excluded from this code.
     Dropout and batch normalization layers were studied but found to not affect the result.
     Initialization of layer parameters is hard-coded relative to training script. TO-DO: update script for flexibility.
    '''
    def __init__(self, mode='two_d', which_device='cuda', hidden_dim=64, output_dim=1, k=16, aggr='add', norm=None,
            loop=True, pool='max', agg_layers=2, mp_layers=2, in_layers=1, out_layers=3):
        super(DynamicReductionNetworkObject, self).__init__()

        self.flag = False

        self.message_layer = None

        self.loop = loop

        if norm is None:
            norm = torch.ones(input_dim)
        self.datanorm = nn.Parameter(norm)

        self.k = k

        self.device = torch.device(which_device)

        if(which_device != 'cpu'):
            torch.cuda.reset_max_memory_allocated(self.device)

        if(mode!='three_d' and mode!='two_d' and mode!='three_d_message'):
            self.flag = True
            mode = 'two_d'
        if(mode=='three_d'):
            input_dim = 4
        if(mode=='two_d'):
            input_dim = 3
        if(mode=='three_d_message'):
            input_dim = 6
            self.message_layer = GraphMessagePassing(input_dim, hidden_dim)
            self.message_layer.to(self.device)

        self.mode = mode

        self.agg_layers_0 = nn.ModuleList()
        if(mode=='three_d_message'):
            self.agg_layers_1 = nn.ModuleList()
            self.agg_layers_2 = nn.ModuleList()

        print("Pooling with",pool)
        print("Using self-loops" if self.loop else "Not using self-loops")
        print("There are",agg_layers,'aggregation layers')

        # Construct input layer
        in_layers_l_0 = []
        in_layers_l_0 += [nn.Linear(hidden_dim, hidden_dim),
                nn.ELU()]

        for i in range(in_layers-1):
            in_layers_l_0 += [nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU()]

        self.inputnet_0 = nn.Sequential(*in_layers_l_0)

        if(mode=='three_d_message'):
            in_layers_l_1 = []
            in_layers_l_1 += [nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU()]

            for i in range(in_layers-1):
                in_layers_l_1 += [nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU()]

            self.inputnet_1 = nn.Sequential(*in_layers_l_1)

            in_layers_l_2 = []
            in_layers_l_2 += [nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU()]

            for i in range(in_layers-1):
                in_layers_l_2 += [nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU()]

            self.inputnet_2 = nn.Sequential(*in_layers_l_2)

        # Construct Aggregation Layers
        for i in range(agg_layers):
            mp_layers_l = []

            for j in range(mp_layers-1):
                mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                        nn.ELU()]

            mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ELU()]

            convnn = nn.Sequential(*mp_layers_l)

            self.agg_layers_0.append(EdgeConv(nn=convnn, aggr=aggr))

        if(mode=='three_d_message'):
            for i in range(agg_layers):
                mp_layers_l = []

                for j in range(mp_layers-1):
                    mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                            nn.ELU()]

                mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                        nn.ELU()]

                convnn = nn.Sequential(*mp_layers_l)

                self.agg_layers_1.append(EdgeConv(nn=convnn, aggr=aggr))

            for i in range(agg_layers):
                mp_layers_l = []

                for j in range(mp_layers-1):
                    mp_layers_l += [nn.Linear(2*hidden_dim, 2*hidden_dim),
                            nn.ELU()]

                mp_layers_l += [nn.Linear(2*hidden_dim, hidden_dim),
                        nn.ELU()]

                convnn = nn.Sequential(*mp_layers_l)

                self.agg_layers_2.append(EdgeConv(nn=convnn, aggr=aggr))

        # Construct Output Layers
        out_layers_l_0 = []

        for i in range(out_layers-1):
            out_layers_l_0 += [nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU()]

        out_layers_l_0 += [nn.Linear(hidden_dim, output_dim)]

        self.output_0 = nn.Sequential(*out_layers_l_0)

        if(mode=='three_d_message'):
            out_layers_l_1 = []

            for i in range(out_layers-1):
                out_layers_l_1 += [nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU()]

            out_layers_l_1 += [nn.Linear(hidden_dim, output_dim)]

            self.output_1 = nn.Sequential(*out_layers_l_1)

            out_layers_l_2 = []

            for i in range(out_layers-1):
                out_layers_l_2 += [nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU()]

            out_layers_l_2 += [nn.Linear(hidden_dim, output_dim)]

            self.output_2 = nn.Sequential(*out_layers_l_2)

            # "Pooling" linear layer for 3 views
            out_layers_l = []

            out_layers_l += [nn.Linear(3, 1)]

            self.output = nn.Sequential(*out_layers_l)

        # Use appropriate pooling method
        if pool == 'max':
            self.poolfunc = max_pool
            self.x_poolfunc = max_pool_x
            self.global_poolfunc = global_max_pool
        elif pool == 'mean':
            self.poolfunc = avg_pool
            self.x_poolfunc = avg_pool_x
            self.global_poolfunc = global_mean_pool
        elif pool == 'add':
            self.poolfunc = avg_pool
            self.x_poolfunc = avg_pool_x
            self.global_poolfunc = global_add_pool
        else:
            print("ERROR: INVALID POOLING")

    def doLayer(self, data, i, j):
        '''
        Do one aggregation layer
            data: current batch object
            i: the index of the layer to be done
            j: which network if training on all 3 views

        Returns the transformed batch object. 
            if this is the last layer, instead returns (data.x, data.batch)
        '''
        edgeconv = None
        if(j==0):
            edgeconv = self.agg_layers_0[i]
        if(j==1):
            edgeconv = self.agg_layers_1[i]
        if(j==2):
            edgeconv = self.agg_layers_2[i]

        knn = knn_graph(data.x, self.k, data.batch, loop=self.loop, flow=edgeconv.flow)
        data.edge_index = to_undirected(knn)
        data.x = edgeconv(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        
        if i != len(self.agg_layers_0)-1:
            data.edge_attr = None
            return self.poolfunc(cluster, data)

        else:
            return self.x_poolfunc(cluster, data.x, data.batch)

    def forward(self,data0,data1,data2):
        #Push the batch through the network

        if(self.mode=='three_d_message'):
            '''
             If three_d_message mode, construct a message passing graph connecting
                  hits in 3 views through 3D space point
            '''
            num_nodes_1 = data0.x.size(dim=0)
            num_nodes_2 = data1.x.size(dim=0)
            num_nodes_3 = data2.x.size(dim=0)

            # Ugly....
            source_index = []
            target_index = []
            for i in range(num_nodes_1):
                source_index.append(i)
            for i in range(num_nodes_2):
                source_index.append(num_nodes_1+i)
            for i in range(num_nodes_3):
                source_index.append(num_nodes_1+num_nodes_2+i)
            for i in range(num_nodes_1):
                source_index.append(i)
            for i in range(num_nodes_2):
                source_index.append(num_nodes_1+i)
            for i in range(num_nodes_3):
                source_index.append(num_nodes_1+num_nodes_2+i)
            for i in range(num_nodes_1):
                source_index.append(i)
            for i in range(num_nodes_2):
                source_index.append(num_nodes_1+i)
            for i in range(num_nodes_3):
                source_index.append(num_nodes_1+num_nodes_2+i)

            for i in range(3):
                for j in range(num_nodes_1):
                    target_index.append(j)
            for i in range(3):
                for j in range(num_nodes_2):
                    target_index.append(num_nodes_1+j)
            for i in range(3):
                for j in range(num_nodes_3):
                    target_index.append(num_nodes_1+num_nodes_2+j)

            edge_indicies = [source_index,target_index]
            edge_indicies = torch.tensor(edge_indicies)

            new_graph = torch.cat((data0.x, data1.x, data2.x), 0)

            edge_indicies = edge_indicies.to(self.device)
            new_graph = new_graph.to(self.device)

            output = self.message_layer(new_graph,edge_indicies)
            output = torch.split(output,[num_nodes_1,num_nodes_2,num_nodes_3]) 

            data0.x = self.datanorm * output[0]
            data0.x = self.inputnet_0(data0.x)

        if(self.mode!='three_d_message'):
            data0.x = self.datanorm * data0.x
            data0.x = self.inputnet_0(data0.x)           

        for i in range(len(self.agg_layers_0)):
            data0 = self.doLayer(data0, i,0)

        ###########################################
        if len(self.agg_layers_0)==0: #if there are no layers, format data appropriately 
            data0 = data0.x, data0.batch

        x0 = self.global_poolfunc(*data0)

        x0 = self.output_0(x0).squeeze(-1)

        if(self.mode!='three_d_message'):
            return x0

        data1.x = self.datanorm * output[1]
        data1.x = self.inputnet_1(data1.x)

        for i in range(len(self.agg_layers_1)):
            data1 = self.doLayer(data1, i,1)

        ###########################################
        if len(self.agg_layers_1)==0: #if there are no layers, format data appropriately 
            data1 = data1.x, data1.batch

        x1 = self.global_poolfunc(*data1)

        x1 = self.output_1(x1).squeeze(-1)

        data2.x = self.datanorm * output[2]
        data2.x = self.inputnet_2(data2.x)

        for i in range(len(self.agg_layers_2)):
            data2 = self.doLayer(data2, i,2)

        ###########################################
        if len(self.agg_layers_2)==0: #if there are no layers, format data appropriately 
            data2 = data2.x, data2.batch

        x2 = self.global_poolfunc(*data2)

        x2 = self.output_2(x2).squeeze(-1)

        output = torch.cat((torch.unsqueeze(x0,dim=0), torch.unsqueeze(x1,dim=0), torch.unsqueeze(x2,dim=0)), dim=0)
        output = torch.transpose(output,0,1)
        batch_output = self.output(output).squeeze(-1)
        return batch_output, self.flag
