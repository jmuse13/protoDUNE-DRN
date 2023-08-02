from time import time
from tqdm import tqdm,trange

import numpy as np
import math
import os
import h5py
import argparse

import torch
import torch.autograd
import torch.nn as nn
from torch import Tensor
from torch import optim
from torch.utils.data import Subset
from torch.nn.functional import softplus

import torch_geometric
from torch_geometric.nn import global_max_pool, to_hetero
from torch_geometric.data import Data,HeteroData
from torch_geometric.loader import DataLoader

from DynamicReductionNetworkObject import DynamicReductionNetworkObject as DRN

class Dataset(torch_geometric.data.Dataset):
    # Custom Dataset for DataLoader object which randomizes input and formats data and target
    def __init__(self, train_valid, in_regions, energy_type, in_file0, in_file1, in_file2):
        self.index_map = []
        self.energy_map = []
        self.f0 = in_file0
        self.f1 = in_file1
        self.f2 = in_file2
        self.train_valid = train_valid
        self.in_regions = in_regions
        self.energy_type = energy_type

        for i in self.in_regions:
            self.get_data(i)

    def get_data(self, in_region):
        energies = self.f2[in_region+'_'+self.train_valid+'/'+self.energy_type]
        for j in list(self.f2[in_region+'_'+self.train_valid].keys()):
            if(j=='box' or j=='reco' or j=='nom_energy' or j=='energy'):
                continue
            self.index_map.append(in_region+'_'+self.train_valid+'/'+j)
            self.energy_map.append(energies[int(j.split('_')[1])])
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        out = {}

        if(self.f0!=None):
            out['zero'] = Data(x=torch.from_numpy(np.asarray(self.f0[self.index_map[index]]).astype(np.float32)),y=torch.from_numpy(np.asarray(self.energy_map[index]).astype(np.float32)))
        if(self.f1!=None):
            out['one'] = Data(x=torch.from_numpy(np.asarray(self.f1[self.index_map[index]]).astype(np.float32)),y=torch.from_numpy(np.asarray(self.energy_map[index]).astype(np.float32)))
        if(self.f0==None):
            out['zero'] = []
        if(self.f1==None):
            out['one'] = []
        out['two'] = Data(x=torch.from_numpy(np.asarray(self.f2[self.index_map[index]]).astype(np.float32)),y=torch.from_numpy(np.asarray(self.energy_map[index]).astype(np.float32)))

        return out

def loadFeatures(train_valid, in_regions, energy_type, in_file0, in_file1, in_file2, batch_size):
    print("loading in features...")
    t0 = time()

    params = {'batch_size':batch_size,'num_workers':1,'shuffle':False, 'pin_memory':True}

    data_set = Dataset(train_valid, in_regions, energy_type, in_file0,in_file1,in_file2)
    loader = DataLoader(data_set, **params)

    print("\tTook %0.3f seconds"%(time()-t0))

    return loader

parser = argparse.ArgumentParser()
parser.add_argument("--i",type=str,default="null",help="Input File Directory")
parser.add_argument("--o",type=str,default="out",help="Model Directory")
parser.add_argument("--e",type=str,default="energy",help="Energy Target")
parser.add_argument("--batch_size",type=int,default=50,help="Batch Size")
parser.add_argument("--ep",type=int,default=0,help="Epoch Iteration to Apply")
parser.add_argument("--device",type=str,default="cuda",help="Which Device?")
parser.add_argument("--mode",type=str,default="two_d",help="Training Mode")
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

# Set parameters
output_dir = args.o
in_folder_top = args.i
energy_type = args.e

batch_size = args.batch_size

epoch = int(args.ep)

which_device = args.device

# Harded-coded region definitions (per beam energy point)- flexibility is an algorithmic cost/benefit analysis
in_regions = ['zero_point_three_gev','zero_point_five_gev','one_gev_0','two_gev','three_gev_0','six_gev_0','seven_gev']

# Load files for all 3 views
in_file0 = None
in_file1 = None
in_file2 = None

if(args.mode=='three_d_message'):
    try:
        in_file0 = h5py.File(in_folder_top+'view_zero.h5','r')
    except:
        print('No input file for induction plane 0')
    try:
        in_file1 = h5py.File(in_folder_top+'view_one.h5','r')
    except:
        print('No input file for induction plane 1')
    try:
        in_file2 = h5py.File(in_folder_top+'view_two.h5','r')
    except:
        print('No input file for collection plane')
else:
    try:
        in_file2 = h5py.File(in_folder_top+'view_two.h5','r')
    except:
        print('No input file for collection plane')    

device = torch.device(which_device)
if(which_device != 'cpu'):
    torch.cuda.reset_max_memory_allocated(device)

# Initialize model and load epoch that will be used to apply the model
model = DRN(mode=args.mode,which_device=which_device)
model.load_state_dict(torch.load(output_dir+'model_'+epoch+'.pt', map_location=torch.device(which_device)))
model.to(device)

model.eval()

with torch.no_grad():
    # Apply the model and pull out true values for comparison with regression 
    # ('box' is nominal energy reco method used in protoDUNE)
    for top_direct in top_directories:
        output_dir_ = output_dir+top_direct+'/'
        os.system('mkdir '+output_dir_)
        y_pred = []
        y_true = []
        y_reco = []
        y_box = []
        predname = '%s/pred.pickle'%(output_dir_)
        truename = '%s/true.pickle'%(output_dir_)
        reconame = '%s/reco.pickle'%(output_dir_)
        boxname = '%s/box.pickle'%(output_dir_)

        loader = loadFeatures('apply',top_direct, energy_type, in_file0, in_file1, in_file2, batch_size)
        total = len(loader.dataset)
        t = tqdm(enumerate(loader),total=int(math.ceil(total/batch_size)))

        t_true_energies = in_file2[top_direct+'_apply'+'/'+energy_type]
        t_true_recos = in_file2[top_direct+'_apply'+'/reco']
        t_true_boxes = in_file2[top_direct+'_apply'+'/box']
        for i,data in t:
            data0 = None
            data1 = None
            data2 = None

            if(in_file0==None and in_file1==None):
                data0 = data['two']
                data0 = data0.to(device)
            else:
                data0 = data['zero']
                data0 = data0.to(device)

                data1 = data['one']
                data1 = data1.to(device)

                data2 = data['two']
                data2 = data2.to(device)

            result = model(data0,data1,data2)
            y_pred.append(result)
            y_true.append(t_true_energies[i])
            y_reco.append(t_true_recos[i])
            y_box.append(t_true_boxes[i])

        # Yes, I am basically transforming HDF5 files to pickle format for no apparent reason
        with open(predname, 'wb') as f:
            pickle.dump(y_pred, f, protocol = 4)
        with open(truename, 'wb') as f:
            pickle.dump(y_true, f, protocol = 4)
        with open(reconame, 'wb') as f:
            pickle.dump(y_reco, f, protocol = 4)
        with open(boxname, 'wb') as f:
            pickle.dump(y_box, f, protocol = 4)
in_file0.close()
in_file1.close()
in_file2.close()
