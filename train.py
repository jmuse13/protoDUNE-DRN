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

    params = {'batch_size':batch_size,'num_workers':24,'shuffle':True, 'pin_memory':True}

    data_set = Dataset(train_valid, in_regions, energy_type, in_file0,in_file1,in_file2)
    loader = DataLoader(data_set, **params)

    print("\tTook %0.3f seconds"%(time()-t0))

    return loader

parser = argparse.ArgumentParser()
parser.add_argument("--i",type=str,default="null",help="Input File Directory")
parser.add_argument("--o",type=str,default="out",help="Output Directory")
parser.add_argument("--e",type=str,default="energy",help="Energy Target")
parser.add_argument("--valid_batch_size",type=int,default=50,help="Validation Batch Size")
parser.add_argument("--train_batch_size",type=int,default=50,help="Validation Batch Size")
parser.add_argument("--ep",type=int,default=150,help="Number of Epochs")
parser.add_argument("--device",type=str,default="cuda",help="Which Device?")
parser.add_argument("--max_lr",type=float,default=0.0001,help="Nominal Learning Rate")
parser.add_argument("--min_lr",type=float,default=1e-07,help="Minimum Learning Rate")
parser.add_argument("--mode",type=str,default="two_d",help="Training Mode")
args = parser.parse_args()

# Set parameters
output_dir = args.o
in_folder_top = args.i
energy_type = args.e

valid_batch_size = args.valid_batch_size
train_batch_size = args.train_batch_size

num_epochs = args.ep

max_lr =  args.max_lr
min_lr =  args.min_lr

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

train_loader = loadFeatures('train',in_regions, energy_type, in_file0,in_file1,in_file2, train_batch_size)
valid_loader = loadFeatures('valid',in_regions, energy_type, in_file0,in_file1,in_file2, valid_batch_size)

device = torch.device(which_device)
if(which_device != 'cpu'):
    torch.cuda.reset_max_memory_allocated(device)

# Initialize model
model = DRN(mode=args.mode,which_device=which_device)
model.to(device)
print("model device:", next(model.parameters()).device)

# Initialize loss function and optimizer/scheduler.
# These were all optimized previously and are left hard-coded
optimizer = optim.Adam(model.parameters(), lr=max_lr)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=min_lr)
mae_loss_v = nn.L1Loss()
mae_loss = nn.L1Loss()

min_valid_loss = np.inf

print('0.0001, ADAM, L1')

for ep in range(num_epochs):
    # Train the model
    train_loss = 0.0
    training_errors = 0

    print('EPOCH:', ep,'LR:', scheduler.get_last_lr())

    model.train() 

    total = len(train_loader.dataset)
    t = tqdm(enumerate(train_loader),total=int(math.ceil(total/train_batch_size)))

    for i,data in t:
        optimizer.zero_grad()

        batch_target = data['two'].y
        batch_target = batch_target.to(device)

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

        batch_output,flag = model(data0,data1,data2)
        if(flag==True):
            print('Issue with input to model')
            training_errors += 1
            continue
        batch_loss = mae_loss(batch_output,batch_target)
        train_loss += batch_loss.item()

        optimizer.step()

    # Validate the model

    valid_loss = 0.0
    valid_errors = 0

    model.eval()

    total = len(valid_loader.dataset)
    v = tqdm(enumerate(valid_loader),total=int(math.ceil(total/valid_batch_size)))

    for i, data in v:
        batch_target = data['two'].y
        batch_target = batch_target.to(device)

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

        batch_output,flag = model(data0,data1,data2)
        if(flag==True):
            print('Issue with input to model')
            valid_errors += 1
            continue
        batch_loss = mae_loss_v(batch_output,batch_target)
        valid_loss += batch_loss.item()

    scheduler.step()

    print('Epoch {} LOSS train {} valid {}'.format(ep,(train_loss/len(train_loader)), (valid_loss/len(valid_loader))))
    print('Training errors: {} Validation errors: {}'.format(training_errors,valid_errors))
    if(min_valid_loss > valid_loss):
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(),output_dir+'model_'+str(ep)+'.pt')
in_file0.close()
in_file1.close()
in_file2.close()
