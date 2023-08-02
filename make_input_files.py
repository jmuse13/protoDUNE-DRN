import pandas as pd
import numpy as np
import torch
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--i",type=str,default="null",help="Input File Directory")
parser.add_argument("--o",type=str,default="out",help="Output Directory")
parser.add_argument("--mode",type=str,default="two_d",help="Training Mode")
args = parser.parse_args()

# Set parameters
in_path = args.i
out_directory = args.o
mode = args.mode

tree = 'out_events.root:pdAnaTree/EventTree'

# Define input energy samples
top_energies = []
top_energies.append(['zero_point_three_gev'])
top_energies.append(['zero_point_five_gev'])
top_energies.append(['one_gev_0','one_gev_1','one_gev_2','one_gev_3','one_gev_4','one_gev_5','one_gev_6','one_gev_7','one_gev_8','one_gev_9'])
top_energies.append(['two_gev'])
top_energies.append(['three_gev_0','three_gev_1','three_gev_2'])
top_energies.append(['six_gev_0','six_gev_1','six_gev_2','six_gev_3','six_gev_4','six_gev_5','six_gev_6','six_gev_7','six_gev_8','six_gev_9'])
top_energies.append(['seven_gev'])

# Define output files
hf0 = h5py.File(out_directory'+/view_zero.h5', 'w')
hf1 = h5py.File(out_directory'+/view_one.h5', 'w')
hf2 = h5py.File(out_directory'+/view_two.h5', 'w')

variables = ['event','is_shower','directionY','directionZ','startZ','showerStartX','startY','startZ',
            'energy','particle_energy_full','hit0w','hit0t','hit0q','hit1w','hit1t','hit1q','hit2w',
            'hit2t','hit2q','primary_Shower_sim_energy_new','pointx','pointy','pointz']

# Loop over input energies
for fo in range(len(top_energies)):
    bottom_directories = []
    nominal_energy_list = []
    log_energy_list = []
    sum_charge_list = []
    box_energy_list = []
    out_data0 = []
    out_data1 = []
    out_data2 = []

    tot_num_events = 0

    # For each energy...
    for top_dir in top_energies[fo]:
        # Ugly but effective way to pull out multiple root files in multiple subdirectories
        for dirpath,dirnames,filenames in os.walk(in_path+top_dir,topdown=False):
            for i in range(len(dirnames)):
                bottom_directories.append(dirnames[i])
        for bot_dir in bottom_directories:
            folder = in_path+top_dir+'/'+bot_dir+'/'
            in_file = None
            try:
                in_file = uproot.open(folder+tree)
            except:
                print('No file found')
                continue
            # Get input file
            in_data = in_file.arrays(variables,library="np")

            # Loop over events
            for i in range(len(in_data["event"])):
                # Require primary particle is a EM shower and daughter is either shower or track->shower
                if(len(in_data["is_shower"][i]) > 2):
                    continue
                if(len(in_data["is_shower"][i]) == 0):
                    continue
                if(len(in_data["is_shower"][i]) == 1):
                    if(in_data["is_shower"][i][0] != 1):
                        continue
                if(len(in_data["is_shower"][i]) == 2):
                    if(in_data["is_shower"][i][0] != 0):
                        continue
                    if(in_data["is_shower"][i][1] != 1):
                        continue
                # Cuts to ensure shower is consistent with beamline direction
                if(in_data["directionY"][i][0]>0 or in_data["directionY"][i][0]<-0.60):
                    continue
                if(in_data["directionZ"][i][0]<0.75):
                    continue
                if(in_data["startZ"][i][0]>45 or in_data["startZ"][i][0]<25):
                    continue
                direction_vector = np.array([in_data["showerStartX"][i][0],in_data["startY"][i][0],in_data["startZ"][i][0]])
                direction_vector = vector / np.linalg.norm(direction_vector)
                beam_vector = np.array([0,0,1])
                if(np.clip(np.dot(direction_vector,beam_vector),-1,1)>0.15):
                    continue

                temp_energy = in_data["energy"][i]
                nominal_energy_list.append(in_data["energy"][i])
                temp_charge = 0
   
                data_array0 = []
                data_array1 = []
                data_array2 = []

                # Build feature vectors
                for j in range(len(in_data["hit2w"][i])):
                    if(mode=='two_d'):
                        data_array2.append([in_data['hit2w'][i][j],in_data['hit2t'][i][j],in_data['hit2q'][i][j]])
                    if(mode=='three_d'):
                        data_array2.append([in_data['pointx'][i][j],in_data['pointy'][i][j],in_data['pointz'][i][j],in_data['hit2q'][i][j]])
                    if(mode=='three_d_message')
                        data_array2.append([in_data['hit2w'][i][j],in_data['hit2t'][i][j],in_data['hit2q'][i][j],in_data['pointx'][i][j],
                                            in_data['pointy'][i][j],in_data['pointz'][i][j]])
                    temp_charge = temp_charge + in_data["hit2q"][i][j]

                for j in range(len(in_data["hit1w"][i])):
                    if(mode=='two_d'):
                        data_array1.append([in_data['hit1w'][i][j],in_data['hit1t'][i][j],in_data['hit1q'][i][j]])
                    if(mode=='three_d'):
                        data_array1.append([in_data['pointx'][i][j],in_data['pointy'][i][j],in_data['pointz'][i][j],in_data['hit1q'][i][j]])
                    if(mode=='three_d_message')
                        data_array1.append([in_data['hit1w'][i][j],in_data['hit1t'][i][j],in_data['hit1q'][i][j],in_data['pointx'][i][j],
                                            in_data['pointy'][i][j],in_data['pointz'][i][j]])

                for j in range(len(in_data["hit0w"][i])):
                    if(mode=='two_d'):
                        data_array0.append([in_data['hit0w'][i][j],in_data['hit0t'][i][j],in_data['hit0q'][i][j]])
                    if(mode=='three_d'):
                        data_array0.append([in_data['pointx'][i][j],in_data['pointy'][i][j],in_data['pointz'][i][j],in_data['hit0q'][i][j]])
                    if(mode=='three_d_message')
                        data_array0.append([in_data['hit0w'][i][j],in_data['hit0t'][i][j],in_data['hit0q'][i][j],in_data['pointx'][i][j],
                                            in_data['pointy'][i][j],in_data['pointz'][i][j]])

                # Get log-based target (follows that this target distribution follows a double-sided crystal ball and is energy independent)
                logratioflip = np.log(temp_charge/temp_energy)
                if(np.isfinite(logratioflip)==False):
                    continue
                log_energy_list.append(logratioflip)
                sum_charge_list.append(temp_charge)

                # Save feature vectors, count event, get box energy (nominal reco method in protoDUNE for comparison)
                out_data0.append(data_array0)
                out_data1.append(data_array1)
                out_data2.append(data_array2)
                
                tot_num_events = tot_num_events + 1
                
                temp_box = 0
                for j in range(len(in_data["particle_energy_full"][i])):
                    temp_box = temp_box + in_data["particle_energy_full"][i][j]
                box_energy_list.append(temp_box)

    # Here, split events into training and validation samples randomly, but ensure input energies have ~same num events
    choosen_idx = np.random.choice(tot_num_events,5750,replace=False)
    train_idx = choosen_idx[0:4600]
    valid_idx = choosen_idx[4600:5750]

    v_energy_list = []
    v_nom_energy_list = []
    v_reco_list = []
    v_features0 = []
    v_features1 = []
    v_features2 = []

    t_energy_list = []
    t_nom_energy_list = []
    t_reco_list = []
    t_features0 = []
    t_features1 = []
    t_features2 = []

    a_energy_list = []
    a_nom_energy_list = []
    a_reco_list = []
    a_features0 = []
    a_features1 = []
    a_features2 = []
    a_box = []

    vv0 = hf0.create_group(top_energies[fo][0]+'_valid')
    tt0 = hf0.create_group(top_energies[fo][0]+'_train')
    aa0 = hf0.create_group(top_energies[fo][0]+'_apply')

    vv1 = hf1.create_group(top_energies[fo][0]+'_valid')
    tt1 = hf1.create_group(top_energies[fo][0]+'_train')
    aa1 = hf1.create_group(top_energies[fo][0]+'_apply')

    vv2 = hf2.create_group(top_energies[fo][0]+'_valid')
    tt2 = hf2.create_group(top_energies[fo][0]+'_train')
    aa2 = hf2.create_group(top_energies[fo][0]+'_apply')

    # Fill validation sample
    count_v = 0
    for v in valid_idx:
        v_energy_list.append(log_energy_list[v])
        v_nom_energy_list.append(nominal_energy_list[v])
        v_reco_list.append(sum_charge_list[v])
        vv0.create_dataset('features_'+str(count_v),data=np.asarray(out_data0[v]),dtype=np.float32)
        vv1.create_dataset('features_'+str(count_v),data=np.asarray(out_data1[v]),dtype=np.float32)
        vv2.create_dataset('features_'+str(count_v),data=np.asarray(out_data2[v]),dtype=np.float32)
        count_v += 1
    vv2.create_dataset('energy',data=np.asarray(v_energy_list,dtype=np.float32)) 
    vv2.create_dataset('nom_energy',data=np.asarray(v_nom_energy_list,dtype=np.float32))
    vv2.create_dataset('reco',data=np.asarray(v_reco_list,dtype=np.float32))

    # Fill train sample
    count_t = 0
    for t in train_idx:
        t_energy_list.append(log_energy_list[t])
        t_nom_energy_list.append(nominal_energy_list[t])
        t_reco_list.append(sum_charge_list[t])
        tt0.create_dataset('features_'+str(count_t),data=np.asarray(out_data0[t]),dtype=np.float32)
        tt1.create_dataset('features_'+str(count_t),data=np.asarray(out_data1[t]),dtype=np.float32)
        tt2.create_dataset('features_'+str(count_t),data=np.asarray(out_data2[t]),dtype=np.float32)
        count_t += 1
    tt2.create_dataset('energy',data=np.asarray(t_energy_list,dtype=np.float32))      
    tt2.create_dataset('nom_energy',data=np.asarray(t_nom_energy_list,dtype=np.float32))
    tt2.create_dataset('reco',data=np.asarray(t_reco_list,dtype=np.float32))

    # Fill all events for application (yes, this double counts but easier to keep track without much memory use)
    for a in range(len(log_energy_list)):
        a_energy_list.append(log_energy_list[a])
        a_nom_energy_list.append(nominal_energy_list[a])
        a_reco_list.append(sum_charge_list[a])
        a_box.append(box_energy_list[a])
        aa0.create_dataset('features_'+str(a),data=np.asarray(out_data0[a]),dtype=np.float32)
        aa1.create_dataset('features_'+str(a),data=np.asarray(out_data1[a]),dtype=np.float32)
        aa2.create_dataset('features_'+str(a),data=np.asarray(out_data2[a]),dtype=np.float32)
    aa2.create_dataset('energy',data=np.asarray(a_energy_list,dtype=np.float32))
    aa2.create_dataset('nom_energy',data=np.asarray(a_nom_energy_list,dtype=np.float32))
    aa2.create_dataset('reco',data=np.asarray(a_reco_list,dtype=np.float32))
    aa2.create_dataset('box',data=np.asarray(a_box,dtype=np.float32))

hf0.close()
hf1.close()
hf2.close()
