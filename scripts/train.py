import sys
import os
from os.path import join, realpath, dirname
#from importlib.machinery import SourceFileLoader

import h5py
import numpy as np
import time

## import modules
dir_path = dirname(realpath(__file__))
sys.path.append(os.path.abspath(join(dir_path,'../src')))
from cvae import GWDataset, Bayesian_Regressor, CVAE

## define file names
data_file = join(dir_path,'../data/'+sys.argv[1])
noise_file = join(dir_path,'../data/noises.hdf5')

## define global variables
device = 'cuda'
dim_train = int(1e5)
dim_target = 3
nbins = 128

## output file name
save_as = sys.argv[2]
           
## load data
with h5py.File(data_file,'r') as f:
  X_train = f['train/data'][()][:dim_train,:nbins]
  y_train = f['train/target'][()][:dim_train,:dim_target]
  snr0 = f['train/snr'][()][:dim_train]
trainset = GWDataset(X_train,y_train,snr0)

## load noises
with h5py.File(noise_file,'r') as f:
  noises = f['ET/noise'][()][:dim_train,:nbins]

## load the network
network_type == 'cnn'
print('network type: %s'%network_type)

if network_type == 'mlp':
  netparams = {'din':nbins,'dlin':[1024,1024,1024],\
    'zdim':10,'dout':dim_target}
  net = CVAE_MLP(**netparams).to(device)
    
elif network_type == 'cnn':
  netparams = {'din':nbins,\
    'ksize':[8,8,8],'nkernel':[16,16,16],'dpool':[2,2,2],
    'dlin':[64,64],'zdim':8,'dout':dim_target}
  net = CVAE(**netparams).to(device)

## print trainable params
total_params = sum(p.numel() for p in net.parameters()\
                   if p.requires_grad)
print('trainable params: %d'%total_params)

## load the model
modelargs = {'batch_size':512,'epochs':500,'lr':1e-4,\
  'early_stop':True,'tol':1e-3,'lr_decay':True,'lr_step':80,\
  'annealing':np.array(3*[1e-5,1/3,2/3,1,1,1]),\
  'validation':0.1,'device':device}
model = Bayesian_Regressor(net,**modelargs)
## set target rescalings
model.ynorms = np.array([1.,0.01,0.1,1.])[:dim_target]

## init the model
pre_init = False
if pre_init:
  pre_init_name = 'name.pkl'
  pre_init_model = join(dir_path,'../models/'+pre_init_name)
  model.load(pre_init_model)
  print('model initialized to: ',pre_init_name)
else:
  print('Model is not preinitialized')
  
## set the snr
set_snr = True
if set_snr:
  snr_min = 40
  snr_max = 80
  print('SNR between %d and %d'%(snr_min,snr_max))
  snr = np.random.uniform(snr_min,snr_max,len(trainset))
else:
  print('No snr is imposed')
  snr = None

# fit the model
t0 = time.time()
losses = model.fit(trainset,noises,snr)
t1 = time.time()
train_time = t1 - t0

print('Training time: %.2f mins'%(train_time/60))

# save the model
saveargs = {'params':netparams,'train_time':train_time,'losses':losses}
model.save(saveargs,join(dir_path,'../models/'+save_as))