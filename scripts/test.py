import os
import sys
from os.path import join, realpath, dirname

import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import time

## import modules
dir_path = dirname(realpath(__file__))
sys.path.append(os.path.abspath(join(dir_path,'../src')))
from cvae import GWDataset, Bayesian_Regressor, CVAE
from cvae import PP_plot

## define file names
data_file = join(dir_path,'../data/'+sys.argv[1])
noise_file = join(dir_path,'../data/noises.hdf5')
model_name = sys.argv[2]
model_file = join(dir_path,'../models/' + model_name)

## define global variables
device = 'cuda'
dim_target = 3
nbins = 128
  
## load network
checkpoint = torch.load(model_file)
net = CVAE(**checkpoint['params']).to(device)
model = Bayesian_Regressor(net)
model.load(model_file)

## print some info
total_params = sum(p.numel() for p in net.parameters()\
                   if p.requires_grad)
print('trainable params: %d'%total_params)

print('Training time: %.0f mins'%(checkpoint['train_time']/60))

## plot losses
losses = checkpoint['losses']
tL = losses['train_L']
tKL = losses['train_KL']
vL = losses['valid_L']
vKL = losses['valid_KL']
epochs = list(range(len(tL)))
plt.plot(epochs,tL,label='$\mathcal{L}_{recon}$ (training)')
plt.plot(epochs,vL,label='$\mathcal{L}_{recon}$ (validation)')
plt.plot(epochs,tKL,label='$\mathcal{L}_{KL}$ (training)')
plt.plot(epochs,vKL,label='$\mathcal{L}_{KL}$ (validation)')
plt.xlabel('epochs',fontsize=13)
plt.legend(fontsize=12)
plt.yscale('log')
save_as = join(dir_path,'../results/losses.png')
plt.savefig(save_as,bbox_inches='tight')

with h5py.File(data_file,'r') as f:
  X_test = f['test/data'][()][:,:nbins]
  y_test = f['test/target'][()][:,:dim_target]
  snr0 = f['test/snr'][()]
testset = GWDataset(X_test,y_test,snr0)

## load noises
with h5py.File(noise_file,'r') as f:
  noises = f['ET/noise'][()][-len(testset):,:nbins]
  
## add noise
SNR = np.random.uniform(40,80,len(testset))
X = testset.add_noise(noises,snr=SNR)

## sample
t0 = time.time()
n_samples = int(1e4)
samples = model.sample(X,n_samples)
t1 = time.time()
sample_time = t1 - t0

print('Sampling time: %.3f secs'%sample_time)
print('model score: ',model.score(samples,testset.y))

## plot PP plot
names = ['final mass $M$','final spin $\chi_f$','mass ratio $q$']
CL = np.linspace(0,1,100)
scores, KS = PP_plot(samples,testset.y,CL=CL);
plt.figure(figsize=(5,5))
for i in range(len(scores)):
  plt.plot(CL,scores[i],label=names[i])
plt.plot(CL,CL,'--',color='black')
plt.legend(fontsize=12)
plt.xlabel('$p$',fontsize=13)
plt.ylabel('$CDF\,(p)$',fontsize=13)
plt.grid(linestyle=':',alpha=0.9)
save_as = join(dir_path,'../results/pp_plot.png')
plt.savefig(save_as,bbox_inches='tight')