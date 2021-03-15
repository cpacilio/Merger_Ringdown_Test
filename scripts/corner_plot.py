import os
import sys
from os.path import join, realpath, dirname

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager
import torch
import corner

## import modules
dir_path = dirname(realpath(__file__))
sys.path.append(os.path.abspath(join(dir_path,'../src')))
from cvae import GWDataset, Bayesian_Regressor, CVAE

## define file names
data_file = join(dir_path,'../data/'+sys.argv[1])
noise_file = join(dir_path,'../data/noises.hdf5')
model_name = sys.argv[2]
model_file = join(dir_path,'../models/' + model_name)

## define global variables
device = 'cuda'
dim_train = int(1e5)
dim_target = 3
nbins = 128

## load network
checkpoint = torch.load(model_file)
net = CVAE(**checkpoint['params']).to(device)
model = Bayesian_Regressor(net)
model.load(model_file)

with h5py.File(data_file,'r') as f:
  X_test = f['test/data'][()][:,:nbins]
  y_test = f['test/target'][()][:,:dim_target]
  snr0 = f['test/snr'][()]
testset = GWDataset(X_test,y_test,snr0)

## load noises
with h5py.File(noise_file,'r') as f:
  noises = f['ET/noise'][()][-len(testset):,:nbins]
  
## add noise
SNR = 60
X = testset.add_noise(noises,snr=SNR)

## sample
idx = np.argsort(testset.y[:,0],axis=0)
N = 500
n_samples = int(1e4)
sample = model.sample(X[idx[N]].reshape(1,-1),n_samples)[0]

## make corner plot
p = 0.95
a = min((1-p)/2,(1+p)/2)
b = 1-a
print('ground truth: ',testset.y[idx[N]][:dim_target])
figure = corner.corner(sample.T,titles=['$M$','$\chi_f$','$q$'],\
                       labels = None,\
                       quantiles=[a,b],\
                       show_titles=True,title_kwargs={'fontsize':13},\
                       truths=testset.y[idx[N]],truth_color='steelblue')
figure.set_size_inches((8,8))

# plot the signal
matplotlib.pyplot.rcParams["font.weight"] = 'normal'
matplotlib.pyplot.rcParams["font.family"] = 'DejaVu Sans'

left, bottom, width, height = [0.129, -0.15, 0.82, 0.2]
ax2 = figure.add_axes([left, bottom, width, height])
hfont = {'fontfamily':'monospace'}
ax2.plot(X[idx[N]],label='SNR=%d'%60,color='steelblue')
ax2.set_facecolor('white')
A = 60/testset.snr[idx[N]]
ax2.plot(A*testset.x[idx[N]],label='noiseless',\
  color='orange',linestyle='--')
plt.legend(fontsize=12,frameon=False)
ax2.set_ylabel('strain',fontsize=13,**hfont)
ax2.set_xlabel('t (s)',fontsize=13)
ax2.set_xticks(np.linspace(0,128,5))
ax2.set_xticklabels(np.linspace(0,128/4096,5))
ax2.grid(linestyle=':',alpha=0.9)

save_as = dir_path + '/../results/corner.png'
figure.savefig(save_as,bbox_inches='tight');