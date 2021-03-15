import sys
import os
from os.path import join, realpath, dirname

import h5py
import numpy as np
import pandas as pd 
from scipy.interpolate import interp1d

dir_path = dirname(realpath(__file__))
sys.path.append(os.path.abspath(join(dir_path,'../src')))
import ringdown as rd

save_as = sys.argv[1]#'datafile.hdf5'
filename = join(dir_path,'../data/' + save_as)
print(dir_path)
print(filename)

## generating function
def generate_data(n_samples,masses,spins,qs,modes,phases,iota,spin_relation=False):
  data = []
  target = []
  snr = []
  for i in range(n_samples):
    mass = np.random.uniform(*masses)
    q = np.random.uniform(*qs)
    if spin_relation:
      spin = rd.spin_fit(q)
    else:
      spin = np.random.uniform(*spins)
    distance = 1000
    phase = np.random.uniform(*phases)
    hp,_ = rd.RDwaveform(mass,spin,q,modes,times,iota,phase,distance,method='numerical')
    data.append(hp)
    target.append([mass,spin,q,distance])
    ## compute SNR
    snr.append(rd.SNR(mass,spin,q,iota,distance,modes,psd,method='numerical')[0])
  ## convert to np
  data = np.array(data)
  target = np.array(target)
  snr = np.array(snr)
  ## output
  return data, target, snr
  
## write data
def save_data(group,data,target,snr):
  target_labels = ['mass','spin','mratio','dL']
  with h5py.File(filename,'a') as f:
    X = f.create_dataset(group+'/data',dtype=np.float,data=data)
    y = f.create_dataset(group+'/target',dtype=np.float,data=target)
    y.attrs['labels'] = target_labels
    snr = f.create_dataset(group+'/snr',dtype=np.float,data=snr)
 
  
## set time array
sampling_rate = 4096
dt = 1/sampling_rate
bins = 128
times = np.linspace(0,bins*dt,bins)

## set prior ranges
masses = [25,100]
spins = [0,0.9]
qs = [1,8]
modes = [(2,2),(3,3),(2,1)]
phases = [0,2*np.pi]
iota = np.pi/3

## load psd for SNR
psd_name = dir_path + '/../data/ETD_sensitivity.txt'
df = pd.read_csv(psd_name,sep='   ',engine='python',\
  names=['f','s1','s2','stot'])
x, y = df['f'], df['stot']**2
psd = interp1d(x,y)

## generate train data
n_samples = 10**5
data, target, snr = generate_data(n_samples,masses,spins,qs,modes,phases,iota)
save_data('train',data,target,snr)

## generate test data
n_samples = 10**3
data, target, snr = generate_data(n_samples,masses,spins,qs,modes,phases,iota)
save_data('test',data,target,snr)

## generate test2 data
n_samples = 10**3
data, target, snr = generate_data(n_samples,masses,spins,qs,modes,phases,iota,\
  spin_relation=True)
save_data('test2',data,target,snr)