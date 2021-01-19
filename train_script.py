import sys
from os.path import join
from importlib.machinery import SourceFileLoader

## read command line arguments
PROJECT_PATH = sys.argv[1]
DEVICE = sys.argv[2]
DATA_NAME = sys.argv[3]
nx = int(sys.argv[4])
ny = int(sys.argv[5])
modelname = sys.argv[6]

# import modules
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

mymodule_path = join(PROJECT_PATH,'Modules/qnm_cvae.py')
mymodule = SourceFileLoader('mymodule',mymodule_path).load_module()
from mymodule import GWDataset, Regressor_Bayes, CVAE
mymodule.DEVICE = DEVICE

NBINS = 128

netparams = {'din':NBINS,'ksize':[8,8,8],\
             'nkernel':[16,16,16],'dpool':[2,2,2],\
             'dlin':[64,64],'zdim':8,'dout':ny}
          
modelargs = {'batch_size':512,'epochs':600,'lr':1e-4,\
           'early_stop':True,'tol':0.01,\
           'lr_decay':True,'lr_step':80,\
           'annealing':np.array(3*[1e-5,1/3,2/3,1,1,1]),\
           'validation':0.1}

# load data
DATA_FILE = join(PROJECT_PATH,'Data/'+DATA_NAME)
with h5py.File(DATA_FILE,'r') as f:
  X_train = f['train/data'][()][:nx,:NBINS]
  y_train = f['train/target'][()][:nx,:ny]
  SNR0 = f['train/snr'][()][:nx]
trainset = GWDataset(X_train,y_train,SNR0)

# load noises
NOISE_FILE = PROJECT_PATH+'/Data/Noises.hdf5'
with h5py.File(NOISE_FILE,'r') as f:
  noises = f['ET/noise'][()][:nx,:NBINS]

# load net and model
net = CVAE(**netparams).to(DEVICE)
model = Regressor_Bayes(net,**modelargs)
model.ynorms = np.array([1.,0.01,0.1,1.])[:ny]

# init the model
try:
  model0name = sys.argv[7]
  model.load(PROJECT_PATH+'/cvae_models/'+model0name)
  print('model init to '+model0name)
except:
  print('Model is not preinitialized')
  
# set the SNR
try:
  SNRm = int(sys.argv[8])
  SNRp = int(sys.argv[9])
  print('SNR between %d and %d'%(SNRm,SNRp))
  SNR = np.random.uniform(SNRm,SNRp,len(trainset))
except:
  print('No SNR is imposed')
  SNR = None

# fit the model
t0 = time.time()
trainset.reset()
losses = model.fit(trainset,noises,SNR=SNR)
trainset.reset()
t1 = time.time()
train_time = t1 - t0

# save the model
saveargs = {'params':netparams,'train_time':train_time,'losses':losses}
model.save(saveargs,PROJECT_PATH+'/cvae_models/'+modelname)
