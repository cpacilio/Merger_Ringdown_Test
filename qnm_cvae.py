import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.base import BaseEstimator,TransformerMixin

#from scipy.stats import kstest

DEVICE = 'cuda'


## class to standardize the dataset
## note: the inheritances are essential to call fit_transform method
class Standardizer(BaseEstimator,TransformerMixin):
  def __init__(self):
    return None

  def fit(self,X,y=None):
    self.max = torch.max(X,dim=1)[0].view(-1,1)
    self.mu = torch.mean(X/self.max,1).view(-1,1)
    self.loc = torch.std(X/self.max,1).view(-1,1)
    return self
  
  def transform(self,X,y=None):
    out = (X/self.max-self.mu)/self.loc
    return out
    

## class to manage the datasets
class GWDataset():
  def __init__(self,X,y,z=None):
    self.x0 = X
    self.x = deepcopy(self.x0)
    self.y = y
    if z is not None:
      self.snr = z
    
  def reset(self):
    self.x = deepcopy(self.x0)
    return self
    
  def __len__(self):
    return len(self.x0)
    
  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]
  
  def add_noise(self,noises,SNR=None):
    if SNR is not None:
      A = SNR/self.snr
      self.x *= A.reshape(-1,1)
    self.x += noises
    return self
    

## Convolutional neural network for the cvae
AFFINE = True

class Encoder(nn.Module):
  def __init__(self,din,nkernel,ksize,dpool,dlin,dout,cat=None):
    super(Encoder,self).__init__()
    self.cat = cat
    self.dout = dout
    ## compute dflat
    dflat = din
    for i in range(len(ksize)):
      dflat = int((dflat-ksize[i]+1)/dpool[i])
    self.flat_dim = dflat*nkernel[-1]
    ## make layers
    self.conv = self.make_conv_layers(nkernel,ksize,dpool)
    self.lin = self.make_lin_layers(dlin)
    self.z1 = nn.Linear(dlin[-1],dout)
    self.z2 = nn.Linear(dlin[-1],dout)
    
  def forward(self,X,c=None):
    x = deepcopy(X)
    x = x.view(x.shape[0],1,x.shape[1])
    x = self.conv(x)
    x = x.view(x.shape[0],self.flat_dim)
    if self.cat:
      x = torch.cat((x,c),dim=1)
    x = self.lin(x)
    mu, logvar = self.z1(x), self.z2(x)
    return mu, logvar
  
  def make_conv_layers(self,nkernel,ksize,dpool):
    conv = []
    cin = 1
    for i in range(len(nkernel)):
      conv.append(nn.Conv1d(cin,nkernel[i],ksize[i]))
      conv.append(nn.MaxPool1d(dpool[i]))
      conv.append(nn.GroupNorm(1,nkernel[i],affine=AFFINE))
      conv.append(nn.ReLU())
      cin = nkernel[i]
    return nn.Sequential(*conv)
    
  def make_lin_layers(self,dlin):
    lin = []
    din = self.flat_dim
    if self.cat:
      din += self.cat
    for k in dlin:
      lin.append(nn.Linear(din,k))
      lin.append(nn.GroupNorm(1,k,affine=AFFINE))
      lin.append(nn.ReLU())
      din = k
    return nn.Sequential(*lin)
    
    
class CVAE(nn.Module):
  def __init__(self,din,nkernel,ksize,dpool,dlin,zdim,dout):
    super(CVAE,self).__init__()
    self.input_dim = din
    self.output_dim = dout
    self.latent_dim = zdim
    self.encoder = Encoder(din,nkernel,ksize,dpool,dlin,zdim)
    self.guide = Encoder(din,nkernel,ksize,dpool,dlin,zdim,cat=dout)
    self.decoder = Encoder(din,nkernel,ksize,dpool,dlin,dout,cat=zdim)
    
  def forward(self,data,target):
    mu0, logvar0 = self.guide(data,target)
    mu1, logvar1 = self.encoder(data)
    z = self.reparametrize(mu0,logvar0)
    mu2, logvar2 = self.decoder(data,z)
    return mu0,logvar0,mu1,logvar1,mu2,logvar2
 
  def reparametrize(self,mu,logvar):
    sigma = torch.exp(0.5*logvar)
    eps = torch.randn_like(sigma)
    z = mu + eps*sigma
    return z
  
  def sample(self,data):
    mu1, logvar1 = self.encoder(data)
    sigma1 = torch.exp(0.5*logvar1)
    z = torch.normal(mu1,sigma1).to(DEVICE)
    mu2, logvar2 = self.decoder(data,z)
    sigma2 = torch.exp(0.5*logvar2)
    out = torch.normal(mu2,logvar2).to(DEVICE)
    return out
    
    
## bayesian regressor class
def ELBO(target,mu0,logvar0,mu1,logvar1,mu2,logvar2):
  L = 0.5*torch.sum(1.8378770664093453+logvar2+(mu2-target)**2/torch.exp(logvar2))
  KL = -0.5*torch.sum(1+(logvar0-logvar1)-(mu0-mu1)**2/torch.exp(logvar1)-torch.exp(logvar0-logvar1))
  return L/len(target), KL/len(target)
  
def annealing(step,betas):
  step += 1
  if step >= len(betas):
    return 1, step
  else:
    return betas[step], step

class Regressor_Bayes():
  default_dict = {'epochs':100,'batch_size':512,'criterion':ELBO,\
                  'optim':optim.Adam,'lr':1e-4,'annealing':None,\
                  'early_stop':True,'tol':1e-2,'n_iter_no_change':5,\
                  'validation':0.2,'lr_decay':False,'lr_step':100}

  def __init__(self,net,**kwargs):
    self.__dict__ = {**self.default_dict,**kwargs}
    self.net = net
    self.sc = Standardizer()
    self.ynorms = 1.
    
  def prepare_data_loaders(self,dataset):
    data, target = dataset.x, dataset.y/self.ynorms
    data = torch.from_numpy(data).float().to(DEVICE)
    target = torch.from_numpy(target).float().to(DEVICE)
    trainset = TensorDataset(data,target)
    valid_len = int(self.validation*len(trainset))
    train_len = len(trainset) - valid_len
    trainset, validset = random_split(trainset,[train_len,valid_len])
    train_loader = DataLoader(trainset,batch_size=self.batch_size,shuffle=True)
    valid_loader = DataLoader(validset,batch_size=self.batch_size,shuffle=True)
    return train_loader, valid_loader
    
  def fit(self,dataset,noises=None,SNR=None):
    ## set optimizer
    optimizer = self.optim(self.net.parameters(),lr=self.lr)
    if self.lr_decay:
      self.scheduler = StepLR(optimizer,step_size=self.lr_step,gamma=0.5)
    
    if noises is not None:
      noise = torch.from_numpy(noises).float().to(DEVICE)
    if SNR is not None:
      dataset.x /= dataset.snr.reshape(-1,1)
      SNR = torch.from_numpy(SNR).float().to(DEVICE)
      
    ## prepare data
    train_loader, valid_loader = self.prepare_data_loaders(dataset)
    
    ## init useful variables
    losses = {}
    losses['train_tot'] = []
    losses['valid_tot'] = []
    losses['train_L'] = []
    losses['valid_L'] = []
    losses['train_KL'] = []
    losses['valid_KL'] = []
    
    epochs_no_improve = 0
    if self.annealing is not None:
      beta = 0
      beta_step = 0
    else:
      beta = 1
    
    ## training loop
    for epoch in range(self.epochs):
      ## training step
      self.net.train()
      train_loss = 0
      train_loss_L = 0
      train_loss_KL = 0
      for batch_idx, (data,target) in enumerate(train_loader):
        ## add noise
        if noises is not None:
          idx = torch.randperm(noise.size()[0])[:len(data)]
          nn = noise[idx]
          A = 1.
          if SNR is not None:
            idx = torch.randperm(SNR.size()[0])[:len(data)]
            A = SNR[idx]
            A = A.view(-1,1)
          input = data*A + nn
        else:
          input = data
        ## standardize
        input = self.sc.fit_transform(input)
        ## backprop
        optimizer.zero_grad()
        output = self.net(input,target)
        L, KL = self.criterion(target,*output)
        loss = L + KL*beta
        loss.backward()
        optimizer.step()
        ## store losses
        train_loss += loss.item()*len(data)/len(train_loader.dataset)
        train_loss_L += L.item()*len(data)/len(train_loader.dataset)
        train_loss_KL += KL.item()*len(data)/len(train_loader.dataset)
      losses['train_tot'].append(train_loss)
      losses['train_L'].append(train_loss_L)
      losses['train_KL'].append(train_loss_KL)
      
      ## validation step
      self.net.eval()
      valid_loss = 0
      valid_loss_L = 0
      valid_loss_KL = 0
      with torch.no_grad():
        for batch_idx, (data,target) in enumerate(valid_loader):
          ## add noise
          if noises is not None:
            idx = torch.randperm(noise.size()[0])[:len(data)]
            nn = noise[idx]
            A = 1.
            if SNR is not None:
              idx = torch.randperm(SNR.size()[0])[:len(data)]
              A = SNR[idx]
              A = A.view(-1,1)
            input = data*A + nn         
          else:
            input = data
          ## standardize
          input = self.sc.fit_transform(input)
          ## compute loss
          output = self.net(input,target)
          L, KL = self.criterion(target,*output)
          loss = L + KL*beta
          ## store losses
          valid_loss += loss.item()*len(data)/len(valid_loader.dataset)
          valid_loss_L += L.item()*len(data)/len(valid_loader.dataset)
          valid_loss_KL += KL.item()*len(data)/len(valid_loader.dataset)
      losses['valid_tot'].append(valid_loss)
      losses['valid_L'].append(valid_loss_L)
      losses['valid_KL'].append(valid_loss_KL)
      
      ## step necessary for early stop
      if epoch == 0:
        valid_loss_min = valid_loss
        checkpoint = self.net.state_dict()
      
      ## LR scheduler
      if self.lr_decay:
        self.scheduler.step()
        
      ## annealing
      if self.annealing is not None:
        beta, beta_step = annealing(beta_step,self.annealing)

      ## early stop criterion
      if self.early_stop and epoch!=0 and epoch%20==0:
        if valid_loss + self.tol < valid_loss_min:
          epochs_no_improve = 0
          valid_loss_min = valid_loss
          checkpoint = self.net.state_dict()
        else:
          epochs_no_improve += 1
        
        if epochs_no_improve >= self.n_iter_no_change:
          self.net.load_state_dict(checkpoint)
          break
    
    dataset = dataset.reset()
    return losses
    
  def predict(self,X):
    x = deepcopy(X)
    x = torch.from_numpy(x).float().to(DEVICE)
    x = self.sc.fit_transform(x)
    with torch.no_grad():
      self.net.eval()
      out = self.net.sample(x)
      if DEVICE == 'cuda':
        out = out.cpu().numpy()
      else:
        out = out.numpy()
    return out*self.ynorms
    
  def sample(self,X,N=1000):
    dim = (len(X),self.net.output_dim,N)
    samples = np.zeros(dim)
    for i in range(N):
      samples[:,:,i] = self.predict(X)
    return samples
    
  def save(self,kwargs,save_path):
    params = {'state_dict':self.net.state_dict(),\
              'ynorms':self.ynorms}
    params = {**params,**kwargs}
    torch.save(params,save_path)
    return None
    
  def score(self,samples,ytrue):
    ## returns the KS statistics
    CL = np.linspace(0,1,100) 
    _, KS = PP_plot_median(samples,ytrue,CL)
    return KS
   
  def load(self,load_path,map_location=DEVICE):
    checkpoint = torch.load(load_path,map_location=map_location)
    self.net.load_state_dict(checkpoint['state_dict'])
    self.ynorms = checkpoint['ynorms']
    return None
    
    
## evaluation functions    

def PP_plot_median(samples,target,CL):
  sorted_samples = np.sort(samples,axis=-1)
  scores = np.zeros((target.shape[1],len(CL)))
  L = samples.shape[-1]
  for i in range(len(CL)):
    p = CL[i]
    a = min(int(0.5*L*(1-p)), int(0.5*L*(1+p)-1))
    b = max(int(0.5*L*(1-p)), int(0.5*L*(1+p)-1))
    condition1 = target >= sorted_samples[:,:,a]
    condition2 = target <= sorted_samples[:,:,b]
    scores[:,i] = np.mean(condition1*condition2,axis=0)
  KS = np.max(np.abs(scores-\
   np.repeat(CL.reshape(1,-1),3,axis=0)),axis=1)
  return scores, KS
  
def PP_plot_tail(samples,target,CL):
  sorted_samples = np.sort(samples,axis=-1)
  scores = np.zeros((target.shape[1],len(CL)))
  L = samples.shape[-1]
  for i in range(len(CL)):
    p = CL[i]
    a = max(0,int(L*p)-1)
    condition = target <= sorted_samples[:,:,a]
    scores[:,i] = np.mean(condition,axis=0)
  KS = np.max(np.abs(scores-\
   np.repeat(CL.reshape(1,-1),3,axis=0)),axis=1)
  return scores, KS
