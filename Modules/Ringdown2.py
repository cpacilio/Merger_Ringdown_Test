import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from importlib.machinery import SourceFileLoader
from scipy.interpolate import interp1d

## import modules
from os.path import join, realpath, dirname
full_path = realpath(__file__)
dir_path = dirname(full_path)
harmonics = SourceFileLoader('harmonics',join(dir_path,'harmonics.py')).load_module()
from harmonics import sYlm

## constants
G = 6.67408*(10.**(-11))
cc = 299792458.
Msun = 1.98855*(10.**30)
Rsun = Msun*G/cc**2
Megapc = 3.08*(10.**22)


## spherical harmonics
def spherical_harmonics(angle,mode):
  l, m = mode
  Yp = np.real(sYlm(-2,l,m,angle,0)+(-1)**l*sYlm(-2,l,-m,angle,0))
  Yc = np.real(sYlm(-2,l,m,angle,0)-(-1)**l*sYlm(-2,l,-m,angle,0))
  return Yp, Yc

## load qnm numerical data
qnm_interp_dict = {}
qnm_data_dict = {(2,2):'n1l2m2.dat',(2,1):'n1l2m1.dat',(3,3):'n1l3m3.dat',(4,4):'n1l4m4.dat'}

def interp_qnm(mode):
  df = pd.read_csv(join(dir_path,qnm_data_dict[mode]),sep=' ',\
    engine='python',names=['spin','omegaR','omegaIm','arg1','arg2'])
  x, y1, y2 = np.array(df['spin']), np.array(df['omegaR']), np.array(-df['omegaIm'])
  outR, outI = interp1d(x,y1,'cubic'), interp1d(x,y2,'cubic')
  qnm_interp_dict[mode] = (outR,outI)
  return None  

for k in qnm_data_dict.keys():
  interp_qnm(k)


def qnm_Kerr(mass,spin,mode,method='numerical'):
  '''
  mass: in solar masses
  spin: dimensionless
  '''
  conversion_factor = Rsun/cc
  
  if method == 'numerical':    
    ## interpolate modes from data given in
    ## https://pages.jh.edu/~eberti2/ringdown/
    omegaR = qnm_interp_dict[mode][0](spin)/mass/conversion_factor
    omegaI = qnm_interp_dict[mode][1](spin)/mass/conversion_factor
    tau = 1/omegaI
    
  elif method == 'fit':    
    ## use qnm fits from
    ## https://arxiv.org/abs/gr-qc/0512160
    coeff = {}
    coeff[(2,1)] = [0.6,-0.2339,0.4175,-0.3,2.3561,-0.2277]
    coeff[(2,2)] = [1.5251,-1.1568,0.1292,0.7,1.4187,-0.4990]
    coeff[(3,3)] = [1.8956,-1.3043,0.1818,0.9,2.3430,-0.4810]
    coeff[(4,4)] = [2.3,-1.5056,0.2244,1.1929,3.1191,-0.4825]
    
    f = coeff[mode][:3]
    q = coeff[mode][3:]

    omegaR = (f[0]+f[1]*(1-spin)**f[2])/mass
    omegaR /= conversion_factor
    Q = (q[0]+q[1]*(1-spin)**q[2])
    tau = 2*Q/omegaR
  
  return omegaR/2/np.pi, tau  


def qnm_amplitudes(q,mode,method='numerical'):
  '''
  q: mass ratio
  '''
  A = {}
  
  if method == 'fit':
    ## use fits from https://arxiv.org/abs/1111.5819 
    eta = q/(1+q)**2
    A[(2,2)] = 0.864*eta
    A[(2,1)] = 0.52*(1-4*eta)**0.71*A[(2,2)]
    A[(3,3)] = 0.44*(1-4*eta)**0.45*A[(2,2)]
    A[(4,4)] = (5.4*(eta-0.22)**2+0.04)*A[(2,2)]
  
  elif method == 'numerical':
    ## use fits from
    ## in their updated form
    eta = q/(1+q)**2
    A[(2,2)] = 0.864*eta
    A[(2,1)] = A[(2,2)]*(0.472881 - 1.1035/q + 1.03775/q**2 - 0.407131/q**3)
    A[(3,3)] = A[(2,2)]*(0.433253 - 0.555401/q + 0.0845934/q**2 + 0.0375546/q**3)
  
  return A[mode]

def qnm_phases(q,mode,phase,method='numerical'):
  '''
  q: mass ratio
  '''
  if method == 'numerical':
    ## use fits from 
    ## https://arxiv.org/abs/2005.03260
    ## in their updated form
    P = {}
    P[(2,2)] = phase
    P[(2,1)] = P[(2,2)] - (1.80298 - 9.70704/(9.77376 + q**2))
    P[(3,3)] = P[(2,2)] - (2.63521 + 8.09316/(8.32479 + q**2))
    out = P[mode]

  elif method == 'fit':
    out = mode[1]*phase
    
  return out
  

def spin_fit(q):
  ## from https://arxiv.org/abs/1106.1021
  '''
  q: mass ratio
  spin: dimensionless
  '''
  eta = q/(1+q)**2
  spin = 2*np.sqrt(3)*eta - 3.871*eta**2 + 4.028*eta**3
  return spin


def RDwaveform(mass,spin,q,modes,times,iota,phase,distance,method='numerical'):
  ## following the conventions in 
  ## https://arxiv.org/abs/1111.5819 
  ## and in https://arxiv.org/abs/2005.03260
  '''
  distance: in Mpc
  ''' 
  conversion_factor = mass*Rsun/distance/Megapc

  hp = np.zeros_like(times)
  hc = np.zeros_like(times)
  for mode in modes:
    A = qnm_amplitudes(q,mode,method=method)
    freq, tau = qnm_Kerr(mass,spin,mode,method=method)
    Yp, Yc = spherical_harmonics(iota,mode)
    phi = qnm_phases(q,mode,phase,method=method)
    hp += A*np.exp(-times/tau)*Yp*np.cos(2*np.pi*freq*times - phi)
    hc += A*np.exp(-times/tau)*Yc*np.sin(2*np.pi*freq*times - phi)
  hp *= conversion_factor
  hc *= -conversion_factor
  return hp, hc


def SNR(mass,spin,q,iota,distance,modes,noise_curve,method='numerical'):
  ## following Eq.(15) in 
  ## https://arxiv.org/abs/1809.03500
  conversion_factor = mass*Rsun/distance/Megapc
  rhop = 0.
  rhoc = 0.
  for mode in modes:
    A = qnm_amplitudes(q,mode,method=method)
    freq, tau = qnm_Kerr(mass,spin,mode,method=method)
    Yp, Yc = spherical_harmonics(iota,mode)
    S = noise_curve(freq).item()
    prefactor = (A*conversion_factor)**2*tau/2/S
    rhop += prefactor*Yp**2
    rhoc += prefactor*Yc**2
  
  return np.sqrt(rhop), np.sqrt(rhoc)
