#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



Created on Sun Jan 10 14:50:39 2021

Script to run stochastic model, and copare with Young-Oh's results.

@author: gliu
"""


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time

# Set Options
bboxsim  = [-100,20,-20,90] # Simulation Box
simpoint = [-30,50]
mconfig  = "FULL_HTR"
t_end    = 120000
hfix     = 50
dt       = 3600*24*30
multFAC  = 1
T0       = 0
lags     = np.arange(0,61,1)
usetau   = False
useeta   = False

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'   

 # # Load data from YO's model run
#loc_fname    = "Lon%03d_Lat%03d"  % ('',latf)
loadmod = loadmat(datpath+"yodata/"+"stochrun.mat")

eta = loadmod['eta'].squeeze()
AVG0 = loadmod['AVG0']
AVG1 = loadmod['AVG1']
AVG2 = loadmod['AVG2']
tauall = loadmod['TAUall']

# Load Data (MLD and kprev, damping)
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
LON        = np.squeeze(loaddamp['LON1'])
LAT        = np.squeeze(loaddamp['LAT'])

if mconfig == "SLAB_PIC":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
elif mconfig=="FULL_HTR":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")

# Restrict to Region
dampingr,lonr,latr = proc.sel_region(damping,LON,LAT,bboxsim)
hclim,_,_ = proc.sel_region(mld,LON,LAT,bboxsim)
kprev,_,_ = proc.sel_region(kprevall,LON,LAT,bboxsim)

# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]

# Calculate values
o,a = proc.find_latlon(-30,50,lonr,latr)
if usetau:
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim[o,a,:],tauall.mean(1),dt,ND=0,hfix=hfix)
    yolbd = 1/tauall.mean(1)
    yofac = (1 -np.exp(-yolbd))/yolbd
else:
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,hfix=hfix)
    


# Make Forcing
if useeta:
    randts = eta.squeeze()
    t_end = len(randts)
else:
    randts = np.random.normal(0,1,t_end)/4

# Run models
o,a = proc.find_latlon(-30,50,lonr,latr)
sst = {}

for i in range(3):
    if usetau:

        #sst[i],_,_ = scm.noentrain(t_end,lbd[i],T0,randts,FAC[i],multFAC=multFAC,debug=True)
        sst[i],_,_ = scm.noentrain(t_end,yolbd,T0,randts,yofac,multFAC=multFAC,debug=True)
    else:
        sst[i],_,_ = scm.noentrain(t_end,lbd[i][o,a,:],T0,randts,FAC[i][o,a,:],multFAC=multFAC,debug=True)


if usetau:
   # sst[3],_,_,_,_=scm.entrain(t_end,lbd[3],T0,randts,beta,hclim[o,a,:],
                                            # kprev[o,a,:],FAC[3],multFAC=multFAC,debug=True)
   sst[3],_,_,_,_=scm.entrain(t_end,yolbd,T0,randts,beta,hclim[o,a,:],
       kprev[o,a,:],yofac,multFAC=multFAC,debug=True)
else:
    sst[3],_,_,_,_=scm.entrain(t_end,lbd[3][o,a,:],T0,randts,beta[o,a,:],hclim[o,a,:],
                                         kprev[o,a,:],FAC[3][o,a,:],multFAC=multFAC,
                                         debug=True)


# Calculate autocorrelation
kmonth = hclim[o,a,:].argmax() # kmonth is the INDEX of the mongth
autocorr = {}
for model in range(4):
    
    # Get the data
    tsmodel = sst[model]
    tsmodel = proc.year2mon(tsmodel) # mon x year
    
    # Deseason (No Seasonal Cycle to Remove)
    tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
    
    # Plot
    autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)

#%% Make Test Plot

fig,ax = plt.subplots(1,1)

#ax.plot(lags,AVG0,color='b',label='CESM1 (YO)')
ax.plot(lags,AVG1,color='c',label='No-Entrain (YO)')
ax.plot(lags,AVG2,color='g',label='Entrain (YO)')
ax.plot(lags,autocorr[2],'-r',label='No-Entrain (GL)')
ax.plot(lags,np.roll(autocorr[3],0),':r',label='Entrain (GL)')
ax.legend()

titlestr = 'SST Autocorrelation; Month'+str(kmonth+1) + '\n Lon: ' + "-30" + ' Lat: '+ "50"
ax.set(xlabel='Lag (months)',
       ylabel='Correlation',
       ylim=(-0.5,1.1),
       title=titlestr )
#outname = outpath+'SSTAC_usetau'+str(usetau)+'_mldlinterp_'+ loc_fname +'.png'

ax.set_xlim(0,37)



fig,ax=plt.subplots(1,1)
ax.plot(lbdpic,label="PIC lbd")
ax.plot(lbdhtr,label="HTR lbd")
ax.legend()