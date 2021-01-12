#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test Synthetic Stochastic Model

Created on Tue Jan 12 03:46:45 2021

@author: gliu
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs

#%% Set Options
#bboxsim  = [-100,20,-20,90] # Simulation Box
query     = [-30,50]
pointmode = 1
mconfig = "SLAB_PIC"
t_end   = 120000
hfix    = 50
dt      = 3600*24*30
multFAC = 1
T0      = 0
lags    = np.arange(0,61,1)
fstd = 1/4
multFAC = 1

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'

#%%

# Load Data (MLD and kprev, damping)
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
lon        = np.squeeze(loaddamp['LON1'])
lat        = np.squeeze(loaddamp['LAT'])
if mconfig == "SLAB_PIC":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
elif mconfig=="FULL_HTR":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")

# Make up forcing for now
#F = np.random.normal(0,1,size=mld.shape)
xtks = np.arange(1,13,1)
F = np.ones(mld.shape)# * np.sin(-1*np.pi*xtks/6)[None,None,:]

# Restrict to point
params = scm.get_data(pointmode,query,lat,lon,damping,mld,kprevall,F)
[o,a],damppt,hclim,kprev,Fpt = params

# Visualize points
fig,ax = viz.summarize_params(lat,lon,params)

# Convert Parameters
lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,damppt,dt,ND=False,hfix=hfix)

# Prepare Forcing
randts = np.random.normal(0,fstd,t_end)
Fh     = randts * np.tile(Fpt,int(t_end/12))

# Run the stochastic model
sst = {}
for i in range(3):
    sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
sst[3]=scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False)

# Calculate Autocorrelation and plot
autocorr = scm.calc_autocorr(sst,lags,hclim.argmax()+1)
fig,ax = plt.subplots(1,1)
for i in range(4):
    ax.plot(autocorr[i])
    





