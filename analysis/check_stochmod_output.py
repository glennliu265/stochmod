#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

stochmod_region_outputcheck

Created on Mon Aug 17 13:00:33 2020

@author: gliu
"""



import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.io import loadmat

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc

#%% 

# Set run parameters to test
runid    = "002"
fscale   = 100
funiform = 4   
nyrs     = 1000

# Point to Plot
lonf = -30
latf = 65

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200818/'

# Autocorrelation Options
lags = np.arange(0,61,1)

# String Names Set
expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
modelname = ("Fixed","Max", "Seasonal", "Entrain")
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$")

#%% Script Start
loadstart= time.time()

# Read in Stochmod SST Data
sst = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s_fscale%03d.npy"%(nyrs,funiform,runid,fscale),allow_pickle=True).item()
sst[3] = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s_fscale%03d.npy"%(nyrs,funiform,runid,fscale))
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Load data from CESM
loadmod = loadmat(datpath+"stochrun_Lon330_Lat050.mat")
eta = loadmod['eta']
AVG0 = loadmod['AVG0']
AVG1 = loadmod['AVG1']
AVG2 = loadmod['AVG2']
tauall = loadmod['TAUall']

# Load MLD Data
mld = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD

# Read in damping data for the coordinates
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])

# Make some strings
print("Data Loaded in %.2fs"%(time.time()-loadstart))

#%% Check at the single point (lon 330, lat 50)

klon,klat = proc.find_latlon(-30,50,lon,lat)
klonr,klatr = proc.find_latlon(-30,50,lonr,latr)

# Calculate Autocorrelation
autocorr = {}
for model in range(4):
       
       # Get sst and havg
       tsmodel = sst[model][klonr,klatr,:]
       
       # Find kmonth
       kmonth = mld[klon,klat,:].argmax() + 1
       
       # Take regional average
       tsmodel = proc.year2mon(tsmodel) # mon x year
       
       # Deseason (No Seasonal Cycle to Remove)
       tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
       
       # Plot
       autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth,0)
    

# Make the figure
f1 = plt.figure(figsize=(5,3))
ax = plt.axes()
plt.style.use("seaborn")
plt.style.use("seaborn-bright")
ax.plot(lags,AVG0,color='b',label='CESM1 (YO)')
ax.plot(lags,AVG1,color='c',label='No-Entrain (YO)')
ax.plot(lags,AVG2,color='g',label='Entrain (YO)')
ax.plot(lags,autocorr[2],'-r',label='No-Entrain (GL)')
ax.plot(lags,autocorr[3],':r',label='Entrain (GL)')
ax.legend()
plt.grid(True)

titlestr = "SST Autocorrelation Month @ Lon:%02d lat:%02d \n Forcing: %s Scale:%03dx MLD: %s" % (-30,50,forcingname[funiform],fscale,modelname[2])


ax.set(xlabel='Lag (months)',
       ylabel='Correlation',
       ylim=(-0.5,1.1),
       title=titlestr )
ax.set_xticks(np.arange(0,66,6))
outname = outpath+'StochmodComparisonYO_%s.png' % expid
plt.savefig(outname, bbox_inches="tight",dpi=200)
