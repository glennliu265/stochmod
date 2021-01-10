#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

loop_stochmod_region
Created on Sun Aug 23 17:10:35 2020

@author: gliu
"""

import sys
import time
import glob
#%% Determine System
startall = time.time()
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
#    scriptpath  = projpath + '03_Scripts/stochmod/'
    datpath     = projpath + '01_Data/'
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")


elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

#%%

import stochmod_region as sr

# # of runs
pointmode = 1 # Set to 1 to output data for the point speficied below
points=[-30,50] # Lon, Lat for pointmode
naoscale = 10 # Number to scale NAO and other forcings by

#pointlon-30_lat50
locstring = "pointlon%i_lat%i"%(points[0],points[1])

# Integration Options
nyr      = 10000        # Number of years to integrate over
fstd     = 0.3         # Standard deviation of the forcing
bboxsim  = [-100,20,-20,90] # Simulation Box

# Do a stormtrackloop
runid    = "101"
funiforms = [1]
applyfacs  = [0,1,2]
mconfig   = "SLAB_PIC"

fscale = naoscale


#%%
for applyfac in applyfacs:
    
    for funiform in funiforms:
        
        if len(glob.glob(datpath+'model_output/' + "stoch_output_%iyr_run%s_randts.npy"%(nyr,runid)))==0:
            genrand=1
        else:
            
            genrand=0
        
        
        sr.stochmod_region(pointmode,funiform,fscale,runid,genrand,nyr,fstd,bboxsim,stormtrack,mconfig=mconfig,applyfac=applyfac)
        print("Completed run %s funiform %s (Runtime Total: %.2fs)" % (runid,funiform,time.time()-startall))
        
#%% Post Process
import numpy as np
from amv import proc,viz
import matplotlib.pyplot as plt

# Set Lags
lags = np.arange(0,37,1)
funiform = 1
# Load MLD

ssts = {} # Stored by applyfac, model
for i,applyfac in enumerate(applyfacs):
    expid = "%s_%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(locstring,mconfig,nyr,funiform,runid,fscale,applyfac)

    ssts[i] = np.load(datpath+"model_output/"+ "stoch_output_%s.npz"%(expid),allow_pickle=True)['sst'].item()
    
    if i == 0:
        mld = np.load(datpath+"model_output/"+ "stoch_output_%s.npz"%(expid),allow_pickle=True)['hclim']


# Calculate Autocorrelation for each cast
kmonth = mld.argmax()
acall = {}
for i,applyfac in enumerate(applyfacs):
    ac = {}
    for m in range(4):
        
        sst = ssts[i][m]
        sst = np.roll(sst,-1) # Roll since first month is feb (use jan, no-entrain 1d)
        
        tsmodel = proc.year2mon(sst) # mon x year
        
        # Deseason
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Compute autocorrelation and save data for region
        ac[m] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
        
    acall[i] = ac.copy()
        

#%% Plot autocorrelation (Plot by Model)

applyfaclab = ["Forcing Only","Incl. Seasonal MLD","Incl. Seasonal MLD and Integration Factor"]
modelname = ("MLD Fixed","MLD Max", "MLD Seasonal", "MLD Entrain")
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monname=('January','February','March','April','May','June','July','August','September','October','November','December')

mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
mons3tile = np.roll(mons3tile,-kmonth)
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20201123_AMVTele/"

xlm = [0,36]
xtk = np.arange(0,39,3)
xtk2 =  np.arange(xlm[0],xlm[1],3)

locstringtitle = "LON: %i LAT: %i" % (points[0],points[1])

slabac = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM-SLAB_PIC_autocorrelation_pointlon330_lat50.npy")

for m in [2,3]: # only do for variny mld
    
    
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use('seaborn')
    
    ax2 = ax.twiny()
    ax2.set_xlim(xlm)
    ax2.set_xticks(xtk2)
    ax2.set_xticklabels(mons3tile[xtk2], rotation = 45)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    
    ax.plot(lags,slabac,color='k',label="CESM1-SLAB (PIC)")
    
    for f in range(3):
        
        acplot = acall[f][m]
        
        ax.plot(lags,acplot,label=applyfaclab[f])
    ax.legend()
    
    ax.set_xticks(xtk)
    ax.set_xlim(xlm)
    ax.set_title("%s SST Autocorrelation at %s, Model: %s" % (monname[kmonth],locstringtitle,modelname[m]))
    ax.set_xlabel("Lags (Months)")
    ax.set_ylabel("Correlation")
    plt.tight_layout()
    plt.savefig("%sApplyfac_compare_model%s_%s.png"%(outfigpath,modelname[m],expid),dpi=200)
        


#%% Plot model intercomparison
    
for f in range(3):
    
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use('seaborn')
    plt.style.use('seaborn-bright')
    
    ax2 = ax.twiny()
    ax2.set_xlim(xlm)
    ax2.set_xticks(xtk2)
    ax2.set_xticklabels(mons3tile[xtk2], rotation = 45)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    
    ax.plot(lags,slabac,color='k',label="CESM1-SLAB (PIC)")
    
    for m in range(4): # only do for variny mld
        acplot = acall[f][m]
        ax.plot(lags,acplot,label=modelname[m])
    ax.legend()
    
    ax.set_xticks(xtk)
    ax.set_xlim(xlm)
    ax.set_title("%s SST Autocorrelation at %s,\n %s" % (monname[kmonth],locstringtitle,applyfaclab[f]))
    ax.set_xlabel("Lags (Months)")
    ax.set_ylabel("Correlation")
    plt.tight_layout()
    plt.savefig("%sApplyfac%s_compare_model%s.png"%(outfigpath,applyfaclab[f],expid),dpi=200)