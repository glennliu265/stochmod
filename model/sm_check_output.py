#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------
Quick Checks on Stochastic Model Output
---------------------------------------
For new format (10/04/2021) where models are saved separately
and separate forcing/damping values are applied

Created on Mon Oct  4 15:04:16 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tqdm
import time

from scipy import signal


#%% Set the location

stormtrack = 0

if stormtrack == 0:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    
    input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"

    figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20211004/"
elif stormtrack == 1:
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/" 

from amv import proc,viz
import scm

#%%

# Plotting params
mnames = ["h (constant)","h (vary)","h (entraining)","CESM-FULL","CESM-SLAB"]
mcol   = ["r","magenta","orange","k","gray"]

#%%
# Set Experiment Parameters
frcname   = 'flxeof_090pct_SLAB-PIC_eofcorr2'
runid      = "010"
t_end      = 12000
ampq       = 3

# Load SST in (new, separate forcing)
ssts = []
for i in tqdm.tqdm(range(3)):
    # Set experiment name
    expname    = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i_model%i.npz" % (output_path,frcname,int(t_end/12),runid,ampq,i) 
    if i > 0:
        expname = expname.replace("SLAB","FULL")
    print("Loading " + expname)
    
    ld = np.load(expname,allow_pickle=True)
    if i == 0:
        lonr = ld['lon']
        latr = ld['lat']
    sst = ld['sst']
    ssts.append(sst)
    

#%% Load SST in (old, same forcing)
expnameold = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i.npz" % (output_path,frcname,int(t_end/12),"009",ampq) 
ld         = np.load(expnameold)
sstold     = ld['sst']
lonold     = ld['lon']
latold     = ld['lat']

#%% Check Autocorrelation at SPG Point
kmonth = 1
lags   = np.arange(0,37,1)
xtk2   = np.arange(0,37,2)
lonf   = -30
latf   = 50
klon,klat = proc.find_latlon(lonf,latf,lonr,latr)
klono,klato  = proc.find_latlon(lonf,latf,lonold,latold)

#%% Load CESM

st = time.time()
sstfull,sstslab = scm.load_cesm_pt(input_path+"../",loadname='both',grabpoint=[lonf,latf])
print("Loaded in %.2fs" % (time.time()-st))

#%%

sstpts    = []
ssto      = []
for i in range(3):
    sstpt = ssts[i][klon,klat,1200:]
    #sstpt = signal.detrend(sstpt,type='linear')
    sstpts.append(sstpt)
    ssto.append(sstold[i][klono,klato,1200:])
sstpts.append(sstfull)
sstpts.append(sstslab)

acs = scm.calc_autocorr(sstpts,lags,kmonth+1)
aco = scm.calc_autocorr(ssto,lags,kmonth+1)

fig,ax = plt.subplots(1,1)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="Autocorrelation at SPG Point")
for i in range(len(acs)):
    ax.plot(lags,acs[i],label=mnames[i],color=mcol[i])
    
for i in range(len(aco)):
    ax.plot(lags,aco[i],label=mnames[i]+" (old)",color=mcol[i],ls='dashed')
    
ax.legend(ncol=2)
plt.savefig("%sTesting_CESM-Full_Damping_Forcing_autocorr_skip100beg.png"%figpath,dpi=150)


#%% Postprocess new output

expid = "forcing%s_%iyr_run%s_ampq%i_varyDF" % (frcname,t_end/12,runid,ampq)
lags    = np.arange(0,37,1)
preload = [lonr,latr,ssts]
scm.postprocess_stochoutput(expid,output_path,input_path,output_path+"proc/",lags,preload=preload,mask_pacific=True)

