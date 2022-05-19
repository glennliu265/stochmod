#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Analyze the F' Variable calculated from CESM1

Created on Wed May 18 17:49:05 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import proc,viz
import scm
import yo_box as ybx
#%%

datpath  = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/'
fnames   = "Fprime_PIC_%s_rolln0.nc"
mconfigs = ["FULL","SLAB"]


#%% Load whole Dataset



i      = 1

ds     = xr.open_dataset(datpath+fnames % (mconfigs[i]))
st     = time.time()
fprime = ds.Fprime.values
times  = ds.time.values
lat    = ds.lat.values
lon    = ds.lon.values
print("Loaded in %.2fs"%(time.time()-st))

#%%
# Just Load  A Point ())

lonf = 330
latf = 50

fprimes = []
for i in range(2):
    
    ds     = xr.open_dataset(datpath+fnames % (mconfigs[i]))
    ds     = ds.sel(lon=lonf,lat=latf,method='nearest')
    st     = time.time()
    fprime = ds.Fprime.values
    times  = ds.time.values
    lat    = ds.lat.values
    lon    = ds.lon.values
    fprimes.append(fprime)
    print("Loaded in %.2fs"%(time.time()-st))
    
#%% Anomalize (or check anomalization)


i = 1

scycle,ts_monyr = proc.calc_clim(fprimes[i],0,returnts=1)
    
#%% Now Do Spectra Analysis

from tqdm import tqdm

def make_wn(ntime,stdev,nsamples):
    wn_ts_all = []#np.zeros((nsamples,ntime)) # [Sample][Time]
    for m in tqdm(range(nsamples)):
        
        # Make and append timeseries
        ts = np.random.normal(0,stdev,ntime)
        wn_ts_all.append(ts)
    return wn_ts_all


def mc_wn_spectra(ts,nsamples,nsmooth,pct,return_pcts="ALL",tails="both"):
    
    # Make the white noise timeseries
    stdev     = np.std(ts)
    ntime     = len(ts)
    wn_ts_all = make_wn(ntime,stdev,nsamples)
    
    # Compute the spectra
    specs,freqs,_,_,_=scm.quick_spectrum(wn_ts_all,nsmooth,pct)
    
    # Calculate the spectra
    specs = np.array(specs) # [sample, freq]
    freq  = freqs[0]
    
    # Sort by order and select top N %
    sortspec = np.sort(specs,axis=0)
    if isinstance(return_pcts,str): # Return all values
        return sortspec,freq
    else:
        nthres = len(return_pcts)
        cfspec = np.zeros([2,nthres,len(freq)]) # [lower(0)/upper(1) x thres x freq]
        for i in range(2):
            if i == 0: #Lower
                pcnts = np.array(return_pcts)
            elif i == 1: #Upper
                pcnts = 1 - np.array(return_pcts)
            # Select rank
            idsel         = np.floor(pcnts*nsamples).astype(int)
            selection     = sortspec[idsel,:]
            cfspec[i,:,:] = selection
        if tails == 'upper':
            cfspec = cfspec[1,...]
        if tails == 'lower':
            cfspec = cfspec[0,...]
        return cfspec,freq
        
        
            
            
            
        
            
        
    
    
    
    
    
    
    
    
    
    

# Calculate Spectra
nsmooth = 100
pct     = 0.10
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(fprimes,nsmooth,pct)
dtplot  = 3600*24*365


# Compute White Noise confidence
nsamples = 10000
pcts     = [0.025,0.05,0.5]
cfspecs = []
for i in tqdm(range(2)):
    cfspec,freq = mc_wn_spectra(fprimes[i],nsamples,nsmooth,pct,return_pcts=pcts,tails="both")
    cfspecs.append(cfspec)

#%%

# Settings
xper    = np.array([100,20,10,5,2])
xtk     = 1/xper
xlm     = (xtk[0],xtk[-1])

pctid   = 0


# Plot Results
fig,ax = plt.subplots(1,1)
for mc in range(2):
    ax.plot(freqs[mc]*dtplot,specs[mc]/dtplot,label=mconfigs[mc])
    
    # Plot the confidence
    ax.fill_between(freqs[mc]*dtplot,cfspecs[mc][0,pctid,:]/dtplot,
                    cfspecs[mc][1,pctid,:]/dtplot,alpha=0.25)
    
ax.legend()

ax.set_xticks(xtk)
ax.set_xlim(xlm)

ax2 = ax.twiny()
ax2.set_xticks(xtk)
ax2.set_xlim(xlm)
ax.set_xticklabels(xper)
    


#viz.plot_freqlin()
