#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calulate desired quantities from the CESM output

Uses output processed from remove_ENSO, calc_ENSO...


#data is on stormtrack located in... 
/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_PIC_SLAB/02_ENSOREM/FULL/ENSOREM_TS_lag1_pcs2_monwin3.npz

@author: gliu
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs
from scipy import signal

#%%

mconfig = "FULL_PIC"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
outpath = ""
lonf    = 330
latf    = 50
lags    = np.arange(0,37,1)
kmonth  = 1

savefiles = False
#%% Functions (each one already as ENSO Removed...)
def load_data(mconfig,datpath):
    st  = time.time()
    ld  = np.load(datpath+"%s_ENSOREM_TS_lag1_pcs2_monwin3.npz" % mconfig,allow_pickle=True)
    sst = ld['TS']
    lat = ld['lat']
    lon = ld['lon']
    print("Loaded in %.2fs"%(time.time()-st))
    return sst,lat,lon

#%%

# Load Files, Restrict to Point, Calculate Autocorrelation
ssts = []
sstpts = []
acs = []
for config in ["SLAB_PIC","FULL_PIC"]:
    
    # Load Data
    sst,lat,lon = load_data(config,datpath)
    
    ssts.append(sst)
    
    # Restrict to point
    klon,klat = proc.find_latlon(lonf,latf,lon,lat)
    locfn,loctitle=proc.make_locstring(lon[klon],lat[klat])
    sstpt = sst[:,klat,klon]
    sstpts.append(sstpt)
    if savefiles:
        outname = "%s%s_SST_%s.npy" % (datpath,mconfig,locfn)
        np.save(outname,sstpt)
        print("Saved file to %s" % outname)


    # Calculate Autocorrelation
    nmon    = sstpt.shape[0]
    nyr     = int(nmon/12)
    sstrs   = sstpt.reshape(nyr,12)
    ac      = scm.calc_autocorr([sstrs],lags,kmonth+1)
    acs.append(ac)
    if savefiles:
        outname = "%s%s_autocorr_%s_lags%ito%i_month%i.npy" % (datpath,mconfig,locfn,lags[0],lags[-1],kmonth+1)
        np.save(outname,ac[0])
        print("Saved file to %s" % outname)
    
    plt.scatter(lags,ac[0]),plt.plot(lags,ac[0],label=config),plt.title("Autocorrelation at %s" % loctitle),plt.grid(True,ls='dotted'),plt.legend()


# Make some comparison plots

#%% Calculate Autocorrelation for all points

#sst,lat,lon = load_data(mconfig,datpath)
sst = ssts[1].copy()

nmon,nlatr,nlonr = sst.shape #[time x lat x lon]

# Loop through each point. and calculate an autocorrelation curve
pointsize = nlonr*nlatr
startloop = time.time()


sst = sst.transpose(0,2,1)
sstrs = sst.reshape(nmon,pointsize)


# Preallocate 
enstime = time.time()

# Get ensemble [time x space]
sstens = sstrs

# Isolate non-nan points, summing along dimension zero
oksst,knan,okpts = proc.find_nan(sstens,0)

# Get dimensions and reshape the time to [month x yr x space]
timedim,spacedim = oksst.shape 
oksst = np.reshape(oksst,(int(timedim/12),12,spacedim))
oksst = np.transpose(oksst,(1,0,2))

# Preallocate and loop for each month...
autocorrm = np.ones((12,len(lags),spacedim)) * np.nan

# Loop for the months
for m in range(12):

    # Calculate autocorrelation for that month
    autocorrm[m,:,:] = proc.calc_lagcovar_nd(oksst,oksst,lags,m+1,0)

    msg = "Completed Mon %02d (Elapsed: %.2fs)" % (m+1,time.time()-startloop)
    print(msg,end="\r",flush=True)

#end month loop

# Preallocate array to re-insert nan points
autocorr = np.ones((12,len(lags),pointsize)) * np.nan
autocorr[:,:,okpts] = autocorrm

autocorr = np.reshape(autocorr,(12,len(lags),nlonr,nlatr))

    
# Save output

expid = mconfig
np.save(datpath+"Stochmod_Autocorrelation_%s.npy"%(expid),autocorr)
#print("\nCompleted ENS %02d in %.2fs" % (e+1,time.time()-enstime))

### Note the above took forever to run, need to speed it up
plt.plot(autocorr[1,:,klat,klon])

#%% Plot ac to check



#%% Calculate Power Spectra

