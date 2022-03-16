#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take Output from NHFLX_EOF_monthly.py and make it into stochastic model input

Created on Fri Mar 11 18:57:32 2022

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm

import cmocean
#%%
stormtrack = 0

if stormtrack == 1:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
elif stormtrack == 0:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    #datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220113/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"

from amv import proc,viz
import scm

def sel_regionxr(ds,bbox):
    """
    Select region from xr.dataset with 'lon' (degrees East), 'lat'.
    Current supports crossing the prime meridian

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    bbox : TYPE
        DESCRIPTION.

    Returns
    -------
    dsreg : TYPE
        DESCRIPTION.

    """
    # Select lon (lon360)
    if bbox[0] > bbox[1]: # Crossing Prime Meridian
        print("Crossing Prime Meridian!")
        dsreg = ds.isel(lon =(ds.lon < bbox[1]) + (ds.lon>bbox[0]))
        dsreg = dsreg.sel(lat=slice(bbox[2],bbox[3]))
    else:
        dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    return dsreg

#%% User Edits

# Set your configuration
mconfig = "PIC_SLAB"

# Set up names (need to change this at some point, silly coding...)
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
elif mconfig == "PIC_FULL":
    mcname = "FULL-PIC"

# Bounding Box of Plotting and EOF Analysis
bbox    = [260,20,0,65]
bboxeof = [280,20,0,65]

# Debuggin params
debug   = True
lonf    = -30
latf    = 50

# EOF parameters
N_mode = 200

# Plotting params
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
blabels=[0,0,0,0]

# Correction Mode
correction     = True # Set to True to use [Qnet + lambda*T], rather than [Qnet]
rolln          = 0

if rolln != 1: # Add rolln to end of file name
    print("Adding extra string for indexing")
    rollnstr       = "_rolln%s" % str(rolln)
    correction_str = "_Fprime_rolln0"
else: # No rolln for case where it is T(t-1)
    rollnstr       = ""
    correction_str = "_Fprime"

#%% Load EOFs and Correct
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
elif mconfig == "PIC_FULL":
    mcname = "FULL-PIC"

bboxtext  = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr   = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename  = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)
ld        = np.load(savename,allow_pickle=True)

eofall    = ld['eofall']
eofslp    = ld['eofslp']
pcall     = ld['pcall']
varexpall = ld['varexpall']

lon = ld['lon']
lat = ld['lat']

lon360        = np.load(datpath+"../CESM_lon360.npy")
lon180,_        = scm.load_latlon()


nlon,nlat,nmon,nmode = eofall.shape
#%% Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG)

spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP

N_modeplot = 5
for N in tqdm(range(N_modeplot)):
    if N == 1:
        chkbox = eapbox # Shift coordinates west
    else:
        chkbox = spgbox
    for m in range(12):
        
        
        
        
        sumflx = proc.sel_region(eofall[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
        sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
        
        if sumflx > 0:
            print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
            eofall[:,:,m,N]*=-1
            pcall[N,m,:] *= -1
        if sumslp > 0:
            print("Flipping sign for SLP, mode %i month %i" % (N+1,m+1))
            eofslp[:,:,m,N]*=-1


    
#%% Load file from NHFLX_EOF_monthly.py

savename     = datpath + "../NHFLX_EOF_Ratios_%s.npz" % mcname
if correction:
    savename = proc.addstrtoext(savename,correction_str)
ld           = np.load(savename)
varflx_EOF   = ld['varflx_EOF']
varflx_ori   = ld['varflx_ori']
varflx_ratio = ld['varflx_ratio']
vratio_alone = ld['varflx_ratio_alone']

eof_corr     = eofall * 1/vratio_alone

# ------------------------------------------------
#%% Calculate/plot cumulative variance explained
# ------------------------------------------------
# mode number (x-axis) vs. %-variance explained (y-axis)
# separate lines used for each month

# Calculate cumulative variance at each EOF
cvarall = np.zeros(varexpall.shape)
for i in range(N_mode):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Plot Params
N_modeplot = 50
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)

for m in range(12):
    plt.plot(modes,cvarall[:,m]*100,label="Month %i"% (m+1),marker="o",markersize=4)
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
savename = "%s%s_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,mcname,N_modeplot,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)
plt.savefig(savename,dpi=150)

# ------------------------------------------------
# %% Make the Forcings...
# ------------------------------------------------

# -----------------------------
#% Save a select number of EOFs
# -----------------------------
eofcorr       = 0
N_mode_choose = 2

# Select the mode
eofforce      = eofall.copy()
eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
eofforce      = eofforce[:,:,:N_mode_choose,:]

# Prepare correction based on variance explained
if eofcorr == 1: # Correct based on EOF "basinwide" variance explained
    perc_in   = varexpall[:N_mode_choose,:].sum(0) # Sum variance for selected modes
    ampfactor = 1/perc_in # [Months]
    eofforce  *= ampfactor
elif eofcorr == 2: # Correct based on the LOCAL variance explained at each point...
    perc_in   = np.zeros([nlon,nlat,12]) * np.nan
    for im in range(12): # Loop for each nonth...
        perc_in[:,:,im] =  varflx_ratio[:,:,im,N_mode_choose]
    ampfactor = 1/perc_in
    eofforce  *= ampfactor[:,:,None,:] 

savenamefrc   = "%sflxeof_%ieofs_%s_eofcorr%i.npy" % (datpath,N_mode_choose,mcname,eofcorr)
if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)

# -----------------------------
#%% Save a given EOF
# -----------------------------

eofcorr       = 0
N_mode_choose = 0

# Select the mode
eofforce = eofall.copy()
eofforce = eofforce.transpose(0,1,3,2)   # lon x lat x pc x mon
eofforce = eofforce[:,:,N_mode_choose,:] # lon x lat x mon

# Prepare correction based on variance explained
if eofcorr == 1: # Correct based on EOF "basinwide" variance explained
    perc_in   = varexpall[N_mode_choose,:] # Sum variance for selected modes
    ampfactor = 1/perc_in # [Months]
    eofforce  *= ampfactor
elif eofcorr == 2: # Correct based on the LOCAL variance explained at each point...

    EOF_var = eofforce[:,:,:].copy() **2 # Variance of that EOF mode at each point
    perc_in = EOF_var/varflx_ori.squeeze() # Percentage of original variance
    ampfactor = 1/perc_in
    eofforce *= ampfactor

eofforce = eofforce[:,:,None,:] # lon x lat x pc x mon

savenamefrc   = "%sflxeof_EOF%i_%s_eofcorr%i.npy" % (datpath,N_mode_choose+1,mcname,eofcorr)
if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)

np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)

#%% Monthly Forcing

# Load the Forcing
vthres  = 0.90
eofcorr = 2


savenamefrc = "%sflxeof_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,mcname,eofcorr)
if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
eofforce = np.load(savenamefrc)


monids   = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
monnames = ("DJF","MAM","JJA","SON")
for s in tqdm(range(4)):
    # Calculate seasonal average
    eofseas = np.mean(eofforce[:,:,:,monids[s]],-1,keepdims=True)
    
    # Save the output
    savenamefrc = "%sflxeof_%03ipct_%s_eofcorr%i_%s.npy" % (datpath,vthres*100,mcname,eofcorr,monnames[s])
    if correction:
        savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
    print("Saving to %s"%savenamefrc)
    np.save(savenamefrc,eofseas)
    
    