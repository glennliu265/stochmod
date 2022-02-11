#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qcorr simple

Based on Claude's suggestion, recalculate the 
stochastic forcing amplitude based on a simpler approach using 
T', Qnet, and Lambda'


For Qnet
- Removes mean
- Removes seasonal cycle

Created on Thu Feb 10 08:43:38 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220210/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

#%% Used Edits

lonf    = -30
latf    = 50

mconfig    = "FULL"
frcname    = "flxeof_090pct_FULL-PIC_eofcorr2"
input_path = rawpath

debug = True

"""
#%% Next Few Sections, Load in the Data...
"""


    


#%% Load in the Temperature Anomalies
# [lon x lat x time]

# Load in from dataset
ds      = xr.open_dataset(datpath+"../CESM_proc/TS_anom_PIC_%s.nc" % (mconfig))
T        = ds.TS.values
lon360   = ds.lon.values
ds.close()

# Transpose dimensions
T        = T.transpose(2,1,0) # [time x lat x lon] --> [lon x lat x time]
lon180,T = proc.lon360to180(lon360,T)  

# Get dimension lengths
nlon,nlat,ntime = T.shape
nyrs            = int(ntime/12)

#%% Load in Qnet

# Load in from NumPY (Need to update this)
Q   = np.load(datpath+"../NHFLX_PIC_Full.npy") # [year x month x lat x lon]
Q   = Q.reshape(ntime,nlat,nlon) # [time x lat x lon]
Q   = Q.transpose(2,1,0) # [lon x lat x time]

# Flip to lon 180
_,Q = proc.lon360to180(lon360,Q)  

# Remove Time Mean
Q   = Q - Q.mean(2)[:,:,None]

# Remove Scycle
Qseas,Q = proc.calc_clim(Q,2,returnts=1)
Q       = Q - Qseas[:,:,None,:]
Q = Q.reshape(nlon,nlat,ntime)

#%% Use the function used for sm_rewrite.py, load in lambda
# [lon x lat x month]

inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
if mconfig == 'SLAB':
    lbd = dampingslab
elif mconfig == 'FULL':
    lbd = dampingfull

# Tile Lambda
lbd = np.tile(lbd,nyrs)
#%% Compute Stochastic Forcing
"""
Qnet = -lambda*T + F'

F' = Qnet + lambda*T

"""
# Test the phase
fig,ax =plt.subplots(1,1)
test = np.arange(0,12,1)
ax.plot(test,label='normal',color='k')
ax.plot(np.roll(test,-1),label="-1",color='b')
ax.plot(np.roll(test,1),label="1",color='r')
ax.legend()
ax.grid(True)
ax.set_xlim([0,3])

# Uh oh what lag of lambda to use?
rolln    = 1
if rolln == 0.5: # ([T(t) + T(t-1)]/2)
    dampterm = (np.roll(T,1,axis=2)+T)/2 * lbd
else:
    dampterm = np.roll(T,rolln,axis=2) * lbd
Fprime = Q + dampterm 

# Plot some test plots
if debug:
    plt.pcolormesh(Fprime[...,0].T,vmin=-400,vmax=400),plt.colorbar()
    plt.show()
    plt.pcolormesh(Q[...,0].T,vmin=-400,vmax=400),plt.colorbar()
    plt.show()
    plt.pcolormesh(dampterm[...,0].T,vmin=-50,vmax=50),plt.colorbar()
    plt.show()
    
#%% Do some Exploratory Analysis
normspec = False
labels    = ("$Q_{net}$","F'","$\lambda T$","-$\lambda T$ + F'")
klon,klat = proc.find_latlon(lonf,latf,lon180,lat)
locstring = "Lon %.f, Lat %.f" % (lon180[klon],lat[klat])

# 1) Compute the power spectra
nsmooth = 100
pct     = 0.00
dtplot  = 3600*24*365 
xtks    = [1/100,1/50,1/25,1/10,1/2.5,1/1]
xper    = [1/x for x in xtks]
xlm     = [xtks[0],xtks[-1]]

inflx     = [Q[klon,klat,:],Fprime[klon,klat,:],dampterm[klon,klat,:],Fprime[klon,klat,:]-dampterm[klon,klat,:]]
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inflx,nsmooth,pct)
if normspec:
    ylm = [0,3]
    specs = [specs[i]/np.var(inflx[i]) for i in range(len(inflx))]
else:
    ylm = [0,2000]

fig,ax = plt.subplots(1,1)
for i in range(len(inflx)):
    if i == (len(inflx)-1):
        ls = 'dashed'
    else:
        ls = 'solid'
    ax.plot(freqs[i]*dtplot,specs[i]/dtplot,label=labels[i],lw=3,ls=ls)

plotspec = specs[1] + specs[2]
ax.plot(freqs[i]*dtplot,plotspec/dtplot,label=labels[i] + "(Add Spectra)",lw=3,ls=ls)
ax.legend()
ax.set_xticks(xtks)
ax.set_xticklabels(xper)
ax.set_xlim(xlm)
ax.set_ylim(ylm)
ax.set_title("Power Spectra Estimates @ %s " % locstring )
ax.grid(True,ls='dotted')

#%% Save the updated forcing for the EOF Analysis

"""
Fprime is [lon x lat x time]

"""
# 
Fprime = proc.flipdims(Fprime) #  Flip to [time x lat x lon]
times   = xr.cftime_range(start="0400-01-01",periods=Fprime.shape[0],freq="MS")


# Save as a netCDF 
savename = "%s../Fprime_PIC_%s.nc" % (datpath,mconfig)
proc.numpy_to_da(Fprime,times,lat,lon180,'Fprime',savenetcdf=savename)


#%%



