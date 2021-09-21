#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:03:07 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import time
from scipy.io import loadmat

from tqdm import tqdm
import matplotlib.pyplot as plt


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz

#%%
fn      = "pop_frc.b.e11.B1850C5CN.f09_g16.005.150217.nc"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
latlonfn = "tlat_tlon.npz"


#%% Function

def getpt_pop_array(lonf,latf,invar,tlon,tlat,searchdeg=0.75,printfind=True,verbose=False):
    
    """
    IMPT: assumes input variable is of the shape [lat x lon x otherdims]
    tlon = ARRAY [lat x lon]
    tlat = ARRAY [lat x lon]
    """
    
    if lonf < 0:# Convet longitude to degrees East
        lonf += 360
    
    # Query Points
    quer = np.where((lonf-searchdeg < tlon) & (tlon < lonf+searchdeg) & (latf-searchdeg < tlat) & (tlat < latf+searchdeg))
    latid,lonid = quer
    
    if printfind:
        print("Closest LAT to %.1f was %s" % (latf,tlat[quer]))
        print("Closest LON to %.1f was %s" % (lonf,tlon[quer]))
        
    if (len(latid)==0) | (len(lonid)==0):
        if verbose:
            print("Returning NaN because no points were found for LAT%.1f LON%.1f"%(latf,lonf))
        return np.nan
        exit
    
    
    # Locate points on variable
    if invar.shape[:2] != tlon.shape:
        print("Warning, dimensions do not line up. Make sure invar is Lat x Lon x Otherdims")
        exit
    
    return invar[latid,lonid,:].mean(0) # Take mean along first dimension



#%%
ds = xr.open_dataset(datpath+fn)
ds

# Load in data
st   = time.time()
qdp  = ds.qdp.values
hblt = ds.hblt.values
print("Loaded data in %.2fs"%(time.time()-st))


# Load in Tlat/Tlon coordinates
ld = np.load(datpath+latlonfn,allow_pickle=True)
tlon = ld['tlon']
tlat = ld['tlat']

# Load in target CESM1LE coordinates
ll =  loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")
lat = ll['LAT'].squeeze()
lon = ll["LON"].squeeze()
lon1 = np.hstack([lon[lon>=180]-360,lon[lon<180]])

# Transpose the data
hblt = hblt.transpose(1,2,0) # [384,320,time]
qdp  = qdp.transpose(1,2,0)

#%% Interpolate for each point
start = time.time()
lonsize,latsize = lon.shape[0],lat.shape[0]
hblt_atm = np.zeros((lonsize,latsize,12))*np.nan
qdp_atm  = hblt_atm.copy()

for o in tqdm(range(lonsize)):
    
    lonf = lon1[o]
    if lonf < 0:
        lonf += 360
    #print(lonf)
    
    for a in range(latsize):
        latf = lat[a]
        
        # Get point
        hval = getpt_pop_array(lonf,latf,hblt,tlon,tlat,printfind=False)
        qval = getpt_pop_array(lonf,latf,qdp,tlon,tlat,printfind=False)
        
        
        if np.any(np.isnan(hval)):
            msg = "Land Point @ lon %f lat %f" % (lonf,latf)
            continue
        else:
            hblt_atm[o,a,:] = hval.copy()
            qdp_atm[o,a,:] = qval.copy()
print("Finished in %f seconds" % (time.time()-start))  
 
       
plt.pcolormesh(lon,lat,hblt_atm[:,:,0].T),plt.colorbar()
plt.pcolormesh(lon,lat,qdp_atm[:,:,0].T),plt.colorbar()

np.save(datpath+"SLAB_PIC_qdp.npy",qdp_atm)
np.save(datpath+"SLAB_PIC_hblt.npy",hblt_atm)


