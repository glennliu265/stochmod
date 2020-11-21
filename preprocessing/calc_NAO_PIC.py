#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF Damping Calculations: Compute ENAO and EAP from PIC SLAB
Created on Fri Oct 30 12:08:19 2020

@author: gliu
"""
import xarray as xr
import numpy as np
import glob
import time

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
from amv import proc

#%% User Edits

# Mode
mode = 'SLAB' # "SLAB or FULL"

# TS [time lat lon]
varkeep = ['PSL','time','lat','lon','lev'] 

# PCs to calculate
N_mode = 3# User edited variable!

# Subset data for enso index calculation
bbox = [-90+360, 40, 20, 80]

# Mode:
mode = 'DJFM' # Mode is 'DJFM',or 'Monthly'
debug = True # Set to true to make figure at end


# Outpath
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_PIC_SLAB/NAO/"
outname = "EOF_NAO_%s_PIC_SLAB.npz" % (mode)

#%% Functions 
def check_NAO_sign(eofs,pcs,lon,lat,tol=5,verbose=True):
    """
    checks sign of EOF for NAO/EAP and flips sign by check the sum over
    several boxes (see below for more information)
    
    checks to see if the sum within the region and flips if sum < 0
    
    inputs:
        1) eofs [space (latxlon), PC] EOF spatial pattern from eof_simple
        2) pcs  [time, PC] PC timeseries calculated from eof_simple
        3) lat  [lat] Latitude values
        4) lon  [lon] Longitude values
        5) tol  [INT] Tolerance (in degrees) for searching
    
    outputs:
        1) eofs [space, PC]
        2) pcs  [time, PC]
    
    """   
    # Set EOF boxes to check over (west,east,south,north)
    lonLisb = -9.1398+360
    latLisb = 38.7223
    lonReyk = -21.9426+360
    latReyk = 64.146621
    naobox1 = [lonReyk-tol,lonReyk+tol,latReyk-tol,latReyk+tol] #EOF1 sign check Lat/Lon Box (Currently using nino3.4)
    naobox2 = [lonLisb-tol,lonLisb+tol,latLisb-tol,latLisb+tol]
    eapbox = [320,15,35,60] #EAP Box
    
    # Find dimensions and separate out space
    nlon = lon.shape[0]
    nlat = lat.shape[0]
    npcs = eofs.shape[1] 
    eofs = eofs.reshape(nlat,nlon,npcs)
    eofs = eofs.transpose(1,0,2) # [lon x lat x npcs]
        
    for n in range(npcs):
        
        if n == 0: # EOF 1
            chk = proc.sel_region(eofs,lon,lat,naobox1,reg_sum=1)[n]
        
            if chk > 0:
                if verbose:
                    print("Flipping EOF %i (Reykjavik Check) because sum is %.2f"%(n,chk))
        
                eofs[:,:,n] *= -1
                pcs[:,n] *= -1
            
            chk = proc.sel_region(eofs,lon,lat,naobox2,reg_sum=1)[n]
        
            if chk < 0:
                if verbose:
                    print("Flipping EOF %i (Lisbon Check) because sum is %.2f"%(n,chk))
        
                eofs[:,:,n] *= -1
                pcs[:,n] *= -1
        else:
            chk = proc.sel_region(eofs,lon,lat,eapbox,reg_sum=1)[n]
        
            if chk > 0:
                if verbose:
                    print("Flipping EOF %i (EAP Check) because sum is %.2f"%(n,chk))
        
                eofs[:,:,n] *= -1
                pcs[:,n] *= -1
                
    
    
    eofs = eofs.transpose(1,0,2).reshape(nlat*nlon,npcs) # Switch back to lat x lon x pcs
    
    return eofs,pcs
    
# Define preprocessing variable
def preprocess(ds,varlist=varkeep):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)
    
    # Select the ground level
    ds = ds.isel(lev=-1)
    
    # # Correct first month (Note this isn't working)
    # if ds.time.values[0].month != 1:
    #     startyr = str(ds.time.values[0].year)
    #     endyr = str(ds.time.values[-1].year)
    #     correctedtime = xr.cftime_range(start=startyr,end=endyr,freq="MS",calendar="noleap") 
    #     ds = ds.assign_coords(time=correctedtime) 
    
    return ds

def combineflux(fluxname,filepath,downward_positive=True,verbose=False):
    """  
    Combine flux files from CESM to get the following variables...
    (Note expressions below assume downwards_positive, but can set to False to
     flip values). CESM1 Flux values are given as upwards positive.
     1) Net Heat Flux (NHFLX) = FSNS - (LHFLX + SHFLX + FLNS) 
     2) Radiative Heat Flux (RHFLX) = FSNS - FLNS
     3) Turbulent Heat Flux (THFLX) = - (LHFLX + SHFLX)
    
    Parameters
    ----------
    fluxname : STR
        Desired flux. Can be:
            'NHFLX' (Net Heat Flux)
            'RHFLX' (Radiative Heat Flux)
            'THFLX' (Turbulent Heat Flux)
            
    filepaths : STR
        File path and name, with "%s" indicating where
        the flux name is. Note that this is assumed to be
        an npz file.
        
        ex: "/PathtoFile/ENSOREM_%s_HISTORICAL.nc"
    OPTIONAL ---
    downward_positive : BOOL
        Set to TRUE for positive flux into the ocean
        
    verbose : BOOL
        Print added values to check
    
    Returns
    -------
    combineflx : ARRAY
        Numpy Array containing the combined flux

    """

    fluxes = ['FSNS','FLNS','LHFLX','SHFLX']
    
    if fluxname == 'NHFLX':
        inflx = fluxes
    elif fluxname == 'THFLX':
        inflx = fluxes[2:]
    elif fluxname == 'RHFLX':
        inflx = fluxes[:]
    
    i = 0
    for f in inflx:
        
        # Open the file [time x lat x lon]
        flux = np.load(filepath%f,allow_pickle=True)[f]
        
        if f == 'FSNS':
            flux *= -1
        
        # Save value for printing
        savevaladd = flux[0,44,44].copy()
        
        # Append
        if i == 0:
            fluxout = flux.copy()
        else:
            savevalori = fluxout[0,44,44].copy()
            fluxout += flux
            if verbose:
                print("%.2f + %.2f = %.2f"%(savevalori,savevaladd,fluxout[0,44,44].copy()))
        i+= 1
    if downward_positive:
        fluxout *= -1
    return fluxout

#%% Script Start
# Get List of nc files for preindustrial control
ncpath = r'/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/PSL/'
if mode == 'SLAB':
    ncsearch = 'e.e11.E1850C5CN.f09_g16.001.cam.h0.PSL.*.nc'
elif mode == 'FULL':
    ncsearch = 'b.b11.E1850C5CN.f09_g16.001.cam.h0.PSL.*.nc'
nclist = glob.glob(ncpath+ncsearch)
nclist.sort()
nclist

# Open dataset
st = time.time()
dsall = xr.open_mfdataset(nclist,concat_dim='time',preprocess=preprocess)
print("Opened in %.2fs"%(time.time()-st))

# Slice to region
dsreg = dsall.sel(lat=slice(bbox[2],bbox[3]))
dsreg = dsreg.where((dsreg.lon >= bbox[0]) | (dsreg.lon <= bbox[1]),drop=True)

# Read out the variables
st = time.time()
psl = dsreg.PSL.values # [time x lat x lon]
lon = dsreg.lon.values
lat = dsreg.lat.values
times = dsreg.time.values
print("Data loaded in %.2fs"%(time.time()-st))

#%% Calculate ENSO

# Apply Area Weight
_,Y = np.meshgrid(lon,lat)
wgt = np.sqrt(np.cos(np.radians(Y))) # [lat x lon]
psl = psl * wgt[None,:,:]

# Reshape for NAO calculations
ntime,nlat,nlon = psl.shape 
psl = psl.reshape(ntime,nlat*nlon) # [time x space]
psl = psl.T #[space x time]

# Remove NaN points
okdata,knan,okpts = proc.find_nan(psl,1) # Find Non-Nan Points
oksize = okdata.shape[0]

# Calcuate monthly anomalies
okdata = okdata.reshape(oksize,int(ntime/12),12) # [space x yr x mon]
manom = okdata.mean(1)
tsanom = okdata - manom[:,None,:]
#tsanom = tsanom.reshape(nlat*nlon,ntime)
nyr = tsanom.shape[1]

if mode == 'Monthly':
    eofall = np.zeros((nlat*nlon,12,N_mode)) *np.nan# [space x month x pc]
    pcall  = np.zeros((nyr,12,N_mode)) *np.nan# [year x month x pc]
    varexpall  = np.zeros((12,N_mode)) * np.nan #[month x pc]
    
    # Compute EOF!!
    for m in range(12):
        
        # Perform EOF
        st = time.time()
        eofsok,pcs,varexp=proc.eof_simple(tsanom[:,:,m],N_mode,1)
        #print("Performed EOF in %.2fs"%(time.time()-st))
    
        # Place back into full array
        eofs = np.zeros((nlat*nlon,N_mode)) * np.nan
        eofs[okpts,:] = eofsok   
    
        # Correct ENSO Signs
        eofs,pcs = check_NAO_sign(eofs,pcs,lon,lat,verbose=True)
        
        
        # Save variables
        eofall[:,m,:] = eofs.copy()
        pcall[:,m,:] = pcs.copy()
        varexpall[m,:] = varexp.copy()
        
        print("Completed month %i in %.2fs"%(m+1,time.time()-st))
elif mode == 'DJFM':
    
    # Take DJFM Mean
    tsanom = tsanom[:,:,[-1,0,1,2]].mean(2)
    
    # Preallocate
    eofall = np.zeros((nlat*nlon,N_mode)) *np.nan# [space x pc]
    pcall  = np.zeros((nyr,N_mode)) *np.nan# [year x pc]
    varexpall  = np.zeros((N_mode)) * np.nan #[pc]
    
    # Perform EOF
    st = time.time()
    eofsok,pcs,varexpall=proc.eof_simple(tsanom,N_mode,1)
    #print("Performed EOF in %.2fs"%(time.time()-st))
    
    # Place back into full array
    eofs = np.zeros((nlat*nlon,N_mode)) * np.nan
    eofs[okpts,:] = eofsok   
    
    # Correct ENSO Signs
    eofall,pcall = check_NAO_sign(eofs,pcs,lon,lat,verbose=True)
        
    print("Completed EOF in %.2fs"%(time.time()-st))
    

if debug:
    import matplotlib.pyplot as plt
    import cmocean  
    cmap=cmocean.cm.balance 
    eofall = eofall.reshape(nlat,nlon,3) 
    plt.pcolormesh(lon,lat,eofall[:,:,2],cmap=cmap),plt.colorbar(),plt.show()




#%% Regress DJFM NAO to DJFM Net Heat Flux Feedback

# NHFLX Settings
flux = 'NHFLX'
pcrem   = 2
ensolag = 1
emonwin = 3
datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_PIC_SLAB/02_ENSOREM/" # Output Path

# Set File Names
ensoexp = "lag%i_pcs%i_monwin%i" % (ensolag,pcrem,emonwin)
filepath = datpath+"ENSOREM_%s_"+ensoexp + ".npz"

# Load Flux File
flx = combineflux(flux,filepath,verbose=True)

# Calculate DJFM mean
ntime,nlat,nlon = flx.shape
flx = flx.reshape(int(ntime/12),12,nlat*nlon)
flx = flx[:,[-1,0,1,2],:].mean(1) # Select DJFM

# Restrict PC (drop first 2 years due to ensolag + 3monwin, and end year due to 3monwin)
pcin = pcall[2:-1,:]

# Standardize PC
pcstd = pcin / np.std(pcin,0)

# Preallocate
pattern = np.zeros((nlat*nlon,N_mode)) # [space x pc]

for p in range(N_mode):
    
    pc = pcstd[:,p]
    
    pattern[:,p],_ = proc.regress_2d(pc,flx,nanwarn=1)
    print("Regressed to PC %i"% (p+1))

pattern = pattern.reshape(nlat,nlon,N_mode)

if debug:
    plt.pcolormesh(lon,lat,pattern[:,:,0],cmap=cmap),plt.colorbar(),plt.title("EOF 1"),plt.show()
    
    
# Save Output
st = time.time()
np.savez(outpath+outname,**{
         'eofs': eofall,
         'pcs': pcall,
         'varexp': varexpall,
         'nhflx_pattern': pattern,
         'lon': lon,
         'lat':lat,
         'times':times}
        )   

print("Data saved in %.2fs"%(time.time()-st))