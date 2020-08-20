#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to preprocess raw SLP data, perform EOF analysis, and regress back to
NHFLX anomaly data.


Created on Wed Aug 19 18:12:13 2020

@author: gliu
"""
import xarray as xr
import time
import glob

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc,viz

#%% # Functions

# Preprocessing function (need to edit to make it work within open_mfdataset)
def preprocess(ds,varkeep):
    """correct time dimension and drop unwanted variables"""
    
    # Correct time dimension to start in Jan rather than Feb
    if ds.time.values[0].month != 1:
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
        print("\tCorrected Start to: %s; End to: %s" %  (str(ds.time.values[0]),str(ds.time.values[-1])))

    
    # Load proper latitude to accomodate different indexing
    from scipy.io import loadmat
    lat = np.squeeze(loadmat("/home/glliu/01_Data/CESM1_LATLON.mat")['LAT'])
    if np.any(~(lat == ds.lat.values)):
        ds = ds.assign_coords(lat=lat)
        print("\tReassigning latitude values ")
    
    # Drop variables unless it is in "varkeep"
    dsvars = list(ds.variables)
    varrem = [i for i in dsvars if i not in varkeep]
    ds = ds.drop(varrem)
    
    return ds

def xrdeseason(ds):
    """ Remove seasonal cycle..."""
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')

#%% User Edits

# Path to data (NHFLX)
datpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NAO_Forcing_DataProc/"
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NHFLX/"


#Path to SLP Data and glob expression
varname = 'PSL'
datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/%s/" % varname
ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.%s.*.nc" % varname

# Variables to keep
varkeep = ['PSL','lon','lat','time','ensemble'] # Variables to Keep

# Bounding Box to compute eof over
lonW = -90
lonE = 40
latS = 20
latN = 80

# EOF flipping options
N_mode = 3 # Number of EOFs to calculate
station = "Reykjavik"
flipeof = 1
lonLisbon = -9.1398 + 360
latLisbon = 38.7223
lonReykjavik = -21.9426 + 360
latReykjavik = 64.146621
tol = 5 # in degrees
eapbox = [320,15,35,60]
#%% Read in NHFLX

# Open Datasets (NHFLX) [1.53s]
nclist = glob.glob(outpath+"NHFLX_ens*.nc")
nclist = [x for x in nclist if "mean" not in x]
ds = xr.open_mfdataset(nclist,concat_dim='ensemble',
                       combine='nested',
                       compat='identical', # seems to be strictest setting...not sure if necessary
                       parallel="True",
                       join="exact" # another strict selection...
                      )
# Add ensemble as a dimension
ds = ds.assign_coords({'ensemble':np.arange(1,43,1)})


# Deseason, then Detrend (Deforce) [1m46s]
# Note: we had already deforced and deseasoned the data in the script where NHFLX was generated.
# Oddly enough, there still appears to be a difference between the two...
start = time.time()
ds2 = ds.groupby('time.month') - ds.groupby('time.month').mean('time')
ds2 = ds2 - ds2.mean('ensemble')
nhflxall = ds2.NHFLX.values
print("Deseasoned and Deforced in %.2fs" % (time.time()-start) )

#%% Read in raw SLP Data

# Get list of ncfiles
globby = datpath+ncsearch
nclist =glob.glob(globby)
nclist = [x for x in nclist if "OIC" not in x]
nclist.sort()

# Read in and concatenate each file
for e in range(len(nclist)):
    startloop = time.time()
    psl = xr.open_dataset(nclist[e])
    psl = preprocess(psl,varkeep)
    if e == 0:
        pslall = psl.copy()
    else:
        pslall = xr.concat([pslall,psl],dim="ensemble")
    print("Completed ensemble %i of 42 in %.2fs" % (e+1,time.time()-startloop))


#% Select time period after 1920-01-01 and remove seasonal and ensemble mean [2m8s]
dsna   = pslall.sel(time=slice('1920-01-01','2005-12-31'))
dsna2 = xrdeseason(dsna)
dsna2 = dsna2 - dsna2.mean('ensemble')

# Read out values from netcdf (global) [3.13s]
start = time.time()
pslglo = dsna2.PSL.values
lon    = dsna2.lon.values
lat    = dsna2.lat.values
pslglo /= 100 # Convert from Pa to hPa
print("Read in files from netcdf in %.2f" % (time.time()-start))

# Slice to region for eof calc [1m37s]
invar = np.reshape(pslglo.transpose(3,2,0,1),(288,192,1032*42)) #reshape to [lon,lat,time x ens]
lonW = -90 + 360
lonE = 40
latS = 20
latN = 80
invar,rlon,rlat = proc.sel_region(invar,lon,lat,[lonW,lonE,latS,latN]) 
invar = invar.reshape(105,64,1032,42).transpose(3,2,1,0) # reshape to [ens x time x lat x lon]

# Get dimension sizes (regional)
nyr  = int(dsna2.time.shape[0] / 12)
nens = len(nclist)
nlon = len(rlon)
nlat = len(rlat)

# Grand Loop for EOF calculation and regression [2m55s]
# Preallocate [ens x mon x year]
pcall      = np.zeros((nens,12,int(nyr),N_mode))
varexpall  = np.zeros((nens,12,N_mode))
flxpattern = np.zeros((nens,12,192,288,N_mode))
psleofall  = np.zeros((nens,12,192,288,N_mode))

# Reshape to separate months and years and combine lat/lon
invar    = np.reshape(invar,(42,nyr,12,nlat*nlon))
nhflxall = np.reshape(nhflxall,(42,nyr,12,192*288))
pslglo   = np.reshape(pslglo,(42,nyr,12,192*288)) 

# Flip so that things are POSITIVE DOWNWARDS
nhflxall *= -1

startloop = time.time()
for e in range(nens):
    for m in range(12):
        
        # Transpose to match input requirements
        varmon = invar[e,:,m,:].T    # Transpose to [Space x Time]
        flxmon = nhflxall[e,:,m,:].T # Do same for net heat flux
        pslmon = pslglo[e,:,m,:].T   # Repeat for global psl

        # Get month indexes
        okdata,knan,okpts    = proc.find_nan(varmon,1)
        okdataf,knanf,okptsf = proc.find_nan(flxmon,1)
        okdatap,knanp,okptsp = proc.find_nan(pslmon,1)

        #% Perform EOF -------------
        _,pcs,varexpall[e,m,:]= proc.eof_simple(okdata,N_mode,1)

        # Standardize pc before regression along the time dimension
        pcstd = np.squeeze(pcs/np.std(pcs,0))

        #Loop for each mode... (NOte, can vectorize this to speed it up,.)
        flxpats = np.ones((192,288,3))*np.nan
        psleofs = np.ones((192,288,3))*np.nan
        
        for pcn in range(N_mode):

            # Regress back to SLP (NOT flux)
            eofpatokp,_ = proc.regress_2d(pcstd[:,pcn],okdatap)
            eofpatokf,_ = proc.regress_2d(pcstd[:,pcn],okdataf)

            # Reshape regression pattern and put back (for psl)
            eofpatp = np.ones((192*288))*np.nan
            eofpatp[okptsp] = eofpatokp
            eofpatp = np.reshape(eofpatp,(192,288))

            # Reshape regression pattern and put back (for flx)
            eofpatf = np.ones((192*288))*np.nan
            eofpatf[okptsf] = eofpatokf
            eofpatf = np.reshape(eofpatf,(192,288))

            if pcn == 0:

                # Check the signs and flip if necessary
                # EOF 1 only (need to figure out the other EOFs..)
                # First check using reykjavik
                rboxlon = np.where((lon >= lonReykjavik-tol) & (lon <= lonReykjavik+tol))[0]
                rboxlat = np.where((lat >= latReykjavik-tol) & (lat <= latReykjavik+tol))[0]

                chksum = np.nansum(eofpatp[rboxlat[:,None],rboxlon[None,:]],(0,1))
                if chksum > 0:
                    print("\t Flipping sign based on Reykjavik, Month %i Ens%i" % (m+1,e+1))
                    eofpatp *= -1
                    eofpatf *= -1
                    pcs[:,pcn] = -1


                # Double Check with Lisbon
                lboxlon = np.where((lon >= lonLisbon-tol) & (lon <= lonLisbon+tol))[0]
                lboxlat = np.where((lat >= latLisbon-tol) & (lat <= latLisbon+tol))[0]

                chksum = np.nansum(eofpatp[lboxlat[:,None],lboxlon[None,:]],(0,1))
                if chksum < 0:
                    print("\t Flipping sign based on Lisbon, Month %i Ens%i" % (m+1,e+1))
                    eofpatp *= -1
                    eofpatf *= -1
                    pcs[:,pcn] *= -1

            elif (pcn == 1) | (pcn == 2):
                
                rsum = proc.sel_region(eofpatp.T,lon,lat,eapbox,reg_sum=1)
                if rsum > 0:
                    print("\t Flipping sign based on EAP, PC%i onth %i Ens%i" % (pcn+1,m+1,e+1))
                    eofpatp *= -1
                    eofpatf *= -1
                    pcs[:,pcn] *= -1
                 
                 
            flxpats[:,:,pcn] = eofpatf
            psleofs[:,:,pcn] = eofpatp

        #% Assign to outside variable -------------
        flxpattern[e,m,:,:,:] = flxpats # ENS x Mon x Lon x Lat x PC
        psleofall[e,m,:,:,:]  = psleofs  # ENS x Mon x Lon x Lat x PC
        pcall[e,m,:] = np.squeeze(pcs)   # ENS x Mon x PC


        print("\rCalculated EOF for month %02d of ens %02d in %.2fs" % (m+1,e+1,time.time()-startloop),end="\r",flush=True)

# Save output
np.savez("%sNAO_Monthly_Regression_PC123.npz"%(datpath1),pcall=pcall,flxpattern=flxpattern,psleofall=psleofall,varexpall=varexpall)
print("saved to %sNAO_Monthly_Regression_PC123.npz" %(datpath1))