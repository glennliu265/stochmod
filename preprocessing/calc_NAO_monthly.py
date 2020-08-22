#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_NAO_monthly:
    
Script to preprocess raw SLP data, perform EOF analysis, and regress back to
NHFLX anomaly data. Works on stormtrack and flips EOF signs, checking for 
NAO in EOF1 and EAP in EOF2 and EOF3. Indicate the bounding boxes and points
in the user edits section.

Inputs:
    1. NHFLX anomalies, generated via [preproc_NAO_monthly.py] stored as 
        NHFLX_ens%03d.nc
    2. Raw PSL files from CESM (42 ensemble members)
    

Output:
    npz file [NAO_Monthly_Regression_PC123.npz] that contains
    1. pcall      (array: ens x mon x yr x pc): 
        PC timeseries for each month and ensemble
        
    2. flxpattern (array: ens x mon x lat x lon x pc):
        NHFLX pattern regressed back to PC
        
    3. psleofall  (array: ens x mon x lat x lon x pc):
        PSL pattern regressed back to PC
        
    4. varexpall  (array: ens x mon x pc):
        Variance explained 
    
Dependencies:
    xarray as xr
    numpy as np
    sys
    time
    glob
    
    amv.proc

Notes to self: Parts of the code are really slow, due to the differences in
latitude from ens 35 onwards, and hence the need to manually read everything in.
Need to find a way to optimize that (perhaps calculate SLP anomalies beforehand)

Created on Wed Aug 19 18:12:13 2020

@author: gliu
"""

import xarray as xr
import time
import glob
import numpy as np
import time

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model")
from amv import proc
import scm

#%% User Edits
allstart = time.time()

# NAO Calculation Type
#   0 = NAO[DJFM-avg] regressed to NHFLX[DJFM-avg]
#   1 = NAO[DJFM-avg] regressed to NHFLX[Monthly]
#   2 = NAO[Monthly]  regressed to NHFLX[Monthly]
naotype = 0

# Bounding Box to compute eof over
bboxNAO = [-90,40,20,80] # NAO BOX BASED ON DESER ET AL PAPER
#bboxNAO = [-90,20,20,75]
djfm    = [11,0,1,2] # Months to average over for NAO calculation

# EOF flipping options
N_mode = 3 # Number of EOFs to calculate
station = "Reykjavik"
flipeof = 1
lonLisbon = -9.1398 + 360
latLisbon = 38.7223
lonReykjavik = -21.9426 + 360
latReykjavik = 64.146621
tol = 5 # in degrees
eapbox = [320,15,35,60] # Box to check EAP pattern

# Variables to load in 
varnames = ('NHFLX','SLP')

# Ensemble numbers
mnum = np.concatenate([np.arange(1,36),np.arange(101,108,1)])

outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NAO_Forcing_DataProc/"

print("Running calc_NAO_monthly in mode %i ..." % (naotype))
#%% Read in SLP and NHFLX Data
# Currently designed to read the output for preproc_SLP_monthly.py and 
# preproc_NHFLX_monthly.py

for v in range(len(varnames)):
    vstart = time.time()
    # Get variable name and path
    vn = varnames[v]
    datpath  =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % vn
    
    # Create list of variables
    nclist = ["%s%s_ens%03d.nc" % (datpath,vn,e) for e in mnum]
    
    # Open dataset
    ds = xr.open_mfdataset(nclist,
                           concat_dim='ensemble',
                           combine='nested',
                           compat='identical', # seems to be strictest setting...not sure if necessary
                           parallel="True",
                           join="exact" # another strict selection...
                           )
    
    # Add ensemble as a dimension
    ds = ds.assign_coords({'ensemble':np.arange(1,len(mnum)+1,1)})
    
    # Merge variables to Dataset (assuming they have the same coordinates)
    if v == 0:
        dsall = ds.copy()
    else:
        dsall = xr.merge([dsall,ds])

#%% Get the DJFM and Regional cuts for EOF calculation
cuttime = time.time()

# Read in the data # [Ens x Time x Lat d Lon]
pslglo = dsall.PSL.values / 100 # divide by 100 to conver to hPa
lat    = dsall.lat.values 
lon    = dsall.lon.values
times  = dsall.time.values

# Get dimensions
nlon  = len(lon)
nlat  = len(lat)
ntime = len(times)
nens  = len(mnum)
nyr   = int(ntime/12)

# Reshape to [lon,lat,time x ens]
invar = np.reshape(pslglo.transpose(3,2,0,1),(nlon,nlat,ntime*nens)) 

# Correct if expressed in degrees west
if bboxNAO[0] < 0: bboxNAO[0] += 360
if bboxNAO[1] < 0: bboxNAO[0] += 360

# Select NAO region and reshape to original dimensions
pslnao,rlon,rlat = proc.sel_region(invar,lon,lat,bboxNAO)
nlatr = len(rlat)
nlonr = len(rlon)
pslnao = pslnao.reshape(nlonr,nlatr,1032,42).transpose(3,2,1,0) # reshape to [ens x time x lat x lon]

# Separate out the month and year dimensions and combine lat/lon, then transpose to [ens x space x yr x mon]
pslnao = pslnao.reshape(nens,nyr,12,nlatr*nlonr).transpose(0,3,1,2) #[ens x space x yr x mon]
pslglo = pslglo.reshape(nens,nyr,12,nlat*nlon).transpose(0,3,1,2)  

print("Variables loaded for EOF calculation in %.2fs" % (time.time()-cuttime))
#%% Perform EOF Analysis and calculate the PC

# Just calculate EOF for each ensemble member
if naotype < 2:
    
    # Take the DJFM mean 
    pslnao = np.mean(pslnao[:,:,:,djfm]).mean(3) # [ens x space x yr]
    pslglo = np.mean(pslglo[:,:,:,djfm]).mean(3) # [ens x space x yr]
    
    # Preallocate
    pcall      = np.zeros((nens,nyr,N_mode))     # [ens x year x pc]
    varexpall  = np.zeros((nens,N_mode))         # [ens x pc]
    eofall     = np.zeros((nens,nlat,nlon,N_mode)) # [ens x lat x lon x pc]
    
    for e in range(nens):
        startloop = time.time()
        
        # Select ensemble [Space x Time]
        varens = pslnao[e,:,:]  
        varglo = pslglo[e,:,:]
        
        # Get rid of NaN points
        okdata,knan,okpts    = proc.find_nan(varens,1)
        okdatap,knanp,okptsp = proc.find_nan(varglo,1)
        
        #% Perform EOF -------------
        _,pcs,varexpall[e,:]= proc.eof_simple(okdata,N_mode,1)
        
        
        # Standardize pc before regression along the time dimension
        pcstd = np.squeeze(pcs/np.std(pcs,0))
        
        #Loop for each mode... (NOte, can vectorize this to speed it up,.)
        psleofs = np.ones((192,288,3))
        for pcn in range(N_mode):

            # Regress back to SLP 
            eofpatokp,_ = proc.regress_2d(pcstd[:,pcn],okdatap)

            # Reshape regression pattern and put back (for psl)
            eofpatp = np.ones((192*288))*np.nan
            eofpatp[okptsp] = eofpatokp
            eofpatp = np.reshape(eofpatp,(192,288))

            # ----- Check Sign start -----
            if pcn == 0:
                # Check the signs and flip if necessary
                # EOF 1 only (need to figure out the other EOFs..)
                # First check using reykjavik
                rboxlon = np.where((lon >= lonReykjavik-tol) & (lon <= lonReykjavik+tol))[0]
                rboxlat = np.where((lat >= latReykjavik-tol) & (lat <= latReykjavik+tol))[0]
            
                chksum = np.nansum(eofpatp[rboxlat[:,None],rboxlon[None,:]],(0,1))
                if chksum > 0:
                    #print("\t Flipping sign based on Reykjavik, Ens%i" % (e+1))
                    eofpatp *= -1
                    pcs[:,pcn] *= -1


                # Double Check with Lisbon
                lboxlon = np.where((lon >= lonLisbon-tol) & (lon <= lonLisbon+tol))[0]
                lboxlat = np.where((lat >= latLisbon-tol) & (lat <= latLisbon+tol))[0]
                
                chksum = np.nansum(eofpatp[lboxlat[:,None],lboxlon[None,:]],(0,1))
                if chksum < 0:
                    #print("\t Flipping sign based on Lisbon,Ens%i" % (e+1))
                    eofpatp *= -1
                    pcs[:,pcn] *= -1
                
            # Check for EAP Pattern and Flip (EO#F2 and EOF3)
            elif (pcn == 1) | (pcn == 2):
                
                rsum = proc.sel_region(eofpatp.T,lon,lat,eapbox,reg_sum=1)
                if rsum > 0:
                    #print("\t Flipping sign based on EAP, PC%i Ens%i" % (pcn+1,e+1))
                    eofpatp    *= -1
                    pcs[:,pcn] *= -1
             
            # ----- Check Sign end -----
            
            psleofs[:,:,pcn] = eofpatp
        # End PC loop
        #% Assign to outside variable -------------
        eofall[e,:,:,:]     = psleofs           # ENS x Lon x Lat x PC
        pcall[e,:,:]        = np.squeeze(pcs)   # ENS x Yr x PC
        print("\rCalculated EOFs for ens %02d in %.2fs" % (e+1,time.time()-startloop),end="\r",flush=True)
        # End ensemble loop

# Calculate Monthly EOF
else:
    
    # Preallocate [ens x mon x year]
    pcall      = np.zeros((nens,12,nyr,N_mode))     # [ens x mon x year x pc]
    varexpall  = np.zeros((nens,12,N_mode))         # [ens x mon x pc]
    eofall     = np.zeros((nens,12,192,288,N_mode)) # [ens x mon x lat x lon x pc]
    
    for e in range(nens):
        startloop = time.time()
        for m in range(12):
            # [Space x Time]
            varens = pslnao[e,:,:,m]
            varglo = pslglo[e,:,:,m]
             
             # Get rid of NaN points
            okdata,knan,okpts    = proc.find_nan(varens,1)
            okdatap,knanp,okptsp = proc.find_nan(varglo,1)
             
            #% Perform EOF -------------
            _,pcs,varexpall[e,m,:]= proc.eof_simple(okdata,N_mode,1)
            
            # Standardize pc before regression along the time dimension
            pcstd = np.squeeze(pcs/np.std(pcs,0))
            
            #Loop for each mode... (NOte, can vectorize this to speed it up,.)
            psleofs = np.ones((192,288,3))
            
            for pcn in range(N_mode):

                # Regress back to SLP 
                eofpatokp,_ = proc.regress_2d(pcstd[:,pcn],okdatap)

                # Reshape regression pattern and put back (for psl)
                eofpatp = np.ones((192*288))*np.nan
                eofpatp[okptsp] = eofpatokp
                eofpatp = np.reshape(eofpatp,(192,288))
                
                # ----- Check Sign start -----
                if pcn == 0:
                
                    # Check the signs and flip if necessary
                    # EOF 1 only (need to figure out the other EOFs..)
                    # First check using reykjavik
                    rboxlon = np.where((lon >= lonReykjavik-tol) & (lon <= lonReykjavik+tol))[0]
                    rboxlat = np.where((lat >= latReykjavik-tol) & (lat <= latReykjavik+tol))[0]
                
                    chksum = np.nansum(eofpatp[rboxlat[:,None],rboxlon[None,:]],(0,1))
                    if chksum > 0:
                        #print("\t Flipping sign based on Reykjavik, Month %i Ens%i" % (m+1,e+1))
                        eofpatp *= -1
                        pcs[:,pcn] *= -1


                    # Double Check with Lisbon
                    lboxlon = np.where((lon >= lonLisbon-tol) & (lon <= lonLisbon+tol))[0]
                    lboxlat = np.where((lat >= latLisbon-tol) & (lat <= latLisbon+tol))[0]
             
                    chksum = np.nansum(eofpatp[lboxlat[:,None],lboxlon[None,:]],(0,1))
                    if chksum < 0:
                        #print("\t Flipping sign based on Lisbon, Month %i Ens%i" % (m+1,e+1))
                        eofpatp *= -1
                        pcs[:,pcn] *= -1
                
                # Check for EAP Pattern and Flip (EOF2 and EOF3)
                elif (pcn == 1) | (pcn == 2):
                    
                    rsum = proc.sel_region(eofpatp.T,lon,lat,eapbox,reg_sum=1)
                    if rsum > 0:
                        #print("\t Flipping sign based on EAP, PC%i onth %i Ens%i" % (pcn+1,m+1,e+1))
                        eofpatp *= -1
                        pcs[:,pcn] *= -1
                 
                # ----- Check Sign end -----
                psleofs[:,:,pcn] = eofpatp
            # End PC loop
            #% Assign to outside variable -------------
            eofall[e,m,:,:,:]     = psleofs           # ENS x Mon x Lon x Lat x PC
            pcall[e,m,:,:]        = np.squeeze(pcs)   # ENS x Mon x Yr x PC
            print("\rCalculated EOFs for month %02d of ens %02d in %.2fs" % (m+1,e+1,time.time()-startloop),end="\r",flush=True)
        # End Month Loop
    # End Ensemble Loop






#%% Regress NHFLX Data

# Load NHFLX Data
lstart = time.time()
if naotype == 0:
    dsw = scm.cut_NAOregion(ds,mons=[12,1,2,3])
    nhflx = dsw.NHFLX.values * -1  # Flip so that things are positive downwards   
else:
    nhflx = dsall.NHFLX.values * -1 # Flip so that things are positive downwards
# Move ensemble dimension to the front
if nhflx.shape[0] != 42:
    ensdim = nhflx.shape.index(nens)
    dimsbefore = np.arange(0,ensdim,1)
    dimsafter  = np.arange(ensdim+1,len(nhflx.shape),1)
    nhflx = nhflx.transpose(np.concatenate([ensdim,dimsbefore,dimsafter]))
    print("Warning: Moving ensemble to first dimension")

# Separate out the month and year dimensions and combine lat/lon, then transpose to [ens x space x yr x mon]
nhflx = nhflx.reshape(nens,nyr,12,nlat*nlon).transpose(0,3,1,2)  

    
# Perform regression based on the 
if naotype == 0:
    
    # Take djf mean
    nhflx = nhflx[:,:,:,djfm].mean(3)
    
    # Preallocate
    flxpattern = np.zeros((nens,192*288,N_mode))
    
    # Loop for each pc
    for n in range(N_mode):
        for e in range(nens):
            
            # Get PC # Time
            pcin = pcall[e,:,n]
            
            # Get variable # Space x Time
            varin = nhflx[e,:,:]
            
            # Perform regression
            flxpattern[e,:,n],_ = proc.regress_2d(pcin,varin)
            
            msg = '\rCompleted Regression for PC %02i/%02i, ENS %02i/%02i' % (n+1,N_mode,e+1,nens)
            print(msg,end="\r",flush=True)


    # Reshape variable [pc, ens, lat, lon]
    flxpattern = np.reshape(flxpattern,(nens,nlat,nlon,N_mode))
    
else:
    
    #Preallocate 
    flxpattern = np.zeros((nens,12,192*288,N_mode))
    
    # Loop for each pc
    for n in range(N_mode):
        for e in range(nens):
            for m in range(12):
                
                # Get PC
                if naotype == 1: # No month dimension for DJFM NAO
                    pcin = pcall[e,:,n]
                else:
                    pcin = pcall[e,m,:,n]
                
                # Get variable
                varin = nhflx[e,:,:,m]
                
                # Perform regression
                flxpattern[e,m,:,n],_ = proc.regress_2d(pcin,varin)
                msg = '\rCompleted Regression for Mon %02d/12 PC %02i/%02i, ENS %02i/%02i' % (m+1,n+1,N_mode,e+1,nens)
                print(msg,end="\r",flush=True)
    flxpattern = flxpattern.reshape(nens,12,nlat,nlon,N_mode)
print("Completed NHFLX Regression in %.2f" % (time.time()-lstart))

# Save output
savename = "%sNAO_Monthly_Regression_PC123_naotype%i.npz"%(outpath,naotype) 
np.savez(savename,pcall=pcall,flxpattern=flxpattern,psleofall=eofall,varexpall=varexpall)
print("saved to %s. Script complete in %.2f" %(savename,time.time()-allstart))