#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

prep_MLD


Script to calculate MLD variables and month of previous entrainment


"""
import time
import numpy as np
import xarray as xr
from scipy.io import loadmat
#%% Functions

def find_kprev(h):
    """
    Script to find the previous, entraining month via linear interpolation
    
    """
    # Preallocate
    kprev = np.zeros(12)
    hout = np.zeros(12)
    
    # Month Array
    monthx = np.arange(1,13,1)  
    
    # Determine if the mixed layer is deepening (true) or shoaling (false)--
    dz = h / np.roll(h,1) 
    dz = dz > 1
    #dz = dz.values
    
        
        
        
    for m in monthx:
        
        
        # Quick Indexing Fixes ------------------
        im = m-1 # Month Index (Pythonic)
        m0 = m-1 # Previous month
        im0 = m0-1 # M-1 Index
        
        # Fix settings for january
        if m0 < 1:
            m0 = 12
            im0 = m0-1
        
        # Set values for minimun/maximum -----------------------------------------
        if im == h.argmax() or im== h.argmin():
            #print("Ignoring %i, max/min" % m)
            kprev[im] = m
            hout[im] = h[im]
            continue
        
    
        
        # Ignore detrainment months
        if dz[im] == False:
            #print("Ignoring %i, shoaling month" % m)
            continue
        
        # For all other entraining months.., search backwards
        findmld = h[im]  # Target MLD   
        hdiff = h - findmld
          
        searchflag = 0
        ifindm = im0
        
        
        while searchflag == 0:
                
            hfind= hdiff[ifindm]
            
            # For the first month greater than the target MLD,
            # grab this value and the value before it
            if hfind > 0:
                # Set searchflag to 1 to prepare for exit
                searchflag = 1
                
                # record MLD values
                h_before = h[ifindm+1]
                h_after  = h[ifindm]
                m_before = monthx[ifindm+1]
                m_after =  monthx[ifindm]
                
                # For months between Dec/Jan, assign value between 0 and 1
                if ifindm < 0 and ifindm == -1:
                    m_after = ifindm+1
                
                # For even more negative indices
                
                #print("Found kprev for month %i it is %f!" % (m,np.interp(findmld,[h_before,h_after],[m_before,m_after])))
                kprev[im] = np.interp(findmld,[h_before,h_after],[m_before,m_after])
                hout[im] = findmld
            
            # Go back one month
            ifindm -= 1
    
    return kprev, hout


#%% User Edited Settings

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'

# Mixed Layer Variable Name
mldnc = "HMXL_HTR_clim.nc"

stol  = 0.75 # Search tolerance for curivilinear grid (degrees) <Note there is sensitivity to this>....

# Target Lon/Lat
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])
lonsize = lon.shape[0]
latsize = lat.shape[0]
damping = loaddamp['ensavg']


# Out Names
hout     = "HMXL_hclim.npy"
kprevout = "HMXL_kprev.npy"

#%% Script Start

# Load in Data
ds = xr.open_dataset(datpath+mldnc)

# Take ensemble mean
h_ensmean = ds.HMXL.mean('ensemble')/100

# This portion of the data unfortunately has an ensemble dimension
tlon = ds.TLONG.mean('ensemble')
tlat = ds.TLAT.mean('ensemble')

# Preallocate and specify search tolerance 
hclim = np.zeros((lonsize,latsize,12),dtype=float)
kprev = np.zeros((lonsize,latsize,12),dtype=float)

# Assign new names to coordinates
h_ensmean = h_ensmean.assign_coords(TLONG=tlon)
h_ensmean = h_ensmean.assign_coords(TLAT=tlat)

def getpt_pop(lonf,latf,ds,searchdeg=0.5,returnarray=1):
    """ Quick script to read in a xr.Dataset (ds)
        and return the value for the point specified by lonf,latf
        
    
    """
    
    
    # Do same for curivilinear grid
    if lonf < 0:
        lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates
    else:
        lonfc = lonf
        
    # Find the specified point on curvilinear grid and average values
    selectmld = ds.where((lonfc-searchdeg < ds.TLONG) & (ds.TLONG < lonfc+searchdeg)
                    & (latf-searchdeg < ds.TLAT) & (ds.TLAT < latf+searchdeg),drop=True)
    
    pmean = selectmld.mean(('nlon','nlat'))
    
    if returnarray ==1:
        h = np.squeeze(pmean.values)
        return h
    else:
        return pmean
    
#%% Old calculation method using xarray

start = time.time()
icount= 0
for o in range(0,lonsize):
    # Get Longitude Value
    lonf = lon[o]
    
    # Convert to degrees Easth
    if lonf < 0:
        lonf = lonf + 360
    
    for a in range(0,latsize):
        
        
        # Get latitude indices
        latf = lat[a]
        
        
        # Skip if the point is land
        if np.isnan(np.mean(damping[o,a,:])):
            msg = "Land Point @ lon %f lat %f" % (lonf,latf)
            #print(msg)
            hclim[o,a,:] = np.ones(12)*np.nan
            kprev[o,a,:] = np.ones(12)*np.nan
            continue
        
        
        # Get point
        hclim[o,a,:] = getpt_pop(lonf,latf,h_ensmean,searchdeg=stol)
        

        
        # Find Entraining Months
        kprev[o,a,:],_ = find_kprev(hclim[o,a,:])
        icount +=1
        print("Completed %i of %i" % (icount,lonsize*latsize))
        
        
        
print("Finished in %f seconds" % (time.time()-start))
        


#%% # New Calculation method (Seems to be faster)

tlon = tlon.values
tlat = tlat.values
invar = h_ensmean.transpose(1,2,0)


def getpt_pop_array(lonf,latf,invar,tlon,tlat,searchdeg=0.75,printfind=True):
    
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
        print("Returning NaN because no points were found for LAT%.1f LON%.1f"%(latf,lonf))
        return np.nan
        exit
    
    
    # Locate points on variable
    if invar.shape[:2] != tlon.shape:
        print("Warning, dimensions do not line up. Make sure invar is Lat x Lon x Otherdims")
        exit
    
    return invar[latid,lonid,:].mean(0) # Take mean along first dimension
    
    


start = time.time()
icount= 0
for o in range(0,lonsize):
    # Get Longitude Value
    lonf = lon[o]
    
    # Convert to degrees Easth
    if lonf < 0:
        lonf = lonf + 360
    
    for a in range(0,latsize):
        
        
        # Get latitude indices
        latf = lat[a]
        
        # Get point
        value = getpt_pop_array(lonf,latf,invar,tlon,tlat,searchdeg=stol,printfind=False)
        if np.any(np.isnan(value)):
            msg = "Land Point @ lon %f lat %f" % (lonf,latf)
            hclim[o,a,:] = np.ones(12)*np.nan
            kprev[o,a,:] = np.ones(12)*np.nan
            
        else:
            hclim[o,a,:] = value.copy()
            # Find Entraining Months
            kprev[o,a,:],_ = find_kprev(hclim[o,a,:])
        icount +=1
        print("Completed %i of %i" % (icount,lonsize*latsize))
        
        
print("Finished in %f seconds" % (time.time()-start))  
    
    














#%%
np.save(datpath+hout,hclim)
np.save(datpath+kprevout,kprev)




