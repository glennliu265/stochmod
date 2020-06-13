#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:32:33 2020

@author: gliu
"""

import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
# Script to find the last MLD Entrainment depths



## Month to find the previous month for Td
# Depdendencies: numpy as np, xarray as xr
# Input: h (dataarray of mixed layer depths)
# Output: (1) kprev - month last time this depth was achieved 
# (2) hout - associated mixed layer pdeths
def find_kprev(h):
    
    # Preallocate
    kprev = np.zeros(12)
    hout = np.zeros(12)
    
    # Month Array
    monthx = np.arange(1,13,1)  
    
    # Determine if the mixed layer is deepening (true) or shoaling (false)--
    dz = h / np.roll(h,1) 
    dz = dz > 1
    dz = dz.values
    
        
        
        
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
        if h[im] == h.max() or h[im]== h.min():
            print("Ignoring %i, max/min" % m)
            kprev[im] = m
            hout[im] = h[im]
            continue
        
    
        
        # Ignore detrainment months
        if dz[im] == False:
            print("Ignoring %i, shoaling month" % m)
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
                
                print("Found kprev for month %i it is %f!" % (m,np.interp(findmld,[h_before,h_after],[m_before,m_after])))
                kprev[im] = np.interp(findmld,[h_before,h_after],[m_before,m_after])
                hout[im] = findmld
            
            # Go back one month
            ifindm -= 1
    
    return kprev, hout



# Set Point and month
lonf    = -0
latf    = 7


# Set up title naming conventions
loc_figtitle = "Lon: %i Lat: %i" % (lonf,latf)
if lonf < 0:
    lonstr = lonf + 360
loc_fname    = "Lon%03d_Lat%03d"  % (lonstr,latf)


#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath = projpath + '01_Data/'

# Load Mixed layer variables
mldnc = "HMXL_HTR_clim.nc"
ds = xr.open_dataset(datpath+mldnc)


if lonf < 0:
    lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates

# Find the specified point on curvilinear grid and average values
selectmld = ds.where((lonfc-0.5 < ds.TLONG) & (ds.TLONG < lonfc+0.5)
                    & (latf-0.5 < ds.TLAT) & (ds.TLAT < latf+0.5),drop=True)
# Select accordingly 
ensmean = selectmld.mean(('ensemble','nlon','nlat'))/100
h = np.squeeze(ensmean.to_array())



# -----------------------
# # Start Calculations
# -----------------------
# # Preallocate
# kprev = np.zeros(12)
# hout = np.zeros(12)

# # Month Array
# monthx = np.arange(1,13,1)      

# # Find max
# maxh = h.max()
# minh = h.min()

# # Determine if the mixed layer is deepening (true) or shoaling (false)--
# dz = h / np.roll(h,1) 
# dz = dz > 1
# dz = dz.values

# for m in monthx:

kprev,hout=find_kprev(h)



    
    
    # Quick Indexing Fixes ------------------
    im = m-1 # Month Index (Pythonic)
    m0 = m-1 # Previous month
    im0 = m0-1 # M-1 Index
    
    # Fix settings for january
    if m0 < 1:
        m0 = 12
        im0 = m0-1
    
    # Set values for minimun/maximum -----------------------------------------
    if h[im] == h.max() or h[im]== h.min():
        print("Ignoring %i, max/min" % m)
        kprev[im] = m
        hout[im] = h[im]
        continue
    

    
    # Ignore detrainment months
    if dz[im] == False:
        print("Ignoring %i, shoaling month" % m)
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
            
            print("Found kprev for month %i it is %f!" % (m,np.interp(findmld,[h_before,h_after],[m_before,m_after])))
            kprev[im] = np.interp(findmld,[h_before,h_after],[m_before,m_after])
            hout[im] = findmld
        
        # Go back one month
        ifindm -= 1

# *********************************************************************
# Make array for plotting (to see connections between entrainment lines)
# *********************************************************************
connex = np.zeros((12,2,2))

connex =[]
for pt in monthx:
    im = pt-1
    
    
    #onth x monthloc x mldval
    
    if kprev[im] != 0:
        
        x = (pt,kprev[im])
        y = (h.item(im),hout[im])
        
        connex.append((x,y))
        
        
        
        # connex[im,0,0] = im# Month 
        # connex[im,0,1] = h[im]
        
        # connex[im,1,0] = kprev[im]
        # connex[im,1,1] = hout[im]

foundmon = kprev[kprev!=0]
foundmld = hout[kprev!=0]

# Add extramonths
plotmonthx = np.arange(0,14,1)
plotmld    = xr.concat([h[-1],h,h[0]],dim="month")


# Create masked array to highlight entrainment period
#dz = dz.values
# dzm = ma.array(dz)

# maxid = np.argmax(h.values,0)
# minid = np.argmin(h.vakues,0)
# dzm[mask_start] = ma.masked

# *********************
# Sample Plot
# ********************* 

f1 = plt.figure()
ax = plt.axes()
#sns.set('paper','whitegrid','bright')
ax.plot(plotmonthx,plotmld,color='k',label='MLD Cycle')
#ax.plot(monthx[dz],h[dzm],color='b')


for m in range(0,len(connex)):
    ax.plot(connex[m][0],connex[m][1])

ax.scatter(foundmon,foundmld,marker="x")

ax.set(xlabel='Month',
       ylabel='Mixed Layer Depth',
       xlim=(0,12),
       title="Mixed Layer Depth Seasonal Cycle \n" + loc_figtitle
       )

ax.set_xticks(range(1,14,1))


    

    
    
    