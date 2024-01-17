#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Stochastic Model Base Code...
Created on Mon Jul 27 11:49:57 2020

@author: gliu
"""

# %%Dependencies
import numpy as np
import xarray as xr
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats,signal
from tqdm import tqdm
import glob

import sys
stormtrack = 1
if stormtrack == 0:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
elif stormtrack == 1:
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc
import time
import yo_box as ybx
import tbx
import pandas as pd

#%% Helper Functions/Utilities

def calc_Td(t,index,values,prevmon=False,debug=False):
    """
    Calculate entraining temperature (Td) using linear interpolation.
    
    Parameters
    ----------
    t : INT
        Timestep (in months, where t(0) = Jan)
    index : ARRAY [12]
        Time of entraining month
    values : ARRAY [t]
        Array of values to interpolate
    prevmon : BOOL, optional
        Set to True to calculate Td0. The default is False.
    debug : TYPE, optional
        Set to True to print outputs. The default is False.
    Returns
    -------
    Td: INT if prevmon=False, Tuple (Td1,Td0) if prevmon=False
    """
    
    # Initialize month array
    months = []
    m1  = (t+1)%12
    #m1=t%12
    if m1 == 0:
        m1 = 12
    months.append(m1)
    
    # Option to include previous month
    if prevmon:
        m0 = m1-1
        if m0==0:
            m0 = 12
        months.append(m0)
    if debug:
        print("t=%i, Month is %i"%(t,m1))

    # Loop for each month
    Td = []
    mcnts = [1,0] # Amount to add to get the index
    for mcount,m in enumerate(months):
        mcnt = mcnts[mcount]
        if debug:
            print("\tCalculating Td for m=%i"%m)
        
        # # For m0, check if index=0 and skip if so (first entraining month)
        # if (len(months)>1) and (m==months[-1]):
        #     if index[m-1] == 0:
        #         return 0,0
        #         # Td.append(Td[0])
        #         # if debug:
        #         #     print("\t\tSince m0=%i, or first entraining month, Td0=Td1" % m)
        #         # continue
        
        # Find # of months since the anomaly was formed
        k1m = (m1 - np.floor(index[m-1])) % 12
        if k1m == 0:
            k1m = 12
        
        # Get Index in t
        kp1 = int(t+mcnt - k1m)
        if debug:
            print("\t\tkprev is %.2f for month %i, or %i months ago at t=%i"% (index[m-1],m,k1m,kp1))
        
        # Retrieve value between 0 and 1
        kval = index[m-1]-np.floor(index[m-1])
        
        # Interpolate
        Td1 = np.interp(kval,[0,1],[values[kp1],values[kp1+1]])
        if debug:
            print("\t\tsince %.2f is between %i and %i... "%(kval,0,1))
            print("\t\t\tTd is interpolated to %.2f, between %.2f and %.2f"%(Td1,values[kp1],values[kp1+1]))
        Td.append(Td1)
    
    if prevmon: # return Td=[Td1,Td0]
        return Td
    else: # Just return Td1
        return Td[0]

def cut_NAOregion(ds,bbox=[0,360,-90,90],mons=None,lonname='lon',latname='lat'):
    """
    Prepares input DataArray for NAO Calculation by cutting region and taking
    an optional average over specified range of months
    Dependencies
        xarray as xr
        numpy as np
    Inputs
        1) ds [DataArray]                    - Input DataArray
        2) bbox [Array: lonW,lonE,latS,latN] - Bounding Boxes
        OPTIONAL
        3) mons [Array: months]              - Months to include in averaging
                ex. For DJFM average, mons = [12,1,2,3]
                Default --> No Monthly Average
        4) lonname [str] - name of longitude in dataarray
        5) latname [str] - name of latitude in dataarray
    Outputs
        ds [DataArray] - Output dataarray with time mean and region
    """
    # Average over specified months
    if mons is not None:
        # Take the DJFM Mean
        season = ds.sel(time=np.in1d(ds['time.month'],mons))
        dsw = season.groupby('time.year').mean('time')
    else:
        dsw = ds.copy()
    
    # Cut to specified NAO region
    lonW,lonE,latS,latN = bbox
    
    # Convert to degrees East
    if lonW < 0:
        lonW += 360
    if lonE < 0:
        lonE += 360
    
    # Select North Atlantic Region for NAO Calculation...
    if lonW > lonE: # Cases crossing the prime meridian
        #print("Crossing Prime Meridian!")
        dsna = dsw.where((dsw[lonname]>=lonW) | (dsw[lonname]<=lonE),drop=True).sel(lat=slice(latS,latN))
    else:
        dsna = dsw.sel(lon=slice(lonW,lonE),lat=slice(latS,latN))
        
    return dsna

def make_naoforcing(NAOF,randts,fscale,nyr):
    """
    Makes forcing timeseries, given NAO Forcing Pattern for 3 different
    treatments of MLD (NAOF), a whiite noise time series, and an scaling 
    parameter.
    
    Inputs:
        1) randts [Array] - white noise timeseries varying between -1 and 1
        3) NAOF   [Array] - NAO forcing [Lon x Lat x Mon] in Watts/m2
        4) fscale         - multiplier to scale white noise forcing\
        5) nyr    [int]   - Number of years to tile the forcing
    Dependencies: 
        1) 
    
    Outputs
        1) F{} [DICT] - [lon x lat x t_end] - Full forcing time series
        2) Fseas{} [DICT] - [lon x lat x 12 or 1] - seasonal forcing time series
    """
    
    # Make dictionary
    F = {}
    Fseas = {}

    
    
    # Check if there is a month component 
    if len(NAOF[0]) > 2:
        
        # Fixed MLD
        tilecount = int(12/NAOF[0].shape[2]*nyr)
        F[0] = np.tile(NAOF[0],tilecount) *randts[None,None,:] * fscale 
            
        # Max MLD
        tilecount = int(12/NAOF[1].shape[2]*nyr)
        F[1] = np.tile(NAOF[1],tilecount) *randts[None,None,:] * fscale 
        
        Fseas[0] = NAOF[0] * fscale 
        Fseas[1] = NAOF[1] * fscale 
        
    else:
        
        # Fixed MLD 
        F[0] = NAOF[0][:,:,None] *randts[None,None,:] * fscale
        Fseas[0] = NAOF[0][:,:,None] * fscale
        
        # Max MLD
        F[0] = NAOF[1][:,:,None] *randts[None,None,:] * fscale
        Fseas[1] = NAOF[1][:,:,None] * fscale
    
    # Seasonally varying mld...
    F[2]     = np.tile(NAOF[2],nyr) * randts[None,None,:] * fscale 
    Fseas[2] =  NAOF[2][:,:,:] * fscale 
    
            
    return F,Fseas

def set_stochparams(h,damping,dt,ND=True,rho=1000,cp0=4218,hfix=50,usemax=False,hmean=None):
    """
    Given MLD and Heat Flux Feedback, Calculate Parameters
    
    Inputs:
        1) h: Array [Lon x Lat x Mon]
            Mixed Layer depths (climatological)
        2) damping: Array [Lon x Lat x Mon]
            Heat Flux Feedbacks (W/m2)
        3) dt: INT
            Model timestep in seconds
        4) ND: Boolean
            Set to 1 to process 2D data, rather than data at 1 point
        Optional Arguments:
            rho   - density of water [kg/m3]
            cp0   - specific heat of water [J/(K*kg)]
            hfix  - fixed mixed layer depth [m]
            usemax - use seasonal maximum MLD
    
    
    Outputs:
        1) lbd: DICT [hvarmode] [Lon x Lat x Mon]
            Dictionary of damping values for each MLD treatment 
        2) lbd_entr: Array [Lon x Lat x Mon]
            Damping for entraining model
        3) FAC: Array [Lon x Lat x Mon]
            Seasonal Reduction Factor
        4) Beta: Array [Lon x Lat x Mon]
            Entraining Term
    
    """    
    
    # Calculate Beta
    if ND == True:
        beta = np.log( h / np.roll(h,1,axis=2) ) # Roll along time axis
        
        # Find Maximum MLD during the year
        hmax = np.nanmax(np.abs(h),axis=2)
        hmax = hmax[:,:,None]
        
        ## Find Mean MLD during the year
        if hmean is None:
            hmean = np.nanmean(h,axis=2)
            hmean = hmean[:,:,None]
        else:
            print("Using Slab MLD %f = 54.61" % hmean[56,74])
        
    else:
        beta = np.log( h / np.roll(h,1,axis=0) )
        
        # Find Maximum MLD during the year
        hmax = np.nanmax(np.abs(h))
        if hmean is None:
            hmean = np.nanmean(h)
    
    # Set non-entraining months to zero
    beta[beta<0] = 0
    
    # Replace Nans with Zeros in beta
    #beta = np.nan_to_num(beta)
    
    # Preallocate lambda variable 
    lbd = {}
    
    # Fixed MLD
    lbd[0] = damping / (rho*cp0*hfix) * dt
    
    # Mean MLD
    lbd[1] = damping / (rho*cp0*hmean) * dt
    
    if usemax:
        # Maximum MLD
        lbd[1] = damping / (rho*cp0*hmax) * dt
    
    # Seasonal MLD
    lbd[2] = damping / (rho*cp0*h) * dt
    
    # Calculate Damping (with entrainment)
    lbd_entr = np.copy(lbd[2]) + beta
    lbd[3] = lbd_entr.copy()
    
    
    # Compute reduction factor
    FAC = {}  
    for i in range(4):
        fac = (1-np.exp(-lbd[i]))/lbd[i]
        fac = np.nan_to_num((1-np.exp(-lbd[i]))/lbd[i])
        fac[fac==0] = 1 # Change all zero FAC values to 1
        FAC[i] = fac.copy()
    
    return lbd,lbd_entr,FAC,beta


def find_kprev(h,debug=False):
    """
    Script to find the month of detrainment, given a seasonal
    cycle of mixed layer depths

    Parameters
    ----------
    h : ARRAY [12,]
        Seasonal Mixed layer depth cycle

    Returns
    -------
    kprev : [12,]
        Detrainment Month
    hout : [12,]
        Output for plotting(?)

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
        if im == h.argmax(): #or im== h.argmin():
            if debug:
                print("Ignoring %i, max/min" % m)
            kprev[im] = m
            hout[im] = h[im]
            continue

        # Ignore detrainment months
        if dz[im] == False:
            if debug:
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
                if debug:
                    print("Found kprev for month %i it is %f!" % (m,np.interp(findmld,[h_before,h_after],[m_before,m_after])))
                kprev[im] = np.interp(findmld,[h_before,h_after],[m_before,m_after])
                hout[im] = findmld
            
            # Go back one month
            ifindm -= 1
    
    return kprev, hout

def calc_kprev_lin(h,entrain1=-1,entrain0=0):
    """
    Estimate detrainment indices given a timeseries of mixed-layer
    depths. Uses/Assumptions:
        - Linear interpolation
        - Forward direction for entrainment/detrainment (h[t], h[t+1])
        - For last value, assumes t+1 is the first value (periodic)
    
    Inputs
    ------
        h : ARRAY [time]
            Timeseries of mixed layer depth
        
        --- Optional ---
        
        entrain1 : Numeric (Default = -1)
            Placeholder value for first time MLD reaches a given depth
        entrain0 : Numeric (Default = 0)
            Placeholder value for detraining months
    Output
    ------
        kprev : ARRAY [time]
            Indices where detrainment occurred    
    """
    # Preallocate, get dimensions
    ntime = h.shape[0]
    kprev = np.zeros(ntime)*np.nan
    
    # Looping for each step, get index of previous step
    for t in range(ntime):
        
        # Wrap around for end value
        if t >= (ntime-1):
            dt = h[t]/h[0]
        else: # Forward step comparison 
            dt = h[t]/h[t+1]
        
        # Skip points where the mixed layer is detraining or unchanging
        if dt >= 1:
            kprev[t] = entrain0
            continue
        
        
        # Find the last index where h had the same value
        hdiff = h - h[t]
        hdiff = hdiff[:t] # Restrict to values before current timestep
        kgreat = np.where(hdiff > 0)[0] # Find values deeper than current MLD
        if len(kgreat) == 0:  # If no values are found, assume first time entraining to this depth
            kprev[t] = entrain1
            continue
        else:
            kd  = kgreat[-1] # Take index of most recent value
            # Linear interpolate to approximate index
            kfind = np.interp(h[t],[h[kd],h[kd+1]][::-1],[kd,kd+1][::-1])
            
            if kfind == float(t):
                kprev[t] = entrain1
            else:
                kprev[t] = kfind
        # End Loop
    return kprev

def convert_NAO(hclim,naopattern,dt,rho=1000,cp0=4218,hfix=50,usemax=False,hmean=None):
    """
    Convert NAO forcing pattern [naopattern] from (W/m2) to (degC/S) 
    given seasonal MLD (hclim)
    
    Inputs:
        1) hclim [3d Array]      - climatological MLD [Mons]
        2) NAOF [2d or 3d Array] - NAO forcing [Lon x Lat] in Watts/m2
        3) dt                    - timestep in seconds
        4) rho                   - Density of water [kg/m3]
        5) cp0                   - Specific Heat of water [J/(K*kg)]
        6) hfix                  - Fixed Mixed layer Depth
        7) usemax (optional)     - Set to True to use max seasonal MLD
        8) hmean (optional) [3-d array] - MLD vaue to use
    Output:
        1) NAOF [dict]    - Dictionary of arrays [lon x lat x mon], where 
            0 = fixed MLD
            1 = maximum MLD
            2 = seasonal MLD
    
    """
    # Check if forcing pattern is 3D
    patshape = naopattern.shape
    if len(patshape) != 3: 
        naopattern = naopattern[:,:,None]
    
    # Set up MLD[lon x lat x mon x hvarmode]
    mld = np.ones((patshape[0],patshape[1],12,3))
    mld[:,:,:,0]  *= hfix # Fixed MLD
    

    if hmean is None:
        mld[:,:,:,1]  = np.tile(hclim.mean(2)[:,:,None],12) # Mean MLD
        if usemax:
            mld[:,:,:,1]  = np.tile(hclim.max(2)[:,:,None],12) # Max MLD
    else:
        mld[:,:,:,1]  = np.tile(hmean,12) # Mean MLD (Slab)
        print("Using Slab MLD %f = 54.61" % mld[56,74,0,1])
    
    mld[:,:,:,2]  = hclim.copy() # Clim MLD
    
    # Convert NAO to correct units...
    NAOF = {}
    for i in range(3):
        
        hchoose = mld[:,:,:,i]
        NAOF[i] = naopattern * dt / cp0 / rho / hchoose
    
    return NAOF

def get_data(pointmode,query,lat,lon,damping,mld,kprev,F):
    """
    Wrapper to return data based on pointmode
    
    Parameters
    ----------
    pointmode : INT
        0 = Return variables as is
        1 = Return variables at a point
        2 = Return area-averaged variables
    query : ARRAY
        [lon,lat] if pointmode is 1
        [lonW,lonE,latS,latN] if pointmode is 2
    lat : ARRAY
        latitude coordinates
    lon : ARRAY
        longitude coordinates
    damping : ARRAY [lat,lon,month]
        heat flux feedback (W/m2)
    mld : ARRAY [lat,lon,month]
        mixed layer depth (m)
    kprev : ARRAY [lat,lon,month]
        detrainment month
    F : ARRAY [lat,lon,month]
        forcing pattern
    
    Returns
    -------
        1. point indices ARRAY
        2. damping at point ARRAY
        3. mld at point ARRAY
        4. kprev at point ARRAY
        5. forcing at point ARRAY
    """
    
    if pointmode == 1:
        o,a = proc.find_latlon(query[0],query[1],lon,lat)
        return [o,a],damping[o,a],mld[o,a],kprev[o,a],F[o,a]
    else:
        inparams = [damping,mld,kprev,F]
        outparams = []
        for param in inparams:
            
            if pointmode == 2:
                var = proc.sel_region(param,lon,lat,query,reg_avg=1)
                lonr=[0,] # Dummy Variable
                latr=[0,] # Dummy Variable
            else:    
                var,lonr,latr = proc.sel_region()
            outparams.append(var)
        return np.concatenate([lonr,latr,outparams])


#%% Stochastic Model Code

"""
SST Stochastic Model, no Entrainment
Integrated with the forward method
assuming lambda at a constant monthly timestep

Dependencies: 
    - numpy as np

 Inputs
 1) t_end : timestep to integrate until (monthly)
 2) lbd   : seasonally varying decay term (lambda)
 3) T0    : Initial temperature
 4) F     : Forcing term
    
"""
def noentrain(t_end,lbd,T0,F,FAC,multFAC=1,debug=False):
    
    # Preallocate
    temp_ts = np.zeros(t_end)
    
    if debug:
        damp_ts = np.zeros(t_end)
        noise_ts = np.zeros(t_end)
        
    # Set value for first timestep
    #temp_ts[0] = T0 #"DEC"
    
    # Prepare the entrainment term
    explbd = np.exp(-lbd)
    #explbd[explbd==1] = 0
    
    if (multFAC == 1) & (F.shape[0] != FAC.shape[0]):
        F *= np.tile(FAC,int(t_end/12)) # Tile FAC and scale forcing
    
    # Loop for integration period (indexing convention from matlab)
    for t in range(t_end):
        
        # Get the month
        m = (t+1)%12 # First t+dt is Feb, t is therefore Jan
        #m = t%12 # First t+dt is Jan, t is therefor dec...
        if m == 0:
            m = 12
        #print("For t = %i month is %i"%(t,m))
        
        # Get Noise/Forcing Term (first step will be Jan...)
        # Isn't t-1 since t-1 = -1, which would be dec forcing..
        # UPDATE: So dec forcing generates Jan SST, etc...???
        noise_term = F[t]
        
        # Form the damping term with temp from previous timestep
        if t == 0:
            damp_term = explbd[m-1]*T0
        else:
            damp_term = explbd[m-1]*temp_ts[t-1]
        
        # Compute the temperature
        temp_ts[t] = damp_term + noise_term  
        
        # Save other variables
        if debug:
            noise_ts[t] = np.copy(noise_term)
            damp_ts[t]  = np.copy(damp_term) 
    if debug:
        return temp_ts,noise_ts,damp_ts
    return temp_ts

# Entrain Model (Single Point)
def entrain(t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC=1,debug=False,debugprint=False,
            Tdgrab=None,add_F=None,return_dict=False,Tdexp=None):
    
    """
    SST Stochastic Model, with Entrainment
    Integrated with the forward method at a single point
    assuming lambda at a constant monthly timestep
    If Tdgrab is given, appends to front for Td calculations, then removes.
    
    Parameters
    ----------
    t_end : INT
        Length of integration, in months
    lbd : ARRAY [12,]
        Heat Flux Feedback (degC/sec)
    T0 : INT
        Initial Temperature (degC)
    F : ARRAY [t_end,]
        Forcing term (white noise time series) (degC/sec)
    beta : ARRAY [12,]
        Entrainment term coefficient (log(h(t+1)/h(t)))
    h : ARRAY [12,]
        Mixed Layer Depth (meters)
    kprev : ARRAY [12,]
        Month of detrainment (calculated through find_kprev)
    FAC : ARRAY [12,]
        Integration Factor ((1-exp(-lbd))/lbd)
    multFAC : BOOL, optional
        Set to true to apply integration factor to forcing and entrain term. 
        The default is TRUE.
    debug : BOOL, optional
        Set to true to output each term separately
    debugprint : BOOL, optional
        Set to true to print messages at each timestep. The default is False.
    Td0 : Array of len[Tdgrab]
        Initial temperatures below the mixed layer from previous experiments to append
    add_F : ARRAY [tend,]
        Additional forcing to add directly intot he equation (multiple by integration factor if chosen)
    Tdexp : ARRAY [12]
        Decay time applied to Td detrained/formed at each given month (Td * exp(-Tdexp * kmonth)). Default is 0 (no decay_)
        
    
    Returns
    -------
    temp_ts : ARRAY [t_end,]
        Resultant Temperature timeseries
    if debug is True, returns the follow ARRAYs [t_end,]
        damp_ts    - Damping term
        noise_ts   - Noise term
        entrain_ts - Entrainment Term
        Td_ts      - Entraining Temperature
        
    """
    
    # Preallocate
    temp_ts = np.zeros(t_end) * T0
    
    # If Tdgrab is on, Grab the temperatures
    if Tdgrab is None:
        t0  = 0 # Start from 0
        Td0 = None
    else: # New length is t_end + len(Tdgrab)
        t0  = len(Tdgrab)
        Td0 = None # Set to this initially
        t_end += len(Tdgrab) # Append years to beginning of simulation
        temp_ts = np.concatenate([Tdgrab,temp_ts])
        
    if Tdexp is None:
        Tdexp = np.zeros(12)
        
        
    if debug:
        noise_ts   = np.zeros(t_end)
        damp_ts    = np.zeros(t_end)
        entrain_ts = np.zeros(t_end)
        Td_ts      = np.zeros(t_end)
        
    entrain_term = np.zeros(t_end)
    
    # Prepare the entrainment term
    explbd = np.exp(np.copy(-lbd))
    #explbd[explbd==1] = 0
    
    # Loop for integration period (indexing convention from matlab)
    for t in np.arange(t0,t_end,1):
        
        # Get the month (start from Jan, so +1)
        m  = (t+1)%12
        #m=t%12
        if m == 0:
            m = 12
        
        # --------------------------
        # Calculate entrainment term
        # --------------------------
        if (t<12) and (Tdgrab is None): # Start Entrainment term after first 12 months
            if debugprint:
                print("Skipping t=%i" % t)
            entrain_term = 0
        else: # Otherwise, 
            if beta[m-1] == 0: # For months with no entrainment
                entrain_term = 0
                Td0          = None # Reset Td0 term
                if debugprint:
                    print("No entrainment on month %i"%m)
                    print("--------------------\n")
            else:
                if (Td0 is None) & (h.argmin()==m-2) :# For first entraining month
                    Td1 = calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
                    Td0 = Td1 # Previous month does not have entrainment!
                if Td0 is None: # Calculate Td0 
                    Td1,Td0 = calc_Td(t,kprev,temp_ts,prevmon=True,debug=debugprint)
                else: # Use Td0 from last timestep
                    Td1 = calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
                
                # Implement decay to Td
                detrain_m1 = kprev[m-1]               # Month of Detrainment for that anomaly
                idt1       = int(np.floor(detrain_m1)) - 1 # Get indices for Td decay timescale (rounding down to nearest month)
                detrain_m0 = kprev[m-2]               # Repeat for previous month
                idt0       = int(np.floor(detrain_m0)) - 1
                Td0 = Td0 * np.exp(-Tdexp[idt0] * kprev[m-2])
                Td1 = Td1 * np.exp(-Tdexp[idt1] * kprev[m-1])
                
                # Compute Td (in future, could implement month-dependent decay..)
                Td = (Td1+Td0)/2
                if debugprint:
                    print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0)) 
                    print("--------------------\n")
                Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
                
                # Calculate entrainment term
                entrain_term = beta[m-1]*Td
        
        # ----------------------
        # Get Noise/Forcing Term
        # ----------------------
        noise_term = F[t]
        if add_F is not None:
            noise_term = F[t] + add_F[t]
        
        # ----------------------
        # Calculate damping term
        # ----------------------
        if t == 0:
            damp_term = explbd[m-1]*T0
        else:
            damp_term = explbd[m-1]*temp_ts[t-1]
        
        # ------------------------
        # Check Integration Factor
        # ------------------------
        if multFAC:
            integration_factor = FAC[m-1]
        else:
            integration_factor = 1
        
        # -----------------------
        # Compute the temperature
        # -----------------------
        temp_ts[t] = damp_term + (noise_term + entrain_term) * integration_factor

        # ----------------------------------
        # Save other variables in debug mode
        # ----------------------------------
        if debug:
            damp_ts[t]    = damp_term
            noise_ts[t]   = noise_term * integration_factor
            entrain_ts[t] = entrain_term * integration_factor
    
    # Now Remove initial values from variables
    temp_ts = temp_ts[t0:] # Remove intial Tdgrab values
    
    if debug:
        damp_ts    = damp_ts[t0:]
        noise_ts   = noise_ts[t0:]
        entrain_ts = entrain_ts[t0:]
        Td_ts      = Td_ts[t0:]
        if return_dict:
            output_dict = {
                'damping_term'       : damp_ts,
                'noise_term'         : noise_ts,
                'F_only'             : noise_ts - add_F,
                'entrain_term'       : entrain_ts,
                'integration_factor' : FAC,
                'Td'                 : Td_ts
                }
            return output_dict
        return temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts
    return temp_ts

# Entrain Model (Single Point)
def entrain_parallel(inputs):
    """
    SST Stochastic Model, with Entrainment rewritten to take all inputs as 1
    t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC = inputs
    
    Parameters
    ----------
    t_end : INT
        Length of integration, in months
    lbd : ARRAY [12,]
        Heat Flux Feedback (degC/sec)
    T0 : INT
        Initial Temperature (degC)
    F : ARRAY [t_end,]
        Forcing term (white noise time series) (degC/sec)
    beta : ARRAY [12,]
        Entrainment term coefficient (log(h(t+1)/h(t)))
    h : ARRAY [12,]
        Mixed Layer Depth (meters)
    kprev : ARRAY [12,]
        Month of detrainment (calculated through find_kprev)
    FAC : ARRAY [12,]
        Integration Factor ((1-exp(-lbd))/lbd)
    multFAC : BOOL, optional
        Set to true to apply integration factor to forcing and entrain term. 
        The default is TRUE.
    debug : BOOL, optional
        Set to true to output each term separately
    debugprint : BOOL, optional
        Set to true to print messages at each timestep. The default is False.

    Returns
    -------
    temp_ts : ARRAY [t_end,]
        Resultant Temperature timeseries
    if debug is True, returns the follow ARRAYs [t_end,]
        damp_ts    - Damping term
        noise_ts   - Noise term
        entrain_ts - Entrainment Term
        Td_ts      - Entraining Temperature
        
    """
    t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC = inputs
    debug=False
    debugprint=False
    
    # Preallocate
    temp_ts = np.zeros(t_end)
    
    if debug:
        noise_ts = np.zeros(t_end)
        damp_ts = np.zeros(t_end)
        entrain_ts = np.zeros(t_end)
        Td_ts   = np.zeros(t_end)
        
    entrain_term = np.zeros(t_end)
    
    # Prepare the entrainment term
    explbd = np.exp(np.copy(-lbd))
    #explbd[explbd==1] = 0
    
    Td0 = None # Set Td=None Initially
    # Loop for integration period (indexing convention from matlab)
    for t in range(t_end):
        
        # Get the month (start from Jan, so +1)
        m  = (t+1)%12
        #m=t%12
        if m == 0:
            m = 12
        
        # --------------------------
        # Calculate entrainment term
        # --------------------------
        if t<12: # Start Entrainment term after first 12 months
            entrain_term = 0
        else:
            if beta[m-1] == 0: # For months with no entrainment
                entrain_term = 0
                Td0 = None # Reset Td0 term
                if debugprint:
                    print("No entrainment on month %i"%m)
                    print("--------------------\n")
            else:
                if (Td0 is None) & (h.argmin()==m-2) :# For first entraining month
                    Td1 = calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
                    Td0 = Td1 # Previous month does not have entrainment!
                if Td0 is None: # Calculate Td0 
                    Td1,Td0 = calc_Td(t,kprev,temp_ts,prevmon=True,debug=debugprint)
                else: # Use Td0 from last timestep
                    Td1 = calc_Td(t,kprev,temp_ts,prevmon=False,debug=debugprint)
                
                Td = (Td1+Td0)/2
                if debugprint:
                    print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0)) 
                    print("--------------------\n")
                Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
                
                # Calculate entrainment term
                entrain_term = beta[m-1]*Td
        
        # ----------------------
        # Get Noise/Forcing Term
        # ----------------------
        noise_term = F[t]
        
        # ----------------------
        # Calculate damping term
        # ----------------------
        if t == 0:
            damp_term = explbd[m-1]*T0
        else:
            damp_term = explbd[m-1]*temp_ts[t-1]
        
        # ------------------------
        # Check Integration Factor
        # ------------------------
        if multFAC:
            integration_factor = FAC[m-1]
        else:
            integration_factor = 1
        
        # -----------------------
        # Compute the temperature
        # -----------------------
        temp_ts[t] = damp_term + (noise_term + entrain_term) * integration_factor

        # ----------------------------------
        # Save other variables in debug mode
        # ----------------------------------
        if debug:
            damp_ts[t] = damp_term
            noise_ts[t] = noise_term * integration_factor
            entrain_ts[t] = entrain_term * integration_factor
    if debug:
        return temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts
    return temp_ts


def noentrain_2d(randts,lbd,T0,F,FAC,multFAC=1,debug=False):
    
    """
    Run the no-entrainment model for all points at once.
    
    Inputs:
        1) randts: 1D ARRAY of random number time series
        2) lbd   : 3D ARRAY [lon x lat x mon] of seasonal damping values, degC/mon
        3) T0    : SCALAR Initial temperature throughout basin (usually 0 degC)
        4) F     : 3D Array [lon x lat x mon] of seasonal forcing values
    
    
    """
     
    # Determine run length for uniform or patterned forcing
    if len(randts.shape) > 1:
        t_end = randts.shape[2]
    else:
        t_end = len(randts)
    
    # Preallocate
    temp_ts = np.ones((lbd.shape[0],lbd.shape[1],t_end)) 
    damp_term = np.ones((lbd.shape[0],lbd.shape[1],t_end))  * np.nan
    noise_term = np.ones((lbd.shape[0],lbd.shape[1],t_end)) * np.nan
    
    # Prepare the entrainment term
    explbd = np.exp(-lbd)

    # Set the term to zero where damping is insignificant
    #explbd[explbd==1] =0
    
    # Set initial condition
    #temp_ts[:,:,0] = T0
    
    # Multiply forcing by reduction factor if option is set
    if multFAC == 1:
        F *= FAC
    
    # Loop for each timestep (note: using 1 indexing. T0 is from dec pre-simulation)
    for t in tqdm(range(t_end)):
        
        # Get the month
        m = (t+1)%12 # Start from January, params are same month
        #m=t%12 
        if m == 0:
            m = 12
        
        # Form the damping term
        damp_term[:,:,t] = explbd[:,:,m-1] * temp_ts[:,:,t-1]
        
        # Form the noise term
        if F.shape[2] == 12:
            noise_term[:,:,t] = F[:,:,m-1] * randts[None,None,t-1]
        else:
            noise_term[:,:,t] = F[:,:,t] 
                
        # Add with the corresponding forcing term to get the temp
        temp_ts[:,:,t] = damp_term[:,:,t] + noise_term[:,:,t]
        
        #msg = '\rCompleted timestep %i of %i' % (t,t_end-1)
        #print(msg,end="\r",flush=True)
    
    # Apply mask to temp_term
    msk = noise_term.copy()
    msk[~np.isnan(msk)] = 1
    temp_ts *= msk[:,:,:]
    
    if np.all(np.isnan(temp_ts)):
        print("WARNING ALL ARE NAN")
    
    if debug:
        return temp_ts,damp_term,noise_term
    return temp_ts

#%% Postprocessing Utilities

def postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags,
                            returnresults=False,preload=None,
                            mask_pacific=False,savesep=False,
                            useslab=True,mask_damping=False,
                            n_models=None,verbose=False):
    """
    Script to postprocess stochmod output
    
    Inputs:
        1) expid   - "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
        2) datpath - Path to model output
        3) rawpath - Path to raw data that was used as model input
        4) outpathdat - Path to store output data
        5) lags    - lags to compute for autocorrelation
        6) returnresults - option to return results [Bool]
        7) preloaded - option to provide preloaded data. This is a three
             element array[lon,lat,ssts], where ssts is an array of sst for 
             each model [lon180 x lat x time]
        8) mask_pacific - Set out True to mask out the tropical pacific below 20N
    
        9)  savesep - Set to True if data was saved separately (model0,model1,etc)
        10) useslab - Set to True if only CESM-SLAB parameters were used
        11) mask_damping - Set to True to exclude damping values that failed the significance test
        12) n_models - Number of models to loop through. assumes len(sst) that was loaded is the # of models.
        
    
    Based on portions of analyze_stochoutput.py
    
    Dependencies:
        numpy as np
        from scipy.io import loadmat
        amv.proc (sel_region)
        time
    
    """
    
    #% ---- Presets (Can Modify) ----
    allstart = time.time()
    
    # Regional Analysis Settings
    bbox_SP     = [-60,-15,40,65]
    bbox_ST     = [-80,-10,20,40]
    bbox_TR     = [-75,-15,10,20]
    bbox_NA     = [-80,0 ,0,65]
    bbox_NA_new = [-80,0,10,65]
    bbox_ST_w   = [-80,-40,20,40]
    bbox_ST_e   = [-40,-10,20,40]
    bbox_NA_ex  = [-80,0,20,60]
    regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw","NATex")        # Region Names
    bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w,bbox_NA_ex) # Bounding Boxes
    
    #% ---- Read in Data ----
    start = time.time()
    
    # Read in Stochmod SST Data
    if preload is None: # Manually Load the data
        if "forcing" in expid: # Newer Forcing output
            if ~savesep: # Everything was saved in a large file
                ld = np.load(datpath+"stoch_output_%s.npz"%(expid),allow_pickle=True)
                sst = ld["sst"]
            else:
                sst = []
                for modelnum in range(3):
                    # Set load name with model
                    ldname = datpath+"stoch_output_%s_model%i.npz"%(expid,modelnum)
                    if (useslab==0) and (modelnum>0): # Load different forcing for h-vary, entrain
                        ldname = ldname.replace("SLAB","FULL")
                    ld = np.load(ldname)
                    sst.append(ld['sst'])
            lonr = ld['lon']
            latr = ld['lat']
                    
        else: # Older Forcing Output
            sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
            lonr = np.load(datpath+"lon.npy")
            latr = np.load(datpath+"lat.npy")
    else:
        lonr,latr,sst = preload
    if n_models is None:
        n_models = len(sst) # Checks number of SSTs is list by default
    
    # Load MLD Data
    # -------------
    mld = np.load(rawpath+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
    
    # Load Damping Data for lat/lon
    # -----------------------------
    loaddamp = loadmat(rawpath+"ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat")
    lon = np.squeeze(loaddamp['LON1'])
    lat = np.squeeze(loaddamp['LAT'])
    
    # Load damping masks from significance test 
    # (Defaults to method 4) 0 = Failed test
    # Output indicates whicn regions to include
    # ------------------------------------------
    if 'method' in expid:
        moden   = proc.get_stringnum(expid,"method",nchars=1,verbose=verbose) # Get mode number from experiment name
    else:
        moden   = "5"# Use 5 as default
    mskslab = np.load(rawpath+"SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode%s_lag1.npy" % moden)
    mskfull = np.load(rawpath+"FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode%s_lag1.npy"% moden)
    dmskin = [mskslab,mskfull]
    dmsks  = []
    for i in range(2): # select region, append
        mskin = dmskin[i]
        mskreg,_,_ = proc.sel_region(mskin,lon,lat,[lonr[0],lonr[-1],latr[0],latr[-1]])
        mskreg[mskreg==0] = np.nan
        #mskreg = proc.nan_inv(mskreg) # Change so that 1 = insignificant points
        #mskreg = np.array(mskreg,dtype=bool)
        dmsks.append(mskreg.prod(-1))
    dmsks.append(dmsks[-1]) # Append again for entraining
    # if len(sst) != 3: # For cases where preloaded SSTs are provided...
    #     mskall = dmsks[0] * dmsks[1]
    #     for i in range(len(sst)):
    #         dmsk
    
    # Load and apply mask if option is set
    if mask_pacific:
        # Load the mask
        msk = np.load(rawpath+"pacific_limask_180global.npy")
        
        # Select the region
        mskreg,_,_ = proc.sel_region(msk,lon,lat,[lonr[0],lonr[-1],latr[0],latr[-1]])
        
        # Apply the mask to SST
        if preload is None:
            if len(sst.shape) < 4: # Include model dimension (?)
                print("Adding model dimension because sst size is %i" % len(sst.shape))
                sst = sst[None,...]
            sst *= mskreg[None,:,:,None]
            
        else:
            for i in range(len(sst)):
                sst[i] *= mskreg[:,:,None]
            #sst[1] *= msk[:,:,None]
    
    print("Data loaded in %.2fs" % (time.time()-start))
    
    #% ---- Get Regional Data ----
    start = time.time()
    nregion = len(regions)
    sstregion = {}
    for r in range(nregion):
        bbox = bboxes[r]
        
        sstr = {}
        for model in range(n_models):
            tsmodel = sst[model]
            sstr[model],_,_=proc.sel_region(tsmodel,lonr,latr,bbox)
        sstregion[r] = sstr
    
    # ---- Calculate autocorrelation and Regional avg SST ----
    kmonths = {}
    autocorr_region = {}
    sstavg_region   = {}
    for r in range(nregion):
        bbox = bboxes[r]
        
        autocorr = {}
        sstavg = {}
        for model in range(n_models):
            
            # Get sst and havg
            tsmodel = sstregion[r][model]
            havg,_,_= proc.sel_region(mld,lon,lat,bbox)
            
            # Find kmonth
            havg = np.nanmean(havg,(0,1))
            kmonth     = havg.argmax()
            kmonths[r] = kmonth
            
            # Take regional average 
            #tsmodel = np.nanmean(tsmodel,(0,1))
            # Take area-weighted regional average
            if mask_damping:
                tsmodel   = proc.sel_region(sst[model]*dmsks[model][...,None],lonr,latr,bbox,reg_avg=1,awgt=1)
            else:
                tsmodel   = proc.sel_region(sst[model],lonr,latr,bbox,reg_avg=1,awgt=1)
            
            sstavg[model] = np.copy(tsmodel)
            tsmodel = proc.year2mon(tsmodel) # mon x year
            
            # Deseason (No Seasonal Cycle to Remove)
            tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
            
            # Compute autocorrelation and save data for region
            autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)

        autocorr_region[r] = autocorr.copy()
        sstavg_region[r] = sstavg.copy()
        
    # Save Regional Autocorrelation Data
    np.savez("%sSST_Region_Autocorrelation_%s.npz"%(outpathdat,expid),autocorr_region=autocorr_region,kmonths=kmonths)
    
    # Save Regional Average SST 
    np.save("%sSST_RegionAvg_%s.npy"%(outpathdat,expid),sstavg_region)
    
    print("Completed autocorrelation and averaging calculations in %.2fs" % (time.time()-start))
    # ---- Calculate different AMVs for each region ----
    start = time.time()
    
    amvbboxes = bboxes
    amvidx_region = {}
    amvpat_region = {}
    for region in range(nregion):
        
        #% Calculate AMV Index
        amvtime = time.time()
        amvidx = {}
        amvpat = {}
        
        for model in range(n_models):
            if mask_damping:
                amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbboxes[region],dropedge=5,mask=dmsks[model])
            else:
                amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbboxes[region],dropedge=5)
        print("Calculated AMV variables for region %s in %.2f" % (regions[region],time.time()-amvtime))
        
        amvidx_region[region] = amvidx
        amvpat_region[region] = amvpat
        
    # Save Regional Autocorrelation Data
    np.savez("%sAMV_Region_%s.npz"%(outpathdat,expid),amvidx_region=amvidx_region,amvpat_region=amvpat_region)
    print("AMV Calculations done in %.2fs" % (time.time()-start))
    print("Stochmod Post-processing Complete in %.2fs" % (time.time() - allstart))
    print("Data has been saved to %s" % (outpathdat))
    if returnresults == True:
        return sstregion,autocorr_region,kmonths,sstavg_region,amvidx_region,amvpat_region


def calc_autocorr(sst,lags,basemonth,calc_conf=False,conf=0.95,tails=2,verbose=False):
    """
    Calculate autocorrelation for output of stochastic models
    
    Parameters
    ----------
    sst : DICT
        SST timeseries for each experiment
    lags : ARRAY
        Lags to calculate autocorrelation for
    basemonth : INT
        Month corresponding to lag 0 (ex. Jan=1) (NOT THE INDEX)
    calc_conf : BOOL
        Set to true to calculate confidence intervals
    conf : NUMERIC
        Confidence Level (default = 0.95)
    tails : INT
        Number of tails (1 or 2)
       
    Returns
    -------
    autocorr : DICT
        Autocorrelation stored in same order as sst
    """
    n = len(sst)
    autocorr = {}
    confs = {}
    for model in range(n):
        
        # Get the data
        tsmodel = sst[model]
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Detrend (Linear)
        tsmodel2 = signal.detrend(tsmodel2,axis=1,type='linear')
        
        # Calculate the autocorrelation
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,basemonth,1,debug=verbose)
        
        confs[model] = proc.calc_conflag(autocorr[model],conf,tails,tsmodel.shape[1])
    if calc_conf:
        return autocorr,confs
    return autocorr

#%% Synthetic Stochastic Model Wrapper

def load_data(mconfig,ftype,projpath=None):
    
    """
    Inputs
    ------
    mconfig : STR
        Model Configuration (SLAB_PIC or FULL_HTR)
    ftype : STR
        Forcing Type ('DJFM-MON' or ... )
    projpath : STR (optional)
        Path to project folder (default uses path on laptop)
    
    Outputs
    -------
    mld : ARRAY 
        Monhtly Mean Mixed Layer Depths
    kprevall : ARRAY
        Detrainment Months
    lon : ARRAY
        Longitudes (-180 to 180)
    lat : ARRAY
        Latitudes
    lon360 : ARRAY
        Longitudes (0 to 360)
    cesmslabac : ARRAY
        Autocorrelation at each point in the CESM Slab
    damping : ARRAY
        Monthly ensemble mean Heat flux feedback
    forcing : ARRAY
        Monthly forcing at each point (NAO, EAP, EOF3)
    """
    
    # Set Paths
    if projpath is None:
        projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/'
    input_path  = datpath + 'model_input/'
    
    # Load Data (MLD and kprev) [lon x lat x month]
    if mconfig == "FULL_HTR": # Load ensemble mean historical MLDs
        mld            = np.load(input_path+"%s_HMXL_hclim.npy"% mconfig) # Climatological MLD
        kprevall       = np.load(input_path+"%s_HMXL_kprev.npy"% mconfig) # Entraining Month
    else: # Load PIC MLDs 
        mld            = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
        kprevall       = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
    
    mld1kmean      = np.load(input_path+"FULL_PIC_HMXL_hclim_400to1400mean.npy") # Entraining Month
    
    # Load Lat/Lon, autocorrelation
    dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp       = loadmat(input_path+dampmat)
    lon            = np.squeeze(loaddamp['LON1'])
    lat            = np.squeeze(loaddamp['LAT'])
    cesmslabac     = np.load(datpath+"CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
    lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
    
    # Load damping [lon x lat x mon]
    if mconfig == "SLAB_PIC":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
    elif mconfig == "FULL_PIC":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof1893_mode4.npy")
    elif mconfig =="FULL_HTR":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")
    
    # Load Forcing  [lon x lat x pc x month]
    #forcing = np.load(input_path+mconfig+ "_NAO_EAP_NHFLX_Forcing_%s.npy" % ftype)#[:,:,0,:]
    forcing = np.load(input_path+"SLAB_PIC_NAO_EAP_NHFLX_Forcing_%s.npy" % ftype)#[:,:,0,:]
    return mld,kprevall,lon,lat,lon360,cesmslabac,damping,forcing,mld1kmean

def synth_stochmod(config,verbose=False,viz=False,
                   dt=3600*24*30,rho=1026,cp0=3996,hfix=50,T0=0,projpath=None,
                   specparams=None):
    """
    Parameters
    ----------
    config : DICT
        'mconfig'
        'ftype'
        'genrand'
        'fstd'
        't_end'
        'runid'
        'fname'
    dt : INT (optional)
        Timestep in seconds (monthly timestep by default)
        
'output_path'        

    Returns
    -------
    None.
    """
    debug = False
    if debug:
        verbose=False
        viz=False
        dt=3600*24*30
        rho=1026
        cp0=3996
        hfix=50
        T0=0
        projpath=None
        specparams=None
    
    if projpath is None:
        projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/'
    input_path  = datpath + 'model_input/'
    output_path = datpath + 'model_output/'
    
    # Load data
    # ---------
    mld,kprevall,lon,lat,lon360,cesmslabac,damping,forcing,mld1kmean = load_data(config['mconfig'],config['ftype'])
    hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    
    if verbose:
        print("Loaded Data")
    
    # Generate Random Forcing
    # -----------------------
    if config['genrand']:
        randts = np.random.normal(0,config['fstd'],config['t_end'])
        np.save(input_path + "Forcing_fstd%.2f_%s.npy"% (config['fstd'],config['runid']),randts)
        if verbose:
            print("Generating New Forcing")
    else:
        randts = np.load(input_path + "Forcing_fstd%.2f_%s.npy"% (config['fstd'],config['runid']))
        if verbose:
            print("Loading Old Forcing")
    
    # Select Forcing [lon x lat x mon]
    # --------------------------------
    forcing_flag = False
    if config['fname'] == 'NAO':
        forcing = forcing[:,:,0,:]
    elif config['fname'] == 'EAP':
        forcing = forcing[:,:,1,:]
    elif config['fname'] == 'EOF3':
        forcing = forcing[:,:,2,:]
    elif config['fname'] == 'FLXSTD':
        forcing = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")
    #elif config['fname'] == 'EOF':
    else:
        forcing = np.load(input_path+config['fname']) # [lon180, lat, pc, time]
        forcing_flag = True # Use make forcing
    
    
    # Restrict input parameters to point (or regional average)
    params = get_data(config['pointmode'],config['query'],lat,lon,
                          damping,mld,kprevall,forcing)
    [o,a],damppt,mldpt,kprev,Fpt = params
    kmonth = mldpt.argmax()
    if forcing_flag:# Make special forcing
        # Find the point
        fpt_in = forcing[o,a,...]
        
        # # Average forcing if option is set
        # if config['favg'] is True:
        #     fpt_in = np.ones(12)*fpt_in.mean()
        
        #Fpt = fpt_in
        # params= list(params)
        # params[-1] = fpt_in
        # params = tuple(params)
        Fpt = make_forcing_pt(fpt_in,config['runid'],config['fname'],config['t_end'],input_path,check=False)
        
    print("Restricted Parameters to Point. Kmonth is %i"%kmonth)
    
    # Apply 3 month smoothing if option is set
    if config['smooth_forcing'] == True:
        Fpt = np.convolve(np.hstack([Fpt[-1],Fpt,Fpt[0]]),np.ones(3)/3,mode='valid')
        #params[4] = Fpt
    
    # Check for synthetic points, and assign to variable if it exists
    synthflag = []
    if 'mldpt' in config:
        mldpt = config['mldpt']
        synthflag.append('mld')
    if 'Fpt' in config:
        Fpt = config['Fpt']
        synthflag.append('forcing')
        
        if len(Fpt.shape) > 1:
            forcing_flag = True # Turn on Forcing Flag [Mode x Month]
            Fpt = make_forcing_pt(Fpt,config['runid'],config['fname'],config['t_end'],input_path,check=False)
        else:
            forcing_flag=False # Turn off the forcing Flag
    if 'damppt' in config:
        damppt = config['damppt']
        synthflag.append('damping')
    if verbose:
        print("Detected synthetic forcings for %s"%str(synthflag))
    
    if viz:
        synth = [damppt,mldpt,Fpt]
        fig,ax = viz.summarize_params(lat,lon,params,synth=synth)
    

    
    # Prepare forcing
    mldmean = hblt[o,a,:].mean()
    Fh = {}
    nyrs = int(config['t_end']/12)
    
    if forcing_flag: # Forcing is already set up
        if config['applyfac'] in [0,3]: # White Noise Forcing, unscaled by MLD
            for h in range(3):
                Fh[h] = Fpt
        else:
            for h in range(3):

                
                if h == 0:# Fixed 50 meter MLD
                    hmult = hfix
                if h == 1:
                    hmult = mldmean
                if h == 2:
                    hmult = np.tile(mldpt,nyrs)
                
                Fh[h] = Fpt * (dt/(cp0*rho*hmult))

    else:
        if config['applyfac'] in [0,3]: # White Noise Forcing, unscaled by MLD
            for h in range(3):
                Fh[h] = randts * np.tile(Fpt,nyrs)
        else: # White Noise Forcing + MLD
            for h in range(3):
                if h == 0: # Fixed 50 meter MLD
                    Fh[h] = randts * np.tile(Fpt,nyrs) * (dt/(cp0*rho*hfix))
                elif h == 1: # Seasonal Mean MLD
                    Fh[h] = randts * np.tile(Fpt,nyrs) * (dt/(cp0*rho*mldmean))
                elif h == 2: # Seasonall Varying mean MLD
                    Fh[h] = randts * np.tile(Fpt/mldpt,nyrs) * (dt/(cp0*rho))
    
    # Convert Parameters
    lbd,lbd_entr,FAC,beta = set_stochparams(mldpt,damppt,dt,ND=False,hfix=hfix,hmean=mldmean)
    if verbose:
        print("Completed parameter setup!")
    
    # Apply correction factor (if set)
    if config['method'] == 3:
        for i in range(3):
            underest = method2(lbd[i].mean(),original=False) # Original = uncorrected version with error
            ampmult = 1/underest
            Fh[i] *= ampmult
    
    # Run the stochastic model
    multFAC = 0
    if config['applyfac'] > 1: # Apply Integration factor
        multFAC = 1
    
    sst         = {}
    dampingterm = {}
    forcingterm = {}
    for i in range(3): # No Entrainment Cases
        sst[i],forcingterm[i],dampingterm[i] = noentrain(config['t_end'],lbd[i],T0,Fh[i],FAC[i],multFAC=multFAC,debug=True)
    
    sst[3],dampingterm[3],forcingterm[3],entrainterm,Td=entrain(config['t_end'],
                       lbd[3],T0,Fh[2],
                       beta,mldpt,kprev,
                       FAC[3],multFAC=multFAC,
                       debug=True,debugprint=False)
    if verbose:
        print("Model Runs Complete!")
    
    # Reassign Params
    if forcing_flag:
        Fpt = fpt_in
    params = ([o,a],damppt,mldpt,kprev,Fpt)
    
    ## Basically no effect, so commented out..
    # #Detrend again to be sure
    # for i in range(4):
    #     sst[i] = signal.detrend(sst[i],type='linear')
    
    # Calculate Autocorrelation
    autocorr = calc_autocorr(sst,config['lags'],kmonth+1)
    if verbose:
        print("Autocorrelation Calculations Complete!")
        
        
    # Calculate Spectra
    if specparams is None:
        return autocorr,sst,dampingterm,forcingterm,entrainterm,Td,kmonth,params
    else:
        
        # Unpack SST (dict --> list of arrays)
        ssts  = []
        for i in range(len(sst)):
            ssts.append(sst[i])
        
        # Calculate the spectra
        specout = quick_spectrum(ssts,config['nsmooth'],config['pct'])
        # specs,freqs,CCs,dofs,r1s = specout
        dofs = specout[3]
        bnds = []
        for nu in dofs:
            lower,upper = tbx.confid(config['alpha'],nu*2)
            bnds.append([lower,upper])
        
        specout = specout + (bnds,)
        return autocorr,sst,dampingterm,forcingterm,entrainterm,Td,kmonth,params,specout

def quick_spectrum(sst,nsmooth,pct,
                   opt=1,dt=None,clvl=[.95],verbose=False,return_dict=False,dim=None):
    """
    Quick spectral estimate of an array of timeseries

    Parameters
    ----------
    sst : ARRAY
        Array containing timeseries to look thru [[ts1],[ts2],...]
    nsmooth : INT
        Number of bands to smooth over 
    pct : Numeric
        Percent to taper
    opt : TYPE, optional
        Smoothing option
    dt : INT, optional
        Time Interval. The default is 3600*24*30.
    clvl : ARRAY , optional
        Array of Confidence levels. The default is [.95].
        
    dim : INT, optional
        Specify dimension to compute the spectra over

    Returns
    -------
    specs : ARRAY
        Array containing spectrum for each input series
    freqs : ARRAY
        Corresponding frequencies for each input
    CCs : ARRAY
        Confidence intervals for each input
    dofs : ARRAY
        Degrees of freedom for each input
    r1s : ARRAY
        AR1 parameter used to estimate CC

    """
    
    # -----------------------------------------------------------------
    # Set interval of time series (assumes monthly by default)
    # -----------------------------------------------------------------
    if dt is None:
        dt = np.ones(len(sst)) * 3600*24*30
    else:
        dt = np.ones(len(sst)) * dt
    
    # -----------------------------------------------------------------
    # Calculate and make individual plots for stochastic model output
    # -----------------------------------------------------------------
    #specparams  = []
    specs = []
    freqs = []
    CCs = []
    dofs = []
    r1s = []
    if dim is None:
        n_ts = len(sst)
    else:
        n_ts = sst.shape[dim]
    
    for i in range(n_ts):
        if dim is None:
            sstin = sst[i]
        else:
            sstin = np.take_along_axis(sst,i,dim)
        dt_in = dt[i]
        
        # Calculate Spectrum
        if isinstance(nsmooth,int):
            sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False,verbose=verbose)
        else:
            sps = ybx.yo_spec(sstin,opt,nsmooth[i],pct,debug=False,verbose=verbose)
        
        # Save spectrum and frequency, convert to 1/sec
        P,freq,dof,r1=sps
        specs.append(P*dt_in)
        freqs.append(freq/dt_in)
        dofs.append(dof)
        r1s.append(r1)
        
        # Calculate Confidence Interval
        CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
        CCs.append(CC*dt_in)
        
    if return_dict:
        output_dict = {
            "specs":specs,
            "freqs":freqs,
            "CCs"  : CCs,
            "dofs" :dofs,
            "r1s"  :r1s,
            }
        return output_dict
    return specs,freqs,CCs,dofs,r1s

#%% Data Loading

def load_hadisst(datpath,method=2,startyr=1870,endyr=2018,grabpoint=None):
    
    hadname  = "%sHadISST_detrend%i_startyr%i_endyr%i.npz" % (datpath,method,startyr,endyr)
    ld = np.load(hadname,allow_pickle=True)
    
    hsst = ld['sst']
    hlat = ld['lat']
    hlon = ld['lon']
    
    if grabpoint is None:
        return hsst,hlat,hlon
    else:
        lonf,latf = grabpoint
        khlon,khlat = proc.find_latlon(lonf,latf,hlon,hlat)
        sstpt = hsst[khlon,khlat,:]
        return sstpt

def load_ersst(datpath,method=2,startyr=1854,endyr=2016,grabpoint=None):
    
    hadname  = "%sERSST_detrend%i_startyr%i_endyr%i.npz" % (datpath,method,startyr,endyr)
    ld = np.load(hadname,allow_pickle=True)
    
    hsst = ld['sst']
    hlat = ld['lat']
    hlon = ld['lon']
    
    if grabpoint is None:
        return hsst,hlat,hlon
    else:
        lonf,latf = grabpoint
        if lonf < 0:
            lonf+=360
        khlon,khlat = proc.find_latlon(lonf,latf,hlon,hlat)
        sstpt = hsst[khlon,khlat,:]
        return sstpt


def load_cesm_pt(datpath,loadname='both',grabpoint=None,ensorem=0):
    """
    Load CESM Data
    Inputs:
        1) datpath [STR] : Path to data
        2) loadname [STR] : What to load, default is both: ('both','slab','full')
        3) grabpoint [lonf,latf] : Point to query (optional)
    Returns:
        1) ssts [ARR] : [sstfull,sstslab], depending on loadname
    """
    
    
    st = time.time()
    
    # Get lat/lon
    lat    = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LAT'].squeeze()
    lon360 = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
    
    
    if grabpoint is not None:
        # Query the point
        lonf,latf = grabpoint
        if lonf < 0:
            lonf += 360
        klon360,klat = proc.find_latlon(lonf,latf,lon360,lat)
    
    # Load SSTs
    ssts = []
    if loadname=='both' or loadname=='full': # Load full sst data from model
        print("Loading from CESM-FULL...")
        
        if ensorem == 1:# Load data with ENSO Removed
            ld  = np.load(datpath+"FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
            sstfull = ld['TS']
        else:
            ds = xr.open_dataset(datpath+"CESM_proc/TS_anom_PIC_FULL.nc")
            if grabpoint is not None:
                sstfull = ds.sel(lon=lonf,lat=latf,method='nearest').TS.values
            else:
                sstfull = ds['TS'].values
            
        ssts.append(sstfull)
          
    if loadname=='both' or loadname=='slab': # Load slab sst data
        print("Loading from CESM-SLAB")
        if ensorem == 1:# Load data with ENSO Removed
            ld2 = np.load(datpath+"SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
            sstslab = ld2['TS'] # Time x lat x lon
        else:
            ds = xr.open_dataset(datpath+"CESM_proc/TS_anom_PIC_SLAB.nc")
            if grabpoint is not None:
                sstslab = ds.sel(lon=lonf,lat=latf,method='nearest').TS.values
            else:
                sstslab = ds['TS'].values
        ssts.append(sstslab)
        
    
    print("Loaded PiC Data in %.2fs"%(time.time()-st))
    
    # Retrieve point information
    if grabpoint is None or ensorem==0:
        return ssts
    else:
        sstpts = []
        for sst in ssts:
            sstpt = sst[:,klat,klon360]
            sstpts.append(sstpt)
        return sstpts
    
def load_latlon(datpath=None,lon360=False):
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    
    dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp = loadmat(datpath+dampmat)
    lon   = np.squeeze(loaddamp['LON1'])
    lat   = np.squeeze(loaddamp['LAT'])
    
    if lon360:
        lon = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
    return lon,lat

def load_dmasks(datpath=None,bbox=None):
    # Load Damping Masks
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    mskslab = np.load(datpath+"SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode4_mask.npy")
    mskfull = np.load(datpath+"FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode4_mask.npy")
    dmskin = [mskslab,mskfull]
    dmsks  = []
    for i in range(2): # select region, append
        mskin = dmskin[i]
        if bbox is not None:
            lon,lat = load_latlon(datpath=datpath)
            mskin,_,_ = proc.sel_region(mskin,lon,lat,bbox)
        dmsks.append(mskin.prod(-1)) # multiply for each month
    return dmsks

def load_cesm_le(datpath=None,preprocess=False,return_ds=False):
    """
    Load CESM1.1-LE Historical Data that has been preprocessed for the
    Machine Learning Project (predict_AMV)
    """
    if datpath is None: # Defaults to location on local Mac
        datpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/CESM_data/"
    
    ncname  = "CESM1LE_sst_NAtl_19200101_20051201.nc"
    ds      = xr.open_dataset(datpath+ncname)
    
    if preprocess: # Do some preprocessing
        st = time.time()
        ds = ds.sst - ds.sst.mean('ensemble') # Remove Ensemble mean
        ds = proc.xrdeseason(ds)
        print("Loaded in %.2fs" % (time.time()-st))
    
    if return_ds: # Just return the dataset
        return ds
    
    st = time.time()
    if preprocess:
        sst = ds.values
    else:
        sst = ds.sst.values # [lat x lon x time x ensemble]
    lon = ds.lon.values
    lat = ds.lat.values
    print("Loaded in %.2fs" % (time.time()-st))
    return sst,lon,lat

#%% Heat Flux Feedback Calculations

def indexwindow(invar,m,monwin,combinetime=False,verbose=False):
    """
    index a specific set of months/years for an odd sliding window
    given the following information (see inputs)
    
    drops the first and last years when a the dec-jan boundary
    is crossed, according to the direction of crossing
    time dimension is thus reduced by 2 overall
    
    inputs:
        1) invar [ARRAY: yr x mon x otherdims] : variable to index
        2) m [int] : index of central month in the window
        3) monwin [int]: total size of moving window of months
        4) combinetime [bool]: set to true to combine mons and years into 1 dimension
    
    output:
        1) varout [ARRAY]
            [yr x mon x otherdims] if combinetime=False
            [time x otherdims] if combinetime=True
    
    """
    
    if monwin > 1:  
        winsize = int(np.floor((monwin-1)/2))
        monid = [m-winsize,m,m+winsize]

    
    varmons = []
    msg = []
    for m in monid:

        if m < 0: # Indexing months from previous year
            
            msg.append("Prev Year")
            varmons.append(invar[:-2,m,:])
            
        elif m > 11: # Indexing months from next year
            msg.append("Next Year")
            varmons.append(invar[2:,m-12,:])
            
        else: # Interior years (drop ends)
            msg.append("Interior Year")
            varmons.append(invar[1:-1,m,:])
    if verbose:
        print("Months are %s with years %s"% (str(monid),str(msg)))       
    # Stack together and combine dims, with time in order
    varout = np.stack(varmons) # [mon x yr x otherdims]
    varout = varout.transpose(1,0,2) # [yr x mon x otherdims]
    if combinetime:
        varout = varout.reshape((varout.shape[0]*varout.shape[1],varout.shape[2])) # combine dims
    return varout


def calc_HF(sst,flx,lags,monwin,verbose=True,posatm=True,return_cov=False,
            var_denom=None,return_dict=False):
    """
    damping,autocorr,crosscorr=calc_HF(sst,flx,lags,monwin,verbose=True)
    Calculates the heat flux damping given SST and FLX anomalies using the
    formula:
        lambda = [SST(t),FLX(t+l)] / [SST(t),SST(t+l)]
    
    
    Inputs
    ------
        1) sst     : ARRAY [year x time x lat x lon] 
            sea surface temperature anomalies
        2) flx     : ARRAY [year x time x lat x lon]
            heat flux anomalies
        3) lags    : List of INTs
            lags to calculate for (0-N)
        4) monwin  : INT (odd #)
            Moving window of months centered on target month
            (ex. For Jan, monwin=3 is DJF and monwin=1 = J)
        
        --- OPTIONAL ---
        5) verbose : BOOL
            set to true to display print messages
        6) posatm : BOOL
            check to true to set positive upwards into the atmosphere
        
        7) return_cov : BOOL
            True to return covariance values
        8) var_denom : ARRAY [year x time x lat x lon]
            Rather than autovariance, replace SST(t) with vardenom(t).
        9) return_dict : BOOL
            True to return dictionary
        
    Outputs
    -------     
        1) damping   : ARRAY [month x lag x lat x lon]
            Heat flux damping values
        2) autocorr  : ARRAY [month x lag x lat x lon]
            SST autocorrelation
        3) crosscorr : ARRAY [month x lag x lat x lon]
            SST-FLX cross correlation
    """
    # Reshape variables [time x lat x lon] --> [yr x mon x space]
    nyr,nmon,nlat,nlon = sst.shape
    
    sst = sst.reshape(nyr,12,nlat*nlon)
    flx = flx.reshape(sst.shape)
    if var_denom is not None:
        var_denom = var_denom.reshape(sst.shape)
    #sst = sst.reshape(int(ntime/12),12,nlat*nlon)
    #flx = flx.reshape(sst.shape)
    
    # Preallocate
    nlag      = len(lags)
    damping   = np.zeros((12,nlag,nlat*nlon)) # [month, lag, lat, lon]
    autocorr  = np.zeros(damping.shape)
    crosscorr = np.zeros(damping.shape)
    
    covall    = np.zeros(damping.shape)
    autocovall    = np.zeros(damping.shape)
    
    st = time.time()
    for l in range(nlag):
        lag = lags[l]
        for m in range(12):
            
            lm = m-lag # Get Lag Month
            
            # Restrict to time ----
            flxmon = indexwindow(flx,m,monwin,combinetime=True,verbose=False)
            sstmon = indexwindow(sst,m,monwin,combinetime=True,verbose=False)
            if var_denom is None:
                sstlag = indexwindow(sst,lm,monwin,combinetime=True,verbose=False)
            else:
                sstlag = indexwindow(var_denom,lm,monwin,combinetime=True,verbose=False)
            
            # Compute Correlation Coefficients ----
            crosscorr[m,l,:] = proc.pearsonr_2d(flxmon,sstlag,0) # [space]
            autocorr[m,l,:]  = proc.pearsonr_2d(sstmon,sstlag,0) # [space]
            
            # Calculate covariance ----
            cov     = proc.covariance2d(flxmon,sstlag,0)
            autocov = proc.covariance2d(sstmon,sstlag,0)
            
            # Compute damping
            damping[m,l,:] = cov/autocov
            
            # Save covariances
            covall[m,l,:]     = cov
            autocovall[m,l,:] = autocov
            
            print("Completed Month %02i for Lag %s (t = %.2fs)" % (m+1,lag,time.time()-st))
            
    # Reshape output variables
    damping = damping.reshape(12,nlag,nlat,nlon)  
    autocorr = autocorr.reshape(damping.shape)
    crosscorr = crosscorr.reshape(damping.shape)
    covall = covall.reshape(damping.shape)
    autocovall = autocovall.reshape(damping.shape)
    
    # Check sign
    if posatm:
        if np.nansum(np.sign(crosscorr)) < 0:
            print("WARNING! sst-flx correlation is mostly negative, sign will be flipped")
            crosscorr*=-1
            covall*=-1
    
    if return_dict:
        outdict = {
            'damping':damping,
            'autocorr':autocorr,
            'crosscorr':crosscorr,
            'autocovall':autocovall,
            'covall':covall
            }
        return outdict
    if return_cov:
        return damping,autocorr,crosscorr,autocovall,covall
    return damping,autocorr,crosscorr

def prep_HF(damping,rsst,rflx,p,tails,dof,mode,
            returnall=False,posatm=True,maskval=np.nan):
    """
    
    Mask the damping values using Students T-Test on based on the 
    SST autocorrelation and SST-FLX cross-correlation
    
    Inputs
    ------
        1) damping   : ARRAY [month x lag x lat x lon]
            Heat flux damping values
        2) autocorr  : ARRAY [month x lag x lat x lon]
            SST autocorrelation
        3) crosscorr : ARRAY [month x lag x lat x lon]
            SST-FLX cross correlation
        4) p : NUMERIC
            p-value
        5) tails : INT
            # of tails for t-test (1 or 2)
        6) dof : INT
            Degrees of freedom
        7) mode: INT
            Apply the following significance testing/masking:
            1 --> No Mask
            2 --> SST autocorrelation based
            3 --> SST-FLX cross correlation based
            4 --> Both 2 and 3
            5 --> Both, but replace FULL with SLAB if insignificant
        --- OPTIONAL ---
        8) returnall BOOL
            Set to True to return masks and frequency
        9) posatm : BOOL
            Set to True to ensure positive upwards into the atmosphere
        10) maskval : NUMERIC
            Fill value for failed points
    
    Outputs
    -------
        1) dampingmasked [month x lag x lat x lon]
        
    """
    # Determine correlation threshold
    ptilde    = 1-p/tails
    critval   = stats.t.ppf(ptilde,dof)
    corrthres = np.sqrt(1/ ((dof/np.power(critval,2))+1))
    
    # Check sign
    if posatm:
        if np.nansum(np.sign(rflx)) < 0:
            print("WARNING! sst-flx correlation is mostly negative, sign will be flipped")
            rflx*=-1
    
    # Create Mask
    msst = np.zeros(damping.shape) * maskval#np.nan
    mflx = np.zeros(damping.shape) * maskval#np.nan
    msst[rsst > corrthres] = 1
    mflx[rflx > corrthres] = 1
    if mode == 1:
        mtot = np.ones(damping.shape)     # Total Frequency of successes
        mall = np.copy(mtot)              # Mask that will be applied
        mult = 1
    elif mode == 2:
        mtot = np.copy(msst)
        mall = np.copy(msst)
        mult = 1
    elif mode == 3:
        mtot = np.copy(mflx)
        mall = np.copy(mflx)  
        mult = 1
    elif mode == 4 or mode == 5:
        mtot = msst + mflx
        mall = msst * mflx
        mult = 2
    
    # Apply Significance Mask
    dampingmasked = damping * mall
    
    if returnall:
        return dampingmasked,mtot,mall
    return dampingmasked

def postprocess_HF(dampingmasked,limask,sellags,lon,pos_upward=True):
    
    """
    Apply a land/ice mask and average across the selected lags
    
    """
    
    # Inputs
    ## Dampingmasked [month x lag x lat x lon]
    ## limask [lat x lon]
    
    # Select lags, apply landice mask
    mchoose = dampingmasked[:,sellags,:,:].mean(1) * limask[None,:,:]
    
    # Flip longiude coordinates ([mon lat lon] --> [lon x lat x mon])
    lon1,dampingw = proc.lon360to180(lon,mchoose.transpose(2,1,0))

    # Multiple by 1 to make positive upwards
    if pos_upward:
        dampingw *= -1
    return dampingw

def check_ENSO_sign(eofs,pcs,lon,lat,verbose=True):
    """
    checks sign of EOF for ENSO and flips sign by check the sum over
    conventional ENSO boxes (see below for more information)
    
    checks to see if the sum within the region and flips if sum < 0
    
    inputs:
        1) eofs [space (latxlon), PC] EOF spatial pattern from eof_simple
        2) pcs  [time, PC] PC timeseries calculated from eof_simple
        3) lat  [lat] Latitude values
        4) lon  [lon] Longitude values
    
    outputs:
        1) eofs [space, PC]
        2) pcs  [time, PC]
    
    """   
    # Set EOF boxes to check over (west,east,south,north)
    eofbox1 = [190,240,-5,5] #EOF1 sign check Lat/Lon Box (Currently using nino3.4)
    eofbox2 = [190,240,-5,5] #EOF2 (Frankignoul et al. 2011; extend of positive contours offshore S. Am > 0.1)
    eofbox3 = [200,260,5,5] #EOF3 (Frankignoul et al. 2011; narrow band of positive value around equator)
    chkboxes = [eofbox1,eofbox2,eofbox3]
    
    # Find dimensions and separate out space
    nlon = lon.shape[0]
    nlat = lat.shape[0]
    npcs = eofs.shape[1] 
    eofs = eofs.reshape(nlat,nlon,npcs)
    eofs = eofs.transpose(1,0,2) # [lon x lat x npcs]
        
    for n in range(npcs):
        chk = proc.sel_region(eofs,lon,lat,chkboxes[n],reg_sum=1)[n]
        
        if chk < 0:
            if verbose:
                print("Flipping EOF %i because sum is %.2f"%(n,chk))
    
            eofs[:,:,n] *= -1
            pcs[:,n] *= -1
    

    eofs = eofs.transpose(1,0,2).reshape(nlat*nlon,npcs) # Switch back to lat x lon x pcs
    
    return eofs,pcs

def calc_enso(invar,lon,lat,pcrem,bbox=None):
    """
    Calculate ENSO for input SST (use longitude 0-360)!
    
    Parameters
    ----------
    invar : ARRAY [time x lat x lon360]
        monthly SST variable longitude is 0-360, lat -90 to 90
    lon : [lon]
        Longitude values
    lat : [lat]
        Latitude values
    pcrem : Int
        Number of PCs to alculate
    bbox : List [lonW,lonE,latS,latN], optional
        Bounding box for calculation.
        Default is [120, 290, -20, 20]. The default is None.

    Returns
    -------
    eofall : ARRAY [lat x lon x month x pc]
        EOF Patterns.
    pcall : ARRAY [time x month x pc]
        Principle components.
    varexpall : [month x pc]
        Variance Explained.
        
    Custom dependencies: check_enso_sign, proc.eof_simple
    """
    if bbox is None:
        bbox = [120, 290, -20, 20]
    
    #  Apply Area Weight
    _,Y = np.meshgrid(lon,lat)
    wgt = np.sqrt(np.cos(np.radians(Y))) # [lat x lon]
    ts = invar * wgt[None,:,:]
    
    # Reshape for ENSO calculations
    ntime,nlat,nlon = ts.shape 
    ts = ts.reshape(ntime,nlat*nlon) # [time x space]
    ts = ts.T #[space x time]
    
    # Remove NaN points
    okdata,knan,okpts = proc.find_nan(ts,1) # Find Non-Nan Points
    oksize = okdata.shape[0]

    # Calcuate monthly anomalies
    okdata = okdata.reshape(oksize,int(ntime/12),12) # [space x yr x mon]
    manom  = okdata.mean(1)
    tsanom = okdata - manom[:,None,:]
    nyr    = tsanom.shape[1]
    
    # Preallocate
    eofall = np.zeros((nlat*nlon,12,pcrem)) *np.nan# [space x month x pc]
    pcall  = np.zeros((nyr,12,pcrem)) *np.nan# [year x month x pc]
    varexpall  = np.zeros((12,pcrem)) * np.nan #[month x pc]

    # Compute EOF!!
    for m in range(12):
        
        # Perform EOF
        st = time.time()
        eofsok,pcs,varexp= proc.eof_simple(tsanom[:,:,m],pcrem,1)
        #print("Performed EOF in %.2fs"%(time.time()-st))
    
        # Place back into full array
        eofs = np.zeros((nlat*nlon,pcrem)) * np.nan
        eofs[okpts,:] = eofsok   
    
        # Correct ENSO Signs
        eofs,pcs = check_ENSO_sign(eofs,pcs,lon,lat,verbose=True)
        
        
        # Save variables
        eofall[:,m,:] = eofs.copy()
        pcall[:,m,:] = pcs.copy()
        varexpall[m,:] = varexp.copy()
        
        print("Completed month %i in %.2fs"%(m+1,time.time()-st))


    # Reshape Data
    eofall = eofall.reshape(nlat,nlon,12,pcrem) # [lat x lon x month x pc]
    
    return eofall,pcall,varexpall

def remove_enso(invar,ensoid,ensolag,monwin,reduceyr=True,verbose=True,times=None):
    """
    Remove ENSO via regression of [ensoid] to [invar: time,lat,lon] for each month
    and principle component.
    
    
    Parameters
    ----------
    invar : ARRAY [time x lat x lon]
        Input variable to remove ENSO from
    ensoid : ARRAY [time x month x pc]
        ENSO Index.
    ensolag : INT
        Lag to apply between ENSO Index and Input variable
    monwin : INT
        Size of centered moving window to calculate ENSO for
    reduceyr : BOOL, optional
        Reduce/drop time dimension to account for ensolag. The default is True.
    times : ARRAY[time]
        Array of times to work with
    Returns
    -------
    vout : ARRAY [time x lat x lon]
        Variable with ENSO removed
    ensopattern : ARRAY [month x lat x lon x pc]
        ENSO Component that was removed
    
    """
    allstart = time.time()

    # Standardize the index (stdev = 1)
    ensoid = ensoid/np.std(ensoid,(0,1))
    
    # Shift ENSO by the specified enso lag
    ensoid = np.roll(ensoid,ensolag,axis=1) # (ex: Idx 0 (Jan) is now 11)

    # Reduce years if needed
    if reduceyr: # Since enso leads, drop years from end of period
        dropyr = int(np.fix(ensolag/12) + 1)
        ensoid=ensoid[:-dropyr,:,:]

    # Reshape to combine spatial dimensions
    ntime,nlat,nlon = invar.shape
    nyr             = int(ntime/12)
    vanom           = invar.reshape(nyr,12,nlat*nlon) # [year x mon x space]
    
    # Reduce years if option is set
    if reduceyr:
        vanom=vanom[dropyr:,:,:] # Drop year from beginning of period for lag
        if times is not None:
            times = times.reshape(nyr,12)[dropyr:,:]
    vout            = vanom.copy()
    
    # Variable to save enso pattern
    nyr,_,pcrem = ensoid.shape # Get reduced year, pcs length
    ensopattern        = np.zeros((12,nlat,nlon,pcrem))
    
    # Looping by PC...
    for pc in range(pcrem):
        # Looping for each month
        for m in range(12):
            #print('m loop start, vout size is %s'% str(vout[winsize:-winsize,m,:].shape))
            
            # Set up indexing
            if monwin > 1:  
                winsize = int(np.floor((monwin-1)/2))
                monid = [m-winsize,m,m+winsize]
                if monid[2] > 11: # Reduce end month if needed
                    monid[2] -= 12
            else:
                winsize = 0
                monid = [m]
                
            if reduceyr:
                ensoin = indexwindow(ensoid[:,:,[pc]],m,monwin,combinetime=True).squeeze()
                varin  = indexwindow(vanom,m,monwin,combinetime=True)
                nyr   = int(ensoin.shape[0]/monwin) # Get new year dimension
            else:
                # Index corresponding timeseries
                ensoin = ensoid[:,monid,pc] # [yr * mon]  Lagged ENSO
                varin = vanom[:,monid,:] # [yr * mon * space] Variable
                
                # Re-combine time dimensions
                ensoin = ensoin.reshape(nyr*monwin)
                varin = varin.reshape(nyr*monwin,nlat*nlon)
            
            # Check for points that are all nan
            # delete_points = []
            # for i in range(varin.shape[0]):
            #     if np.all(np.isnan(varin[i,:])):
            #         print("All nan at i=%i for month %i, removing from calculation" % (i,m)) 
            #         delete_points.append(i)
            
            #ensoin = np.delete(ensoin,delete_points)
            #varin  = np.delete(varin,delete_points,axis=0)
                    
            # Regress to obtain coefficients [space]
            varreg,_ = proc.regress_2d(ensoin.squeeze(),varin,nanwarn=1)
            
            # Write to enso pattern
            ensopattern[m,:,:,pc] = varreg.reshape(nlat,nlon).copy()
            
            # Expand and multiply out and take mean for period [space,None]*[None,time] t0o [space x time]
            ensocomp = (varreg[:,None] * ensoin[None,:]).squeeze()
            
            # Separate year and mon and take mean along months
            ensocomp = ensocomp.reshape(nlat*nlon,nyr,monwin).mean(2)
            
            # Remove the enso component for the specified month
            vout[winsize:-winsize,m,:] -= ensocomp.T
            
            if verbose:
                print("Removed ENSO Component for PC %02i | Month %02i (t=%.2fs)" % (pc+1,m+1,time.time()-allstart))
            # < End Mon Loop>
        # < End PC Loop>
        # Get the correct window size, and reshape it to [time x lat x lon]
    vout = vout[winsize:-winsize,:,:].reshape(nyr*12,nlat,nlon)
    if times is not None:
        times = times[winsize:-winsize,:].flatten()
        return vout,ensopattern,times
    return vout,ensopattern

def compute_qnet(datpath,dataset_name,vnames=None,downwards_positive=True,
                 verbose=True):
    """
    Computes dataarray of Qnet (downwards positive) given shortwave, longwave,
    sensible, and latent heat fluxes. Assumes inputs are positive into the ocean.
    
    Looks for files of the form <datpath><dataset name>_<flux name>.nc

    Parameters
    ----------
    datpath : STR
        Path to data
    dataset_name : STR
        Name of dataset
    vnames : List of STR, optional
        Name of shortwave,longwave,sensible, and latent fluxes. The default is:
        ["fsns","flns","hfss","hfls"]
    downwards_positive : BOOL, optional
        Set to False to convert to upwards positive. The default is True.
    verbose : BOOL, optional
        Print debuggin messages. The default is True.
    
    Returns
    -------
    ds_new : TYPE
        DESCRIPTION.

    """
    
    if vnames is None:
        vnames = ["fsns","flns","hfss","hfls"]
    
    flxes = []
    for v in vnames:
        loadname = "%s%s_%s.nc" % (datpath,dataset_name,v)
        if verbose:
            print("Loading %s!" % loadname)
        ds = xr.open_dataset(loadname)
        flxes.append(ds[v])
        
    # Compute Qnet = FSNS + (FLNS + SHFLX + LHFLX)
    ds_new = flxes[0] + (flxes[1]+flxes[2]+flxes[3])
    ds_new = ds_new.rename("qnet")
    
    if downwards_positive == False:
        ds_new *= -1 # Convert to downwards positive
        
    return ds_new

#%% SCM rewritten.
def convert_Wm2(invar,h,dt,cp0=3996,rho=1026,verbose=True,reverse=False):
    """
    outvar = convert_Wm2(invar,h,dt,cp0=3996,rho=1026,verbose=True)
    
    Converts an input array [LON x LAT x TIME] from W/m2 to 1/time
    by multiplying by dt/(rho*cp0*h).
    If input is not 3D, appends dimensions to the front and assumes
    last dimension is time.
    Set reverse=True to revert the conversion.
    
    Parameters
    ----------
    invar : ARRAY [LON x LAT x TIME]
        Variable to convert, in W/m2 or W/m2/degC
    h     : ARRAY [LON X LAT x TIME (or MONTH)]
        Mixed layer depth to use. Can be seasonal or monthly
    dt    : Numeric
        Integration Timestep in seconds
    cp0   : Numeric, optional
        Specific Heat [J/(kg*C)]. The default is 3996.
    rho   : TYPE, optional
        Density of Seawater [kg/m3]. The default is 1026.
    verbose : BOOL, optional
        Set to True to print messages.The default is True
    reverse : BOOL
        Set to True to convert from 1/time to W/m2

    Returns
    -------
    outvar : ARRAY [LON x LAT x TIME]
        Converted variable
    """
    
    # Append dimensions to front, so variables are 3-D
    while len(invar.shape) < 3:
        if verbose:
            print("Warning! invar is not 3D. Appending dims to front.")
        invar = invar[None,:] 
    while len(h.shape) < 3: 
        if verbose:
            print("Warning! h is not 3D. Appending dims to front.")
        h = h[None,:] 
    
    # Check last dimensions of h
    if invar.shape[-1] != h.shape[-1]:
        print("Found seasonal data for h. Tiling to match time for invar")
        ntile = int(invar.shape[-1]/h.shape[-1])
        h     = np.tile(h,ntile)
    
    # Perform Conversion
    if reverse:
        outvar = invar * (rho*cp0*h) / dt
    else:
        outvar = invar * dt / (rho*cp0*h)
    
    return outvar.squeeze()

def load_inputs(mconfig,frcname,input_path,load_both=False,method=4,lagstr="lag1",
                loadpoint=False,ensorem=True):
    """
    lon,lat,mld,kprevall,damping,alpha = load_inputs(mconfig,frcname,input_path)
    
    Load stochastic model inputs from [input_path] directory. This includes
    atmospheric damping, mixed layer depth, entrainment month, and stochastic
    forcing amplitudes. 
    
    Parameters
    ----------
    mconfig : STR
        Model Configuration. Currently supports [SLAB_PIC,FULL_PIC,FULL_HTR]
    frcname : STR
        Name of forcing. Supports ["allrandom","uniform"] or loads "[frcname].npy"
    input_path : STR
        Where all the data is stored
    load_both : BOOL
        Set to true for PIC, loading both FULL and SLAB
    method : [1,2,3,4]
        Statistic test applied to damping
        1 - No test
        2 - SST Autocorr
        3 - SST-FLX Crosscorr
        4 - Both 2 and 3 (Default)
    lagstr : STR
        Lags included (lag1,lag12, or lag123)
    ensorem : BOOL
        Set to True to use damping with ENSO removed (default)
    
    Returns
    -------
    lon : ARRAY [lon,]
        Longitudes
    lat : ARRAY [lat,]
        Latitudes
    h   : ARRAY [lon180 x lat x mon (or time)]
        Mixed Layer depth cycle
    kprevall : ARRAY [lon180 x lat x mon (or time)]
        Indices for detrainment months
    damping  : ARRAY [lon180 x lat x mon (or time)]
        Atmospheric damping
    alpha    : ARRAY [lon180 x lat x pc x mon (or time)]
        Stochastic forcing amplitudes
    
    Adapted from scm.load_data
    """
    # Load lat/lon
    lon           = np.load(input_path+"CESM1_lon180.npy")
    lat           = np.load(input_path+"CESM1_lat.npy")
    nlon,nlat     = len(lon),len(lat)
    if loadpoint: # Get indices if needed
        lonf,latf = loadpoint
        klon,klat = proc.find_latlon(lonf,latf,lon,lat)
    
    # Load Data (MLD and kprev) [lon180 x lat x mon]
    if mconfig == "FULL_HTR": # Load ensemble mean historical MLDs
        h       = np.load(input_path+"%s_HMXL_hclim.npy" % mconfig) # Climatological MLD
        kprevall  = np.load(input_path+"%s_HMXL_kprev.npy" % mconfig) # Entraining Month
    else:
        h       = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
        kprevall  = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
    
    # Load Atmospheric Damping [lon180 x lat x mon]
    if ensorem:
        ensostr = ""
    else:
        print("Loading damping with ENSO present!")
        ensostr = "_ensorem0"
    if mconfig == "SLAB_PIC": # Note older versions used dof894.... why?
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof893_mode%i_%s%s.npy" % (method,lagstr,ensostr))
    elif mconfig == "FULL_PIC":
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof1893_mode%i_%s%s.npy" % (method,lagstr,ensostr))
    elif mconfig =="FULL_HTR":
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode%i_%s%s.npy" % (method,lagstr,ensostr))
    else:
        print("Currently supported damping mconfig are [SLAB_PIC,FULL_PIC,FULL_HTR]")
    
    # Load both damping
    if load_both:  # Note older versions used dof894.... why?
        dampingslab   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof893_mode%i_%s%s.npy" % (method,lagstr,ensostr))
        dampingfull   = np.load(input_path+"FULL_PIC"+"_NHFLX_Damping_monwin3_sig005_dof1893_mode%i_%s%s.npy" % (method,lagstr,ensostr))
        
    # Load Alpha (Forcing Amplitudes) [lon180 x lat x pc x mon], easier for tiling
    if frcname == "allrandom":
        alpha     = np.random.normal(0,1,(nlon,nlat,1,12)) # [lon x lat x 1 x 12]
    elif frcname == "uniform":
        alpha     = np.ones((nlon,nlat,1,12))
    else: # [lon x lat x mon x pc]
        alpha     = np.load(input_path+frcname+".npy")
        if load_both:
            frcnamefull = frcname.replace("SLAB","FULL")
            print(frcnamefull)
            alpha_full  = np.load(input_path+frcnamefull+".npy")
            if loadpoint:
                return lon,lat,h[klon,klat,:],kprevall[klon,klat,:],dampingslab[klon,klat,:],dampingfull[klon,klat,:],alpha[klon,klat,...],alpha_full[klon,klat,...]
            return lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full
    
    # Load point data
    if loadpoint:
        return lon,lat,h[klon,klat,:],kprevall[klon,klat,:],damping[klon,klat,:],alpha[klon,klat,...],alpha_full[klon,klat,...]
    return lon,lat,h,kprevall,damping,alpha,alpha_full

def make_forcing(alpha,runid,frcname,t_end,input_path,check=True,alpha_full=None):
    """
    forcing = make_forcing(alpha,runid,frcname,t_end,input_path)
    
    Scale a white noise time series N(0,1) of length [t_end] by the
    given forcing amplitude [alpha]. Checks for existing file in input_path
    based on runid. "allrandom" forcings will have a separately generated
    timeseries for each point.
    
    Parameters
    ----------
    alpha : ARRAY [lon180 x lat x pc x mon (or time)]
        Stochastic forcing amplitudes
    runid : STR
        ID of the run (user-assigned)
    frcname : STR
        Name of forcing. Supports ["allrandom","uniform"] or loads "[frcname].npy"
    t_end : NUMERIC
        Length of simulation (in units of dt)
    input_path : STR
        Path to load/save white noise timeseries
    check : BOOL
        Set to True to check before ovewriting forcing
    
    Returns
    -------
    forcing : ARRAY [lon180 x lat x pc x time]
        Forcing for stochastic model, in units of W/m2

    """
    if alpha_full is not None:
        flag=True
    else:
        flag=False
    # Get the dimensions
    nlon,nlat,N_mode,nmon = alpha.shape
    
    if flag: # Get shape for FULL run
        _,_,N_mode_full,_ = alpha_full.shape
        N_mode_in = np.max([N_mode,N_mode_full])
    else:
        N_mode_in = N_mode
    
    # Append extra symbols for "allrandom" forcing, make filename
    if frcname == "allrandom":
        runid += "_2D"
    outname = "%srandts_%s_%imon.npy" % (input_path,runid,t_end)
    
    # Check/make random stochastic time series
    query = glob.glob(outname)
    
    # Generate NEW random timeseries if nothing is found ---------------------
    if len(query) < 1: 
        print("Generating new forcing for runid %s..."% runid)
        
        # Generate white noise
        if frcname == "allrandom": # for each point
            randts = np.random.normal(0,1,(nlon,nlat,1,t_end)) # [lon x lat x pc x time]
        else: # Uniform thruout basin, for each PC
            if flag: # Create timeseries with larger # of modes
                randts = np.random.normal(0,1,(1,1,N_mode_in,t_end)) # [1 x 1 x pc,time]
            else:
                randts = np.random.normal(0,1,(1,1,N_mode_in,t_end)) # [1 x 1 x pc,time]
        
        # Save forcing
        np.save(outname,randts)
    
    # Either overwrite or load new timeseries, based on user prompt ----------
    else: 
        if check:
            overwrite = input("Found existing file(s) \n %s. \n Overwite? (y/n)" % (str(query)))
        else:
            overwrite = "n" # Don't overwrite for loops, dont ask for prompt
        
        if overwrite == "y": # Generate new timeseries
            print("Generating new forcing for runid %s..."% runid)
            
            # Generate white noise
            if frcname == "allrandom": # for each point
                randts = np.random.normal(0,1,(nlon,nlat,1,t_end)) # [lon x lat x pc x time]
            else: # Uniform thruout basin, for each PC
                if flag: # Create timeseries with larger # of modes
                    randts = np.random.normal(0,1,(1,1,N_mode_in,t_end)) # [1 x 1 x pc,time]
                else:
                    randts = np.random.normal(0,1,(1,1,N_mode_in,t_end)) # [1 x 1 x pc,time]
            # Save forcing
            np.save(outname,randts)
        elif overwrite == "n": # Load old series
            print("Loading existing random timeseries")
            randts = np.load(outname)
            # Check to make sure there is enough timeseries
            if randts.shape[2] < N_mode_in:
                Nmode_add = N_mode_in - randts.shape[2]
                print("Warning... adding and saving %i random timeseries" % (Nmode_add))
                randtsadd = np.random.normal(0,1,(1,1,Nmode_add,t_end))
                randts = np.concatenate([randts,randtsadd],axis=2)
                np.save(outname,randts)
            
        else:
            print("Invalid input, must be 'y' or 'n'")
    # Resultant randts should be 4-D [lon x lat x pc x time]
    
    # Scale the forcing with the timeseries
    forcing = tile_forcing(alpha,randts)
    if flag:
        forcing_full = tile_forcing(alpha_full,randts)
        return forcing,forcing_full
    return forcing

def tile_forcing(alpha,randts):
    """
    Parameters
    ----------
    alpha : [lon x lat x N_mode x month ]
        Forcing Amplitudes
    randts : TARRAY [1 x 1 x N_mode_max x simulation length]
        Random Timeseries

    Returns
    -------
    forcing : ARRAY [lon x lat x month]

    """
    # Get Dimensions
    nlat,nlon,N_mode,nmon = alpha.shape
    _,_,modemax,t_end = randts.shape
    
    # Scale the forcing with the timeseries
    alpha_tile = alpha.copy()
    if t_end != nmon:
        ntile      = int(t_end/nmon)
        alpha_tile = np.tile(alpha_tile,ntile) #[lon x lat x pc x time]
    forcing = alpha_tile * randts[:,:,:N_mode,:] # Only take needed values

    # Sum the PCs for the forcing
    if N_mode > 1:
        forcing = forcing.sum(2) # Sum along N_mode axis
    forcing = forcing.squeeze()
    
    if len(forcing.shape)<3:
        forcing = forcing[None,None,:] # [lon x lat x mon]
        
    return forcing
    


def make_forcing_pt(alpha,runid,frcname,t_end,input_path,check=True):
    """
    1-D version of make_forcing_pt
    forcing = make_forcing(alpha,runid,frcname,t_end,input_path)
    
    Scale a white noise time series N(0,1) of length [t_end] by the
    given forcing amplitude [alpha]. Checks for existing file in input_path
    based on runid. "allrandom" forcings will have a separately generated
    timeseries for each point.
    
    Parameters
    ----------
    alpha : ARRAY [pc x mon (or time)]
        Stochastic forcing amplitudes
    runid : STR
        ID of the run (user-assigned)
    frcname : STR
        Name of forcing. Supports ["allrandom","uniform"] or loads "[frcname].npy"
    t_end : NUMERIC
        Length of simulation (in units of dt)
    input_path : STR
        Path to load/save white noise timeseries
    check : BOOL
        Set to True to check before ovewriting forcing
    
    Returns
    -------
    forcing : ARRAY [lon180 x lat x pc x time]
        Forcing for stochastic model, in units of W/m2

    """
    
    # Get the dimensions
    N_mode,nmon = alpha.shape
    
    # Append extra symbols for "allrandom" forcing, make filename
    outname = "%srandts_%s_%imon_pt.npy" % (input_path,runid,t_end)
    
    # Check/make random stochastic time series
    query = glob.glob(outname)
    if len(query) < 1: # Generate NEW random timeseries if nothing is found
        print("Generating new forcing for runid %s..."% runid)
        
        # Generate white noise
        randts = np.random.normal(0,1,(N_mode,t_end)) # [pc,time]
        
        # Save forcing
        np.save(outname,randts)
        
    else: # Either overwrite or load new timeseries, based on user prompt
        if check:
            overwrite = input("Found existing file(s) \n %s. \n Overwite? (y/n)" % (str(query)))
        else:
            overwrite = "n" # Don't overwrite for loops, dont ask for prompt
        
        if overwrite == "y": # Generate new timeseries
            print("Generating new forcing for runid %s..."% runid)
            
            # Generate white noise
            randts = np.random.normal(0,1,(N_mode,t_end)) # [1 x 1 x pc x time]
            
            # Save forcing
            np.save(outname,randts)
        elif overwrite == "n": # Load old series
            print("Loading existing random timeseries")
            randts = np.load(outname)
        else:
            print("Invalid input, must be 'y' or 'n'")
    # Resultant randts should be 4-D [lon x lat x pc x time]
    
    # Scale the forcing with the timeseries
    alpha_tile = alpha.copy()
    if t_end != nmon:
        ntile      = int(t_end/nmon)
        alpha_tile = np.tile(alpha_tile,ntile) #[lon x lat x pc x time]
    forcing = alpha_tile * randts[:N_mode,:]
    
    # Sum the PCs for the forcing
    if N_mode > 1:
        forcing = forcing.sum(0)
    forcing = forcing.squeeze()
    return forcing
    
def cut_regions(inputs,lon,lat,bboxsim,pointmode,points=[-30,50],awgt=1):
    """
    outputs,[lonr],[latr] = cut_regions(inputs,lon,lat,bboxsim,pointmode,points=[-30,50],awgt=1)
    
    Cut input variables ([lon x lat x otherdims]) to region specified by bboxsim
    [lonw, lonE, latS, latN] or individual point, determined by pointmode.
    Automatically handles dimensions > 3 by reshaping.

    Parameters
    ----------
    inputs : ARRAY of ARRAYs [lon180 x lat x otherdims]
        Arrays containing variables to cut/crop
    lon : ARRAY [lon,]
        Longitudes
    lat : ARRAY [lat,]
        Latitudes
    bboxsim : ARRAY [lonW,lonE,latS,latN]
        Bounding box of target region
    pointmode : INT [0, 1, or 2]
        0 = Cut to region [bboxsim]
        1 = Select points = [lonf,latf]
        2 = Area-weighted average over [bboxsim]
    points : [ARRAY[lonf,latf], optional
        Coordinates of point to restrict to . The default is [-30,50].
    awgt : INT (0,1,2), optional
        Area weighting (see the area_avg function). The default is 1.

    Returns
    -------
    outputs : ARRAY of ARRAYs [lon180 x lat x otherdims]
        Cropped variables
    if pointmode == 0, also returns lonr, latr for the region
        
    """
    
    # Looping for each variable
    outputs = []
    
    for v in range(len(inputs)):
        invar = inputs[v]
        
        # Temporarily combine dims >= 3
        reshapeflag = False
        if len(invar.shape) > 3: 
            print("Combining dims...")
            reshapeflag = True
            shapevar = invar.shape
            nlon     = shapevar[0]
            nlat     = shapevar[1]
            nother   = np.prod(shapevar[2:])
            invar    = invar.reshape(nlon,nlat,nother)
        
        # Restrict to region/point
        if pointmode == 0: # Restrict to region
            varr,lonr,latr = proc.sel_region(invar,lon,lat,bboxsim)
        elif pointmode == 1: # Restrict to point
            lonf,latf = points
            klon,klat = proc.find_latlon(lonf,latf,lon,lat)
            varr = invar[klon,klat,:]
        elif pointmode == 2: # Area weighted average (cos-weighting)
            varr = proc.area_avg(invar,bboxsim,lon,lat,awgt)
        else:
            print("Invalid pointmode (accepts 0,1, or 2)")
        
        # Reshape back to original
        if reshapeflag:
            if pointmode == 0:
                newshape = np.hstack([[len(lonr),len(latr)],shapevar[2:]])
            else:
                newshape = shapevar[2:]
                #newshape = np.hstack([[1,1],shapevar[2:]])
                #varr = varr[None,None,...] # Add lat/lon singleton dims
            print("Reshaping to size %s" % (str(newshape)))
            varr = varr.reshape(newshape)
        if pointmode != 0:
            varr = varr[None,None,...]
        
        
        # Append to output
        outputs.append(varr)
    if pointmode == 0:
        return outputs,lonr,latr
    else:
        return outputs

def calc_FAC(lbd,correct=True):
    FAC         = np.nan_to_num((1-np.exp(-lbd))/lbd)
    if correct:
        FAC[FAC==0] = 1 # Change all zero FAC values to 1
    return FAC

def calc_beta(h):
    beta = np.log( h / np.roll(h,1,axis=2) ) # h(t)/h(t-1) Is the time correct? Need to test effect
    beta[beta<0] = 0 # Set non-entraining months to zero
    return beta

def integrate_Q(lbd,F,T,mld,cp0=3996,rho=1026,dt=3600*24*30,debug=False):
    """
    Q = integrate_Q(lbd,F,T)\
        
        
    lbd is in 1/mon
    F is in 1/mon
    T is in degC
    
    Integrate the heat flux applied to the stochastic model, and calculate the ratio
    
    """
    nlon,nlat,ntime = F.shape
    mld_in = np.tile(mld,int(ntime/12))
    
    lbd_ori_units = np.tile(lbd,int(ntime/12))*(rho*cp0*mld_in)/dt # convert back to W/m2 per degC
    q_ori_units   = F*(rho*cp0*mld_in)/dt # convert back to W/m2
    
    
    Q           = np.zeros((nlon,nlat,ntime)) * np.nan
    q           = Q.copy()
    lbdT        = Q.copy()
    for t in tqdm(range(ntime)):
        q[:,:,t]    = q_ori_units[:,:,t]
        lbdT[:,:,t] = -lbd_ori_units[:,:,t]*(T[:,:,t]+T[:,:,t-1])/2
        Q[:,:,t]    = q[:,:,t] + lbdT[:,:,t]
    if debug:
        return Q,q,lbdT
    return Q

def integrate_noentrain(lbd,F,T0=0,multFAC=True,debug=False):
    """
    T,[damping_term],[forcing_term] = integrate_noentrain(lbd,F,T0=0,multFAC=True,debug=False)
    
    Integrate non-entraining stochastic model.
    T(t) = exp(-lbd)*T(t-1) + (1-exp(-lbd))/lbd * F(t)
    
    Parameters
    ----------
    lbd : ARRAY [lon x lat x mon]
        Atmospheric damping in units of [1/mon]
    F : ARRAY [lon x lat x time]
        Forcing term in units of [degC/mon]
    T0 : Numeric, optional
        Initial Temperature. The default is 0.
    multFAC : BOOL, optional
        Multiply by integration factor. The default is True.
    debug : BOOL, optional
        Set to true to return all terms separately. The default is False.

    Returns
    -------
    T : ARRAY [lon x lat x time]
        Output temperature
    damping_term : ARRAY [lon x lat x time]
        exp(-lbd)*T(t-1)
    forcing_term : ARRAY [lon x lat x time]
        FAC*F(t)

    """
    nlon,nlat,ntime = F.shape
    
    # Calculate Integration Factor
    FAC = 1
    if multFAC:
        FAC = calc_FAC(lbd)
    
    # Prepare other terms
    explbd = np.exp(-lbd)
    #explbd[explbd==1] =0 # Set the term to zero where damping is insignificant
    
    # Preallocate
    T            = np.zeros((nlon,nlat,ntime))
    T[:,:,0]     = T0 # Set first value
    damping_term = T.copy()
    forcing_term = T.copy()
    
    # Integrate Forward
    for t in tqdm(range(ntime)):
        
        # Get the month 
        m = (t+1)%12 # Start from January, params are same month
        if m == 0:
            m = 12
        
        # Form the terms and step forrward
        damping_term[:,:,t] = explbd[:,:,m-1] * T[:,:,t-1]
        forcing_term[:,:,t] = FAC[:,:,m-1] * F[:,:,t]
        T[:,:,t]            = damping_term[:,:,t] + forcing_term[:,:,t]
    
    # Apply masked based on forcing term
    msk = F.sum(2)
    msk[~np.isnan(msk)] = 1
    T *= msk[:,:,None]
    if np.all(np.isnan(T)):
        print("WARNING ALL ARE NAN")
    
    if debug:
        return T,damping_term,forcing_term
    return T

def integrate_entrain(h,kprev,lbd_a,F,T0=0,multFAC=True,debug=False,Td0=False,add_F=None,return_dict=False,Tdexp=None):
    """
    
    Parameters
    ----------
    h : ARRAY [lon x lat x 12,]
        MIXED LAYER DEPTHS (M)
    kprev : ARRAY [lon x lat x 12]
        INDICES OF DETRAINMENT MONTH
    lbd_a : ARRAY [lon x lat x 12]
        Atmospheric heat flux damping, units of deg/sec.
    F     : ARRAY [lon x lat x 12]
        Forcing, in units of deg/sec
    T0 : Initial temperature, optional
        DESCRIPTION. The default is 0.
    multFAC : TYPE, optional
        DESCRIPTION. The default is True.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    Td0 : ARRAY [len(Tdgrab)]
        DESCRIPTION. The default is False.
    add_F : ARRAYs [lon x lat x time]
        Additional forcing to add to the expression. This is directly
        passed into the function for each point and nothing further is done...
        
    Returns
    -------
    TYPE
        DESCRIPTION.
        
    """
    
    nlon,nlat,ntime = F.shape
    
    # Calculate beta
    beta = calc_beta(h)
    
    # Add entrainment damping, set up lbd terms
    lbd = lbd_a + beta
    #explbd = np.exp(-lbd)
    
    # Calculate Integration Factor
    FAC = 1
    if multFAC:
        FAC = calc_FAC(lbd)
        
    # Set Td damping timescale
    if Tdexp is None:
        Tdexp = np.zeros(FAC.shape)
    
    # Preallocate
    T            = np.zeros((nlon,nlat,ntime))
    damping_term = T.copy()
    forcing_term = T.copy()
    entrain_term = T.copy()
    Td           = T.copy()
    # Loop for each point
    for a in tqdm(range(nlat)):
        
        for o in range(nlon):
            
            # Skip land/ice points, checking the forcing term
            if np.any(np.isnan(F[o,a,:])):
                continue
            
            # Get Last [Tdgrab] values for selected point
            if Td0 is not False:
                Tdgrab = Td0[o,a,:]
            else:
                Tdgrab = None
                
            # If initial values are provided, get value
            T0_in = T0[o,a]
            
            # Check for additional forcing
            if add_F is not None:
                add_F_in = add_F[o,a,:]
            else:
                add_F_in = None
                
            # Integrate in time
            temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts = entrain(ntime,lbd[o,a,:],T0_in,F[o,a,:],beta[o,a,:],h[o,a,:],kprev[o,a,:],FAC[o,a,:],
                                                                multFAC=multFAC,debug=True,debugprint=False,Tdgrab=Tdgrab,add_F=add_F_in,Tdexp=Tdexp[o,a,:])
            
            # Save outputs
            T[o,a,:]            = temp_ts.copy()
            damping_term[o,a,:] = damp_ts.copy()
            forcing_term[o,a,:] = noise_ts.copy()
            entrain_term[o,a,:] = entrain_ts.copy()
            Td[o,a,:]           = Td_ts.copy()
    
    # Apply masked based on forcing term
    msk = F.sum(2)
    msk[~np.isnan(msk)] = 1
    T *= msk[:,:,None]
    if np.all(np.isnan(T)):
        print("WARNING ALL ARE NAN")
    
    if debug:
        if return_dict:
            output_dict = {
                'T'            : T,
                'damping_term' : damping_term,
                'forcing_term' : forcing_term,
                'entrain_term' : entrain_term,
                "Td"           : Td,
                "beta"         : beta,
                "FAC"          : FAC,
                "lbd"          : lbd,
                }
            return output_dict
        return T,damping_term,forcing_term,entrain_term,Td
    return T

def method1(lbd,include_b=True):
    a = 1-lbd
    b = (1-np.exp(-lbd))/lbd
    
    if include_b:
        mult = ((1-b)**2 + (1-b**2)*a + 2*b*a**2)/(1+a)
    else:
        mult = (1+a+b**2*(1-a)-2*(1-a**2))/(1+a)
    return mult

def method2(lbd,include_b=True,original=True):
    a = 1-lbd
    b = (1-np.exp(-lbd))/lbd
    
    # Calculate variance of Q
    if original:
        mid_term  = (b**2 * (1-a)) / (2 * (1+a)**2)
    else:
        mid_term  = (b**2 * (1-a)) / (2)
    
    if include_b:
        last_term = b*(1-a)
    else:
        last_term = (1-a)
    
    mult = 1 + mid_term - last_term
    return mult
    
def run_sm_rewrite(expname,mconfig,input_path,limaskname,
                   runid,t_end,frcname,ampq,
                   bboxsim,pointmode,points=[-30,50],
                   dt=3600*24*30,
                   debug=False,check=True,
                   useslab=0,savesep=False,
                   intgrQ=False,
                   method=4,chk_damping=False,
                   custom_params = {},
                   hconfigs=[0,1,2],
                   continue_run=False,Tdgrab=24,lagstr="lag1",
                   budget=False,
                   ensorem=True
                   ):
    
    """
    Inputs
    ------
    
    1. expname [STR]: Name of experiment output
    2. mconfig [STR]: Model Configuration
    3. input_path [STR]: Where input parameters are contained
    4. limaskname [STR]: Name of file containing land-ice mask
    5. runid [STR]: Name of the white noise timeseries
    6. t_end [INT]: Number of months to integrate over
    7. frcname [INT]: Name of forcing file
    8. ampq [INT]: Type of forcing correction to use
    9. bboxsim [ARRAY]: Simulation bounding box [lonW,lonE,latS,latN]    
    
    ...
    need to finish adding documentation
    10. pointmode [INT]:
    11. points [ARRAY] : Point to run simulation at [lon,lat]
    12. dt [INT]       : Simulation step (default is monthly)
    13. debug [BOOL   ]: Set to true to just run 10-year simulation
    14. check [BOOL]   :
    15. useslab [BOOL] : Use only slab parameters
        0 : Use parameters for respective configuration
        1 : Use SLAB Forcing and Damping
        2 : Use SLAB Damping
        3 : Use SLAB Forcing
        4 : Use FULL Damping
        5 : Use FULL Forcing
    16. savesep [BOOL] :
    17. intgrQ [BOOL]  :
    18. method [INT]   : Heat Flux Significance Testing Method
    ...
    
    19. chk_damping [BOOL]: set True to set negative damping values to 0
    20. custom_params [DICT]: set custom arrays for forcing, damping, mld,q_ek
                Dict Key | Expected Input (Arrays)
                -------- | -----------------------
                "h"       - Mixed Layer Depth :: [LON x LAT x 12]
                "forcing" - Stochastic Forcing Amplitude :: [LON x LAT x EOF_MODE x 12]
                "lambda"  - Atmospheric Heat Flux Feedback :: [LON x LAT x 12]
                "q_ek"    - Name of Ekman Forcing
    21. hconfigs [ARRAY]: Mixed Layer Experiments to run. Options: [0,1,2]
            0 : non-entrain, constant mld, slab parameters
            1 : non-entrain, seasonal mld, full parameters
            2 : entraining , seasonal mld, full parameters
    22. continue_run [STR]: Full path+Name of output experiment to take T0 and Td' from.'
    23. Tdgrab [INT]: Number of previous months to take from previous run for Td' calculation'
    24. budget [BOOL]: Set to true to save each term separately. Currently only available for entrain.
    25. ensorem [BOOL]: Set to false to use damping without enso removed
    """
    
    start = time.time()
    print("Running the following MLD configurations: %s" % (hconfigs))
    
    if debug:
        t_end = 120 # Just run 10 yr
    
    # Load data in
    # ------------
    lon,lat,h,kprevall,damping,dampingfull,alpha,alpha_full = load_inputs(mconfig,frcname,input_path,
                                                                          load_both=True,method=method,
                                                                          lagstr=lagstr,ensorem=ensorem)
    hblt = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
    hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]
    
    # Check if there are custom values
    # Replaces both full and slab values with custom values
    # -----------------------------------------------------
    if "h" in custom_params.keys(): # Replace Mixed Layer
        print("Found custom Mixed Layer Depth!")
        h    = custom_params['h']
        hblt = h.copy()
    if "forcing" in custom_params.keys(): # Replace forcing amplitude
        print("Found custom Forcing!")
        alpha = custom_params['forcing']
        alpha_full = alpha.copy()
    if "lambda" in custom_params.keys():
        print("Found custom Damping!")
        damping = custom_params['lambda']
        dampingfull = damping.copy()
    
    # Apply landice mask to all inputs
    # --------------------------------
    limask    = np.load(input_path+limaskname)
    h           *= limask[:,:,None]
    kprevall    *= limask[:,:,None]
    damping     *= limask[:,:,None]
    dampingfull *= limask[:,:,None]
    alpha       *= limask[:,:,None,None]
    alpha_full  *= limask[:,:,None,None]
    hblt        *= limask[:,:,None]
    
    # Restrict to region or point (Need to fix this section)
    # ---------------------------
    inputs = [h,kprevall,damping,dampingfull,alpha,alpha_full,hblt]
    if pointmode == 0:
        outputs,lonr,latr = cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
    else:
        outputs           = cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
    h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs
    
    
    # Remove negative damping values, if option is set
    # ------------------------------------------------
    if chk_damping:
        dmask = (damping<0)
        dmaskfull = (dampingfull<0)
    
    # Check some params
    # -------------------
    # if debug:
    #     lonf,latf=points
    #     vnames = ["mld","damping","alpha"]
    #     klon,klat = proc.find_latlon(lonf,latf,lonr,latr)
    #     fig,axs = plt.subplots(3,1)
    #     for n,i in enumerate([0,2,3]):
    #         ax = axs.flatten()[n]
    #         if n < 2:
    #             ax.plot(outputs[i][klon,klat])
    #         else:
    #             ax.plot(outputs[i][klon,klat,0,:])
    #         ax.set_title(vnames[n])
    
    # Generate White Noise
    # --------------------
    forcing,forcing_full = make_forcing(alpha,runid,frcname,t_end,input_path,check=check,alpha_full=alpha_full)
    
    # If option is set, repeat procedure to load in ekman forcing
    # ------------------------------------------------------------
    if "q_ek" in custom_params.keys():
        print("Now Including Qek!")
        qek_name = custom_params["q_ek"]
        # Load Qek [lon x lat x mode x month], apply mask -------------
        qek_raw = np.load(input_path+qek_name)
        qek_raw *= limask[:,:,None,None]
        # Restrict to Region ------------------------------------------
        outputs1,_,_ = cut_regions([qek_raw],lon,lat,bboxsim,pointmode,points=points)
        qekr = outputs1[0]
        forcing_qekr = make_forcing(qekr,runid,frcname,t_end,input_path,check=check)
        qek_flag = True
    else:
        qek_flag = False
        
    # If option is set, load Tdexp
    # -----------------------------------------------------------
    if "Tdexp" in custom_params.keys():
        print("Now Including Td Damping")
        Td_name = custom_params["Tdexp"]
        ldTd      = np.load(input_path+Td_name)
        Td_raw  = ldTd['Tddamp']
        # Restrict to Region ------------------------------------------
        outputsTd,_,_ = cut_regions([Td_raw],ldTd['lon'],ldTd['lat'],bboxsim,pointmode,points=points)
        Tdexp = outputsTd[0]
    else:
        Tdexp = None

    # Load data from an older experiment to continue the run
    # ------------------------------------------------------
    if continue_run:
        print("Loading old data from %s"%(continue_run))
        ld = np.load(continue_run,allow_pickle=True)
        T_old = ld['sst']
    else:
        T_old = False
        print("Initializing Experiment from T0=0...")
        
        
    T_all = [] # Run 3 experiments
    for exp in hconfigs:
        if exp == 0: # SLAB Parameters
            h_in = hblt.copy() # Used fixed slab model MLD
            f_in = forcing
            d_in = damping.copy()
        else: # Full Parameters
            h_in = h.copy() # Variable MLD
            f_in = forcing_full
            d_in = dampingfull.copy()
            
        
        # Get old data
        # ------------
        if continue_run:
            T0 = T_old[exp][:,:,-1] # Take the last timestep [lon x lat]
            if exp == 2: #Entraining
                Td0 = T_old[exp][:,:,-Tdgrab:] # Take last Tdgrab years [lon x lat x Tdgrab]
        else:
            T0  = np.zeros((len(lonr),len(latr)))
            Td0 = False
            
        if useslab == 1:
            print("Warning! Using CESM-SLAB Parameters for all cases!")
            f_in = forcing.copy()
            d_in = damping.copy()
        elif useslab == 2:
            print("Warning! Using CESM-SLAB Damping!")
            d_in = damping.copy()
        elif useslab == 3:
            print("Warning! Using CESM-SLAB Forcing!")
            f_in = forcing.copy()
        elif useslab == 4:
            print("Warning! Using CESM-FULL Damping!")
            d_in = dampingfull.copy()
        elif useslab == 5:
            print("Warning! Using CESM-FULL Forcing!")
            f_in = forcing_full.copy()
        
        # Convert to w/m2
        # ---------------
        lbd_a   = convert_Wm2(d_in,h_in,dt)
        F       = convert_Wm2(f_in,h_in,dt) # [lon x lat x time]
        
        #
        # If Option is set, amplitfy F to account for underestimation
        # -----------------------------------------------------------
        if ampq:
            a        = 1-lbd_a
            a        = 1-lbd_a.mean(2)[...,None]
            if ampq == 1:
                print("Doing Old Correction")
                underest = 2*a**2 / (1+a) # Var(Q) = underest*Var(q)
            elif ampq == 2:
                print("Correcting with method 1")
                underest = method1(lbd_a.mean(2)[...,None])
            elif ampq == 3:
                print("Correcting with method 2")
                underest = method2(lbd_a.mean(2)[...,None],original=False) # Original = uncorrected version with error
                
            ntile = int(t_end/a.shape[2])
            ampmult = np.tile(1/np.sqrt(underest),ntile)
            F *= ampmult
        
        # Add Ekman if needed after forcing correction
        # ---------------------------------------------
        if qek_flag:
            if budget:
                F_noek = F.copy() # Save Forcing without Ekman
            Fek    = convert_Wm2(forcing_qekr,h_in,dt)
            F      = F + Fek
            
        # Integrate Stochastic Model
        # --------------------------
        if exp < 2:
            T   = integrate_noentrain(lbd_a,F,T0=T0,multFAC=True,debug=False)
        else:
            if budget:
                print("Saving terms separately for budget analysis. Only the entrain model will run.")
                # Calculate each term separately 
                T,damping_term,forcing_term,entrain_term,Td   = integrate_entrain(h_in,kprev,lbd_a,F,
                                                                                  T0=T0,multFAC=True,debug=True,Td0=Td0,Tdexp=Tdexp)
                
                # Compute some things that were in intergrate_entrain
                beta = calc_beta(h_in)
                lbd  = lbd_a + beta
                FAC  = calc_FAC(lbd_a)
                
            else:
                T   = integrate_entrain(h_in,kprev,lbd_a,F,
                                        T0 = T0,multFAC=True,debug=False,Td0=Td0,Tdexp=Tdexp)
        
        if budget:
            savedict = {
                'sst' : [T,],
                'lon' : lonr,
                'lat' : latr,
                'damping_term' : damping_term,
                'entrain_term' : entrain_term,
                'Td'           : Td,
                'forcing_term' : forcing_term,
                'FAC'          : FAC
                }
            if qek_flag:
                savedict['ekman_term'] = Fek
                savedict['forcing_term'] = F_noek
            
            
            # Save Output (Budget)
            # --------------------
            print(savedict)
            outname = "%s" % (expname)
            np.savez(outname,**savedict,allow_pickle=True)
            print("Saved output to %s in %.2fs" % (outname,time.time()-start))
            
            return
        else:
            T_all.append(T)
        
        if budget:
            print("Error, function has not ended....")
            
        # Integrate Forcing, if the option is set
        # ---------------------------------------
        if exp==0 and intgrQ:
            Q,q,lbdT = integrate_Q(lbd_a,F,T,h_in,debug=True)
            
            # Save output in separate file
            expstr = expname[:-4] + "_model%i_integrQ"%(exp) # Get string without extension, add modelnumber
            np.savez(expstr+".npz",**{
                'lon' : lonr,
                'lat' : latr,
                'Q': Q,
                'q':q,
                'lbdT':lbdT
                },allow_pickle=True)
        
        # Save outputs separately, if option is set
        # -----------------------------------------
        if savesep:
            expstr = expname[:-4] + "_model%i"%(exp) # Get string without extension, add modelnumber
            if exp > 0:
                if useslab == 0: # Only replace if we actually used the separate forcings
                    expstr = expstr.replace("SLAB","FULL") # Replace Slab  with FULL in forcing name
                # Save the results (Not including Q)
            
            np.savez(expstr+".npz",**{
                'sst': T,
                'lon' : lonr,
                'lat' : latr,
                },allow_pickle=True)
            print("Saved results separately to %s"%(expstr))
    
    if (savesep is False) and (budget is False):
        print("Saving output now")
        # Save the results
        np.savez(expname,**{
            'sst': T_all,
            'lon' : lonr,
            'lat' : latr,
            },allow_pickle=True)
        print("Saved output to %s in %.2fs" % (expname,time.time()-start))
    print("Function completed in %.2fs" % (time.time()-start))
    
    


#%% Loading limopt data

def load_limopt_sst(datpath=None,vname="SSTRES"):
    """
    Load SST Data that has been detrended using LIM-opt (Frankignoul et al. 2017).
    Loads the Residual SST (SSTRES) rather than the external trend (SST), change
    this using the vname argument.
    """
    # Set Inputs
    if datpath is None: # Location of lim-opt data
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/lim-opt/"
    dnames = ("COBE","HadISST","ERSST")
    
    # Loop and load into NumPy Arrays
    lons  = []
    lats  = []
    ssts  = []
    times = []
    for d,dname in enumerate(dnames):

        # Read in indices, lat, time
        ds = xr.open_dataset("%s%s-SST-LIM-opt.nc"%(datpath,dname))
        lat = ds.lat.values
        sst = ds[vname].values
        lon360 = ds.lon.values
        times.append(ds.time.values)
        

        # Get Mask Information (HADISST Mask is flipped, need to consider this)
        #dsm = xr.open_dataset("%s%s.MSK.nc"%(datpath,dname))
        #msk = dsm.MSK.values

        # Flip the lon/lat, apply mask, transpose to  time x lat x lon --> lon x lat x time
        lats.append(np.flip(lat))  # Flip lat variable
        sst  = np.flip(sst,axis=1) # Flip latitude axis
        #sst *= msk[None,:,:]       # Apply Mask
        lon180,sst180 = proc.lon360to180(lon360,sst.transpose(2,1,0)) # Transpose 
        ssts.append(sst180)
        lons.append(lon180)
    return ssts,lons,lats,times

def load_limopt_amv(datpath=None):
    """
    Load AMO Data that has been detrended using LIM-opt (Frankignoul et al. 2017).

    """
    # Set Inputs
    if datpath is None: # Location of lim-opt data
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/lim-opt/"
    dnames = ("COBE","HadISST","ERSST")
    
    # Loop and load into NumPy Arrays
    lons  = []
    lats  = []
    ssts  = [] # AMV Pattern
    idxs  = []
    times = []
    for d,dname in enumerate(dnames):

        # Read in indices, lat, time
        ds = xr.open_dataset("%s%s-AMO+PDO-LIMopt.nc"%(datpath,dname))
        lat = ds.lat.values
        sst = ds.SSTAMO.values
        idx = ds.AMO.values
        lon360 = ds.lon.values
        times.append(ds.time.values)
        idxs.append(idx)
        

        # Flip the lon/lat, apply mask, transpose to  time x lat x lon --> lon x lat x time
        lats.append(np.flip(lat))  # Flip lat variable
        sst  = np.flip(sst,axis=0) # Flip latitude axis
        lon180,sst180 = proc.lon360to180(lon360,sst.T) # Transpose 
        sst180[np.where(np.abs(sst180) < 1e-10)] = np.nan # Remove NaN points
        ssts.append(sst180 * np.std(idx)) # Multiply to convert to degC/std_amv
        lons.append(lon180)
        
    return ssts,idxs,lons,lats,times

def load_pathdict(device,csvpath=None,csvname=None):
    """
    Parameters
    ----------
    device : STR
        currently supports 'stormtrack','mbp2019'
    
    csvpath : STR, optional
        Path to csv. The default is None.
    csvname : STR, optional
        Name of csv. The default is None.
    Returns
    -------
    pathdict : DICT
        DICT with keys of the "Category" column and
        entries for the corresponding devices
    """
    
    
    if csvpath is None:
        csvpath = "model/" # Assumed to be in the model folder
    if csvname is None:
        csvname = "sm_proj_path.csv" # Default Name
    df          = pd.read_csv(csvpath+csvname)
    pathdict    = {df['Category'][i]: df[device][i] for i in range(len(df))}
    return pathdict
    
    

#%% Silly Functions to Deal with sm data formats

def unpack_smdict(indict):
    """
    Takes a dict of [run][region][models][OTHERDIMS] and unpacks it into
    an array [unpackaged]

    """
    # Get "Outer Shell" dimensions
    nrun    = len(indict)
    nregion = len(indict[0])
    nmodels = len(indict[0][0])
    
    # For Autocorrelation
    otherdims = indict[0][0][0].shape
    print("Found... Runs (%i) Regions (%i) ; Models (%i) ; Otherdims (%s)" % (nrun,nregion,nmodels,str(otherdims)))
    
    # Preallocate
    newshape = np.concatenate([[nrun,nregion,nmodels],otherdims])
    unpacked = np.zeros(newshape) * np.nan
    
    # Loop thru dict
    for run in range(nrun):
        for reg in range(nregion):
            for mod in range(nmodels):
                unpacked[run,reg,mod,:] = indict[run][reg][mod]
    return unpacked