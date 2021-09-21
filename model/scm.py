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
from scipy import stats
from tqdm import tqdm

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
    explbd[explbd==1] = 0
    
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
def entrain(t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC=1,debug=False,debugprint=False):
    """
    SST Stochastic Model, with Entrainment
    Integrated with the forward method at a single point
    assuming lambda at a constant monthly timestep
    
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
    explbd[explbd==1] = 0
    
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
    explbd[explbd==1] = 0
    
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
    explbd[explbd==1] =0
    
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
                            mask_pacific=False):
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
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    bbox_NA_new = [-80,0,10,65]
    regions = ("SPG","STG","TRO","NAT","NNAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new) # Bounding Boxes
    
    #% ---- Read in Data ----
    start = time.time()
    
    # Read in Stochmod SST Data
    if preload is None: # Manually Load the data
        if "forcing" in expid:
            ld = np.load(datpath+"stoch_output_%s.npz"%(expid),allow_pickle=True)
            sst = ld["sst"]
        else:
            sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
        lonr = np.load(datpath+"lon.npy")
        latr = np.load(datpath+"lat.npy")
    else:
        lonr,latr,sst = preload
    n_models = len(sst)
    
    # Load MLD Data
    mld = np.load(rawpath+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
    
    # Load Damping Data for lat/lon
    loaddamp = loadmat(rawpath+"ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat")
    lon = np.squeeze(loaddamp['LON1'])
    lat = np.squeeze(loaddamp['LAT'])
    
    # Load and apply mask if option is set
    if mask_pacific:
        # Load the mask
        msk = np.load(rawpath+"pacific_limask_180global.npy")
        
        # Select the region
        mskreg,_,_ = proc.sel_region(msk,lon,lat,[lonr[0],lonr[-1],latr[0],latr[-1]])
        
        # Apply the mask to SST
        if preload is None:
            sst *= mskreg[None,:,:,None]
        else:
            sst[0] *= msk[:,:,None]
            sst[1] *= msk[:,:,None]
    
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
            tsmodel   = proc.sel_region(sst[model],lonr,latr,bbox,reg_avg=1,awgt=1)
            
            
            # Commented out below because now first t is Jan.
            ## Temp FIX
            # if model == 3:
            #     tsmodel = np.roll(tsmodel,-1) # First t is feb.
            
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
            amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbboxes[region])
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
    
def calc_autocorr(sst,lags,basemonth):
    """
    Calculate autocorrelation for output of stochastic models
    
    Parameters
    ----------
    sst : DICT
        SST timeseries for each experiment
    lags : ARRAY
        Lags to calculate autocorrelation for
    basemonth : INT
        Month corresponding to lag 0 (ex. Jan=1)
       
    Returns
    -------
    autocorr : DICT
        Autocorrelation stored in same order as sst
    """
    n = len(sst)
    autocorr = {}
    for model in range(n):
        
        # Get the data
        tsmodel = sst[model]
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Plot
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,basemonth,1)
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
    if config['fname'] == 'NAO':
        forcing = forcing[:,:,0,:]
    elif config['fname'] == 'EAP':
        forcing = forcing[:,:,1,:]
    elif config['fname'] == 'EOF3':
        forcing = forcing[:,:,2,:]
    elif config['fname'] == 'FLXSTD':
        forcing = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")
    elif config['fname'] == 'EOF':
        forcing = np.load()
    
    
    # Restrict input parameters to point (or regional average)
    params = get_data(config['pointmode'],config['query'],lat,lon,
                          damping,mld,kprevall,forcing)
    [o,a],damppt,mldpt,kprev,Fpt = params
    kmonth = mldpt.argmax()
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
                   opt=1,dt=3600*24*30,clvl=[.95]):
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
    # Calculate and make individual plots for stochastic model output
    # -----------------------------------------------------------------
    #specparams  = []
    specs = []
    freqs = []
    CCs = []
    dofs = []
    r1s = []
    for i in range(len(sst)):
        sstin = sst[i]
        
        
        # Calculate Spectrum
        if isinstance(nsmooth,int):
            sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
        else:
            sps = ybx.yo_spec(sstin,opt,nsmooth[i],pct,debug=False)

            
        
        # Save spectrum and frequency
        P,freq,dof,r1=sps
        specs.append(P*dt)
        freqs.append(freq/dt)
        dofs.append(dof)
        r1s.append(r1)
        
        # Calculate Confidence Interval
        CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
        CCs.append(CC*dt)
    return specs,freqs,CCs,dofs,r1s

#%% Data Loading

def load_hadisst(datpath,method=2,startyr=1870,grabpoint=None):
    
    hadname  = "%sHadISST_detrend%i_startyr%i.npz" % (datpath,method,startyr)
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

def load_ersst(datpath,method=2,startyr=1854,grabpoint=None):
    
    hadname  = "%sERSST_detrend%i_startyr%i.npz" % (datpath,method,startyr)
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


def load_cesm_pt(datpath,loadname='both',grabpoint=None):
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
    
    
    # Load SSTs
    ssts = []
    if loadname=='both' or loadname=='full': # Load full sst data from model
        ld  = np.load(datpath+"FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
        sstfull = ld['TS']
        ssts.append(sstfull)
        
    if loadname=='both' or loadname=='slab': # Load slab sst data
        ld2 = np.load(datpath+"SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
        sstslab = ld2['TS']
        ssts.append(sstslab)
    
    print("Loaded PiC Data in %.2fs"%(time.time()-st))
    
    # Retrieve point information
    if grabpoint is None:
        return ssts
    else:
        # Query the point
        lonf,latf = grabpoint
        if lonf < 0:
            lonf += 360
        klon360,klat = proc.find_latlon(lonf,latf,lon360,lat)
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


def calc_HF(sst,flx,lags,monwin,verbose=True,posatm=True):
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
    #sst = sst.reshape(int(ntime/12),12,nlat*nlon)
    #flx = flx.reshape(sst.shape)
    
    # Preallocate
    nlag = len(lags)
    damping   = np.zeros((12,nlag,nlat*nlon)) # [month, lag, lat, lon]
    autocorr  = np.zeros(damping.shape)
    crosscorr = np.zeros(damping.shape)
    
    st = time.time()
    for l in range(nlag):
        lag = lags[l]
        for m in range(12):
            lm = m-lag # Get Lag Month
            
            # Restrict to time ----
            flxmon = indexwindow(flx,m,monwin,combinetime=True,verbose=False)
            sstmon = indexwindow(sst,m,monwin,combinetime=True,verbose=False)
            sstlag = indexwindow(sst,lm,monwin,combinetime=True,verbose=False)
            
            # Compute Correlation Coefficients ----
            crosscorr[m,l,:] = proc.pearsonr_2d(flxmon,sstlag,0) # [space]
            autocorr[m,l,:] = proc.pearsonr_2d(sstmon,sstlag,0) # [space]
            
            # Calculate covariance ----
            cov     = proc.covariance2d(flxmon,sstlag,0)
            autocov = proc.covariance2d(sstmon,sstlag,0)
            
            # Compute damping
            damping[m,l,:] = cov/autocov
            
            print("Completed Month %02i for Lag %s (t = %.2fs)" % (m+1,lag,time.time()-st))
            
    # Reshape output variables
    damping = damping.reshape(12,nlag,nlat,nlon)  
    autocorr = autocorr.reshape(damping.shape)
    crosscorr = crosscorr.reshape(damping.shape)
    
    # Check sign
    if posatm:
        if np.nansum(np.sign(crosscorr)) < 0:
            print("WARNING! sst-flx correlation is mostly negative, sign will be flipped")
            crosscorr*=-1
            
    return damping,autocorr,crosscorr

def prep_HF(damping,rsst,rflx,p,tails,dof,mode,returnall=False,posatm=True):
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
        --- OPTIONAL ---
        8) returnall BOOL
            Set to True to return masks and frequency
        6) posatm : BOOL
            Set to True to ensure positive upwards into the atmosphere
    
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
    msst = np.zeros(damping.shape) * np.nan
    mflx = np.zeros(damping.shape) * np.nan
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
    elif mode == 4:
        mtot = msst + mflx
        mall = msst * mflx
        mult = 2
    
    # Apply Significance Mask
    dampingmasked = damping * mall
    
    if returnall:
        return dampingmasked,mtot,mult
    return dampingmasked

def postprocess_HF(dampingmasked,limask,sellags,lon):
    
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
    dampingw *= -1
    return dampingw


#%% SCM, updated equations



