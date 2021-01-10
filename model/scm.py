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

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc
import time

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
    for m in months:
        print("\tCalculating Td for m=%i"%m)
        
        # For m0, check if index=0 and skip if so (first entraining month)
        if (len(months)>1) and (m==months[-1]):
            if index[m-1] == 0:
                Td.append(Td[0])
                print("\t\tSince m0=%i, or first entraining month, Td0=Td1" % m)
                continue
        
        # Find # of months since the anomaly was formed
        k1m = (m1 - np.floor(index[m-1])) % 12
        if k1m == 0:
            k1m = 12
        
        # Get Index in t
        kp1 = int(t - k1m)
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
def noentrain(t_end,lbd,T0,F,FAC,multFAC=1):
    debugmode = 1 # Set to 1 to also save noise and damping time series
    
    # Preallocate
    temp_ts = np.zeros(t_end)
    
    if debugmode == 1:
        damp_ts = np.zeros(t_end)
        noise_ts = np.zeros(t_end)
    else:
        noise_ts = []
        damp_ts = []
        
    # Set value for first timestep
    temp_ts[0] = T0 #"DEC"
    
    # Prepare the entrainment term
    explbd = np.exp(-lbd)
    explbd[explbd==1] = 0
    
    if (multFAC == 1) & (F.shape[0] != FAC.shape[0]):
        F *= np.tile(FAC,int(t_end/12)) # Tile FAC and scale forcing
    
    # Loop for integration period (indexing convention from matlab)
    for t in range(t_end):
        
        # Get the month
        m = (t+1)%12
        if m == 0:
            m = 12
        #print("For t = %i month is %i"%(t,m))

        # Get Noise/Forcing Term
        noise_term = F[t]
        
        # Form the damping term with temp from previous timestep
        if t == 0:
            damp_term = explbd[m-1]*T0
        else:
            damp_term = explbd[m-1]*temp_ts[t-1]
        
        # Compute the temperature
        temp_ts[t] = damp_term + noise_term  
    
        # Save other variables
        if debugmode == 1:
            noise_ts[t] = np.copy(noise_term)
            damp_ts[t]  = np.copy(damp_term)


    # Quick indexing fix
    temp_ts[0] = T0
    if debugmode == 1:
        noise_ts = np.delete(noise_ts,0)
        damp_ts = np.delete(damp_ts,0)
    
    return temp_ts,noise_ts,damp_ts if debugmode ==1 else temp_ts


"""
SST Stochastic Model, with Entrainment
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
def entrain(t_end,lbd,T0,F,beta,h,kprev,FAC,multFAC=1,debug=False):
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
        Set to true to print messages at each timestep. The default is False.

    Returns
    -------
    temp_ts : ARRAY [t_end,]
        Resultant Temperature timeseries
    if debug is True, returns the follow ARRAYs [t_end,]
        noise_ts   - Noise term
        damp_ts    - Damping term
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
                if debug:
                    print("No entrainment on month %i"%m)
                    print("--------------------\n")
            else:
                if Td0 is None: # Calculate Td0 
                    Td1,Td0 = calc_Td(t,kprev,temp_ts,prevmon=True,debug=debug)
                else: # Use Td0 from last timestep
                    Td1 = calc_Td(t,kprev,temp_ts,prevmon=False,debug=debug)
                
                Td = (Td1+Td0)/2
                if debug:
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
            damp_ts[t] = damp_term.copy()
            noise_ts[t] = noise_term.copy() * integration_factor
            entrain_ts[t] = entrain_term.copy() * integration_factor
        
        if debugmode == 1:
            noise_ts[t] = noise_term
            damp_ts[t] = damp_term
            entrain_ts[t] = entrain_term
            Td_ts[t] = Td
    
    if debug:
        return temp_ts,noise_ts,damp_ts,entrain_ts,Td_ts
    return temp_ts



def set_stochparams(h,damping,dt,ND=True,rho=1000,cp0=4218,hfix=50):
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
        
    else:
        beta = np.log( h / np.roll(h,1,axis=0) )
        
        # Find Maximum MLD during the year
        hmax = np.nanmax(np.abs(h))
    
    
    # Set non-entraining months to zero
    beta[beta<0] = 0
    
    # Replace Nans with Zeros in beta
    #beta = np.nan_to_num(beta)
    
    # Preallocate lambda variable 
    lbd = {}
    
    # Fixed MLD
    lbd[0] = damping / (rho*cp0*hfix) * dt
    
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


def find_kprev(h):
    
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



def convert_NAO(hclim,naopattern,dt,rho=1000,cp0=4218,hfix=50):
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
    mld[:,:,:,1]  = np.tile(hclim.max(2)[:,:,None],12) # Max MLD
    mld[:,:,:,2]  = hclim.copy() # Clim MLD
        
        
    # Convert NAO to correct units...
    NAOF = {}
    for i in range(3):
        
        hchoose = mld[:,:,:,i]
        NAOF[i] = naopattern * dt / cp0 / rho / hchoose
    
    return NAOF
    


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

def noentrain_2d(randts,lbd,T0,F,FAC,multFAC=1):
    
    """
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
    for t in range(t_end):
        
        # Get the month
        m = t%12
        if m == 0:
            m = 12
        
        # Form the damping term
        damp_term[:,:,t] = explbd[:,:,m-1] * temp_ts[:,:,t-1]
        
        # Form the noise term
        if F.shape[2] == 12:
            noise_term[:,:,t] = F[:,:,m-1] * randts[None,None,t-1]
        else:
            noise_term[:,:,t] = F[:,:,t-1] 
                
        # Add with the corresponding forcing term to get the temp
        temp_ts[:,:,t] = damp_term[:,:,t] + noise_term[:,:,t]
        
        
        
        msg = '\rCompleted timestep %i of %i' % (t,t_end-1)
        print(msg,end="\r",flush=True)
    
    msk = noise_term.copy()
    msk[~np.isnan(msk)] = 1
    temp_ts *= msk[:,:,:]
    
    if np.all(np.isnan(temp_ts)):
        print("WARNING ALL ARE NAN")
        
    
    return temp_ts,damp_term

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


def postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags,returnresults=False):
    """
    Script to postprocess stochmod output
    
    Inputs:
        1) expid   - "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
        2) datpath - Path to model output
        3) rawpath - Path to raw data that was used as model input
        4) outpathdat - Path to store output data
        5) lags    - lags to compute for autocorrelation
        6) returnresults - option to return results [Bool]
    
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
    regions = ("SPG","STG","TRO","NAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
    
    #% ---- Read in Data ----
    start = time.time()
    
    # Read in Stochmod SST Data
    sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
    lonr = np.load(datpath+"lon.npy")
    latr = np.load(datpath+"lat.npy")
    
    # Load MLD Data
    mld = np.load(rawpath+"HMXL_hclim.npy") # Climatological MLD
    
    # Load Damping Data for lat/lon
    loaddamp = loadmat(rawpath+"ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat")
    lon = np.squeeze(loaddamp['LON1'])
    lat = np.squeeze(loaddamp['LAT'])
    
    print("Data loaded in %.2fs" % (time.time()-start))
    
    #% ---- Get Regional Data ----
    start = time.time()
    nregion = len(regions)
    sstregion = {}
    for r in range(nregion):
        bbox = bboxes[r]
        
        sstr = {}
        for model in range(4):
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
        for model in range(4):
            
            # Get sst and havg
            tsmodel = sstregion[r][model]
            havg,_,_= proc.sel_region(mld,lon,lat,bbox)
            
            # Find kmonth
            havg = np.nanmean(havg,(0,1))
            kmonth     = havg.argmax()
            kmonths[r] = kmonth
            
            
            # Take regional average 
            tsmodel = np.nanmean(tsmodel,(0,1))
            
            # Temp FIX
            if model == 3:
                tsmodel = np.roll(tsmodel,-1) # First t is feb.
            
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
        
        for model in range(4):
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
        