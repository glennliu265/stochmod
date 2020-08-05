#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Stochastic Model Base Code...
Created on Mon Jul 27 11:49:57 2020

@author: gliu
"""

# %%Dependencies

import numpy as np


# %%Functions
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
def noentrain(t_end,lbd,T0,F):
    debugmode = 1 # Set to 1 to also save noise and damping time series
    
    # Preallocate
    temp_ts = np.zeros(t_end)
    
    if debugmode == 1:
        damp_ts = np.zeros(t_end)
        noise_ts = np.zeros(t_end)
    else:
        noise_ts = []
        damp_ts = []
        
    
    # Prepare the entrainment term
    explbd = np.exp(-lbd)

    # Loop for integration period (indexing convention from matlab)
    for t in range(1,t_end):
        
        # Get the month
        m = t%12
        if m == 0:
            m = 12
    
    
        # Get the temperature from the previous step
        if t == 1:
            T = T0
        else:
            T = temp_ts[t-1]
    
        # Get Noise/Forcing Term
        noise_term = F[t-1]
        
        # Form the damping term
        damp_term = explbd[m-1]*T
        
        # Set damping term to zero if damping is insignificant...
        if damp_term == T:
            damp_term=0
        
    
        # Compute the temperature
        temp_ts[t] = damp_term + noise_term  
    
        # Save other variables
        if debugmode == 1:
            noise_ts[t] = np.copy(noise_term)
            damp_ts[t] = np.copy(damp_term)


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
def entrain(t_end,lbd,T0,F,beta,h,kprev,FAC):
    debugmode = 1 # Set to 1 to also save noise,damping,entrain, and Td time series
    linterp   = 1 # Set to 1 to use the kprev variable and linearly interpolate variables
    
    # Preallocate
    temp_ts = np.zeros(t_end)
    if debugmode == 1:
        noise_ts = np.zeros(t_end)
        damp_ts = np.zeros(t_end)
        entrain_ts = np.zeros(t_end)
        Td_ts   = np.zeros(t_end)
    else:
        noise_ts = []
        damp_ts = []
        entrain_ts = []
        Td_ts = []
    
    
    # Prepare the entrainment term
    explbd = np.exp(-lbd)
    
    # Create MLD arrays
    if linterp == 0:
        mlddepths = np.arange(0,np.max(h)+1,1)
        mldtemps = np.zeros(mlddepths.shape)

    # Loop for integration period (indexing convention from matlab)
    for t in range(1,t_end):
        
        
        # Get the month
        m  = t%12

        if m == 0:
            m = 12
        
        # Get the temperature from the previous step
        if t == 1:
            T = T0
        else:
            T = temp_ts[t-1]
            
        
        
        # Calculate entrainment term
        if t<13:
            entrain_term = 0
        else:
            
            # If not an entraining month, skip this step
            if beta[m-1] == 0:
                entrain_term = 0
            else:
                
                
                # Calculate Td
                if linterp == 1:
                    

                    
                    # Get information about the last month
                    m0 = m - 1
                    if m0 == 0:
                        m0 = 12
                    
                    
                    
                    # Find # of months since the anomaly was formed
                    k1m = (m - np.floor(kprev[m-1])) % 12
                    k0m = (m - np.floor(kprev[m0-1])) % 12
                    if k1m == 0:
                        k1m = 12
                    if k0m == 0:
                        k0m = 12
                        
                    
                    
                    
                    # Get the corresponding index month, shifting back for zero indexing
                    kp1 = int(t - k1m)
                    kp0 = int(t - k0m)
    
                                    
                    # Get points (rememebering to shift backwards to account for indexing)
                    # To save computing power, store the Td1 as Td0 for next step?
                    Td1 = np.interp(kprev[m-1],[kp1,kp1+1],[temp_ts[kp1],temp_ts[kp1+1]])
                    if m0-1 == h.argmin():
                        Td0 = Td1
                    else:        
                        Td0 = np.interp(kprev[m0-1],[kp0,kp0+1],[temp_ts[kp0],temp_ts[kp0+1]])
                    
                elif linterp == 0:
                    Td1 = mldtemps[round(h.item(m-1))]
                    Td0 = mldtemps[round(h.item(m0-1))]           
                
                Td = (Td1+Td0)/2
                

                
                # Calculate entrainment term
                entrain_term = beta[m-1]*Td*FAC[m-1]
                
                if debugmode == 1:
                    Td_ts[t] = Td

        
    
        # Get Noise/Forcing Term
        noise_term = F[t-1]
        
        
        # Form the damping term
        damp_term = explbd[m-1]*T
        
        # Set Damping term to zero if feedback is insignificant
        if damp_term == T:
            damp_term = 0
        
        # Compute the temperature
        temp_ts[t] = damp_term + noise_term + entrain_term

        # Save other variables
        if debugmode == 1:
            noise_ts[t] = noise_term
            damp_ts[t] = damp_term
            entrain_ts[t] = entrain_term
        
        
        # Set mixed layer depth tempertures
        if linterp == 0:
            mldtemps[mlddepths<=h.item(m-1)] = temp_ts[t]

    # Quick indexing fix
    temp_ts[0] = T0
    if debugmode == 1:
        noise_ts = np.delete(noise_ts,0)
        damp_ts = np.delete(damp_ts,0)
        entrain_ts = np.delete(entrain_ts,0)
    
    return temp_ts,noise_ts,damp_ts,entrain_ts,Td_ts


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
    beta = np.nan_to_num(beta)
    
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
    
    # Compute reduction factor
    FAC = np.nan_to_num((1-np.exp(-lbd_entr))/lbd_entr)
    
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
        if im == h.argmax() or im== h.argmin():
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
        1) hclim          - climatological MLD [Mons]
        2) NAOF   [Array] - NAO forcing [Lon x Lat] in Watts/m2
        3) dt             - timestep in seconds
        4) rho            - Density of water [kg/m3]
        5) cp0            - Specific Heat of water [J/(K*kg)]
        6) hfix           - Fixed Mixed layer Depth
    
    Output:
        1) NAOF [dict]    - Dictionary of arrays, where 
            0 = fixed MLD
            1 = maximum MLD
            2 = seasonal MLD
    
    """
    
    # Convert NAO to correct units...
    NAOF = {}
    for i in range(3):
    
        # Fixed MLD
        if i == 0:
            hchoose = hfix
        # Max MLD
        elif i == 1:
            hchoose = np.nanmax(np.abs(hclim),axis=2)
        # Varying MLD
        elif i == 2:
            hchoose = np.copy(hclim)
        
        # Compute and restrict to region
        if i == 2:
            
            if len(naopattern.shape) == 2:
                # Monthly Varying Forcing [Lon x Lat x Mon]
                NAOF[i] = naopattern[:,:,None] * dt / cp0 / rho / hchoose
            elif len(naopattern.shape) == 3: # For seasonally varying NAO pattern
                # Monthly Varying Forcing [Lon x Lat x Mon]
                NAOF[i] = naopattern * dt / cp0 / rho / hchoose
        else:
            
            if (len(naopattern.shape) == 3) & (i == 1):
                 NAOF[i] = naopattern * dt / cp0 / rho / hchoose[:,:,None]
            else:
                NAOF[i] = naopattern * dt / cp0 / rho / hchoose
        
    return NAOF
    


def make_naoforcing(NAOF,randts,fscale,nyr):
    """
    Makes forcing timeseries, given NAO Forcing Pattern for 3 different
    treatments of MLD (NAOF), a whiite noise time series, and an scaling 
    parameter
    
    Inputs:
        1) randts [Array] - white noise timeseries varying between -1 and 1
        3) NAOF   [Array] - NAO forcing [Lon x Lat] in Watts/m2
        4) fscale         - multiplier to scale white noise forcing\
        5) nyr    [int]   - Number of years to tile the forcing
    Dependencies: 
        1) 
    
    """
    
    # CMake dictionary
    F = {}
    
    if len(NAOF[0].shape) == 2:
        # Fixed MLD 
        F[0] = NAOF[0][:,:,None]*randts[None,None,:] * fscale
        
        # Max MLD
        F[1] = NAOF[1][:,:,None]*randts[None,None,:] * fscale
        
        # Seasonally varying mld...
        F[2] = np.tile(NAOF[2],nyr) * randts[None,None,:] * fscale
    else:
        for i in range(3):
            
            F[i] = np.tile(NAOF[i],nyr) * randts[None,None,:] * fscale
            
    return F
