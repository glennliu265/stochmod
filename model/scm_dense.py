#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:31:04 2023

@author: gliu
"""

# Entrain Model (Single Point)
def entrain(lbd,T0,F,
            h,
            kprev,
            FAC,
            multFAC=1,
            debug=False,
            debugprint=False,
            Tdgrab=None):
    """
    SST Stochastic Model, with Entrainment
    Integrated with the forward method at a single point
    assuming lambda at a constant monthly timestep
    If Tdgrab is given, appends to front for Td calculations, then removes.
    
    Parameters
    ----------
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
    t_end   = len(F)
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
    
    # Calculate FAC
    if multFAC:
        FAC        = calc_FAC(lbd,)
        
    # Calculate beta
    beta           = calc_beta(h)
    
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
        return temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts
    return temp_ts