#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Stochastic Model Script

- Rewrite of the stochastic model scripts -

Created on Fri Jul 23 12:44:35 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tqdm

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm


# Run Mode
# pointmode = 0 # Set to 1 to output data for the point speficied below
# points=[-30,50] # Lon, Lat for pointmode

# # Forcing Type
# # 0 = completely random in space time
# # 1 = spatially unform forcing, temporally varying
# # 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# # 3 = NAO-like NHFLX Forcing, with NAO (DJFM) and NHFLX (Monthly)
# # 4 = NAO-like NHFLX Forcing, with NAO (Monthly) and NHFLX (Monthly)
# funiform = 0     # Forcing Mode (see options above)
# fscale   = 1     # Value to scale forcing by

# # ID of the run (determines random number sequence if loading or generating)
# runid = "002"

# # White Noise Options. Set to 1 to load data
# genrand   = 1  # Set to 1 to regenerate white noise time series, with runid above
    
# # Integration Options
# nyr      = 1000        # Number of years to integrate over
# fstd     = 0.3         # Standard deviation of the forcing
# bboxsim  = [-100,20,-20,90] # Simulation Box

# # Running Location
# stormtrack = 1 # Set to 1 if running in stormtrack

# applyfac options
# 0) Forcing is just the White Noise For ing
# 1) Forcing is white noise (numerator) and includes MLD
# 2) Forcing includes both MLD seasonal cycle AND integration factor
# 3) Forcing just includes integration factor


# Types of forcing
# "allrandom" : Completely random in space or time
# "uniform"   : Uniform in space, random in time

#%% Functions




def convert_Wm2(invar,h,dt,cp0=3996,rho=1026,verbose=True):
    """
    outvar = convert_Wm2(invar,h,dt,cp0=3996,rho=1026,verbose=True)
    
    Converts an input array [LON x LAT x TIME] from W/m2 to 1/time
    by multiplying by dt/(rho*cp0*h).
    If input is not 3D, appends dimensions to the front and assumes
    last dimension is time.
    
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
    outvar = invar * dt / (rho*cp0*h)
    
    return outvar.squeeze()

def load_inputs(mconfig,frcname,input_path):
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
    
    # Load Data (MLD and kprev) [lon180 x lat x mon]
    if mconfig == "FULL_HTR": # Load ensemble mean historical MLDs
        h       = np.load(input_path+"%s_HMXL_hclim.npy" % mconfig) # Climatological MLD
        kprevall  = np.load(input_path+"%s_HMXL_kprev.npy" % mconfig) # Entraining Month
    else:
        h       = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
        kprevall  = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
    
    # Load Atmospheric Damping [lon180 x lat x mon]
    if mconfig == "SLAB_PIC":
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
    elif mconfig == "FULL_PIC":
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof1893_mode4.npy")
    elif mconfig =="FULL_HTR":
        damping   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")
    else:
        print("Currently supported damping mconfig are [SLAB_PIC,FULL_PIC,FULL_HTR]")
    
    # Load Alpha (Forcing Amplitudes) [lon180 x lat x pc x mon], easier for tiling
    if frcname == "allrandom":
        alpha     = np.random.normal(0,1,(nlon,nlat,1,12)) # [lon x lat x 1 x 12]
    elif frcname == "uniform":
        alpha     = np.ones((nlon,nlat,1,12))
    else: # [lon x lat x mon x pc]
        alpha     = np.load(input_path+frcname+".npy")
    
    return lon,lat,h,kprevall,damping,alpha

def make_forcing(alpha,runid,frcname,t_end,input_path):
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
    
    Returns
    -------
    forcing : ARRAY [lon180 x lat x pc x time]
        Forcing for stochastic model, in units of W/m2

    """
    
    # Get the dimensions
    nlon,nlat,N_mode,nmon = alpha.shape
    
    # Append extra symbols for "allrandom" forcing, make filename
    if frcname == "allrandom":
        runid += "_2D"
    outname = "%srandts_%s_%imon.npy" % (input_path,runid,t_end)
    
    # Check/make random stochastic time series
    query = glob.glob(outname)
    if len(query) < 1: # Generate NEW random timeseries if nothing is found
        print("Generating new forcing for runid %s..."% runid)
        
        # Generate white noise
        if frcname == "allrandom": # for each point
            randts = np.random.normal(0,1,(nlon,nlat,1,t_end)) # [lon x lat x pc x time]
        else: # Uniform thruout basin, for each PC
            randts = np.random.normal(0,1,(1,1,N_mode,t_end)) # [1 x 1 x pc,time]
        
        # Save forcing
        np.save(outname,randts)
        
    else: # Either overwrite or load new timeseries, based on user prompt
        overwrite = input("Found existing file(s) \n %s. \n Overwite? (y/n)" % (str(query)))
        if overwrite == "y": # Generate new timeseries
            print("Generating new forcing for runid %s..."% runid)
            
            # Generate white noise
            if frcname == "allrandom": # for each point
                randts = np.random.normal(0,1,(nlon,nlat,1,t_end)) # [lon x lat x pc x time]
            else: # Uniform thruout basin, for each PC
                randts = np.random.normal(0,1,(1,1,N_mode,t_end)) # [1 x 1 x pc x time]
            
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
    forcing = alpha_tile * randts[:,:,:N_mode,:]
    
    # Sum the PCs for the forcing
    if N_mode > 1:
        forcing = forcing.sum(2)
    forcing = forcing.squeeze()
    
    # Remake into 3D [lon x lat x time]
    if len(forcing.shape)<3:
        forcing = forcing[None,None,:]
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
    for t in tqdm.tqdm(range(ntime)):
        q[:,:,t]    = q_ori_units[:,:,t]
        lbdT[:,:,t] = -lbd_ori_units[:,:,t]*T[:,:,t]
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
    explbd[explbd==1] =0 # Set the term to zero where damping is insignificant
    
    # Preallocate
    T            = np.zeros((nlon,nlat,ntime)) * T0
    damping_term = T.copy()
    forcing_term = T.copy()
    
    # Integrate Forward
    for t in tqdm.tqdm(range(ntime)):
        
        # Get the month 
        m = (t+1)%12 # Start from January, params are same month
        if m == 0:
            m = 12
        
        # Form the terms and step forrward
        damping_term[:,:,t] = explbd[:,:,m-1] * T[:,:,t-1]
        forcing_term[:,:,t] = FAC[:,:,m-1] * F[:,:,t]
        T[:,:,t] = damping_term[:,:,t] + forcing_term[:,:,t]
    
    # Apply masked based on forcing term
    msk = F.sum(2)
    msk[~np.isnan(msk)] = 1
    T *= msk[:,:,None]
    if np.all(np.isnan(T)):
        print("WARNING ALL ARE NAN")
    
    if debug:
        return T,damping_term,forcing_term
    return T

def integrate_entrain(h,kprev,lbd_a,F,T0=0,multFAC=True,debug=False):
    
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
    
    # Preallocate
    T            = np.zeros((nlon,nlat,ntime)) * T0
    damping_term = T.copy()
    forcing_term = T.copy()
    entrain_term = T.copy()
    Td           = T.copy()
    
    # Loop for each point
    for a in tqdm.tqdm(range(nlat)):
        
        for o in range(nlon):
            
            # Skip land/ice points, checking the forcing term
            if np.any(np.isnan(F[o,a,:])):
                continue
            
            # Integrate in time
            temp_ts,damp_ts,noise_ts,entrain_ts,Td_ts = scm.entrain(ntime,lbd[o,a,:],T0,F[o,a,:],beta[o,a,:],h[o,a,:],kprev[o,a,:],FAC[o,a,:],multFAC=multFAC,debug=True,debugprint=False)
            
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
        return T,damping_term,forcing_term,entrain_term,Td
    return T

#%% Testing Inputs

# Directories
input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"

limaskname = "limask180_FULL-HTR.npy" 

# Model Params
ampq       = True # Set to true to multiply stochastic forcing by a set value
mconfig    = "SLAB_PIC"
frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2" 
#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_q-ek_090pct_SLAB-PIC_eofcorr1" #"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_qek_50eofs_SLAB-PIC" #"uniform" "flxeof_5eofs_SLAB-PIC"
#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_5eofs_SLAB-PIC"
#"flxeof_080pct_SLAB-PIC"
#flxeof_qek_50eofs_SLAB-PIC

runid      = "006"
pointmode  = 0 
points     = [-30,50]
bboxsim    = [-100,20,-20,90] # Simulation Box

# Additional Constants
t_end      = 12000 # Sim Length
dt         = 3600*24*30 # Timestep
T0         = 0 # Init Temp


expname    = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i.npz" % (output_path,frcname,int(t_end/12),runid,ampq) 

lonf = -30
latf = 50

debug = True
#%%
# Load data in
# ------------
lon,lat,h,kprevall,damping,alpha = load_inputs(mconfig,frcname,input_path)
hblt = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]

# Apply landice mask to all inputs
# --------------------------------
limask    = np.load(input_path+limaskname)
h        *= limask[:,:,None]
kprevall *= limask[:,:,None]
damping  *= limask[:,:,None]
alpha    *= limask[:,:,None,None]
hblt     *= limask[:,:,None]

# Restrict to region or point
# ---------------------------
inputs = [h,kprevall,damping,alpha,hblt]
if pointmode == 0:
    outputs,lonr,latr = cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
else:
    outputs = cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
h,kprev,damping,alpha,hblt = outputs

# Check some params
# -------------------
if debug:
    vnames = ["mld","damping","alpha"]
    klon,klat = proc.find_latlon(lonf,latf,lonr,latr)
    fig,axs = plt.subplots(3,1)
    for n,i in enumerate([0,2,3]):
        ax = axs.flatten()[n]
        if n < 2:
            ax.plot(outputs[i][klon,klat])
        else:
            ax.plot(outputs[i][klon,klat,0,:])
        ax.set_title(vnames[n])

# Generate White Noise
# --------------------
forcing = make_forcing(alpha,runid,frcname,t_end,input_path)

T_all = [] # Run 3 experiments

for exp in range(3):
    if exp == 0:
        h_in = hblt.copy() # Used fixed slab model MLD
    else:
        h_in = h.copy() # Variable MLD
    
    # Convert to w/m2
    # ---------------
    lbd_a   = convert_Wm2(damping,h_in,dt)
    F       = convert_Wm2(forcing,h_in,dt) # [lon x lat x time]
    
    #
    # If Option is set, amplitfy F to account for underestimation
    # -----------------------------------------------------------
    if ampq:
        a        = 1-lbd_a
        a        = 1-lbd_a.mean(2)[...,None]
        underest = 2*a**2 / (1+a) # Var(Q) = underest*Var(q)
        ntile = int(t_end/a.shape[2])
        ampmult = np.tile(1/np.sqrt(underest),ntile)
        F *= ampmult
    
    # Integrate Stochastic Model
    # --------------------------
    if exp < 2:
        T   = integrate_noentrain(lbd_a,F,T0=0,multFAC=True,debug=False)
    else:
        T   = integrate_entrain(h_in,kprev,lbd_a,F,T0=0,multFAC=True,debug=False)
    
    T_all.append(T)
    
    if exp==0:
        Q,q,lbdT = integrate_Q(lbd_a,F,T,h_in,debug=True)
        
    
# Save the results
np.savez(expname,**{
    'sst': T_all,
    'lon' : lonr,
    'lat' : latr,
    'Q': Q,
    'q':q,
    'lbdT':lbdT
    },allow_pickle=True)

#%% Compare some results

basemonth = 2
kmonth   = basemonth - 1
lags = np.arange(0,37,1)

ssts = [t[klon,klat,:] for t in T_all]
acs = scm.calc_autocorr(ssts,lags,basemonth)

lat    = np.load(input_path+"CESM1_lat.npy")
lon180 = np.load(input_path+"CESM1_lon180.npy")
lon360 = np.load(input_path+"CESM1_lon360.npy")

ko,ka = proc.find_latlon(lonf+360,latf,lon360,lat)
datpath = input_path + "../"
fullauto = np.load(datpath+"CESM_clim/TS_FULL_Autocorrelation.npy") #[mon x lag x lat x lon360]
cesmslabac     = np.load(datpath+"CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon360]
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

# # Restrict to region based on pointmode
# # Test Conversion
# invar     = damping
# converted = convert_Wm2(invar,h,dt)
# #plt.plot(converted[klon,klat])

fig,ax = plt.subplots(1,1)
ax.plot(cesmauto,label="SLAB")
ax.plot(cesmautofull,label="FULL")
ax.plot(acs[0],label="Stoch H-const")
ax.plot(acs[1],label="Stoch H-vary")
ax.plot(acs[2],label="Stoch Entr")
ax.legend()
