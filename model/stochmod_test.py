#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:20:31 2020

# Testing script for stochmod region, pt, etc

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#import seaborn as sns
import xarray as xr
import time
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import viz
import matplotlib.pyplot as plt

import time

#%% Function Inputs

# Point/Region Options
pointmode  = 2
points     = [-30,50]
region     = 0

bboxsim    = [-60,-15,40,65]
# bbox_SP = [-60,-15,40,65]
# bbox_ST = [-80,-10,20,40]
# bbox_TR = [-75,-15,0,20]
# bbox_NA = [-80,0 ,0,65]
# fullsim = [-100,20,-20,90]

# Forcing Options
funiform   = 1
fscale     = 1
runid      = "000"
genrand    = 1
fstd       = 1

# Other integration options
nyr        = 10000
nobeta     = 0 # Set to 1 to not include beta in lbd entrain



# NEW EXPERIMENT SETTINGS 
mldavg = 0 # Use constant (in time) average MLD values for the region
lbdavg = 0 # Use constant (in time) average atmospheric damping for the region

if pointmode == 2:
    # Set region variables
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    regions = ("SPG","STG","TRO","NAT")
    bboxes  = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
    rcol = ('b','r',[0,1,0],'k')
    rcolmem = [np.array([189,202,255])/255,
               np.array([255,134,134])/255,
               np.array([153,255,153])/255,
               [.75,.75,.75]]
    bboxsim = bboxes[region]


# Outside Function setup
stormtrack = 0
startall = time.time()
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
#    scriptpath  = projpath + '03_Scripts/stochmod/'
    datpath     = projpath + '01_Data/'
    outpath = projpath + '02_Figures/Scrap/'
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")


elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")



# --------------
# % Set Parameters--------------------------------------------------------
# --------------
# Unpack Points if in pointmode
lonf,latf = points

# Other intengration Options (not set by user)
t_end    = 12*nyr      # Calculates Integration Period
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
T0       = 0           # Initial temperature [degC]
hfix     = 50          # Fixed MLD value (meters)

# Set Constants
cp0      = 3850 # Specific Heat [J/(kg*C)]
rho      = 1025 # Density of Seawater [kg/m3]

# Set Integration Region
lonW,lonE,latS,latN = bboxsim

# Save Option
saveforcing = 0 # Save Forcing for each point (after scaling, etc)

# Apply fac options
# 0) Forcing is just the White Noise For ing
# 1) Forcing is white noise (numerator) and includes MLD
# 2) Forcing includes both MLD seasonal cycle AND integration factor
applyfac = 1 # Apply integration factor and MLD to scaling

#Set Paths (stormtrack and local)
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/'
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

import scm
from amv import proc
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'   

## ------------ Script Start -------------------------------------------------

print("Now Running stochmod_region with the following settings: \n")
print("funiform  = " + str(funiform))
print("genrand   = " + str(genrand))
print("fstd      = " + str(fstd))
print("runid     = " + runid)
print("pointmode = " + str(pointmode))
print("fscale    = " + str(fscale))
print("nyr       = " + str(nyr))
print("bbox      = " + str(bboxsim))
print("Data will be saved to %s" % datpath)
allstart = time.time()

# Set experiment ID
expid = "%iyr_funiform%i_run%s_fscale%03d" %(nyr,funiform,runid,fscale)

# --------------
# % Load Variables ------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
LON         = np.squeeze(loaddamp['LON1'])
LAT         = np.squeeze(loaddamp['LAT'])
damping     = loaddamp['ensavg']

# Load Mixed layer variables (preprocessed in prep_mld.py)
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# ------------------
# % Restrict to region --------------------------------------------------
# ------------------

# Note: what is the second dimension for?
dampingr,lonr,latr = proc.sel_region(damping,LON,LAT,bboxsim)
hclim,_,_ = proc.sel_region(mld,LON,LAT,bboxsim)
kprev,_,_ = proc.sel_region(kprevall,LON,LAT,bboxsim)

if mldavg == 1:
    hclim = np.ones(hclim.shape) * np.nanmean(hclim,(0,1,2))
if lbdavg == 1:
    dampingr = np.ones(hclim.shape) * np.nanmean(dampingr,(0,1,2))

# Set artificial MLD
# hclim = np.ones(hclim.shape[0:-1])
# # Set artificial MLD (remove later)
# hclima=np.array([1,1,1,
#         1,1,1,
#         1,100,100,
#         100,1,1])
# hclim = hclim[:,:,None] * hclima[None,None,:]


# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]
np.save(datpath+"lat.npy",latr)
np.save(datpath+"lon.npy",lonr)

# ------------------
# %Prep NAO Forcing ----------------------------------------------------
# ------------------
# Load in forcing data and:
#     - standardize format [lon x lat x time]
#     - take appropriate averages.
#     - restrict to region
#     - convert to deg/C (apply MLD cycle if applicable)
#     - Output as dictionary, indexed by MLD config

if funiform > 1: # For NAO-like forcings (and EAP forcings, load in data and setup)

    # Load Longitude for processing
    lon360 =  np.load(datpath+"CESM_lon360.npy")

    if funiform == 2: # Load (NAO-NHFLX)_DJFM Forcing
        
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1 and take ensemble average
        NAO1 = np.mean(naoforcing[0,:,:,:],0) # [Lat x Lon]
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 3: # NAO (DJFM) regressed to monthly NHFLX
        
        # Load NAO Forcing and take ensemble average
        naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
        NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
        
        
    elif funiform == 4: # Monthly NAO and NHFLX
        
        # # Load Forcing and take ensemble average
        # naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
        # NAO1 = np.nanmean(naoforcing,0)
    
          # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
        # Select PC1 Take ensemble average
        NAO1 = naoforcing[:,:,:,:,0].mean(0)
    
    elif funiform == 5: # DJFM EAP and NHFLX 
    
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC2 and take ensemble average
        NAO1 = naoforcing[1,:,:,:].mean(0)# [Lat x Lon] # Take mean along ensemble dimension
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 6: # DJFM NAO+EAP and NHFLX
    
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1-2 and take ensemble average 
        NAO1 = naoforcing[0:2,:,:,:].mean(1)# [PC x Lat x Lon] # Take mean along ensemble dimension
        # Note that PC is in the "time" dimension
        
    # elif funiform == 7: # Monthly NAO+EAP and NHFLX (need to fix this...)
        
    #     # Load Forcing and take ensemble average
    #     naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
    #     # Take ensemble average, then sum EOF 1 and EOF2
    #     NAO1 = naoforcing[:,:,:,:,:2].mean(0) # [ PC x Mon x Lat x Lon]
        
    # # Temporarily reshape to combine PC and mon
    # if funiform > 6:   
    #     NAO1 = NAO1.reshape(24,192,288) # NOTE: need to uncombine later
    
    # Transpose to [Lon x Lat x Time]
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude to Degrees East
    lon180,NAO1 = proc.lon360to180(lon360,NAO1)
    
    # Restrict to region 
    NAO1,_,_ = proc.sel_region(NAO1,LON,LAT,bboxsim)
else: # For funiform= uniform or random forcing, just make array of ones
    NAO1 = np.ones(hclim.shape)
    
# for ____ NAO1.shape = ...
# funiform1 --> 37x27x12, array of ones
# funiform2 --> 37x27x1, fixed pattern
# funiform3 --> 37x27x12, Monthly Pattern

# funiform5 --> 37x27x1, fixed pattern (EAP-DJFM)
# funiform6 --> 36x27x2, fixed pattern (EAP and NAO like forcing)


# Convert NAO from W/m2 to degC/sec. Returns dict with keys 0-2
NAOF  = {}
NAOF1 = {}
if applyfac == 0: # Don't Apply MLD Cycle
    
    if funiform > 1:
        NAO1 = NAO1 * dt / rho / cp0 # Do conversions (minus MLD)
    
    for i in range(3):
        if funiform > 5: 
            NAOF[i]  = NAO1[:,:,0].copy() # NAO Forcing
            NAOF1[i] = NAO1[:,:,1].copy() # EAP Forcing
        else:
            NAOF[i] = NAO1.copy()
        
else: # Apply seasonal MLD cycle and convert
        
        
    if funiform > 5: # Separately convert NAO and EAP forcing
        NAOF  = scm.convert_NAO(hclim,NAO1[:,:,0],dt,rho=rho,cp0=cp0,hfix=hfix) # NAO Forcing
        NAOF1 = scm.convert_NAO(hclim,NAO1[:,:,1],dt,rho=rho,cp0=cp0,hfix=hfix) # EAP Forcing

    else:
        NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)

"""     
# Outformat: Dict. (keys 0-2, representing MLD type) with [lon x lat x mon]
# We have prepared NAO forcing patterns for the 3 different MLD treatments (if
# applyfac is set. All it requires now is scaling by both the chosen factor and
# white nosie timeseries)
"""

# ----------------------------
# % Set-up damping parameters
# ----------------------------
# Converts damping parameters from raw form (Watts/m2) to (deg/sec)
# Also calculates beta and FAC
# Note: Consider combining with NAO Forcing conversion?

lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)


"""
Out Format:
    lbd -> Dict (keys 0-3) representing each mode, damping parameter
    lbd_entr -> array of entrainment damping
    FAC -> Dict (keys 0-3) representing each model, integration factor
    beta ->array [Lon x Lat x Mon]
"""

# ----------------------------
# %% Set Up Forcing           ------------------------------------------------
# ----------------------------


startf = time.time()
    
# Prepare or load random time series
if genrand == 1: # Generate new time series

    print("Generating New Time Series")
    if funiform == 0: # Create entire forcing array [lon x lat x time] and apply scaling factor
        F = np.random.normal(0,fstd,size=(lonsize,latsize,t_end)) * fscale # Removed Divide by 4 to scale between -1 and 1
        # Save Forcing
        np.save(output_path+"stoch_output_%s_Forcing.npy"%(expid),F)
        
    else: # Just generate the time series
        randts = np.random.normal(0,fstd,size=t_end) # Removed Divide by 4 to scale between -1 and 1
        np.save(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)

else: # Load old data

    print("Loading Old Data")
    if funiform == 0:# Directly load full forcing
        F = np.load(output_path+"stoch_output_%s_Forcing.npy"%(expid))
    else: # Load random time series
        randts = np.load(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))


# Generate extra time series for EAP forcing
if funiform in [5,6,7]:
    numforce = 1 # In the future, incoporate forcing for other EOFs
    if genrand == 1:
        randts1 = np.random.normal(0,fstd,size=t_end) # Removed Divide by 4 to scale between -1 and 1
        np.save(output_path+"stoch_output_%iyr_run%s_randts_%03d.npy"%(nyr,runid,numforce),randts)
    else:
        randts1 = np.load(output_path+"stoch_output_%iyr_run%s_randts_%03d.npy"%(nyr,runid,numforce))
        
    if funiform == 5: # Assign EAP Forcing white noise time series
        randts = randts1
    
        

# Use random time series to scale the forcing pattern
if funiform != 0:
    
    
    # if (funiform == 1) & (applyfac==0):# Spatially Uniform Forcing, replicate to domain and apply scaling factor
        
    #     F1 = {}
    #     for hi in range(3):
    #         # Tile monthly data to simulation length and scale by white noise time series
    #         F1[hi] = np.tile(NAOF[hi],nyr) * fscale *  randts[None,None,:]
    #     Fseas1  = NAOF.copy()
        
    if funiform in [5,6,7]: # NAO + EAP Forcing
        F,Fseas   = scm.make_naoforcing(NAOF,randts,fscale,nyr) # Scale NAO Focing
        F1,Fseas1 = scm.make_naoforcing(NAOF1,randts1,fscale,nyr) # Scale EAP forcing
        
        
        # Add the two forcings together
        for hi in range(3):
            F[hi]     += F1[hi]
            Fseas[hi] += Fseas1[hi]
            
    else: # NAO Like Forcing of funiform with mld/lbd factors, apply scaling and randts
    
        F,Fseas = scm.make_naoforcing(NAOF,randts,fscale,nyr)
    

    # Save Forcing if option is set
    if saveforcing == 1:
        np.save(output_path+"stoch_output_%s_Forcing.npy"%(runid),F)
        
print("Forcing Setup in %.2fs" % (time.time() - startf))

"""
Output:
    F - dict (keys = 0-2, representing each MLD treatment) [ lon x lat x time (simulation length)]
    Fseas - dict (keys = 0-2, representing each MLD treatment) [ lon x lat x  month]
    
    
"""
# ----------------------------
# %% Additional setup based on pointmode  ------------------------------------------------
# ----------------------------    

if pointmode == 1: # Find indices for pointmode

    # Get indices for selected point and make string for plotting
    ko,ka = proc.find_latlon(lonf,latf,lonr,latr)
    locstring = "lon%02d_lat%02d" % (lonf,latf)
    
    # Select variable at point
    hclima   = hclim[ko,ka,:]
    dampinga = dampingr[ko,ka,:]
    kpreva   = kprev[ko,ka,:]
    lbd_entr = lbd_entr[ko,ka,:]
    beta     = beta[ko,ka,:]
    naoa     = NAO1[ko,ka,...]
    
    # Do the same for dictionaries indexed by MLD config
    Fa    = {} # Forcing
    Fseasa = {} # Seasonal Forcing pattern
    for hi in range(3):
        Fa[hi] = F[hi][ko,ka,:]
        Fseasa = Fseas[hi][ko,ka,:]
    F = Fa.copy()
    Fseas=Fseasa.copy()
    
    # Do the same but for each model type
    lbda = {}
    FACa = {}
    for model in range(4):
        FACa[model] = FAC[model][ko,ka,:]
        lbda[model] = lbd[model][ko,ka,:]
    lbd = lbda.copy()
    FAC = FACa.copy()

if pointmode == 2: # Take regionally averaged parameters (need to recalculate some things)
    
    # Make string for plotting
    locstring = "lon%02d_%02d_lat%02d_%02d" % (lonW,lonE,latS,latN)
    
    # Current setup: Average raw variables, assuming
    # that bboxsim is the region you want to average over
    hclima    = np.nanmean(hclim,(0,1)) # Take lon,lat mean, ignoring nans
    kpreva    = scm.find_kprev(hclima)[0]
    dampinga  = np.nanmean(dampingr,(0,1)) # Repeat for damping
    naoa      = np.nanmean(NAO1,(0,1)) # Repeat for nao forcing
    
    # Get regionally averaged forcing based on mld config
    rNAOF = {}
    rF    = {}
    for hi in range(3):
        rNAOF[hi] = proc.sel_region(NAOF[hi],lonr,latr,bboxsim,reg_avg=1)
        rF[hi] = randts * np.tile(rNAOF[hi],nyr)
        
    # Add in EAP Forcing [consider making separate file to save?]
    if funiform in [6,7]: # NAO + EAP Forcing
        for hi in range(3):
            rNAOF1 = proc.sel_region(NAOF1[hi],lonr,latr,bboxsim,reg_avg=1)
            rF1 = randts1 * np.tile(rNAOF1,nyr)
            
            # Add to forcing
            rNAOF[hi] += rNAOF1
            rF[hi] += rF1
    # Copy over forcing
    F = rF.copy()
    Fseas = rNAOF.copy()

    # Recalculate parameters based on reigonal averages
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclima,dampinga,dt,ND=0,rho=rho,cp0=cp0,hfix=hfix)
    
    
# Remove entrainment from damping term if option is set
if nobeta == 1:
    lbd[3] = lbd[2]


"""
Output:
    
Dict with keys 0-2 for MLD configuation
    - F (Forcing, full timeseries)
    - Fseas (Forcing, seasonal pattern)

Dict with keys 0-3 for Model Type
    - lbd (damping parameter)
    - FAC (integration factor)

Just Arrays...
    - beta (entrainment velocity)
    - dampinga (atmospheric damping)
    - hclima (mixed layer depth)
    - kpreva (entraining month)
    - naoa (NAO forcing pattern)
    
"""

# ----------
# %%RUN MODELS -----------------------------------------------------------------
# ----------

# FAC  - dict [0-3]
# lbd  - dict [0-4]
# lbd_entr - Array [lon x lat x mon]
# randts - Array [time]
# hclim - Array [lon x lat x mon]
# kprev - Array [lon x lat x mon]


# Set mulFAC condition based on applyfac
if applyfac == 2:
    multFAC = 1 # Don't apply integrationreduction factor if applyfac is set to 0 or 1
    
else:
    multFAC = 0

# Run Model Without Entrainment
sst = {}
# Loop for each Mixed Layer Depth Treatment
for hi in range(3):
    start = time.time()
    
    # Select damping and FAC based on MLD
    FACh = FAC[hi]
    lbdh = lbd[hi]
    
    # Select Forcing
    Fh  = F[hi]
    # if (funiform > 1) | (applyfac==1):
    #     Fh = F[hi]
    # else:
    #     Fh = F
    

        
    if pointmode == 0: #simulate all points
    
        # Match Forcing and FAC shape
        if (len(Fh.shape)>2) & (Fh.shape[2] != FACh.shape[2]):
            FACh = np.tile(FACh,int(t_end/12))
            
        if Fh.shape[2] < 12: # Adjust for cases where Fh is not seasonal
            Fh = np.tile(Fh,12)
        if funiform == 0:
            randts = np.copy(Fh)
        
        sst[hi],_ =  scm.noentrain_2d(randts,lbdh,T0,Fh,FACh,multFAC=multFAC)
        print("Simulation for No Entrain Model, hvarmode %s completed in %s" % (hi,time.time() - start))
    
    else:
        # Run Point Model
        start = time.time()
        
        sst[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)
        
        elapsed = time.time() - start
        tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
        print(tprint)    
        
    # elif pointmode == 1: # simulate for 1 point
    #     start = time.time()

    #     # Run Point Model
    #     sst[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)

    #     elapsed = time.time() - start
    #     tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
    #     print(tprint)
    
    # elif pointmode == 2: # simulate using regionally averaged params
    
    #      # Run Point Model
    #     start = time.time()
    #     sst[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)
        
    #     elapsed = time.time() - start
    #     tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
    #     print(tprint)    
  

# Run Model With Entrainment
start = time.time()

icount = 0
Fh = F[2]
# if (funiform > 1) | (applyfac==1):
#     Fh = np.copy(F[2])
# else:
#     Fh = np.copy(F)
FACh = FAC[3]

if pointmode == 1:
    
    sst[3]= scm.entrain(t_end,lbd[3],T0,Fh,beta,hclima,kpreva,FACh,multFAC=multFAC)
    
elif pointmode == 2:
    
    sst[3]= scm.entrain(t_end,lbd[3],T0,Fh,beta,hclima,kpreva,FACh,multFAC=multFAC)
    
else:
    
    T_entr1 = np.ones((lonsize,latsize,t_end))*np.nan
    for o in range(0,lonsize):
        # Get Longitude Value
        lonf = lonr[o]
        
        # Convert to degrees East
        if lonf < 0:
            lonf = lonf + 360
        
        for a in range(0,latsize):

            # Get latitude indices
            latf = latr[a]
            
            
            # Skip if the point is land
            if np.isnan(np.mean(dampingr[o,a,:])):
                #msg = "Land Point @ lon %f lat %f" % (lonf,latf)
                icount += 1
                continue
                #print(msg)

            else:
                T_entr1[o,a,:] = scm.entrain(t_end,lbd[3][o,a,:],T0,Fh[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:],FACh[o,a,:],multFAC=multFAC)
                
                # lbdin   = np.copy(lbd_entr[o,a,:])
                # Fin     = np.copy(Fh[o,a,:])
                # betain  = np.copy(beta[o,a,:])
                # hclimin = np.copy(hclim[o,a,:])
                # kprevin = np.copy(kprev[o,a,:])
                # FACin   = np.copy(FAC[o,a,:])
                # delayedtask = dask.delayed(scm.entrain)(t_end,lbdin,T0,Fin,betain,hclimin,kprevin,FACin)
                # T_entr1[o,a,:] = delayedtask
            icount += 1
            msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
        #End Latitude Loop
    #End Longitude Loop
    
    # Copy over to sst dictionary
    sst[3] = T_entr1.copy()

#T_entr1 = dask.compute(*T_entr1)
elapsed = time.time() - start
tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        


# %% save output

# if pointmode > 0:
    
#     if pointmode == 1:
#         np.savez(output_path+"stoch_output_point%s_%s.npz"%(locstring,expid),sst=sst,hpt=hclim[ko,ka,:])
        
#     elif pointmode == 2:
#         np.savez(output_path+"stoch_output_point%s_%s.npz"%(locstring,expid),
#                  sst=sst,
#                  hclim=hclima,
#                  kprev=kpreva,
#                  dampping=dampinga,
#                  F=F,
#                  lbd=lbd,
#                  lbd_entr=lbd_entr,
#                  beta=beta,
#                  FAC=FAC,
#                  NAO1=NAO1,
#                  NAOF=NAOF
#                  )
    
# else:
    
#     # SAVE ALL in 1
#     np.save(output_path+"stoch_output_%s.npy"%(expid),sst)
    
#     #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain0_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr0_all)
#     #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain1_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr1)

# print("stochmod_region.py ran in %.2fs"% (time.time()-allstart))
# print("Output saved as %s" + output_path + "stoch_output_%s.npy"%(expid))



#%% Make some  plots


# Set Strings
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$","$EAP_{DJFM}$","(EAP+NAO)_{DJFM}")
regions = ("SPG","STG","TRO","NAT")
modelname = ("MLD Fixed","MLD Max", "MLD CSeasonal", "Entrain")
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monname=('January','February','March','April','May','June','July','August','September','October','November','December')

if pointmode == 1:
    loctitle = "LON %02d Lat %02d" % (lonf,latf)
else:
    loctitle = regions[region]
    locstring = loctitle
    
    
# Set Experiment ID
expid = "run%s_%s_funiform%i_fscale%i_applyfac%i" % (runid,locstring,funiform,fscale,applyfac)


#%% Plot Point

model = 3 # Select Model( 0:hfix || 1:hmax || 2:hvar || 3: entrain)

mon = 6
xl  = [11940,12000]
#yl = [-1e200,1e200] 



sstpt = sst[model]
if model == 3:
    Fpt = F[2]
else:
    Fpt = F[model]

# Calculate Annual Averages
sstptann = proc.ann_avg(sstpt,0)
Fptann   = proc.ann_avg(Fpt,0)

# Set plotting parameters and text
tper = np.arange(0,t_end)
yper = np.arange(0,t_end,12)
fstats = viz.quickstatslabel(Fpt)
tstats = viz.quickstatslabel(sstpt)

# Start plot
fig,ax = plt.subplots(2,1,figsize=(8,6))
plt.style.use('ggplot')

plt.subplot(2,1,1)
plt.plot(tper,Fpt)
plt.plot(yper,Fptann,color='k',label='Ann. Avg')
plt.ylabel("Forcing ($^{\circ}C/s$)",fontsize=10)
plt.legend()
plt.title("%s Forcing at %s with Fscale %.2e \n %s " % (forcingname[funiform],loctitle,fscale,fstats))


plt.subplot(2,1,2)
plt.plot(tper,sstpt)
plt.plot(yper,sstptann,color='k',label='Ann. Avg')
plt.ylabel("SST ($^{\circ}C$)",fontsize=10)
plt.xlabel("Time(Months)",fontsize=10)
plt.legend()
#plt.yscale('log')
#plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.title("SST (%s) \n %s" % (modelname[model],tstats))

plt.xlim(xl)
#plt.ylim(yl)
plt.xticks(np.arange(11940,12006,6))
plt.vlines(np.arange(xl[0]+mon-1,xl[1],12),1,1)


plt.tight_layout()

plt.savefig(outpath+"Stochmodpt_dsdt_SST_model%i_%s.png"%(model,expid),dpi=200)





#%%

fig,ax = plt.subplots(1,1)
ax.plot(tper[5000:100],sstpt[1:100])
#ax.set_yscale('log')
#ax.set_ylim([-2e287,0])

#%% Plot the sst autocorrelation

# Load CESM Data
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()


lags = np.arange(0,37,1)
xlim = [0,36]
xtk =  np.arange(xlim[0],xlim[1]+2,2)
plt.style.use("seaborn-bright")


kmonth = hclima.argmax() # kmonth is the INDEX of the mongth

autocorr = {}
for model in range(4):
    
    # Get the data
    tsmodel = sst[model]
    tsmodel = proc.year2mon(tsmodel) # mon x year
    
    # Deseason (No Seasonal Cycle to Remove)
    tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
    
    # Plot
    autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
    

# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")

# Plot CESM
accesm = cesmauto[region]
ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)

for model in range(4):
    ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale: %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s.png"%(kmonth+1,expid),dpi=200)



# Plot Some relevant parameters
#%% Plot the inputs (RAW)
fig,axs=plt.subplots(3,1,sharex=True,figsize=(6,6))
ax=axs[0]
ax.set_title("Mixed Layer Depth")
ax.set_ylabel("meters")
ax.plot(mons3,hclima,color='b')

ax=axs[1]
ax.set_title("$\lambda_{a}$")
ax.set_ylabel("W$m^{-2}$")
ax.plot(mons3,dampinga,color='r')

ax=axs[2]
ax.set_title("$Forcing$")
ax.set_ylabel("W$m^{-2}$")
ax.plot(mons3,dampinga,color='r')

#%% Plot secondary inputs, post processing


#% Plot the 4 lambdas
fig,axs = plt.subplots(3,1,sharex=True,figsize=(6,6))

ax=axs[0]
for model in range(4):
    ax.plot(mons3,lbd[model],label=modelname[model])
ax.set_ylabel("degC/sec")
ax.set_xlabel("Month")
ax.set_title(r"$\lambda (\rho cp_{0}H)^{-1}$")
ax.legend()
    
ax=axs[1]
for model in range(4):
    ax.plot(mons3,FAC[model],label=modelname[model])
ax.set_title(r"Integration Factor")    

#%%

model = 2


xlims = (0,36)
# Plot Autocorrelation
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn")
#for model in range(4):
ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")

# Plot seasonal autocorrelation
choosevar = "MLD"

if choosevar == "Damping":
    invar = dampinga
    varname = "Damping (ATMOSPHERIC)"
    
elif choosevar == "Beta":
    invar = beta
    varname = "$ln(h(t+1)/h(t))$"
    
elif choosevar == "MLD":
    invar = hclima
    varname = "Mixed Layer Depth"
    
elif choosevar == "Lambda Entrain":
    #invar = np.exp(-lbd_entr*dt/(rho*cp0*hpt))
    invar = lbd_entr
    varname = "Lambda (Entrain)"
    
elif choosevar == "Lambda":
    invar = lbd[model]
    varname = "Lambda Mode %i" % model
    
elif choosevar == "Forcing":
    invar = naopt
    varname = "NAO Forcing (W$m^{-2}$)"
elif choosevar == "Fmag":
    invar = Fseas[model] #* fscale
    varname = "Forcing Magnitude with %s (degC)" % modelname[model]
elif choosevar == "NAO":
    invar = naopt
    varname = "NAO Forcing"
elif choosevar == "Damping Term":

    if hm == 0:
        h = hfix
    elif hm == 1:
        h = np.max(hpt)
    elif hm == 2:
        h = hpt
    
    if hm < 3:
        #invar = np.exp(-damppt*dt/(rho*cp0*h))
        invar = np.exp(-lbd[hm])
    else:
        #invar = np.exp(-lbd_entr*dt/(rho*cp0*hpt))
        invar = np.exp(-lbd_entr)
    varname = "Damping Term (degC/mon) Mode %i" % hm


    
# Find index of maximum and roll so that it is now the first entry (currently set to only repeat for 5 years)
kmonth = hclima.argmax()
maxfirsttile = np.tile(np.roll(invar,-1*kmonth-1),3)
maxfirsttile = np.concatenate((maxfirsttile,[maxfirsttile[0]]))

# Twin axis and plot
ax2 = ax.twinx()
ax2.plot(lags,maxfirsttile,color=[0.6,0.6,0.6])
ax2.grid(False)
ax2.set_ylabel(varname)
plt.xlim(xlims)
plt.tight_layout()
plt.savefig(outpath+"scycle2Fixed_SST_Autocorrelationv%s_Mon%02d_%s.png"%(choosevar,kmonth+1,expid),dpi=200)


fig,ax = plt.subplots(2,1)
ax[0].plot(mons3,hclima)
ax[1].plot(mons3,lbd[2])

#%% Plot to Compare Inclusion of Beta in Damping

# Note: I first ran the script with nobeta=1, and saved the following:
#autocorr_nobeta = np.copy(autocorr[3])
# Make sure to run again with nobeta=0 and run the autocorrelation calculation

modcol = ['b','g','r','darkviolet']

# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")

# Plot CESM
accesm = cesmauto[region]
ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)

# Plot Hseas Model
model =2
ax.plot(lags,autocorr[model],label=modelname[model],color=modcol[model])

# Plot Entrain with beta
model =3
ax.plot(lags,autocorr[model],label=modelname[model]+" with Beta",color=modcol[model],ls="dotted")

# Plot Entrain without beta
ax.plot(lags,autocorr_nobeta,label=modelname[model]+" without Beta",color=modcol[model])

plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale: %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend(ncol=3,fontsize=8)
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s_betacomparison.png"%(kmonth+1,expid),dpi=200)


#%% Tiny plot of beta

fig,ax = plt.subplots(1,1,figsize=(4,2))
ax.plot(mons3,beta)
ax.set_title("Beta")
ax.set_xlabel("Months")
plt.savefig(outpath+"Beta_%s.png"%(locstring),dpi=200)

#%% Comparison plot of the lambda, before and after beta inclusion

modcol = ['b','g','r','darkviolet']
fig,ax = plt.subplots(1,1,figsize=(4,3))
lbdname = [0,0,r"$\lambda (\rho cp_{0}H)^{-1}$",r"$\lambda (\rho cp_{0}H)^{-1}$"+r"+ $w_{e}H^{-1}$"]
for model in [2,3]:
    ax.plot(mons3,lbd[model],label=lbdname[model],color=modcol[model])
ax.set_ylabel("degC/sec")
plt.legend()
plt.savefig(outpath+"Lbd_Seas_comparison_%s.png"%(locstring),dpi=200)



#%% Explore effect of different starting months....

# Load CESM Data
#cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()

lags = np.arange(0,37,1)
xlim = [0,36]
xtk =  np.arange(xlim[0],xlim[1]+2,2)
plt.style.use("seaborn-bright")


for m in range(12):
    kmonth = m






    #kmonth = hclima.argmax() # kmonth is the INDEX of the mongth

    autocorr = {}
    for model in range(4):
        
        # Get the data
        tsmodel = sst[model]
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Plot
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
    

    # plot results
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use("seaborn-bright")
    
    # Plot CESM
    #accesm = cesmauto[region]
    #ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)
    
    for model in range(4):
        ax.plot(lags,autocorr[model],label=modelname[model])
    plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale: %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
    plt.xticks(xtk)
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.style.use("seaborn-bright")
    plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s.png"%(kmonth+1,expid),dpi=200)