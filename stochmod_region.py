# -*- coding: utf-8 -*-
"""
stochmod_sst, Python Version


This is a temporary script file.
"""
from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#import seaborn as sns
import xarray as xr
import time


# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc


#%% User Edits -----------------------------------------------------------------           

# Point Mode
pointmode = 0 # Set to 1 to output data for the point speficied below
lonf = -30
latf = 50

# ID of the run (determines random number sequence if loading or generating)
runid = "001"

# White Noise Options. Set to 1 to load data
genrand   = 1  # Set to 1 to regenerate white noise time series, with runid above

# Forcing Type
# 0 = completely random in space time
# 1 = spatially unform forcing, temporally varying
# 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# 3 = NAO-like NHFLX Forcing, with NAO (DJFM) and NHFLX (Monthly)
# 4 = NAO-like NHFLX Forcing, with NAO (Monthly) and NHFLX (Monthly)
funiform = 0     # Forcing Mode (see options above)
fscale   = 1    # Value to scale forcing by

# Integration Options
nyr      = 1000        # Number of years to integrate over
t_end    = 12*nyr      # Calculates Integration Period
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
T0       = 0           # Initial temperature [degC]
hfix     = 50          # Fixed MLD value (meters)

# Set Constants
cp0      = 3850 # Specific Heat [J/(kg*C)]
rho      = 1000 # Density of Seawater [kg/m3]

# Set Integration Region
lonW = -100
lonE = 20
latS = -20
latN = 90

#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20200723/'





# Set up some strings for labeling
#mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
#monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------
print("Now Running stochmod_region with the following settings: \n")
print("funiform  = " + str(funiform))
print("genrand   = " + str(genrand))
print("runid     = " + runid)
print("pointmode = " + str(pointmode))
print("fscale    = " + str(fscale))
print("nyr       = " + str(nyr))
print("Data will be saved to %s" % datpath)
# --------------
# %% Load Variables -------------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
damppath    = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(damppath+dampmat)
LON         = np.squeeze(loaddamp['LON1'])
LAT         = np.squeeze(loaddamp['LAT'])
damping     = loaddamp['ensavg']

# Load Mixed layer variables (preprocessed in prep_mld.py)
mld         = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(datpath+"HMXL_kprev.npy") # Entraining Month


# Save Options are here
saveforcing = 0 # Save Forcing for each point

# ------------------
# %% Restrict to region ---------------------------------------------------------
# ------------------

# Note: what is the second dimension for?
klat = np.where((LAT >= latS) & (LAT <= latN))[0]
if lonW < 0:
    klon = np.where((LON >= lonW) & (LON <= lonE))[0]
else:
        
    klon = np.where((LON <= lonW) & (LON >= lonE))[0]
          
# Restrict Damping Region
dampingr = damping[klon[:,None],klat[None,:],:]
lonr = np.squeeze(LON[klon])
latr = np.squeeze(LAT[klat])

# Restrict MLD variables to region
hclim = mld[klon[:,None],klat[None,:],:]
kprev = kprevall[klon[:,None],klat[None,:],:]

# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]
np.save(datpath+"lat.npy",latr)
np.save(datpath+"lon.npy",lonr)

# %% Load and Prep NAO Forcing... <Move to separate script?>


if funiform > 1:
    # Load Longitude for processing
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    
    # Load (NAO-NHFLX)_DJFM Forcing
    if funiform == 2:
        
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(datpath+"NAO_NHFLX_Forcing.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1 and take ensemble average
        NAO1 = np.nanmean(naoforcing[0,:,:,:],0) # [Lat x Lon]
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 3:
        
        # Load NAO Forcing and take ensemble average
        naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
        NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
        
        
    elif funiform == 4:
        
        # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
        NAO1 = np.nanmean(naoforcing,0)
    
    
    # Transpose to [Lon x Lat x Time]
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude to Degrees East
    lon180,NAO1 = proc.lon360to180(lon360,NAO1)
    
    # Test Plot
    #plt.pcolormesh(NAO1[:,:,0].T)
    
    NAO1 = NAO1[klon[:,None],klat[None,:],:]
    
    # Convert from W/m2 to C/S for the three different mld options
    NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)
else:
    NAOF = 1
    
    
# ----------------------------
# %% Set-up damping parameters
# ----------------------------

lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)

# ----------------------------
# %% Set Up Forcing           ------------------------------------------------
# ----------------------------

# Load in timeseries or full forcing (for funiform == 0)
if genrand == 1:
    print("Generating New Time Series")
    if funiform == 0:
        # Generate nonuniform forcing [lon x lat x time]
        F = np.random.normal(0,1,size=(lonsize,latsize,t_end))/4 # Divide by 4 to scale between -1 and 1
        
        # Save Forcing
        np.save(datpath+"stoch_output_%iyr_funiform%i_run%s_Forcing.npy"%(nyr,funiform,runid),F)
    
    else:
        
        randts = np.random.normal(0,1,size=t_end)/4 # Divide by 4 to scale between -1 and 1
        np.save(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)
    
else:
    print("Loading Old Data")
    if funiform == 0:
        # Directly load forcing
        F = np.load(datpath+"stoch_output_%iyr_run%s_funiform%i_Forcing.npy"%(nyr,runid,funiform))
    else:
        
        randts = np.load(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))


# Make forcing
if funiform != 0:
    
    # Spatially Uniform Forcing
    if funiform == 1:
        F      = np.ones((lonsize,latsize,t_end))
        F      = np.multiply(F,randts[None,None,:])
        
    # NAO Like Forcing...
    else:
        
        F = scm.make_naoforcing(NAOF,randts,fscale,nyr)
        
        
    # Save Forcing
    if saveforcing == 1:
        np.save(datpath+"stoch_output_%iyr_funiform%i_run%s_Forcing.npy"%(nyr,funiform,runid),F)
      
# ----------
# %%RUN MODELS -----------------------------------------------------------------
# ----------


ko,ka = proc.find_latlon(lonf,latf,lonr,latr)

# Run Model Without Entrainment
T_entr0_all = {}

for hi in range(3):
        
    lbdh = lbd[hi]
    if funiform > 1:
        Fh = F[hi]
    else:
        Fh = F
    
    start = time.time()
    T_entr0 = np.zeros((lonsize,latsize,t_end))
    icount = 0
    
    
    if pointmode == 1:
        T_entr0,_,_=scm.noentrain(t_end,lbdh[ko,ka,:],T0,Fh[ko,ka,:])
    else:    
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
                    msg = "Land Point @ lon %f lat %f" % (lonf,latf)
                    #print(msg)
                    T_entr0[o,a,:] = np.zeros(t_end)*np.nan
                else:
                    
                    T_entr0[o,a,:],_,_ = scm.noentrain(t_end,lbdh[o,a,:],T0,Fh[o,a,:])
                icount += 1
                msg = '\rCompleted No Entrain Run for %i of %i points' % (icount,lonsize*latsize)
                print(msg,end="\r",flush=True)
    
    T_entr0_all[hi] = np.copy(T_entr0)
    
    elapsed = time.time() - start
    tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
    print(tprint)    

        


# Run Model With Entrainment
start = time.time()
T_entr1 = np.zeros((lonsize,latsize,t_end))
icount = 0

if funiform > 1:
    Fh = np.copy(F[2])
else:
    Fh = np.copy(F)


if pointmode == 1:
    T_entr1,_,_,_,_ = scm.entrain(t_end,lbd_entr[ko,ka,:],T0,Fh[ko,ka,:],beta[ko,ka,:],hclim[ko,ka,:],kprev[ko,ka,:],FAC[ko,ka,:])
else:
    
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
                msg = "Land Point @ lon %f lat %f" % (lonf,latf)
                #print(msg)
                T_entr1[o,a,:] = np.zeros(t_end)*np.nan
            else:
                T_entr1[o,a,:],_,_,_,_ = scm.entrain(t_end,lbd_entr[o,a,:],T0,Fh[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:],FAC[o,a,:])
            icount += 1
            msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
        # End Latitude Loop
    # End Longitude Loop
        
elapsed = time.time() - start
tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        


# %% save output


if pointmode == 1:
    
    # Combine entraining and nonentraining models into 1 dictionary
    sst = T_entr0_all.copy()
    sst[3] = T_entr1
    
    np.savez(datpath+"stoch_output_point_%iyr_funiform%i_run%s.npz"%(nyr,funiform,runid),sst=sst,hpt=hclim[ko,ka,:])
else:
    np.save(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s.npy"%(nyr,funiform,runid),T_entr0_all)
    np.save(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s.npy"%(nyr,funiform,runid),T_entr1)
    #np.save(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform),F)

