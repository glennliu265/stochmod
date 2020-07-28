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
import scm

# %% Functions --------------------------------------------------
# Function to calculate lag correlation
# Dependencies: numpy, scipy
def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
    import numpy as np
    from scipy import signal
    from scipy import stats
    
    
    
    debug = 0
    
    if debug == 1:
        basemonth = kmonth
        lags = lags
        var1 = temps
        var2 = temps
        detrendopt = 1
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    base_ts = np.arange(0+leadsize,totyr-lagsize)
    varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    for i in lags:
        
        
        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts]
        
        # Calculate correlation
        corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
            
        if lagm == 3:
            print(i)
            print(corr_ts[i])
            
            
    return corr_ts


#%% User Edits -----------------------------------------------------------------           


runid = "001"
# Set Point and month
kmon    = 3

# Set Variables
cp0      = 4218 # Specific Heat [J/(kg*C)]
rho      = 1000 # Density of Seawater [kg/m3]


# Initial Conditions/Presets
T0       = 0   # Initial Temp [degC]

# Integration Options
nyr      = 1000    # Number of years to integrate over
t_end    = 12*nyr      # Timestep to integrate up to
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
usetau   = 0          # Use tau (estimated damping timescale)
useeta   = 0          # Use eta from YO's model run
usesst   = 0

# Forcing Type
# 0 = completely random in space time
# 1 = spatially unform forcing, temporally varying
# 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# 3 = NAO-like NHFLX Forcing, Monthly

funiform = 2     
fscale   = 10

# hvarmode
hvarmode = 2 # hvar modes (0 - fixe mld, 1 - effective mld, 2 - seasonally varying mld)
hfix     = 50 # Fixed MLD (meters)

# Region options
lonW = -100
lonE = 20
latS = -20
latN = 90


# Correlation Options
detrendopt = 0  # Option to detrend before calculations

# White Noise Options. Set to 1 to load data
genrand   = 0  #

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200723/'


# Set up some strings for labeling
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------



# --------------
# %% Load Variables -------------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
damppath    = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(damppath+dampmat)
LON         = loaddamp['LON1']
LAT         = loaddamp['LAT']
damping     = loaddamp['ensavg']

# Load Mixed layer variables (preprocessed in prep_mld.py)
mld = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD
kprevall = np.load(datpath+"HMXL_kprev.npy") # Entraining Month


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
#

if funiform == 2:
    # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
    naoforcing = np.load(datpath+"NAO_NHFLX_Forcing.npy") #[PC x Ens x Lat x Lon]
    NAO1 = np.nanmean(naoforcing[0,:,:,:],axis=0).T # Take PC1, Ens Mean and Transpose
    
    # Convert Longitude from degrees East
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    NAO1 = np.concatenate((NAO1[kw,:],NAO1[ke,:]),0)
    
    # Restrict to region
    NAO1 = NAO1[klon[:,None],klat[None,:]]
    


elif funiform == 3:
    # Load NAO Forcing (Monthly)
    naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
    NAO1 = np.nanmean(naoforcing,axis=0) # Take PC1, Ens Mean and Transpose
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude from degrees East
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    NAO1 = np.concatenate((NAO1[kw,:,:],NAO1[ke,:,:]),0)
    
    # Restrict to region
    NAO1 = NAO1[klon[:,None],klat[None,:],:]
    
# Convert from W/m2 to C/S for the three different mld options
NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)

        
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
    np.save(datpath+"stoch_output_%iyr_funiform%i_run%s_Forcing.npy"%(nyr,funiform,runid),F)
      
# ----------
# %%RUN MODELS -----------------------------------------------------------------
# ----------

# Run Model Without Entrainment
T_entr0_all = {}
for hi in range(3):
    
    lbdh = lbd[hi]
    if (funiform == 2) | (funiform == 3):
        Fh = F[hi]
    else:
        Fh = F
    
    
    start = time.time()
    T_entr0 = np.zeros((lonsize,latsize,t_end))
    icount = 0
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
if hvarmode == 2:
    start = time.time()
    T_entr1 = np.zeros((lonsize,latsize,t_end))
    icount = 0
    
    if (funiform == 2) | (funiform == 3):
        Fh = np.copy(F[2])
    else:
        Fh = np.copy(F)
    
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
    #
            
    elapsed = time.time() - start
    tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
    print(tprint)    
        


# %% save output

np.save(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s.npy"%(nyr,funiform,runid),T_entr0_all)
np.save(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s.npy"%(nyr,funiform,runid),T_entr1)
#np.save(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform),F)

