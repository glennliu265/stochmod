# -*- coding: utf-8 -*-
"""
stochmod_sst, Python Version


This is a temporary script file.
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

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
def stochmod_noentrain(t_end,lbd,T0,F):
    debugmode = 0 # Set to 1 to also save noise and damping time series
    
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

def stochmod_entrain(t_end,lbd,T0,F,beta,h,kprev,FAC):
    debugmode = 0 # Set to 1 to also save noise,damping,entrain, and Td time series
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


"""
SST Stochastic Model, with/wo Entrainment
Integrated with the forward method
assuming lambda at a constant monthly timestep

Dependencies: 
    - numpy as np

 Inputs
 1) t_end : [scalar]      Timestep to integrate until (monthly)
 2) lbd   : [m x n x 12]  Seasonally varying decay term (lambda)
 3) T0    : [m x n]       Initial temperature
 4) F     : [m x n x t ]  Forcing term
    
"""


#%% User Edits -----------------------------------------------------------------           

# Set Point and month
kmon    = 3

# Set Variables
cp0     = 3850 # Specific Heat [J/(kg*C)]
rho     = 1025 # Density of Seawater [kg/m3]

# Initial Conditions/Presets
T0      = 0   # Initial Temp [degC]

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
# 2 = NAO-like NHFLX Forcing, temporally varying 
funiform = 2        

# hvarmode
hvarmode = 2 # hvar modes (0 - fixe mld, 1 - effective mld, 2 - seasonally varying mld)
hfix     = 50 # Fixed MLD (meters)

# Region options
lonW = -80
lonE = 20
latS = -20
latN = 90


# Correlation Options
detrendopt = 0  # Option to detrend before calculations


# White Noise Options. Set to 1 to load data
genrand   = 1  #

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200617/'


# Set up some strings for labeling
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------



# --------------
# %% Load Variables -------------------------------------------------------------
# --------------


# Load damping variables
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
LON = loaddamp['LON1']
LAT = loaddamp['LAT']
damping = loaddamp['ensavg']

# Load Mixed layer variables
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
# ----------------------
# Select non-nan points ------------------------------------------------------
# ----------------------


# ----------------------------
# %% Generate White Noise Forcing------------------------------------------------
# ----------------------------

# Generate Random White Noise Series (or load existing)
if genrand == 1:
    print("Making New Forcing Data in mode %i"% (funiform))
    # Spatially uniform, temporally varying
    if funiform == 1:
        randts = np.random.normal(0,1,size=t_end)/4
        F      = np.ones((lonsize,latsize,t_end))
        F      = np.multiply(F,randts[None,None,:])
    
    # NAO-like forcing
    elif funiform == 2:
        
        # Load data [PC x ENS x Lat x Lon]
        F = np.load(datpath+"NAO_NHFLX_Forcing.npy")
         
        # Take Ensemble Average and PC1. Change to Lon x Lat
        #F = np.transpose(np.nanmean(F[0,:,:,:],axis=0),(1,0))
        F = np.nanmean(F[0,:,:,:],axis=0).T
        
        # Remap Longitude
        lon360 =  np.load(datpath+"CESM_lon360.npy")
        kw = np.where(lon360 >= 180)[0]
        ke = np.where(lon360 < 180)[0]
        lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
        F = np.concatenate((F[kw,:],F[ke,:]),0)
        
        
        # Restrict to Region
        F = F[klon[:,None],klat[None,:]]
        
        # Scale by a time series
        F = F[:,:,None] * (np.random.normal(0,1,size=t_end)/4)
        
        
    else:
        F = np.random.normal(0,1,size=(lonsize,latsize,t_end))
        
        
    np.save(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform),F)
    
else:
    print("Loading Old Data")
    F = np.load(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform))


# ----------------
# %% Set H based on mode------------------------------------------------------------
# ----------------

if hvarmode == 0:
    
    # Use fixed mixed layer depth for the whole basin
    h = hfix
    
elif hvarmode == 1:
    
    
    # Find maximum mld for each point in basin
    hmax = np.amax(hclim,axis=2)
    h = hmax[:,:,None]
    
elif hvarmode == 2:
    
    # Use spatially and temporally varying h
    h = np.copy(hclim)


# ----------------
# %% Calculate Lambda ---------------------------------------------hvar---------------
# ----------------

if usetau == 1:
    lbd = 1/np.mean(tauall,axis=1)
    
    # DDD try last ensemble
    #lbd = 1 / tauall[:,39]

else:
    lbd = dampingr/ (rho*cp0*h) * dt

# Calculate Entrainment Portion
if hvarmode == 2:
    beta = np.nan_to_num(np.log( h / np.roll(h,1) )) # Currently replacing Nans with zero
    beta[beta<0] = 0
    lbd_entr = lbd + beta


    # -------------------------
    # Prepare Reduction Factor --------------------------------------------------
    # -------------------------
    
    # Calculate Reduction Factor
    FAC_entr = np.nan_to_num((1-np.exp(-lbd_entr))/lbd_entr)


# ----------
# %%RUN MODELS -----------------------------------------------------------------
# ----------

# Run Model Without Entrainment
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
            T_entr0[o,a,:],_,_ = stochmod_noentrain(t_end,lbd[o,a,:],T0,F[o,a,:])
        icount += 1
        msg = '\rCompleted No Entrain Run for %i of %i points' % (icount,lonsize*latsize)
        print(msg,end="\r",flush=True)
#
        
elapsed = time.time() - start
tprint = "\nNo Entrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        
        


# %% Run Model With Entrainment
if hvarmode == 2:
    start = time.time()
    T_entr1 = np.zeros((lonsize,latsize,t_end))
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
                T_entr1[o,a,:] = np.zeros(t_end)*np.nan
            else:
                T_entr1[o,a,:],_,_,_,_ = stochmod_entrain(t_end,lbd_entr[o,a,:],T0,F[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:],FAC_entr[o,a,:])
            icount += 1
            msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
    #
            
    elapsed = time.time() - start
    tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
    print(tprint)    
        


# %% save output

np.save(datpath+"stoch_output_1000yr_funiform%i_entrain0_hvar%i.npy"%(funiform,hvarmode),T_entr0)

if hvarmode == 2:
    np.save(datpath+"stoch_output_1000yr_funiform%i_entrain1_hvar%i.npy"%(funiform,hvarmode),T_entr1)

#np.save(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform),F)

