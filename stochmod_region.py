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
    
    else:
        beta1 = np.log( h / np.roll(h,1,axis=0) )
        
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
    lbd[1] = damping / (rho*cp0*hmax[:,:,None]) * dt
    
    # Seasonal MLD
    lbd[2] = damping / (rho*cp0*h) * dt
    
    # Calculate Damping (with entrainment)
    lbd_entr = np.copy(lbd[2]) + beta    
    
    # Compute reduction factor
    FAC = np.nan_to_num((1-np.exp(-lbd_entr))/lbd_entr)
    
    return lbd,lbd_entr,FAC,beta

#%% User Edits -----------------------------------------------------------------           


runid = "000"
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
# 2 = NAO-like NHFLX Forcing, temporally varying 
funiform = 2     
fscale   = 10   

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
outpath = projpath + '02_Figures/20200723/'


# Set up some strings for labeling
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------



# --------------
# %% Load Variables -------------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
LON = loaddamp['LON1']
LAT = loaddamp['LAT']
damping = loaddamp['ensavg']

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

# Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
naoforcing = np.load(datpath+"NAO_NHFLX_Forcing.npy") #[PC x Ens x Lat x Lon]
NAO1 = np.nanmean(naoforcing[0,:,:,:],axis=0).T # Take PC1, Ens Mean and Transpose

# Convert Longitude from degrees East
lon360 =  np.load(datpath+"CESM_lon360.npy")
kw = np.where(lon360 >= 180)[0]
ke = np.where(lon360 < 180)[0]
lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
NAO1 = np.concatenate((NAO1[kw,:],NAO1[ke,:]),0)

# Convert from W/m2 to C/S for the three different mld options
NAOF = {}#np.zeros((NAO1.shape)+(3,)) * np.nan # [Lon x Lat x Hvarmode]
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
        # Monthly Varying Forcing [Lon x Lat x Mon]
        NAOF[i] = NAO1[klon[:,None],klat[None,:],None] * dt / cp0 / rho / hchoose
    else:
        NAOF[i] = NAO1[klon[:,None],klat[None,:]] * dt / cp0 / rho / hchoose


# ----------------------------
# %% Set-up damping parameters
# ----------------------------

lbd,lbd_entr,FAC,beta = set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)

# ----------------------------
# %% Set Up Forcing           ------------------------------------------------
# ----------------------------

# Load in timeseries or full forcing (for funiform == 0)
if genrand == 1:
    print("Generating New Time Series")
    if funiform == 0:
        # Generate nonuniform forcing [lon x lat x time]
        F = np.random.normal(0,1,size=(lonsize,latsize,t_end)) * fscale
        
        # Save Forcing
        np.save(datpath+"stoch_output_%iyr_run%s_funiform%i_Forcing.npy"%(nyr,runid,funiform),F)
    
    else:
        
        randts = np.random.normal(0,1,size=t_end) * fscale
        np.save(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)
    
else:
    print("Loading Old Data")
    if funiform == 0:
        # Directly load forcing
        F = np.load(datpath+"stoch_output_%iyr_run%s_funiform%i_Forcing.npy"%(nyr,runid,funiform))
    else:
        randts = np.load(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),F)


# Make forcing
if funiform != 0:
    
    # Spatially Uniform Forcing
    if funiform == 1:
        F      = np.ones((lonsize,latsize,t_end))
        F      = np.multiply(F,randts[None,None,:])
        
    # NAO Like Forcing...
    elif funiform == 2:
        
        F = {}
        
        # Fixed MLD 
        F[0] = NAOF[0][:,:,None]*randts[None,None,:]
        
        # Max MLD
        F[1] = NAOF[1][:,:,None]*randts[None,None,:]
        
        # Seasonally varying mld...
        F[2] = np.tile(NAOF[2],nyr) * randts[None,None,:]
    
    # Save Forcing
    np.save(datpath+"stoch_output_%iyr_run%s_funiform%i_Forcing.npy"%(nyr,runid,funiform),F)
      
# ----------
# %%RUN MODELS -----------------------------------------------------------------
# ----------

# Run Model Without Entrainment
T_entr0_all = {}
for hi in range(3):
    
    lbdh = lbd[hi]
    if funiform == 2:
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
                T_entr0[o,a,:],_,_ = stochmod_noentrain(t_end,lbdh[o,a,:],T0,Fh[o,a,:])
            icount += 1
            msg = '\rCompleted No Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
    
    T_entr0_all[hi] = np.copy(T_entr0)
    
    elapsed = time.time() - start
    tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
    print(tprint)    
            
        


# %% Run Model With Entrainment
if hvarmode == 2:
    start = time.time()
    T_entr1 = np.zeros((lonsize,latsize,t_end))
    icount = 0
    
    Fh = np.copy(F[2])
    
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
                T_entr1[o,a,:],_,_,_,_ = stochmod_entrain(t_end,lbd_entr[o,a,:],T0,Fh[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:],FAC[o,a,:])
            icount += 1
            msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
    #
            
    elapsed = time.time() - start
    tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
    print(tprint)    
        


# %% save output

np.save(datpath+"stoch_output_1000yr_funiform%i_entrain0_hvar%i.npy"%(funiform,hvarmode),T_entr0_all)
np.save(datpath+"stoch_output_1000yr_funiform%i_entrain1_hvar%i.npy"%(funiform,hvarmode),T_entr1)

#np.save(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform),F)

