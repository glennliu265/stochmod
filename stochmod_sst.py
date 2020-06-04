# -*- coding: utf-8 -*-
"""
stochmod_sst, Python Version


This is a temporary script file.
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set Point and month
lonf    = -30
latf    = 50
kmon    = 3

# Set Variables
cp0     = 3850 # Specific Heat [J/(kg*C)]
rho     = 1025 # Density of Seawater [kg/m3]

# Initial Conditions/Presets
h       = 150 # Effective MLD [m]
T0      = 0   # Initial Temp [degC]

# Integration Options
t_end   = 12*1000     # Timestep to integrate up to
dt      = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
usetau  = 1           # Use tau (estimated damping timescale)
useeta  = 1           # Use eta from YO's model run
usesst  = 1

# Correlation Options
detrendopt = 0  # Option to detrend before calculations


# White Noise Options
genrand   = 1  #

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200602/'

## ------------ Script Start

# Load damping variables
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
LON = loaddamp['LON1']
LAT = loaddamp['LAT']
damping = loaddamp['ensavg']

# Load data from YO's model run
loadmod = loadmat(datpath+"stochrun.mat")
eta = loadmod['eta']
AVG0 = loadmod['AVG0']
AVG1 = loadmod['AVG1']
AVG2 = loadmod['AVG2']
tauall = loadmod['TAUall']

SST1 = loadmod['SST1']

# Find the corresponding Lat/Lon indices
oid = np.abs(LON-lonf).argmin()
aid = np.abs(LAT-latf).argmin()

# Generate Random White Noise Series (or load existing)
if genrand == 1:
    
    # Mean = 0 , Std = 1, Draw from Gaussian Sample
    F = np.random.normal(0,1,size=t_end)/4
    #plt.plot(F)
    
    np.savetxt(datpath+"randts.csv",F,delimiter=",")
    
else:
    
    F = np.loadtxt(datpath+"randts.csv",delimiter=",")

# Calculate Lambda
if usetau == 1:
    lbd = np.exp(-1 * 1 / np.mean(tauall, axis=1) )
else:
    lbd = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*h) * dt)
    
    
# Preallocate
temp_ts = np.zeros(t_end)
noise_ts = np.zeros(t_end)
damp_ts = np.zeros(t_end)


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
    if useeta == 1:
        noise_term = eta[0,t]
    else:
        noise_term = F[t]
        
    
    # Compute the temperature
    temp_ts[t] = lbd[m-1]*T + noise_term  
    
    # Save other variables
    noise_ts[t] = noise_term
    damp_ts[t] = lbd[m-1]*T

# Quick indexing fix
temp_ts[0] = T0
noise_ts = np.delete(noise_ts,0)
damp_ts = np.delete(damp_ts,0)

# Reshape Time Series to months x year
if usesst == 1:
    temp_ts = SST1
    temps   = np.reshape(temp_ts,(1000,12))
    temps = np.transpose(temps,(1,0))
else:
    temps = np.reshape(temp_ts,(int(np.ceil(temp_ts.size/12)),12))
    temps = np.transpose(temps,(1,0))

#temps = np.reshape(temp_ts,(int(np.ceil(len(temp_ts)/12)),12))
# Reshaping test
test = np.arange(1,37)
rs1 = np.reshape(test,(12,int(len(test)/12)))
rs2 = np.reshape(test,(int(len(test)/12),12)) 
rs2 = np.transpose(rs2,(1,0))

# Calculate autocorrelation --------------------------------------------------
lags = np.arange(0,61)
kmonth = 3;


corr_ts = np.zeros(len(lags))
for i in lags:
    lag_yr = int(np.floor(i+kmonth/12))
    
    baserng = range(kmonth,t_end-lag_yr*12,12)
    lagrng = range(kmonth+i,t_end-lag_yr*12,12)
    corr_ts[i] = stats.pearsonr(temp_ts[0,baserng],temp_ts[0,lagrng])[0]
    
    

# Function to calculate lag correlation
# Dependencies: numpy, scipy
    
def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
    import numpy as np
    from scipy import signal
    from scipy import stats
    
    
    
    debug = 1
    
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
        
        
        
        
    
    
    
    
    
        
    
    
        

    
    
    
  





