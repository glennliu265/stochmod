#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:41:47 2020

@author: gliu


Script to run stochmod at a single point, with a selected set of experiments...

"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from scipy import signal
from scipy import stats


# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/")
import scm

#%% Functions --------------------------------------------------
# Function to calculate lag correlation
# Dependencies: numpy, scipy
def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
    
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
            
            
    return corr_ts

def find_latlon(lonf,latf,lon,lat):
    """
    Find lat and lon indices
    """
    if((np.any(np.where(lon>180)) & (lonf < 0)) or (np.any(np.where(lon<0)) & (lonf > 180))):
        print("Potential mis-match detected between lonf and longitude coordinates")
    
    klon = np.abs(lon - lonf).argmin()
    klat = np.abs(lat - latf).argmin()
    
    msg1 = "Closest lon to %.2f was %.2f" % (lonf,lon[klon])
    msg2 = "Closest lat to %.2f was %.2f" % (latf,lat[klat])
    print(msg1)
    print(msg2)
    
    return klon,klat



    
    
    
    # Calculate Beta
    beta = np.log( h / np.roll(h,1) )
    beta[beta<0] = 0
    
    # Find Maximum MLD during the year
    hmax = np.amax(h)
    
    
    # Preallocate lambda variable
    lbd = {}
    
    
    # Fixed MLD
    lbd[0] = damping / (rho*cp0*hfix) * dt
    
    # Maximum MLD
    lbd[1] = damping / (rho*cp0*hmax) * dt
    
    # Seasonal MLD
    lbd[2] = damping / (rho*cp0*h) * dt
    
    # Calculate Damping (with entrainment)
    lbd_entr = lbd[2] + beta    
    
    # Compute reduction factor
    FAC = (1-np.exp(-lbd_entr))/lbd_entr
    
    return lbd,lbd_entr,FAC,beta

def year2mon(ts):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    ts = ts.T
    return ts




#%% User Edits -----------------------------------------------------------------     

# Set Points
pts = [[-30,50],[-30,60],[-50,12]]
lonf    = -30
latf    = 50

# Experiment Settings
entrain  = 1 # 0 = no entrain; 1 = entrain
hvarmode = 2 # 0 = fixed mld ; 1 = max mld; 2 = clim mld 
funiform = 2 # 0 = nonuniform; 1 = uniform; 2 = NAO-like

# Autocorrelation Parameters#
kmon       = 2                 # Lag 0 base month
lags       = np.arange(0,61,1) # Number of lags to include
detrendopt = 0                 # Detrend before autocorrelation

# Bounding Box for visualization
lonW = -80
lonE = 20
latS = -20
latN = 90
bbox = [lonW,lonE,latS,latN]

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200728/'


# Model Parameters
nyrs = 1000
t_end = 12*nyrs
genrand = 1
hvarmode = 2
funiform = 2
dt = 3600*24*30
T0 = 0

#%% Load Data

# Load Damping Data
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])
damping = loaddamp['ensavg']

# Load MLD data
hclim      = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD
kprevall = np.load(datpath+"HMXL_kprev.npy") # Entraining Month


# Load Random Time Series...
if genrand == 1:
    randts = np.random.normal(0,1,size=t_end)
    np.savetxt(datpath+"randts.csv",randts,delimiter=",")
else:
    randts = np.loadtxt(datpath+"randts.csv",delimiter=",")
    
# Load NAO-like forcing
naoforce = np.load(datpath+"NAO_Forcing_EnsAvg.npy")

#%% Forcing Sensitivity Experiments

fscale = [2.5,1/4,1/40]
ftext = ("10x","1x","0.1x")



# Loop through each selected point
allsst={}
allcorr={}
i = 0
for pt in pts:
    allstart = time.time()
    
    # Find latitude longitude indices
    lonf = pt[0]
    latf = pt[1]
    klon,klat = find_latlon(lonf,latf,lon,lat)
    
    # Get terms for the point
    damppt = damping[klon,klat,:]
    hpt = hclim[klon,klat,:]
    kprev = kprevall[klon,klat,:]
    kmon = hpt.argmax()
    
    
    # Set Damping Parameters
    lbd,lbd_entr,FAC,beta = set_stochparams(hpt,damppt,dt)

    
    

    expsst = {}
    expcorr = {}
    sst={}
    corr={}
    # Loop for each forcing experiment
    for f in range(len(fscale)):
        ff  = fscale[f]
        ftx = ftext[f]
        
        F = randts * ff
        
        
        # Loop nonentraining model
        for l in range(3):
            start = time.time()
            sst[l],_,_ = stochmod_noentrain(t_end,lbd[l],T0,F)
            elapsed = time.time() - start
            tprint = "No Entrain Model, mode %i, ran in %.2fs" % (l,elapsed)
            print(tprint)
        
        # Loop for entrain model
        start = time.time()
        sst[3],_,_,_,_=stochmod_entrain(t_end,lbd_entr,T0,F,beta,hpt,kprev,FAC)
        elapsed = time.time() - start
        tprint = "Entrain Model ran in %.2fs" % (elapsed)
        print(tprint)
       
        # Compute correlation
        for l in range(4):
           
            ts = sst[l]
            ts = year2mon(ts)
           
            corr[l] = calc_lagcovar(ts,ts,lags,kmon+1,detrendopt)
           
        # Store into forcing variable
        expsst[f] = sst
        expcorr[f] = corr
    
       
        
        
        
        
    
    # Save sst based on point
    allsst[i] = expsst
    allcorr[i] = expcorr 
    i += 1
    elapsed = time.time() - allstart
    tprint = "Experiments for pt %i ran in %.2fs" % (i,elapsed)
    print(tprint)
        
    
#%% Forcing Sensitivity Plots


#%% Forcing Plots

tper = np.arange(0,t_end)
fig,ax = plt.subplots(3,1,sharex=True,sharey=False,figsize=(8,6))
plt.style.use("ggplot")

for i in range(len(fscale)):
    
    plt.subplot(3,1,i+1)
    plt.plot(tper,randts*fscale[i])
    plt.title("Forcing %s" %  (ftext[i]))
plt.tight_layout()


#%% Temperature Plots

lname = ("h-fixed","h-max","h-var","entrain")



for pt in range(len(pts)):
    
    
    locstring = "Lon: %i, Lat: %i" % (pts[pt][0],pts[pt][1]) 
    locfname  = "lon%02d_lat%02d" % (pts[pt][0],pts[pt][1]) 
    
    
    for l in range(4):
        ln = lname[l]
        
        # Set up figure
        fig,ax = plt.subplots(3,1,sharex=True,sharey=False,figsize=(8,6))
        plt.style.use("ggplot")
        
        for exp in range(len(fscale)):
            plt.subplot(3,1,exp+1)
            
            
            ts = allsst[pt][exp][i]
            
            plt.plot(tper,ts)
            plt.title("SST at %s for Forcing %s: %s" % (locstring,ftext[exp],ln))
        
        
        # Save Figure
        plt.tight_layout()
        figname = "%s_ldb%s.png" % (locfname,ln)
        plt.savefig(outpath+figname,dpi=200)
            
            
            
    
#%% Testing + Debugging NAO Forcing   

# Other settings
cp0      = 4218
rho      = 1000
dt       = 3600*24*30
T0       = 0
nyr      = 1000
t_end    = nyr*12
fscale   = 10
hvarmode = 2
hfix     = 50
runid    = '000'
genrand  = 0


# Set and Find latitude longitude indices
lonf      = -30
latf      = 65
klon,klat = find_latlon(lonf,latf,lon,lat)


# Get terms for the point
damppt = damping[klon,klat,:]
hpt    = hclim[klon,klat,:]
kprev  = kprevall[klon,klat,:]
kmon   = hpt.argmax()
naopt  = naoforce[klon,klat]

# Set Damping Parameters
lbd,lbd_entr,FAC,beta = scm.set_stochparams(hpt,damppt,dt,ND=False)
     
# Load Random Time Series...
if genrand == 1:
    randts = np.random.normal(0,1,size=t_end)/4
    np.savetxt(datpath+"randts.csv",randts,delimiter=",")
else:
    randts = np.load(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))

# Set up forcing term
if hvarmode == 0:
    hchoose = 50
elif hvarmode == 1:
    hchoose = np.max(np.abs(hpt))
elif hvarmode == 2:
    hchoose = hpt

Fmag = naopt*dt/cp0/rho/hchoose

if hvarmode == 2:
    F = randts * np.tile(Fmag,nyr) * fscale
else:
    F = randts * Fmag * fscale

# Run models
sst = {}

 # Loop nonentraining model
for l in range(3):
    start = time.time()
    sst[l],_,_ = scm.noentrain(t_end,lbd[l],T0,F)
    elapsed = time.time() - start
    tprint = "No Entrain Model, mode %i, ran in %.2fs" % (l,elapsed)
    print(tprint)

# Loop for entrain model
start = time.time()
sst[3],_,_,_,_=scm.entrain(t_end,lbd_entr,T0,F,beta,hpt,kprev,FAC)
elapsed = time.time() - start
tprint = "Entrain Model ran in %.2fs" % (elapsed)
print(tprint)



# Plot Point

hvarmode = 3
sstpt = sst[hvarmode]

tper = np.arange(0,t_end)
fig,ax = plt.subplots(2,1,figsize=(6,4))
plt.style.use('ggplot')


plt.subplot(2,1,1)
plt.plot(tper,F)
plt.ylabel("Forcing ($^{\circ}C/s$)",fontsize=10)
plt.title("Forcing at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(F),np.nanstd(F),np.nanmax(np.abs(F))))



plt.subplot(2,1,2)
plt.plot(tper,sstpt)
plt.ylabel("SST ($^{\circ}C$)",fontsize=10)
plt.xlabel("Time(Months)",fontsize=10)
#plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.title("Detrended, Deseasonalized SST \n Mean: %.2f || Std: %.2f || Max: %.2f" % (np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))


plt.tight_layout()

plt.savefig(outpath+"Stochmodpt_dsdt_SST_lon%02d_lat%02d_hvarmode%i_fscale%i.png"%(lonf,latf,hvarmode,fscale),dpi=200)

