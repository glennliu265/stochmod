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


def set_stochparams(h,damping,dt,rho=1025,cp0=3850,hfix=50):
    """
    Given MLD and Heat Flux Feedback, Calculate Parameters
    
    """    
    
    
    
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
outpath = projpath + '02_Figures/20200721/'


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
cp0 = 4218
rho = 1000
dt  = 3600*24*30
T0  = 0
nyr = 10000
t_end=nyr*12
fscale = 10

# Set and Find latitude longitude indices
lonf = -30
latf = 65
klon,klat = find_latlon(lonf,latf,lon,lat)


# Get terms for the point
damppt = damping[klon,klat,:]
hpt = hclim[klon,klat,:]
kprev = kprevall[klon,klat,:]
kmon = hpt.argmax()
naopt = naoforce[klon,klat]
# Set Damping Parameters
lbd,lbd_entr,FAC,beta = set_stochparams(hpt,damppt,dt)
     
# Load Random Time Series...
if genrand == 1:
    randts = np.random.normal(0,1,size=t_end)
    np.savetxt(datpath+"randts.csv",randts,delimiter=",")
else:
    randts = np.loadtxt(datpath+"randts.csv",delimiter=",")

# Set up forcing term
hchoose = np.max(np.abs(hpt))
Fmag = naopt*dt/cp0/rho/hchoose
F = randts * Fmag * fscale

# Run models
sst = {}

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

plt.savefig(outpath+"HadISST_dsdt_SST_lon%02d_lat%02d_hvarmode%i_fscale%i.png"%(lonf,latf,hvarmode,fscale),dpi=200)

