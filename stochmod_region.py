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

# Functions --------------------------------------------------
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
        
    
        # Compute the temperature
        temp_ts[t] = lbd[m-1]*T + noise_term  
    
        # Save other variables
        if debugmode == 1:
            noise_ts[t] = noise_term
            damp_ts[t] = lbd[m-1]*T


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
def stochmod_entrain(t_end,lbd,T0,F,beta,h,kprev):
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
        
    
    # Create MLD arrays
    if linterp == 0:
        mlddepths = np.arange(0,np.max(h)+1,1)
        mldtemps = np.zeros(mlddepths.shape)

    # Loop for integration period (indexing convention from matlab)
    for t in range(1,t_end):
        
        # Get the month
        m  = t%12
        m0 = m - 1 
        if m == 0:
            m = 12
            m0= 11
        
        
    
        # Get the temperature from the previous step
        if t == 1:
            T = T0
        else:
            T = temp_ts[t-1]
            
        
        
        # Calculate entrainment term
        if t<13:
            entrain_term = 0
        else:
            
            # Calculate Td
            if linterp == 1:
                
                
                
                # Find # of months since the anomaly was formed
                k1m = (m - np.floor(kprev[m-1])) % 12
                k0m = (m - np.floor(kprev[m0-1])) % 12
                
                # Get the corresponding index
                kp1 = int(t - k1m)
                kp0 = int(t - k0m) 

                                
                
                Td1 = np.interp(kprev[m-1] ,[kp1,kp1+1],[temp_ts[kp1],temp_ts[kp1+1]])
                Td0 = np.interp(kprev[m0-1],[kp0,kp0+1],[temp_ts[kp0],temp_ts[kp0+1]])
                
            elif linterp == 0:
                Td1 = mldtemps[round(h.item(m-1))]
                Td0 = mldtemps[round(h.item(m0-1))]           
            
            Td = (Td1+Td0)/2
            
            # Calculate entrainment term
            entrain_term = beta[m-1]*Td
            
            if debugmode == 1:
                Td_ts[t] = Td

        
    
        # Get Noise/Forcing Term
        noise_term = F[t-1]
        
    
        # Compute the temperature
        temp_ts[t] = lbd[m-1]*T + noise_term + entrain_term
    
        # Save other variables
        if debugmode == 1:
            noise_ts[t] = noise_term
            damp_ts[t] = lbd[m-1]*T
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
# def stochmod_entrain(t_end,lbd,T0,F,*args):
    
    
#     # Set entrain mode if args is long
#     entrain = [0]
#     if len(args) > 0:
#         entrain[0] = 1
#         beta  = args[0]
#         h     = args[1]
#         kprev = args[2]
        

#     debugmode = 0 # Set to 1 to also save noise,damping,entrain, and Td time series
 
#     # Preallocate
#     temp_ts = np.zeros(F.shape)

#     # Loop for integration period (indexing convention from matlab)
#     for t in range(1,t_end):
        
#         # Get the month
#         m  = t%12
#         m0 = m - 1 
#         if m == 0:
#             m = 12
#             m0= 11
        
        
    
#         # Get the temperature from the previous step
#         if t == 1:
#             T = T0
#         else:
#             T = temp_ts[:,:,t-1]
            
        
#         # Calculate entrainment term
#         if t<13 or entrain[0]==0:
#             entrain_term = np.zero((lonsize,latsize))
#         else:
            
#             # Calculate Td
#             if linterp == 1:
                            
#                 # Find # of months since the anomaly was formed
#                 k1m = (m - np.floor(kprev[:,:,m-1])) % 12
#                 k0m = (m - np.floor(kprev[:,:,m0-1])) % 12
                
#                 # Get the corresponding index
#                 kp1 = (t - k1m).astype(int)
#                 kp0 = (t - k0m).astype(int)
                
#                 ## AM STUCK HERE..... How to vectorize this? 
#                 # Perform Interpolation
#                 Td1 = np.interp(kprev[m-1] ,[kp1,kp1+1],[temp_ts[kp1],temp_ts[kp1+1]])
#                 Td0 = np.interp(kprev[m0-1],[kp0,kp0+1],[temp_ts[kp0],temp_ts[kp0+1]])
                
      
            
#             Td = (Td1+Td0)/2
            
#             # Calculate entrainment term
#             entrain_term = beta[m-1]*Td
            
#             if debugmode == 1:
#                 Td_ts[t] = Td

        
    
#         # Get Noise/Forcing Term
#         noise_term = F[t-1]
        
    
#         # Compute the temperature
#         temp_ts[t] = lbd[m-1]*T + noise_term + entrain_term
    
#         # Save other variables
#         if debugmode == 1:
#             noise_ts[t] = noise_term
#             damp_ts[t] = lbd[m-1]*T
#             entrain_ts[t] = entrain_term
        
        
#         # Set mixed layer depth tempertures
#         if linterp == 0:
#             mldtemps[mlddepths<=h.item(m-1)] = temp_ts[t]

#     # Quick indexing fix
#     temp_ts[0] = T0
#     if debugmode == 1:
#         noise_ts = np.delete(noise_ts,0)
#         damp_ts = np.delete(damp_ts,0)
#         entrain_ts = np.delete(entrain_ts,0)
    
#     return temp_ts,noise_ts,damp_ts,entrain_ts,Td_ts



def find_kprev_2d(h,lonr,latr):
    
    lonsize = len(lonr)
    latsize = len(latr)
    
    
    # Preallocate
    kprev = np.zeros((lonsize,latsize,12))
    hout = np.zeros((lonsize,latsize,12))
                                                                                                                                                                                                                                                                    
    # Month Array
    monthx = np.arange(1,13,1)  
    
    # Determine if the mixed layer is deepening (true) or shoaling (false)--
    dz = h / np.roll(h,shift=1,axis=2) 
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
        if h[im] == h.max() or h[im]== h.min():
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


def find_kprev(h):
    printst = 0
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
        if h[im] == np.nanmax(h) or h[im]== np.nanmin(h):
            if printst == 1:
                print("Ignoring %i, max/min" % m)
            kprev[im] = m
            hout[im] = h[im]
            continue
        
    
        
        # Ignore detrainment months
        if dz[im] == False:
            if printst == 1:
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
                if printst == 1:
                    print("Found kprev for month %i it is %f!" % (m,np.interp(findmld,[h_before,h_after],[m_before,m_after])))
                kprev[im] = np.interp(findmld,[h_before,h_after],[m_before,m_after])
                hout[im] = findmld
            
            # Go back one month
            ifindm -= 1
    
    return kprev, hout

    

# User Edits -----------------------------------------------------------------           

# Set Point and month
kmon    = 3

# Set Variables
cp0     = 3850 # Specific Heat [J/(kg*C)]
rho     = 1025 # Density of Seawater [kg/m3]

# Initial Conditions/Presets
h       = 150 # Effective MLD [m]
T0      = 0   # Initial Temp [degC]

# Integration Options
nyr     = 1000    # Number of years to integrate over
t_end   = 12*nyr      # Timestep to integrate up to
dt      = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
usetau  = 0          # Use tau (estimated damping timescale)
useeta  = 0          # Use eta from YO's model run
usesst  = 0
hvar    = 1           # seasonally varying h
funiform = 1        # Spatially uniform forcing


# Region options
lonW = -80
lonE = 20
latS = -20
latN = 90


# Correlation Options
detrendopt = 0  # Option to detrend before calculations


# White Noise Options
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
# Load Variables -------------------------------------------------------------
# --------------


# Load damping variables
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
LON = loaddamp['LON1']
LAT = loaddamp['LAT']
damping = loaddamp['ensavg']

# Load Mixed layer variables
mldnc = "HMXL_HTR_clim.nc"
ds = xr.open_dataset(datpath+mldnc)


# ------------------
# Restrict to region ---------------------------------------------------------
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

# Further restrict to non-nan locations



# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]


# ----------------------
# Select non-nan points ------------------------------------------------------
# ----------------------


# ----------------------------
# Generate White Noise Forcing------------------------------------------------
# ----------------------------

# Generate Random White Noise Series (or load existing)
if genrand == 1:
    
    if funiform == 1:
        randts = np.random.normal(0,1,size=t_end)/4
        F      = np.ones((lonsize,latsize,t_end))
    else:
        F = np.random.normal(0,1,size=(lonsize,latsize,t_end))
        
        
    np.save(datpath+"randts_2d.npy",F)
    
else:
    
    F = np.loadtxt(datpath+"randts_2d.npy")

# ---------------------------------------
# Calc Ens Avg Mixed Layer Seasonal Cycle-------------------------------------
# ---------------------------------------

# Preallocate and specify search tolerance 
hclim = np.zeros((lonsize,latsize,12),dtype=float)
kprev = np.zeros((lonsize,latsize,12),dtype=float)

stol  = 0.75 # Search tolerance for curivilinear grid (degrees) <Note there is sensitivity to this>....

# Take ensemble mean
h_ensmean = ds.HMXL.mean('ensemble')

# This portion of the data unfortunately has an ensemble dimension
tlon = ds.TLONG.mean('ensemble')
tlat = ds.TLAT.mean('ensemble')


start = time.time()
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
            hclim[o,a,:] = np.ones(12)*np.nan
            kprev[o,a,:] = np.ones(12)*np.nan
            continue
            
        # Select Points        
        selectmld = h_ensmean.where((lonf-stol < tlon) & (tlon < lonf+stol)
                        & (latf-stol < tlat) & (tlat < latf+stol),drop=True)
        
        # Take the mean of the selected points
        ensmean = selectmld.mean(('nlon','nlat'))/100
        
        # Record values
        hclim[o,a,:] = np.squeeze(ensmean.values)
        
        # Find Entraining Months
        kprev[o,a,:],_ = find_kprev(hclim[o,a,:])
        
        # # Find points out of range the range or on land
        # if selectmld.size == 0 or np.all(xr.ufuncs.isnan(selectmld)):
        #     msg = "Could not find data for lon %f lat %f" % (lonf,latf)
        #     print(msg)
        #     hclim[o,a,:] = np.nan
        # else:
        #     # Take the mean of the points
        #     ensmean = selectmld.mean(('nlon','nlat'))/100
            
            
        #     # Ignore points where there is a nan during the seasonal cycle
        #     if np.any(xr.ufuncs.isnan(ensmean)):
        #         msg = "Due to NaN in seasonal cycle, ignoring lon %f lat %f" % (lonf,latf)
        #         print(msg)
        #         hclim[o,a,:] = np.nan
        #     else:
        #         # Asssign to variable
        #         hclim[o,a,:] = np.squeeze(ensmean.values)

print("Finished in %f seconds" % (time.time()-start))

# Quick visualization to check
plt.contourf(lonr,latr,np.transpose(np.squeeze(hclim[:,:,5])))
plt.contourf(lonr,latr,np.transpose(np.squeeze(kprev[:,:,5])))


# ----------------
# Calculate Lambda------------------------------------------------------------
# ----------------


lbd = np.exp(-1 * dampingr / (rho*cp0*hclim) * dt)
    

# -------------------------
# Prepare Entrainment Terms --------------------------------------------------
# -------------------------

# Compute seasonal correction factor from lambda
FAC = (1-lbd)/ (dampingr / (rho*cp0*hclim))

# Compute the integral of the entrainment term, with dt and correction factor
beta = np.nan_to_num(1/dt * np.log( hclim / np.roll(hclim,1,2) ) * FAC)

# Set term to zero where detrainment is occuring
beta[beta < 0] = 0


 
# Calculate lambda that includes entrainment

lbd_entr = lbd * np.exp(-beta)

    #lbd_entr = np.exp(-1 * (damping[oid,aid,:] / (rho*cp0*h) * dt+ beta ))
## No seasonal correction
#lbd_entr = np.exp(-1 * (damping[oid,aid,:] / (rho*cp0*h) * dt+ np.log( h / np.roll(h,1) )))
   

# ----------
# RUN MODELS -----------------------------------------------------------------
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
tprint = "No Entrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        
        


# Run Model WitEntrainment
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
            T_entr1[o,a,:],_,_ = stochmod_noentrain(t_end,lbd[o,a,:],T0,F[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:])
        
        icount += 1
        msg = '\rCompleted No Entrain Run for %i of %i points' % (icount,lonsize*latsize)
        print(msg,end="\r",flush=True)
#
        
elapsed = time.time() - start
tprint = "No Entrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        



# Run model with entrainment
start = time.time()
#temp_ts,noise_ts,damp_ts,entrain_ts,Td_ts=stochmod_entrain(t_end,lbd_entr,T0,F,beta,h)
T_entr1,_,_,_,_=stochmod_entrain(t_end,lbd_entr,T0,F,beta,h,kprev)
elapsed = time.time() - start
tprint = "Entrain Model ran in %.2fs" % (elapsed)
print(tprint)


# ------------------------------------
# Prepare for Correlation Calculations----------------------------------------
# ------------------------------------


# Reshape Time Series to months x year
if usesst == 1:
    temp_ts = SST1
    #temps   = np.reshape(temp_ts,(1000,12))
    temps = np.reshape(temp_ts,(76,12))
    temps = np.transpose(temps,(1,0))
else:
    
    # No Entrainment
    temps_e0 = np.reshape(T_entr0,(int(np.ceil(T_entr0.size/12)),12))
    temps_e0 = np.transpose(temps_e0,(1,0))
    
    # With Entrainment
    temps_e1 = np.reshape(T_entr1,(int(np.ceil(T_entr1.size/12)),12))
    temps_e1 = np.transpose(temps_e1,(1,0))
    
    F_ts = np.reshape(F,(int(np.ceil(F.size/12)),12))
    F_ts = np.transpose(F_ts,(1,0))

#temps = np.reshape(temp_ts,(int(np.ceil(len(temp_ts)/12)),12))
# Reshaping test
test = np.arange(1,37)
rs1 = np.reshape(test,(12,int(len(test)/12)))
rs2 = np.reshape(test,(int(len(test)/12),12)) 
rs2 = np.transpose(rs2,(1,0))


# -------------------------
# Calculate autocorrelation --------------------------------------------------
# -------------------------


lags = np.arange(0,61)
kmonth = 3;

## YO's Method (Currently not working, need to check indexing)
# corr_ts = np.zeros(len(lags))
# for i in lags:
#     lag_yr = int(np.floor(i+kmonth/12))
    
#     baserng = range(kmonth,t_end-lag_yr*12,12)
#     lagrng = range(kmonth+i,t_end-lag_yr*12,12)
#     corr_ts[i] = stats.pearsonr(temp_ts[0,baserng],temp_ts[0,lagrng])[0]
    
#detrendopt = 1

corr_e0 = calc_lagcovar(temps_e0,temps_e0,lags,kmonth,detrendopt)
corr_e1 = calc_lagcovar(temps_e1,temps_e1,lags,kmonth,detrendopt)
corr_noise = calc_lagcovar(F_ts,F_ts,lags,kmonth,detrendopt)

# --------------
# Make Figures  --------------------------------------------------------------
# --------------

# *********************
# Autocorrelation Plots
# ********************* 
f1 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')

ax.plot(lags,AVG0,color='b',label='CESM1 (YO)')
ax.plot(lags,AVG1,color='c',label='No-Entrain (YO)')
ax.plot(lags,AVG2,color='g',label='Entrain (YO)')
ax.plot(lags,corr_e0,'-r',label='No-Entrain (GL)')
ax.plot(lags,corr_e1,':r',label='Entrain (GL)')
ax.legend()

titlestr = 'SST Autocorrelation; Month'+str(kmonth) + '\n Lon: ' + \
    str(lonf) + ' Lat: '+ str(latf)
ax.set(xlabel='Lag (months)',
       ylabel='Correlation',
       ylim=(-0.5,1.1),
       title=titlestr )
outname = outpath+'SSTAC_usetau'+str(usetau)+'_mldlinterp_'+ loc_fname +'.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


# *********************
# Autocorrelation Plots v2
# ********************* 
f1 = plt.figure()
ax = plt.axes()


ax.plot(lags,AVG0,color='b',lw=3,label='CESM1 (Fully Coupled)')
#ax.plot(lags,corr_noise,'k',lw=3,label='Forcing')
ax.plot(lags,corr_e0,'c',lw=3,label='Stochastic (No-Entrain)')
ax.plot(lags,corr_e1,'g',lw=3,label='Stochastic (Entrain)')
ax.legend(prop=dict(size=16))

titlestr = 'SST Anomaly Autocorrelation; \n Month: '+monsfull[kmonth-1] + ' | Lon: ' + \
    str(lonf) + ' | Lat: '+ str(latf)
    
ax.set_ylabel('Correlation',fontsize=14)
ax.set_ylim(-0.2,1.1)
ax.set_xlabel('Lag (months)',fontsize=14)
#ax.set_yticklabels(labels=np.arange(-0.2,1.2,0.2),fontsize=12)
#ax.set_xticklabels(labels=np.arange(0,65,5),fontsize=12)
ax.set_title(titlestr,fontsize=20)
#plt.style.use('seaborn')
outname = outpath+'SSTAC_usetau'+str(usetau)+'_mldlinterp_'+ loc_fname +'_onlyGL.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


# 
# Noise vs SST
#

f2,axs = plt.subplots(2,1)
axs[0].plot(np.arange(0,101,1),F[0:101])
axs[1].plot(np.arange(0,101,1),T_entr1[0:101])





# *************
# lambda plots ----------------------------------------------------------------
# *************

# Calculate Lambda Values
lbd1 = np.exp(-1 * 1 / np.mean(tauall, axis=1) ) # Tau
lbd2 = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*h) * dt) # Lambda, Constant H
lbd3 = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*150) * dt) # Lambda, s.var H

# Calculate Lambda for all ensemble members
lbd1all =np.exp(-1 * 1 / tauall)

# Month Array
monthx = np.arange(1,13,1)        

# Start Plot
f2 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')

# Plot each ensemble member, then 1 for labeling
for e in range(1,np.shape(tauall)[1]):
    #print(e)
    ax.plot(monthx,lbd1all [:,e],color=(.75,.75,.75))
ln0 = ax.plot(monthx,lbd1all [:,-1],color=(.75,.75,.75),label=r'$exp(-1/\tau)$ : Indv. Member')

# Plot other lines
ln1 = ax.plot(monthx,lbd1,color='k',label=r'$exp(-1/\tau)$ : EnsAvg)')
ln2 = ax.plot(monthx,lbd2,color='b',label=r'$exp(-\lambda_{net} / (\rho cp_0H)\Delta t)$ : Varying MLD')
ln3 = ax.plot(monthx,lbd3,color='r',label=r'$exp(-\lambda_{net} / (\rho cp_0H)\Delta t)$ : Constant MLD')

# Set legend location and fontsize
#ax.legend(loc='best',)
#plt.rc('legend', fontsize=10)    # legend fontsize
lns = ln0 + ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0,bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)

# Title and axis options
titlestr = 'Seasonal Values for Damping Term'
ax.set(xticks=np.arange(1,13,1),
       xlim=(1,12),
       xlabel='Month',
       ylabel='Value ($mon^{-1}$)',
       title=titlestr)
ax.set_xticklabels(mons3)

# Print figure
outname = outpath+"LambdaComp.png"
plt.savefig(outname, bbox_inches="tight",dpi=200)

        
# ******************************
# Plot for a single lambda value
# ******************************

f3 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')
ax.plot(np.arange(1,13,1),damping[oid,aid,:],color='r',label=r'$\lambda_{net}$')
ax.legend(loc='best',)
titlestr = 'Seasonal Values for Damping Term'
ax.set(xticks=np.arange(1,13,1),
       xlabel='Month',
       ylabel='Value ($W/m^{-2}K^{-1}$)',
       title=titlestr )
outname = outpath+'Damping.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


# **********************
# Twin Axis Damping Plot
# **********************
SMALL_SIZE =  12
MEDIUM_SIZE = 14
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
f3,ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('$\lambda_{net} (W m^{-2} K^{-1}$)')
ln1 = ax1.plot(np.arange(1,13,1),damping[oid,aid,:],color='r',label=r'$\lambda_{net}$')
ax1.tick_params(axis='y',labelcolor=color)
ax1.grid(None)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mixed Layer Depth (m)',color=color)
ln2 = ax2.plot(np.arange(1,13,1),h,color='b',label=r'HMXL')
ax2.tick_params(axis='y',labelcolor=color)
ax2.grid(None)


# Set Legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=0)

# Set Title
titlestr = 'Seasonal Values for MLD and $\lambda$ (Ensemble Average) \n Lon: ' + \
    str(lonf) + ' Lat: '+ str(latf)
plt.title(titlestr)


if lonf < 0:
    strlon = 360+lonf
outname = outpath+'Damping_MLD_lon' + str(strlon) + '_lat' + str(latf) + '.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


# **********************
# Xorrelation plots detrended
# **********************
f1 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')

ax.plot(lags,dt_SST0,color='b',label='CESM1 (YO)')
ax.plot(lags,dt_SST1,color='c',label='No-Entrain (YO)')
ax.plot(lags,dt_SST2,color='g',label='Entrain (YO)')
ax.plot(lags,dt_GL,'-r',label='No-Entrain (GL)')
ax.legend()


titlestr = 'Detrended SST Autocorrelation; Month'+str(kmonth) + '\n Lon: ' + \
    str(lonf) + ' Lat: '+ str(latf)
ax.set(xlabel='Lag (months)',
       ylabel='Correlation',
       ylim=(-0.3,1.1),
       title=titlestr )
outname = outpath+'DT_SSTAC_usetau'+str(usetau)+'.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)



# **********************
# Visualize differences between mixed layer variables ------------------------
# **********************

# Load Mixed layer variables and calculate ensemble average
varmx = ('HMXL','XMXL')
mx_ensavg = {}

for mx in varmx:
    mldnc = mx + "_HTR_clim.nc"

    ds = xr.open_dataset(datpath+mldnc)


    # Do same for curivilinear grid
    if lonf < 0:
        lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates
    
    # Find the specified point on curvilinear grid and average values
    selectmld = ds.where((lonfc-0.5 < ds.TLONG) & (ds.TLONG < lonfc+0.5)
                        & (latf-0.5 < ds.TLAT) & (ds.TLAT < latf+0.5),drop=True)
    # Select accordingly 
    ensmean = selectmld.mean(('ensemble','nlon','nlat'))/100
    h = np.squeeze(ensmean.to_array())
    
    
    # Assign to dictionary
    mx_ensavg[mx] = h
    
fmx = plt.figure()
ax = plt.axes()
for mx in varmx:
    ax.plot(range(1,13),mx_ensavg[mx],label=mx)
ax.legend()
plt.title('Mixed Layer Seasonal Cycle (Ens Average) \n' + loc_figtitle)
ax.set(xlim=(1,12),ylim=(0,200),xlabel='Months',ylabel='MLD(m)')
outname = outpath+'HMXL_v_XMXL.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


# Save files
np.savetxt(datpath+"T_entr0.csv",T_entr0,delimiter=",")
np.savetxt(datpath+"T_entr1.csv",T_entr1,delimiter=",")