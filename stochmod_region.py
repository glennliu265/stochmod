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
                if k1m == 0:
                    k1m = 12
                if k0m == 0:
                    k0m = 12
                    
                
                
                
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


# hvarmode
hvarmode = 1 # hvar modes (0 - fixe mld, 1 - effective mld, 2 - seasonally varying mld)
hfix     = 50 # Fixed MLD (meters)




# Region options
lonW = -80
lonE = 20
latS = -20
latN = 90


# Correlation Options
detrendopt = 0  # Option to detrend before calculations


# White Noise Options
genrand   = 0  #

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
np.save(datpath+"lat.npy",latr)
np.save(datpath+"lon.npy",lonr)
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
        F      = np.multiply(F,randts[None,None,:])
        
    else:
        F = np.random.normal(0,1,size=(lonsize,latsize,t_end))
        
        
    np.save(datpath+"randts_2d.npy",F)
    
else:
    
    F = np.load(datpath+"randts_2d.npy")

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

if hvarmode == 0:
    
    # Use fixed mixed layer depth
    
    lbd = np.exp(-1 * dampingr / (rho*cp0*hfix) * dt)
    
elif hvarmode == 1:
    
    
    # Find maximum mld for each point in basin
    
    hmax = np.amax(hclim,axis=2)
    lbd = np.exp(-1 * dampingr / (rho*cp0*hmax[:,:,None]) * dt)
    
elif hvarmode == 2:

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
tprint = "\nNo Entrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        
        


# Run Model With Entrainment
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
            T_entr1[o,a,:],_,_,_,_ = stochmod_entrain(t_end,lbd[o,a,:],T0,F[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:])
        
        icount += 1
        msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
        print(msg,end="\r",flush=True)
#
        
elapsed = time.time() - start
tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
print(tprint)    
        


# save output
np.save(datpath+"stoch_output_1000yr_entrain1_hvar%i.npy"%(hvarmode),T_entr1)
np.save(datpath+"stoch_output_1000yr_entrain0_hvar%i.npy"%(hvarmode),T_entr0)
#np.save(datpath+"stoch_output_1000yr_Forcing_hvar%i.npy",F)

