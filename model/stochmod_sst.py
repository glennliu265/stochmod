# -*- coding: utf-8 -*-
"""
stochmod_sst, Python Version


This is a temporary script file.
"""

# from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from scipy.io import loadmat

#%% Functions --------------------------------------------------
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
    debugmode = 1 # Set to 1 to also save noise,damping,entrain, and Td time series
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


#%% User Edits -----------------------------------------------------------------           

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
nyr     = 1000    # Number of years to integrate over
t_end   = 12*nyr      # Timestep to integrate up to
dt      = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
usetau  = 0          # Use tau (estimated damping timescale)
useeta  = 0          # Use eta from YO's model run
usesst  = 0
hvar    = 1           # seasonally varying h

# Correlation Options
detrendopt = 0  # Option to detrend before calculations

# White Noise Options
genrand    = 1   #

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200721/'


# Set up some strings for labeling
loc_figtitle = "Lon: %i Lat: %i" % (lonf,latf)
if lonf < 0:
    lonstr = lonf + 360
loc_fname    = "Lon%03d_Lat%03d"  % (lonstr,latf)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')
fscale=1/4


## ------------ Script Start -------------------------------------------------

#%% Load Variables -------------------------------------------------------------

# Load damping variables
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
LON = loaddamp['LON1']
LAT = loaddamp['LAT']
damping = loaddamp['ensavg']

# # Load data from YO's model run
# loadmod = loadmat(datpath+"stochrun_"+loc_fname+".mat")
# eta = loadmod['eta']
# AVG0 = loadmod['AVG0']
# AVG1 = loadmod['AVG1']
# AVG2 = loadmod['AVG2']
# tauall = loadmod['TAUall']


loadmld = loadmat(datpath+"mld_lon280_lat005.mat")
yo_kprev40 = loadmld['kprev']
yo_mld     = loadmld['MLDall']

#SST1 = loadmod['SST1']
#SST2 = loadmod['SST2']
#SST0 = loadmod['SST']


# Load Mixed layer variables
mldnc = "HMXL_HTR_clim.nc"
ds = xr.open_dataset(datpath+mldnc)



# ----------------------------
#%% Generate White Noise Forcing------------------------------------------------
# ----------------------------


# Generate Random White Noise Series (or load existing)
if genrand == 1:
    
    # Mean = 0 , Std = 1, Draw from Gaussian Sample
    F = np.random.normal(0,1,size=t_end)*fscale
    #plt.plot(F)
    
    np.savetxt(datpath+"randts.csv",F,delimiter=",")
    
else:
    
    F = np.loadtxt(datpath+"randts.csv",delimiter=",")



# ---------------------------------------
#%% Calc Ens Avg Mixed Layer Seasonal Cycle-------------------------------------
# ---------------------------------------
    
# Find the corresponding Lat/Lon indices
oid = np.abs(LON-lonf).argmin()
aid = np.abs(LAT-latf).argmin()



def getpt_pop(lonf,latf,ds,searchdeg=0.75,returnarray=1):
    """ Quick script to read in a xr.Dataset (ds)
        and return the value for the point specified by lonf,latf
        
    
    """
    
    
    # Do same for curivilinear grid
    if lonf < 0:
        lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates
        
    # Find the specified point on curvilinear grid and average values
    selectmld = ds.where((lonfc-searchdeg < ds.TLONG) & (ds.TLONG < lonfc+searchdeg)
                    & (latf-searchdeg < ds.TLAT) & (ds.TLAT < latf+searchdeg),drop=True)
    
    pmean = selectmld.mean(('nlon','nlat'))
    
    if returnarray ==1:
        h = np.squeeze(pmean)
        return h
    else:
        return pmean
    




if hvar == 1:
    
    # Get value at the point for all ensemble members 
    h = getpt_pop(lonf,latf,ds)
    h2 =  getpt_pop(lonf,latf,h_ensmean)
    
    # Convert to meters and take ensemble average
    h = np.nanmean(h,axis=0)/100


# Calculate kprev values
kprev,_ = find_kprev(h)




# # DDD use YOs calculated mld values
# #h = np.copy(np.nanmean(yo_mld,1))
# #h = np.squeeze(yo_mld[:,-1])




hyo = np.copy(np.nanmean(yo_mld,1))
# # Find previous, entraining month

# # Find the difference

# kprev40,_ = find_kprev(yo_mld[:,39])

fig,ax=plt.subplots(1,1)
ax.plot(np.arange(1,13,1),h,label='used2')
#ax.plot(np.arange(1,13,1),h1clim,label='used1')
ax.plot(np.arange(1,13,1),hyo,label='mldyo',color='k')


# ----------------
#%% Calculate Lambda------------------------------------------------------------
# ----------------


# Calculate Entrainment Portion
beta = np.log( h / np.roll(h,1) )
beta[beta<0] = 0


if usetau == 1:
    lbd = 1/np.mean(tauall,axis=1)
    
    # DDD try last ensemble
    #lbd = 1 / tauall[:,39]

else:
    lbd = damping[oid,aid,:] / (rho*cp0*h) * dt

lbd_entr = lbd + beta


## The old method included the exponentiation directly in the lambda term
# if usetau == 1:
#     lbd = np.exp(-1 * 1 / np.mean(tauall, axis=1) )
    
    
    
#     # For debugging use the value for the last ensemble memner
#     lbd = np.exp(-1 * 1 / tauall[:,-1])
    
# else:
#     lbd = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*h) * dt)

# #lbd = np.exp(-1 * np.ones(12)*15 / (rho*cp0*h) * dt)

# Select which forcing to use
if useeta == 1:
    F = np.copy(eta)
    F = np.squeeze(F)




# -------------------------
#%% Prepare Entrainment Terms --------------------------------------------------
# -------------------------


# Calculate Reduction Factor
#FAC      = (1-np.exp(-lbd))/lbd # I dont think this will be used...
FAC_entr = (1-np.exp(-lbd_entr))/lbd_entr


# # Compute seasonal correction factor from lambda
# if usetau == 1:
#     FAC = np.nan_to_num((1-lbd) / (1 / np.mean(tauall, axis=1)))
#     #FAC = 1
    
#     # Compute the integral of the entrainment term, with dt and correction factor
#     beta = np.log( h / np.roll(h,1) ) * FAC
# else:
    
#     FAC = np.nan_to_num((1-lbd) / (dt*damping[oid,aid,:]/ (rho*cp0*h)))
#     #FAC = 1
#     #FAC = np.nan_to_num((1-lbd) / (dt*np.ones(12)*15/ (rho*cp0*h))) # Testing constant lamda (seems to enhance)
#     # Compute the integral of the entrainment term, with dt and correction factor
#     beta = np.log( h / np.roll(h,1) ) * FAC

# # Set term to zero where detrainment is occuring
# beta[beta < 0] = 0


#  # Calculate lambda that includes entrainment

# lbd_entr = lbd * np.exp(-weh)

#lbd_entr = lbd * np.exp(-beta) #Using this value increases the correlation
    #lbd_entr = np.exp(-1 * (damping[oid,aid,:] / (rho*cp0*h) * dt+ beta ))
## No seasonal correction
#lbd_entr = np.exp(-1 * (damping[oid,aid,:] / (rho*cp0*h) * dt+ np.log( h / np.roll(h,1) )))
   


# fig,ax=plt.subplots(1,1)
# ax.plot(np.arange(1,13,1),lbdtau1,label="Use Tau")
# ax.plot(np.arange(1,13,1),lbdtau0,label="No Use Tau")
# plt.title("Lbd_entr")
# plt.legend()

# fig,ax=plt.subplots(1,1)
# ax.plot(np.arange(1,13,1),FACtau1,label="Use Tau")
# ax.plot(np.arange(1,13,1),FACtau0,label="No Use Tau")
# plt.title("FAC")
# plt.legend()

# Plot to compare the lambda terms and reduction factor
plt.style.use("ggplot")
fig,ax = plt.subplots(1,1)
ax.plot(mons3,lbd,label='no-entrain')
ax.plot(mons3,lbd_entr,label='entrain')
ax.plot(mons3,FAC_entr,label='Reduction Factor')
plt.legend()
ax.set_title("Lambda Comparison")
# ----------
#%% RUN MODELS -----------------------------------------------------------------
# ----------

# For debug mode, set debug to 1 in fuction scripts and write the outputs

# Run Model Without Entrainment
start = time.time()
#temp_ts,noise_ts,damp_ts = stochmod_noentrain(t_end,lbd,T0,F)
T_entr0,npise_ts0,damp_ts0 = stochmod_noentrain(t_end,lbd,T0,F)
elapsed = time.time() - start
tprint = "No Entrain Model ran in %.2fs" % (elapsed)
print(tprint)


# Run model with entrainment
start = time.time()
#temp_ts,noise_ts,damp_ts,entrain_ts,Td_ts=stochmod_entrain(t_end,lbd_entr,T0,F,beta,h)
T_entr1,noise_ts1,damp_ts1,entrain_ts,Td_ts=stochmod_entrain(t_end,lbd_entr,T0,F,beta,h,kprev,FAC_entr)
elapsed = time.time() - start
tprint = "Entrain Model ran in %.2fs" % (elapsed)
print(tprint)

#
# %% Debug visualziations
#

# ------------------------------------
#%% Prepare for Correlation Calculations----------------------------------------
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
#%% Calculate autocorrelation --------------------------------------------------
# -------------------------


lags = np.arange(0,61)
kmonth = h.argmax()+1;

# # YO's method Currently seems to be working)
# def yo_cor(var,lags,t_end,kmonth):
#     corr_ts = np.zeros(len(lags))
    
#     for i in lags:
#         lag_yr = int(np.floor((i+kmonth-1)/12))

#         baserng = np.arange(kmonth-1,t_end-lag_yr*12+2,12)
#         lagrng  = np.arange(kmonth+i-1,t_end-lag_yr*12+2,12)
#         print("Doing %i with lagyr %i"% (i,lag_yr))
#         if len(baserng) > len(lagrng):
#             diffsize = len(baserng)-len(lagrng)
#             baserng = baserng[:0-diffsize:]
#             print("Baserng has %i more than Lagrng"%(diffsize))
        
        
        
        
#         corr_ts[i] = stats.pearsonr(var[baserng],var[lagrng])[0]
    
#     return corr_ts
# corr_e1=yo_cor(T_entr1,lags,t_end,kmonth)


#detrendopt = 1
corr_e0 = calc_lagcovar(temps_e0,temps_e0,lags,kmonth,detrendopt)
corr_e1 = calc_lagcovar(temps_e1,temps_e1,lags,kmonth,detrendopt)
corr_noise = calc_lagcovar(F_ts,F_ts,lags,kmonth,detrendopt)






# testing general lag correlation
# #https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
# import numpy
    
# def autocorr5(x,lags):
#     '''numpy.correlate, non partial'''
#     mean=x.mean()
#     var=numpy.var(x)
#     xp=x-mean
#     corr=numpy.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

#     return corr[:len(lags)]

# corr_e1 = autocorr5(T_entr1,lags)

#
#%% Do some Power Spectral Analysis...
from scipy import signal
fs = 1/(3600*24*30) # Sampling Frequency (1 month in seconds0
#xtk     = [fs/1200,fs/120,fs/12,fs,fs*30]
#xtklabel = ['century','decade','year','mon',"day"]

xtk     = [fs/1200,fs/120,fs/12,fs]
xtklabel = ['century','decade','year','mon']
freqs,pxx_Tentr1 = signal.periodogram(noise_ts1,fs)
                              
plt.plot(freqs, pxx_Tentr1)
#plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [degC**2/Hz]')
plt.xscale('log')
plt.yscale('log')
#plt.ylim([1e1,1e6])
plt.ylim([1e-1,1e11])
plt.xticks(xtk,xtklabel)
plt.show()



plt.title("HadlISST Area-Average SST Anomaly Periodogram")
# --------------
#%% Make Figures  --------------------------------------------------------------
# --------------





# *********************
#%% Autocorrelation Plots
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


# ***********************
#%% Autocorrelation Plots v2
# ***********************
f1 = plt.figure(figsize=(6.56,4.82))
ax = plt.axes()
plt.style.use('ggplot')
#sns.set('paper','whitegrid','bright')


#ax.plot(lags,AVG0,color='b',lw=3,label='CESM1 (Fully Coupled)')
#ax.plot(lags,corr_noise,'k',lw=3,label='Forcing')
ax.plot(lags,corr_e0,'c',lw=3,label='Stochastic (No-Entrain)')
ax.plot(lags,corr_e1,'g',lw=3,label='Stochastic (Entrain)')
ax.plot(lags,corr_noise,color=(0.65,0.65,0.65),lw=3,label='Forcing')
#ax.legend(prop=dict(size=16))
ax.legend()

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
#%% Noise vs SST
#

f2,axs = plt.subplots(2,1)
axs[0].plot(np.arange(0,101,1),F[0:101])
axs[1].plot(np.arange(0,101,1),T_entr1[0:101])





# *************
#%% lambda plots ----------------------------------------------------------------
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
#%% Plot for a single lambda value
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

0


# ******************************
#%% Plot for a 1/tau comparison
# ******************************

f3 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')
# Plot each ensemble member, then 1 for labeling
for e in range(1,np.shape(tauall)[1]):
    #print(e)
    ax.plot(np.arange(1,13,1),1/tauall[:,e],color=(.75,.75,.75))
ln0 = ax.plot(np.arange(1,13,1),1/tauall[:,-1],color=(.75,.75,.75),label=r'$-1/\tau)$ : Indv. Member')


ax.plot(np.arange(1,13,1),damping[oid,aid,:] / (rho*cp0*h) * dt,color='r',label=r'$\lambda_{net}$')
ax.plot(np.arange(1,13,1),1 / np.mean(tauall, axis=1),color='k',label=r'$ 1/\tau (ensemble mean)$')
ax.legend(loc='best',)
titlestr = 'Seasonal Values for Damping Term'
ax.set(xticks=np.arange(1,13,1),
       xlabel='Month',
       ylabel='Value ($W/m^{-2}K^{-1}$)',
       title=titlestr )
outname = outpath+'Damping.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)

savetau = 1/(damping[oid,aid,:] / (rho*cp0*h) * dt)
matdict = {"lambdaastau":savetau}
savemat(outpath+"lambdaastau.mat",matdict)

# ******************************
#%% Plot for a lambda comparison
# ******************************

f3 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')
# Plot each ensemble member, then 1 for labeling
for e in range(1,np.shape(tauall)[1]):
    #print(e)
    ax.plot(np.arange(1,13,1),(rho*cp0*h)/tauall[:,e]/dt,color=(.75,.75,.75))
ln0 = ax.plot(np.arange(1,13,1),(rho*cp0*h)/tauall[:,-1]/dt,color=(.75,.75,.75),label=r'$-1/\tau$ (Indv. Member)')


ax.plot(np.arange(1,13,1),damping[oid,aid,:],color='r',label=r'$\lambda_{net}$')
ax.plot(np.arange(1,13,1),(rho*cp0*h) / np.mean(tauall, axis=1)/dt,color='k',label=r'$ 1/\tau (ensemble mean)$')
ax.legend(loc='best',)
titlestr = 'Seasonal Values for Damping Term'
ax.set(xticks=np.arange(1,13,1),
       xlabel='Month',
       ylabel='Value ($W/m^{-2}K^{-1}$)',
       title=titlestr )
outname = outpath+'tauaslambda_Damping.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)

savetau = 1/(damping[oid,aid,:] / (rho*cp0*h) * dt)
matdict = {"lambdaastau":savetau}
savemat(outpath+"tauaslambda.mat",matdict)
# **********************
#%% Twin Axis Damping Plot
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
#%% Xorrelation plots detrended
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

# %% Visualize some variables


tper= np.arange(0,t_end)
fig,ax = plt.subplots(3,1,sharex=True,sharey=False,figsize=(8,6))
plt.style.use("ggplot")


plt.subplot(3,1,1)
plt.plot(tper,F)
plt.title("Forcing")

plt.subplot(3,1,2)
plt.plot(tper,T_entr0)
plt.title("No Entrain")

plt.subplot(3,1,3)
plt.plot(tper,T_entr1)
plt.title("Entrain")

plt.tight_layout()
plt.savefig(outpath+"ForcingSPG_1x.png",dpi=200)



