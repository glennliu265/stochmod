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

## Functions --------------------------------------------------
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
        noise_term = F[t-1]
        
    
        # Compute the temperature
        temp_ts[t] = lbd[m-1]*T + noise_term  
    
        # Save other variables
        noise_ts[t] = noise_term
        damp_ts[t] = lbd[m-1]*T


    # Quick indexing fix
    temp_ts[0] = T0
    noise_ts = np.delete(noise_ts,0)
    damp_ts = np.delete(damp_ts,0)
    
    return temp_ts,noise_ts,damp_ts

# User Edits -----------------------------------------------------------------           

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
nyr     = 1000      # Number of years to integrate over
t_end   = 12*nyr      # Timestep to integrate up to
dt      = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
usetau  = 0           # Use tau (estimated damping timescale)
useeta  = 0          # Use eta from YO's model run
usesst  = 0
hvar    = 1           # seasonally varying h

# Correlation Options
detrendopt = 0  # Option to detrend before calculations


# White Noise Options
genrand   = 1  #

#Set Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200608/'


# Set up some strings for labeling
loc_figtitle = "Lon: %i Lat: %i" % (lonf,latf)
if lonf < 0:
    lonstr = lonf + 360
loc_fname    = "Lon%03d_Lat%03d"  % (lonstr,latf)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')


## ------------ Script Start -------------------------------------------------

# Load Variables -------------------------------------------------------------

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
SST2 = loadmod['SST2']
SST0 = loadmod['SST']


# Load Mixed layer variables
mldnc = "XMXL_HTR_clim.nc"
ds = xr.open_dataset(datpath+mldnc)




# ----------------------------------------------------------------------------

# Find the corresponding Lat/Lon indices
oid = np.abs(LON-lonf).argmin()
aid = np.abs(LAT-latf).argmin()

# Create seasonally varying h value
if hvar == 1:
    # Do same for curivilinear grid
    if lonf < 0:
        lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates
    
    # Find the specified point on curvilinear grid and average values
    selectmld = ds.where((lonfc-0.5 < ds.TLONG) & (ds.TLONG < lonfc+0.5)
                        & (latf-0.5 < ds.TLAT) & (ds.TLAT < latf+0.5),drop=True)
    # Select accordingly 
    ensmean = selectmld.mean(('ensemble','nlon','nlat'))/100
    h = np.squeeze(ensmean.to_array())



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
    

# Select which forcing to use
if useeta == 1:
    F = np.copy(eta)
    F = np.squeeze(F)

# Mixed Layer Depth term preparations ----------------------------------------

# Compute the integral of the entrainment term
beta = np.log( h / np.roll(h,1) )
beta[beta < 0] = 0
    
# Create a predefined array to represent MLD temps (1 meter resolution)
mlddepths = np.arange(0,np.max(h)+1,1)
mldtemps = np.zeros(mlddepths.shape)
 
# Prepare for Correlation Calculations----------------------------------------
# Run model

# t_start = time.localtime()
# start = time.strftime("%H:%M:%S",t)
start = time.time()


# Try doing the same via function
temp_ts,noise_ts,damp_ts = stochmod_noentrain(t_end,lbd,T0,F)

# t_end = time.localtime()
# end = time.strftime("%H:%M:%S",t)
elapsed = time.time() - start
tprint = "Model ran in %.2fs" % (elapsed)
print(tprint)

# Prepare for Correlation Calculations----------------------------------------
# Reshape Time Series to months x year
if usesst == 1:
    #temp_ts = SST1
    #temps   = np.reshape(temp_ts,(1000,12))
    temps   = np.reshape(temp_ts,(76,12))
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


## YO's Method (Currently not working, need to check indexing)
# corr_ts = np.zeros(len(lags))
# for i in lags:
#     lag_yr = int(np.floor(i+kmonth/12))
    
#     baserng = range(kmonth,t_end-lag_yr*12,12)
#     lagrng = range(kmonth+i,t_end-lag_yr*12,12)
#     corr_ts[i] = stats.pearsonr(temp_ts[0,baserng],temp_ts[0,lagrng])[0]
    
#detrendopt = 1
corr_ts = calc_lagcovar(temps,temps,lags,kmonth,detrendopt)



## Make Figures


# Xorrelation plots
f1 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')

ax.plot(lags,AVG0,color='b',label='CESM1 (YO)')
ax.plot(lags,AVG1,color='c',label='No-Entrain (YO)')
ax.plot(lags,AVG2,color='g',label='Entrain (YO)')
ax.plot(lags,corr_ts,'-r',label='No-Entrain (GL)')
ax.legend()


titlestr = 'SST Autocorrelation; Month'+str(kmonth) + '\n Lon: ' + \
    str(lonf) + ' Lat: '+ str(latf)
ax.set(xlabel='Lag (months)',
       ylabel='Correlation',
       ylim=(-0.3,1.1),
       title=titlestr )
outname = outpath+'SSTAC_usetau_40kyr'+str(usetau)+'.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


    

# lambda plots ----------------------------------------------------------------

lbd1 = np.exp(-1 * 1 / np.mean(tauall, axis=1) )
lbd2 = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*h) * dt)
lbd3 = np.exp(-1 * damping[oid,aid,:] / (rho*cp0*150) * dt)

lbd1all =np.exp(-1 * 1 / tauall)
monthx = np.arange(1,13,1)        




f2 = plt.figure()
ax = plt.axes()
sns.set('paper','whitegrid','bright')
for e in range(1,np.shape(tauall)[1]):
    #print(e)
    ax.plot(monthx,lbd1all [:,e],color=(.75,.75,.75))
ln0 = ax.plot(monthx,lbd1all [:,-1],color=(.75,.75,.75),label=r'$exp(-1/\tau)$ : Indv. Member')
ln1 = ax.plot(monthx,lbd1,color='k',label=r'$exp(-1/\tau)$ : EnsAvg)')
ln2 = ax.plot(monthx,lbd2,color='b',label=r'$exp(-\lambda_{net} / (\rho cp_0H)\Delta t)$ : Varying MLD')
ln3 = ax.plot(monthx,lbd3,color='r',label=r'$exp(-\lambda_{net} / (\rho cp_0H)\Delta t)$ : Constant MLD')
ax.legend(loc='best',)
plt.rc('legend', fontsize=10)    # legend fontsize



lns = ln0 + ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0,bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)

titlestr = 'Seasonal Values for Damping Term'
ax.set(xticks=np.arange(1,13,1),
       xlim=(1,12),
       xlabel='Month',
       ylabel='Value ($mon^{-1}$)',
       title=titlestr)
outname = outpath+"LambdaComp.png"
plt.savefig(outname, bbox_inches="tight",dpi=200)

        


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






# Twin Axis Damping Plot
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






# Xorrelation plots detrended
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




# Visualize differences between mixed layer variables ------------------------


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

