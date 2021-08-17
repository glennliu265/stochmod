#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Model with Net Heat Flux Directly from CESM

Created on Sun Apr 25 08:54:19 2021

@author: gliu
"""


import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs

from scipy import signal
import xarray as xr

#%% User Edits
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210424/"

debug     = True
useanom   = False
pointmode = True
lonf       = -30
latf       = 50
rho        = 1026
cp0        = 3996
dt         = 3600*24*30
mconfig    = "FULL_PIC"
quadrature = False
lags       = np.arange(0,37,1)
mons3      = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
hfix = 50
applyfac = 2
ftype = "DJFM-MON"

labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
expcolors = ('blue','orange','magenta','red')

# testlam = array([169.94615638, 174.40679466, 187.23575848, 188.840277  ,
#        190.71626866, 183.75317341, 177.30749037, 176.80800517,
#        165.06422115, 169.41851357, 178.10587955, 159.28876965])


#%% Functions

def load_data(mconfig,ftype,projpath=None):
    
    """
    Inputs
    ------
    mconfig : STR
        Model Configuration (SLAB_PIC or FULL_HTR)
    ftype : STR
        Forcing Type ('DJFM-MON' or ... )
    projpath : STR (optional)
        Path to project folder (default uses path on laptop)
    
    Outputs
    -------
    mld : ARRAY 
        Monhtly Mean Mixed Layer Depths
    kprevall : ARRAY
        Detrainment Months
    lon : ARRAY
        Longitudes (-180 to 180)
    lat : ARRAY
        Latitudes
    lon360 : ARRAY
        Longitudes (0 to 360)
    cesmslabac : ARRAY
        Autocorrelation at each point in the CESM Slab
    damping : ARRAY
        Monthly ensemble mean Heat flux feedback
    forcing : ARRAY
        Monthly forcing at each point (NAO, EAP, EOF3)
    """
    
    # Set Paths
    if projpath is None:
        projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/'
    input_path  = datpath + 'model_input/'
    
    # Load Data (MLD and kprev) [lon x lat x month]
    if mconfig == "FULL_HTR": # Load ensemble mean historical MLDs
        mld            = np.load(input_path+"%s_HMXL_hclim.npy"% mconfig) # Climatological MLD
        kprevall       = np.load(input_path+"%s_HMXL_kprev.npy"% mconfig) # Entraining Month
    else: # Load PIC MLDs 
        mld            = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
        kprevall       = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
    
    mld1kmean      = np.load(input_path+"FULL_PIC_HMXL_hclim_400to1400mean.npy") # Entraining Month
    
    # Load Lat/Lon, autocorrelation
    dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp       = loadmat(input_path+dampmat)
    lon            = np.squeeze(loaddamp['LON1'])
    lat            = np.squeeze(loaddamp['LAT'])
    cesmslabac     = np.load(datpath+"CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
    lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
    
    # Load damping [lon x lat x mon]
    if mconfig == "SLAB_PIC":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
    elif mconfig=="FULL_HTR":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")
    
    # Load Forcing  [lon x lat x pc x month]
    forcing = np.load(input_path+mconfig+ "_NAO_EAP_NHFLX_Forcing_%s.npy" % ftype)#[:,:,0,:]
    
    return mld,kprevall,lon,lat,lon360,cesmslabac,damping,forcing,mld1kmean
#%% Load Inputs


# ----------------
# Load Fnet and TS
# ----------------
# NHFLX, LON x LAT x TIME
st = time.time()
dsflx = xr.open_dataset(datpath+"NHFLX_PIC.nc") # [288 x 192 x 10766], lon180
flx = dsflx.NHFLX.values
lon = dsflx.lon.values
lat = dsflx.lat.values
print("Loaded NHFLX in %.2fs"% (time.time()-st))

nlon,nlat,ntime =flx.shape


# -----------------------
# Load Mixed Layer Depths and related variables kprev
# -----------------------
st = time.time()
dsmld = xr.open_dataset(datpath+"HMXL_PIC.nc")
mld = dsmld.HMXL.values/100 # Convert to meters
lon = dsmld.lon.values
lat = dsmld.lat.values
print("Loaded MLD in %.2fs"% (time.time()-st))
kprevall       = np.load(datpath+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
hblt  = np.load(datpath+"../SLAB_PIC_hblt.npy")

# ------------
# Load Damping
# ------------
damping = np.load(datpath+"SLAB_PIC"+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")

# ------------------
# Load CESM Spectrum
# ------------------
specnames = "nsmooth%i_taper%i" % (100,10)
ld = np.load("%s../model_output/CESM_PIC_Spectra_%s.npz"%(datpath,specnames),allow_pickle=True)
Pcesmfulla    = ld['specfull']
Pcesmslaba    = ld['specslab']
freqcesmfulla = ld['freqfull']
freqcesmslaba = ld['freqslab']


# -----------------------
# Close Datasets
# -----------------------
dsmld.close()
dsflx.close()



#%% Restrict to point
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
loctitle  = "Lon %.2f Lat %.2f" % (lon[klon],lat[klat])
locfn     = "lon%i_lat%i" % (lonf,latf)

# Calculate 1000 year mean and seasonal MLDs
nlon,nlat,ntimef = mld.shape
hclim      = mld.reshape(nlon,nlat,int(ntimef/12),12)
mldcycle   = hclim.mean(2)
#mld_1kyr   = mld[:,:,:1000]
#mld_1kmean = mld_1kyr.mean(2)

# Retrieve Point ata
params = scm.get_data(1,[lonf,latf],lat,lon,damping,mldcycle,kprevall,flx)
[o,a],damppt,mldpt,kprev,Fpt = params
kmonth = mldpt.argmax()
mldmean = hblt[klon,klat,:].mean()

#%
#%% Run the Model
#%


# Prepare Forcing
Fh = {}
nyrs = int(ntime/12)
for h in range(3):
    if h == 0: # Fixed 50 meter MLD
        Fh[h] = Fpt * (dt/(cp0*rho*hfix))
    elif h == 1: # Seasonal Mean MLD
        Fh[h] = Fpt * (dt/(cp0*rho*mldmean))
    elif h == 2: # Seasonall Varying mean MLD
        Fh[h] = Fpt / np.tile(mldpt,nyrs) * (dt/(cp0*rho))

# Convert Parameters
lbd,lbd_entr,FAC,beta = scm.set_stochparams(mldpt,damppt,dt,ND=False,hfix=hfix,hmean=mldmean)


# Run the stochastic model
multFAC = 0
if applyfac > 1: # Apply Integration factor
    multFAC = 1

sst         = {}
dampingterm = {}
forcingterm = {}
for i in range(3): # No Entrainment Cases
    sst[i],forcingterm[i],dampingterm[i] = scm.noentrain(ntime,lbd[i],0,Fh[i],FAC[i],multFAC=multFAC,debug=True)

sst[3],dampingterm[3],forcingterm[3],entrainterm,Td=scm.entrain(ntime,
                   lbd[3],0,Fh[2],
                   beta,mldpt,kprev,
                   FAC[3],multFAC=multFAC,
                   debug=True,debugprint=False)
    
# Reassign Params
params = ([o,a],damppt,mldpt,kprev,Fpt)
autocorr = scm.calc_autocorr(sst,lags,kmonth+1)

#%%
ac = autocorr

# Get AC Data
query = [lonf,latf]
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = load_data("SLAB_PIC",ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
fullauto = np.load(datpath+"../CESM_clim/TS_FULL_Autocorrelation.npy")
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

#
# Calculate AC for forcing
slabqnet = Fpt*dt/(rho*cp0*mldmean)
autocorrflx = scm.calc_autocorr([slabqnet],lags,kmonth+1)[0]


# Calculate Confidene Intervals
conf  = 0.95
tails = 2
def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags
nlags   = len(lags)
cfstoch = np.zeros([4,nlags,2])
for m in range(4):
    inac = ac[m]
    n = int(len(sst[m])/12)
    cfs = calc_conflag(inac,conf,tails,n)
    cfstoch[m,:,:] = cfs
cfslab = calc_conflag(cesmauto2,conf,tails,898)
cffull = calc_conflag(cesmautofull,conf,tails,1798)
cfflx  = calc_conflag(autocorrflx,conf,tails,898)

# Plot Autocorrelation
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation (%s) \n %s Forcing; Lag 0 = %s" % (locstringtitle, mconfig,mons3[mldpt.argmax()])
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)
ax.plot(lags,autocorrflx[lags],label="$Q_{net}$",color='b')
ax.fill_between(lags,cfflx[lags,0],cfflx[lags,1],color='b',alpha=0.10)
# ax.plot(lags,cesmautofull,color='k',label='CESM Full',ls='dashdot')
# ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)
#for i in range(1,4):
for i in [1]:
    ax.plot(lags,ac[i],label=labels[i],color=expcolors[i])
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=expcolors[i],alpha=0.25)
ax.legend()
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outfigpath+"QnetIntegration_%s_Autocorrelation_CF_%s.png"% (mconfig,locstring),dpi=200)


#%% Calculate Spectrum
# Parameters
pct     = 0.10
nsmooth = 100
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1
specnames = "nsmooth%i_taper%i" % (nsmooth,pct*100)

# Retrieve Data For Point (CESM)
lonf = query[0]
latf = query[1]
if lonf < 0:
    lonf += 360
klon360,klat   = proc.find_latlon(lonf,latf,lon360,lat) # Global, 360 lon

Pcesmfull    = Pcesmfulla[:,klat,klon360]
Pcesmslab    = Pcesmslaba[:,klat,klon360]
freqcesmfull = freqcesmfulla[:,klat,klon360]
freqcesmslab = freqcesmslaba[:,klat,klon360]

# ----------------------------------
# Calculate spectrum for the forcing
# ----------------------------------
sps      = ybx.yo_spec(slabqnet,opt,nsmooth,pct,debug=False)
Pflx,freqflx,dof,r1=sps
Pflx *= dt
freqflx /= dt

# -------------------------------------------
# calculate for the Individual experiments
# -------------------------------------------
sstall=sst
#nsmooth=nsmooth*10/2
specparams  = []
splotparams = []
specs = []
freqs = []
for i in range(4):
    sstin = sstall[i]
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    specs.append(P*dt)
    freqs.append(freq/dt)


#%% Plot Spectrum together    
expcolors = ('blue','orange','magenta','red')

# Set up variance preserving plot
freq = freqs[0]
fig,ax = plt.subplots(1,1)
ax.set_ylabel("Frequency x Power",fontsize=13)

#for i in np.arange(1,4):
for i in [1]:
    print(specs[i].sum())
    ax.semilogx(freqs[i],freqs[i]*specs[i],label=labels[i],color=expcolors[i],lw=0.75)
#ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='k',lw=0.75)
ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
ax.semilogx(freqflx,Pflx*freqflx,label="$Q_{net}$",color='b',lw=0.75)


xmin = 10**(np.floor(np.log10(np.min(freq))))

ax.set_xlim([xmin,0.5/dt])
ax.grid(True,ls='dotted')
freqtick = ax.get_xticks()
yy = ax.get_ylim()
period =1/freq
htax   =ax.twiny()
htax.set_xscale("log")
htax.set_yscale("linear")
xtkl = ["%.1f"% s for s in np.fix(1/freqtick/dt)]
htax.set_xlim([xmin,0.5/dt])
htax.set_xlabel("Period (%s)"%tunit,fontsize=13)
ax.legend()

ax,htax=viz.make_axtime(ax,htax)

#ax.semilogx(freqcesmfull,freqcesmfull*Pcesmfull,'gray',label="CESM-FULL")
#vlv = [1/(100*dt*12),1/(10*12*dt),1/(12*dt)]
vlv = [1/(100*365*24*3600),1/(10*365*24*3600),1/(365*24*3600)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    ax.grid(True,which='both',ls='dotted')

#ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum at %s \n" % (locstringtitle) + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i_%s_fluxintegration.png"%(outfigpath,'COMPARISON',nsmooth,pct*100,axopt,locstring),dpi=200)

