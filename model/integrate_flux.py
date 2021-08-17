#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Model where T(t+1) = T(t) + F(t)/(rho*cp0*h)

Created on Wed Feb  3 14:11:57 2021
@author: gliu
"""

import numpy as np
import xarray as xr
from tqdm import tqdm
import time
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.io import loadmat

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz

import yo_box as ybx
import tbx
import scm
from scipy import signal
#%% User Edits
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210322_AMVTeleconf/"

debug     = True
useanom   = False
pointmode = True
lonf     = -30
latf     = 50
rho      = 1026
cp0      = 3996
dt       = 3600*24*30
quadrature = False
lags       = np.arange(0,37,1)
mons3      = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# testlam = array([169.94615638, 174.40679466, 187.23575848, 188.840277  ,
#        190.71626866, 183.75317341, 177.30749037, 176.80800517,
#        165.06422115, 169.41851357, 178.10587955, 159.28876965])



cmbal = cmocean.cm.balance
#%% Load MLD, TS, Fnet

# ----------------
# Load Fnet and TS
# ----------------
if useanom:
    # NHFLX, LON x LAT x TIME
    st = time.time()
    dsflx = xr.open_dataset(datpath+"NHFLX_PIC.nc") # [288 x 192 x 10766], lon180
    flx = dsflx.NHFLX.values
    lon = dsflx.lon.values
    lat = dsflx.lat.values
    print("Loaded NHFLX in %.2fs"% (time.time()-st))
    
    
    #%Load in TS from slab
    #fn1 = datpath + "../TS_SLAB_withENSO.npy"
    fn2 = datpath + "../ENSOREM_TS_lag1_pcs2_monwin3.npz"
    #tsenso1 = np.load(fn1)
    ld2 = np.load(fn2)
    tsenso0 = ld2['TS']
    lon360  = ld2['lon']
    lat     = ld2['lat']
    
    # Remap lon360 --> 180
    nmon,nlat,nlon = tsenso0.shape
    tsenso0 = tsenso0.transpose(2,1,0) # lon x lat x mon
    lon1,tsenso180 = proc.lon360to180(lon360,tsenso0) # Flip longitudes

else:
    # NHFLX, LON x LAT x TIME
    st = time.time()
    dsflx = xr.open_dataset(datpath+"NHFLX_PIC_SLAB_raw.nc") # [288 x 192 x 10766], lon180
    flx = dsflx.NHFLX.values # [10812, 192, 288]
    lon360 = dsflx.lon.values
    lat = dsflx.lat.values
    
    # Remap lon360 --> 180
    flx = flx.transpose(2,1,0)
    lon1,flx = proc.lon360to180(lon360,flx)
    print("Loaded NHFLX in %.2fs"% (time.time()-st))
    
    # Load TS
    st     = time.time()
    dssst  = xr.open_dataset(datpath+"TS_PIC_SLAB_raw.nc") # [288 x 192 x 10766], lon180
    ts     = dssst.TS.values # [10812, 192, 288]
    
    # Remap lon360 --> 180
    ts = ts.transpose(2,1,0)
    lon1,ts = proc.lon360to180(lon360,ts)
    print("Loaded TS in %.2fs"% (time.time()-st))

# -----------------------
# Load Mixed Layer Depths
# -----------------------
st = time.time()
dsmld = xr.open_dataset(datpath+"HMXL_PIC.nc")
mld = dsmld.HMXL.values/100 # Convert to meters
lon = dsmld.lon.values
lat = dsmld.lat.values
print("Loaded MLD in %.2fs"% (time.time()-st))

# -----------------------
# Close Datasets
# -----------------------
dsmld.close()
dsflx.close()
dssst.close()

# -------------------------------------
#%% Calculate 1kmean Mixed Layer Depth
# -------------------------------------

klon,klat = proc.find_latlon(lonf,latf,lon1,lat)
loctitle  = "Lon %.2f Lat %.2f" % (lon1[klon],lat[klat])
locfn     = "lon%i_lat%i" % (lonf,latf)

# Calculate 1000 year mean and seasonal MLDs
nlon,nlat,ntimef = mld.shape
hclim      = mld.reshape(nlon,nlat,int(ntimef/12),12)
mldcycle   = hclim.mean(2) 
mld_1kyr   = mld[:,:,:1000]
mld_1kmean = mld_1kyr.mean(2)

print("1000 yr mean mld is : %.3f m" % (mld_1kmean[klon,klat]))
print("Mean of scycle is   : %.3f m" % (mldcycle.mean(2)[klon,klat]))

if debug:
    fig,ax  = plt.subplots(1,1)
    for i in range(10):
        ax.plot(hclim[klon,klat,i,:],label='year %i'%i,alpha=0.5)
    ax.plot(mldcycle[klon,klat,...],label='mean cycle',color='k')
    ax.hlines([mld_1kmean[klon,klat]],xmin=0,xmax=12,label="1kyr Mean",color='gray',ls='dashed')
    ax.set_title("Seasonal MLD Cycle at %s"%loctitle)
    ax.grid(True,ls='dotted')
    ax.legend()

# -------------------------------
#%% Load Qcorrection and Slab MLD
# -------------------------------

hblt  = np.load(datpath+"../SLAB_PIC_hblt.npy")
qdp   = np.load(datpath+"../SLAB_PIC_qdp.npy") 

qdpa = -1*(qdp -qdp.mean(2)[:,:,None])

if debug:
    fig,axs = plt.subplots(2,2)
    
    axlist = axs.flatten()
    
    ax = axlist[0]
    pcm = ax.pcolormesh(mld_1kmean.T,vmin=0,vmax=500)
    ax.set_title("MLD (Mean, yr400-1400)")
    fig.colorbar(pcm,ax=ax)
    
    
    ax = axlist[1]
    pcm = ax.pcolormesh(flx[:,:,0].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_title("Qnet (400-01)")
    fig.colorbar(pcm,ax=ax)
    
    ax = axlist[2]
    pcm = ax.pcolormesh(hblt[:,:,0].T,vmin=0,vmax=500)
    ax.set_title("hblt (Jan)")
    fig.colorbar(pcm,ax=ax)
    
    ax = axlist[3]
    pcm = ax.pcolormesh(qdpa[:,:,0].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_title("qdp (Jan)")
    fig.colorbar(pcm,ax=ax)
    
    plt.tight_layout()
    
    plt.savefig(outfigpath+"QIntegration_Inputs.png",dpi=200)

# --------------------------------------------------------------
#%% Plot Monthly Patterns of Qflx Correction over north atlantic
# --------------------------------------------------------------

fig,axs= plt.subplots(4,3,figsize=(8,8))

for i in range(12):
    ax  = axs.flatten()[i]
    pcm = ax.pcolormesh(lon1,lat,qdpa[:,:,i].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_xlim([-80,20])
    ax.set_ylim([0,80])
    ax.set_title(mons3[i])
    fig.colorbar(pcm,ax=ax)

    plt.tight_layout()

plt.savefig(outfigpath+"qdp_anomaly.png",dpi=200)

# --------------------------------------------------------------
#%% Load in Snow Contribution for single point
# --------------------------------------------------------------


# Load Data
precsc = np.load(datpath+"../PRECSC_PIC_SLAB_lon330_lat50.npy")
precsl = np.load(datpath+"../PRECSL_PIC_SLAB_lon330_lat50.npy")


# Convert to W/m2
lhf_snow = 3.33e5 # J/kg, Latent heat of fusion of snow
rho_fw   = 1e3 # kg/m3, Density of Fresh Water
precsc = precsc * (lhf_snow*rho_fw)
precsl = precsl * (lhf_snow*rho_fw)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot The Distribution for each month
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig,axs=plt.subplots(2,1)
ax = axs[0]
ax.plot(mons3,precsc.reshape(901,12).T,label="",color='b',alpha=0.1)
ax.plot(mons3,precsc.reshape(901,12)[0,:],label="Indv. Year",color='b',alpha=0.1)
ax.plot(mons3,precsc.reshape(901,12).mean(0),label="Mean",color='k',alpha=1)
ax.set_title("Convective snow rate (water equivalent) $(W/m^{2})$")
ax.grid(True,ls='dotted')
ax.set_xlim([0,11])
ax.set_ylabel("$W/m^{2}$")
ax.legend()

ax = axs[1]
ax.plot(mons3,precsl.reshape(901,12).T,label="",color='r',alpha=0.1)
ax.plot(mons3,precsl.reshape(901,12)[0,:],label="Indv. Year",color='r',alpha=0.1)
ax.plot(mons3,precsl.reshape(901,12).mean(0),label="Mean",color='k',alpha=1)
ax.set_title("Large-scale (stable) snow rate (water equivalent)  $(W/m^{2})$")
ax.grid(True,ls='dotted')
ax.set_xlim([0,11])
ax.set_ylabel("$W/m^{2}$")
ax.legend()

plt.tight_layout()
plt.savefig(outfigpath+"SNOW_Contribution_50N_330E.png",dpi=200)



# Plot cumulative contribution
acontrib = (precsc+precsl).reshape(901,12)
mcontrib = acontrib.mean(0)


xtklsnow = ["%s \n (%.2f) " % (mons3[i],mcontrib[i]) for i in range(12)]

fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.plot(mons3,acontrib.T,alpha=0.25,color='k')
ax.plot(mons3,acontrib.mean(0),color='r',label="Mean")
ax.legend()
ax.grid(True,ls='dotted')
ax.set_xticklabels(xtklsnow)
ax.set_title(r"$Q_{snow}$")
ax.set_ylabel("W/m2")
plt.savefig(outfigpath+"MeanContrib.png",dpi=200)

print(acontrib.mean(0))

snowcorr =acontrib

# ---------------------------------
#%% Plot monthly Fnet at that target point
# ---------------------------------

flxmon = flx[klon,klat,...].reshape(int(flx.shape[2]/12),12)

fig,ax = plt.subplots(1,1)
ax.plot(mons3,flxmon.T,color='gray',alpha=0.1,label="")
#ax.plot(mons3,flxmon.std(0),color='k',ls='dashed',label=r"$\pm 1 \sigma$")

#ax.plot(mons3,flxmon.mean(0),color='k',ls='dotted')
ax.plot(mons3,-1*flxmon.std(0),color='k',ls='dashed',label="")


#ax.plot(mons3,qdp[klon,klat,:],color='r',ls='solid',label="Q-Correction")

ax.set_ylim([-200,200])
ax.grid(True,ls='dotted')
ax.legend()
ax.set_title("NHFLX Seasonal Cycle at %s"%loctitle)
plt.savefig("%sNHFLXa_Scycle_SLAB_PIC_%s.png"%(outfigpath,locfn),dpi=200)


# ------------------------------
#%% Plot result after correction
# ------------------------------

flxmon = (flx[klon,klat,...]+np.tile(qdp[klon,klat,...],int(10812/12))).reshape(int(flx.shape[2]/12),12)

fig,ax = plt.subplots(1,1)
ax.plot(mons3,flxmon.T,color='gray',alpha=0.1,label="")
ax.plot(mons3,flxmon.std(0)+flxmon.mean(0),color='k',ls='dashed',label=r"$\pm 1 \sigma$")
#ax.plot(mons3,flxmon.mean(0),color='k',ls='dotted')
ax.plot(mons3,-1*flxmon.std(0)+flxmon.mean(0),color='k',ls='dashed',label="")
ax.plot(mons3,(-1*qdp[klon,klat,:]),color='r',ls='solid',label="Q-Correction")
ax.set_ylim([-200,200])
ax.grid(True,ls='dotted')
ax.legend()

ax.set_title("NHFLX Seasonal Cycle at %s (Corrected)"%loctitle)
plt.savefig("%sNHFLX_Scycle_SLAB_PIC_%s_corr.png"%(outfigpath,locfn),dpi=200)


# ----------------------------------------------------------------------------------------
#%% Interpolate values (move forcing to the beginning rather than the middle of the month)
# ----------------------------------------------------------------------------------------

print("Interpolating Values")
flxquad = np.zeros(flx.shape)
tsquad = np.zeros(flx.shape)
for i in tqdm(np.arange(1,flx.shape[2]-1)):
    flxquad[:,:,i] = (flx[:,:,i-1] + flx[:,:,i]) / 2 # Take mean of 2 values
    tsquad[:,:,i] = (ts[:,:,i-1] + ts[:,:,i]) / 2 # Take mean of 2 values
    #print("%f is between %f and %f"% (flxquad[222,111,i-1],flx[222,111,i-1],flx[222,111,i]))
flxquad[:,:,0] = flx[:,:,0]
tsquad[:,:,0] = ts[:,:,0]

# Interpolate qdp as well
qdpquad = np.roll(qdp,1,axis=2)
qdpquad = (qdpquad+qdp)/2
fig,ax = plt.subplots(1,1)
ax.set_title("QDP Interpolation")
ax.plot(np.arange(1.5,13.5,1),qdp[klon,klat,:],label="ori",color='k',marker='d')
ax.plot(np.arange(1,13,1),qdpquad[klon,klat,:],label="shift",marker="o",color='red')
ax.set_xticks(np.arange(1,13,1))
ax.grid(True,ls="dotted")
#ax.set_xticklabels()

# Plot Differences
fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.plot(np.arange(0.5,13.5,1),flx[222,111,:13],label="Original Timeseries",color='k',zorder=-2,marker="d")
ax.scatter(np.arange(1,13,1),flxquad[222,111,1:13],label="Interpolated Timeseries",marker="o",color='red')
ax.plot(np.arange(1,13,1),flxquad[222,111,1:13],label="",marker="o",color='red',ls='dashed')
ax.set_xticks(np.arange(0,13,1))
ax.set_xticklabels(np.hstack([mons3,mons3[0]]))
ax.grid(True,ls='dotted')
ax.legend()
ax.set_xlabel("Month")
ax.set_ylabel("Forcing (W/m2)")
ax.set_title("Interpolated Qnet")
plt.savefig(outfigpath+"Interpolated_Qnet.png",dpi=200)

# ---------------------
#%% Integrate the model
# ---------------------

# Set Inputs
flxin   = flx
qdpin   = -qdp
mldmean = hblt.mean(2)

dp = 1

_,_,tsteps = flx.shape
if pointmode == 0:
    # Preallocate
    sstq  = np.zeros((nlon,nlat,tsteps))
    sstq[:,:,0] = ts[:,:,0]
    sstn = sstq.copy()
    ssts = sstq.copy()
    for t in tqdm(range(1,tsteps)):
        
        m = (t)%12
        
        # SST with correction
        sstq[:,:,t] = sstq[:,:,t-1]  + (flxin[:,:,t] + qdpin[:,:,m]) * (dt / (rho*cp0*mldmean))
        
        # SST without correction
        sstn[:,:,t] = sstn[:,:,t-1]  + flxin[:,:,t]                  * (dt / (rho*cp0*mldmean))
        
        # SST in quadrature (without correction)
        ssts[:,:,t] = ssts[:,:,t-1]  + (flxquad[:,:,t]+ -1*qdpquad[:,:,m]) * (dt / (rho*cp0*mldmean))
        
        
        # Plot results
        fig,ax = plt.subplots(1,1)
        ax.plot(sstq[klon,klat,:],label="With Correction")
        ax.plot(sstn[klon,klat,:],label="Without Correction")
        ax.legend()
        
else:
    
    # Select variables at point
    flxin   = flxin[klon,klat]
    qdpin   = qdpin[klon,klat]
    mldmean = mldmean[klon,klat]
    
    
    # Preallocate
    sstq    = np.zeros(tsteps)
    sstq[0] = ts[klon,klat,0]
    sstn    = sstq.copy()
    ssts    = sstq.copy()
    sstcd   = sstq.copy()
    #sstcd[0:2] = ts[klon,klat,0:2]
    for t in tqdm(range(1,tsteps)):
        
        m = (t)%12
        
        # SST with correction
        sstq[t]  = sstq[t-1]*dp  + (flxin[t] + qdpin[m]) * (dt / (rho*cp0*mldmean))
        
        # SST without correction
        sstn[t]  = sstn[t-1]*dp  + flxin[t]              * (dt / (rho*cp0*mldmean))
        
        # SST in quadrature (wit correction)
        ssts[t]  = ssts[t-1]*dp  + (flxquad[klon,klat,t] + -1*qdpquad[klon,klat,m]) * (dt / (rho*cp0*mldmean))
        
        # SST with correction, centered difference
        #if t >=2:
        #sstcd[t] = sstq[t-1]*dp  + (flxin[t] + qdpin[m] - (snowcorr.max(0)[m])) * (dt / (rho*cp0*mldmean))
        sstcd[t] = sstcd[t-1]*dp  + (flxin[t] + qdpin[m] - (precsl[t] + precsc[t])) * (dt / (rho*cp0*mldmean))
            #sstcd[t] = sstcd[t-2]*dp + (flxquad[klon,klat,t] + -1*qdpquad[klon,klat,m]) * (2*dt / (rho*cp0*mldmean))
        
    fig,ax=plt.subplots(1,1)
    
    ax.set_title("Temperature Integration Output (Undetrended)")
    ax.plot(sstq,label="SST Q-Correction")
    ax.plot(sstcd,label="With Q-correction and Snow Contribution",alpha=0.9,ls='solid')
    ax.plot(ts[klon,klat,:],label="CESM",color='k')
    #ax.plot(sstn,label="Without Correction",color="red")
    ax.legend()
    #ax.set_xlim([10260,10262])
    ax.grid(True,ls='dotted')
    plt.savefig(outfigpath+"Undetrended_Output_1dIntegrationlast300.png",dpi=200)
    
    plt.show()
    plt.plot(sstq-sstcd,color='r')
    plt.title("Differences (With - Without Snow Effects)")
    plt.savefig(outfigpath+"Snow_Differences.png",dpi=200)


#%% Plot Corrected Forcing inputs


rhs = (flxin+np.tile(qdpin,901)-(precsl + precsc)).reshape(901,12)

fig,ax=plt.subplots(1,1)
ax.plot(mons3,flxin.reshape(901,12).T,color='gray',alpha=0.1)
ax.plot(mons3,rhs.T,color='orange',alpha=0.05,label="")
#ax.fill_between(mons3,rhs.min(0),rhs.max(0),color='orange',alpha=0.4)
ax.plot(mons3,flxin.reshape(901,12).mean(0),color='k',label="Fnet")
ax.plot(mons3,qdpin,color='r',label="Qcorr")
ax.plot(mons3,-1*(precsl + precsc).reshape(901,12).T,color='cornflowerblue',alpha=0.05,label="")
ax.plot(mons3,-1*(precsl + precsc).reshape(901,12).mean(0),color='b',alpha=1,label=r"$Q_{Snow}$")
ax.plot(mons3,rhs.mean(0),color='gold',label="Corrected Forcing")
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("W/m2")
ax.set_ylim([-350,350])
ax.set_title("1-D Temperature Integration Inputs")
plt.savefig(outfigpath+"Integration_Inputs.png",dpi=200)

# -----------------------
#%% Post Processing
# -----------------------
def postproc_sst(sst,manom=True,detrend=True):
    sst = sst-sst.mean(0)
    
    if detrend:
        # Detrend Data
        sst = signal.detrend(sst)
    
    if manom:
        # Calculate Monthly Anomalies
        ntime = sst.shape[0]
        sst    = sst.reshape(int(ntime/12),12)
        sst = sst - sst.mean(0)[None,:]
    return sst.flatten()

# Set Experiment Colors/Parameters
expcolors = ['k','b','cyan','r','m','orange']
expnames  = ('$SST_{CESM}$',
             r"$SST_{Qcorr}$",
             r"$SST$",
             r"$SST_{Qcorr + Shift}$",
             "$Q_{net}$",
             r"$SST (with Q_{snow})$"
             )
expls = ('solid','solid','dashed','solid','solid','solid') 
if pointmode:
    sstin   = [ts[klon,klat,:],sstq,sstn,ssts,flxin,sstcd]
else:
    sstin = [ts[klon,klat,:],sstq[klon,klat,:],sstn[klon,klat,:],ssts[klon,klat,:],flxin[klon,klat,:],sstcd[klon,klat,:]]

# # Detrend each timeseries
# if useanom:
#     sstin = [ts[klon,klat,:],sstq[klon,klat,:],sstn[klon,klat,:],ssts[klon,klat,:],flxin[klon,klat,:]]
# else:
#     sstin   = [ts[klon,klat,:],sstq,sstn,ssts,flxin]
sstproc = []

for sst in sstin:
    sstproc.append(postproc_sst(sst,manom=True))

sstann = []
for sst in sstproc:
    nyr = sst.shape[0]
    sst = sst.reshape(int(nyr/12),12)
    sstann.append(sst.mean(1))
    

# Remove a common seasonal cycle (from cesm slab)
# manoms = sstproc[0].reshape(int(sstproc[0].shape[0]/12),12).mean(0)
# for i in range(len(sstproc)):
#     sstma = sstproc[i].reshape(int(sstproc[0].shape[0]/12),12) - manoms[None,:]
    
#     sstproc[i] = sstma.flatten()


#%% Plot Differences for each timeseries


diffs = []
diffsann=[]
for s,sst in enumerate(sstproc):
    diff = (sstproc[0]-sst)**2
    diffa =(sstann[0]-sstann[s])**2
    diffs.append(diff)
    diffsann.append(diffa)
    
    print("RMSE For %i is %.3f"% (s,np.mean(diff)))
    
# Plot Differences w.r.t. time
fig,ax = plt.subplots(1,1)
for i in [0,1,2,3,5]:
    ax.plot(diffs[i],color=expcolors[i],label=expnames[i],ls=expls[i])
ax.legend()

np.argmax(diffs[3])

# Check if differences occur at any particular month
chkdiff = diffs[3].reshape(int(10812/12),12)
fig,ax = plt.subplots(1,1)
ax.set_title("$(SST_{CESM} - SST_{Qcorr+Shift})^{2}$")
ax.set_ylabel("SST (degC)")
ax.plot(mons3,chkdiff.T,color='k',alpha=0.1,label='')
ax.plot(mons3,chkdiff.std(0),color='w',ls='dotted',label="stdev of differences")
plt.savefig(outfigpath + "Differences_QcorrShift.png",dpi=200)



#
#%% Plot the results

invars = [sstproc,sstann]
invars2 = [diff,diffsann]
inlabs   = ["Monthly","Annual"]
tunits2   = ["Month","Year"]
ylm = [-4,4]
xtks = [np.arange(0,11000,500),np.arange(0,950,50)]



rngs2 = [[0,50],[150,250],[850,900]]
rngs1 = [[0,120],[1500,2500],[9800,10800]]
rngall = [rngs1,rngs2]

v = 1
invar = invars[v]
indiff = invars2[v]
inlab = inlabs[v]
rngs  = rngall[v]



# Plot full monthly timeseries
fig,ax = plt.subplots(1,1,figsize=(16,4))
ax2 = ax.twinx()
for i in [0,1,5]:
    lw = 1.5
    if i == 3:
        lw = 1.25
    ax.plot(invar[i],color=expcolors[i],label=expnames[i],ls=expls[i],lw=lw)
    
    if i == 5:
        ax2.bar(np.arange(0,invar[i].shape[0],1),indiff[i],color=expcolors[i],alpha=1)
ax.legend(ncol=4,fontsize=12)

ax.vlines([np.array(rngs).flatten()],ymin=ylm[0],ymax=ylm[-1],color='k',lw=1.25,label="",ls='dashed')
#ax.set_xlim([1600,1800])
ax.set_title(r"$Q_{net}$ Integration SST Anomaly at " + loctitle,fontsize=14)
ax.set_xlabel("Time (%s)" % inlab,fontsize=12)
ax.set_ylabel("SST (degC)",fontsize=12)
ax.set_ylim(ylm)
ax2.set_ylim([0,4])
ax2.set_ylabel("Squared Error (degC)")
ax.set_xticks(xtks[v])
ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig(outfigpath+"SST_Timeseries_%s.png"%inlab,dpi=200)



for rng in rngs:
    fig,ax = plt.subplots(1,1,figsize=(4,3))
    for i in [0,1,5]:#[0,1,3,5]:
        lw = 1
        if i == 3:
            lw = 0.75
        plotrng = np.arange(rng[0],rng[1])
        ax.plot(plotrng,invar[i][rng[0]:rng[1]],color=expcolors[i],label=expnames[i],ls=expls[i],lw=lw)
    ax.set_title("%s %i to %i" % (tunits2[v],rng[0],rng[1]))
    ax.grid(True,ls='dotted')
    plt.tight_layout()
    plt.savefig(outfigpath+"SST_Timeseries_%s_%ito%i.png" % (inlab,rng[0],rng[1]),dpi=200)

# ---------------------------
#%% Calculate Autocorrelation
# ---------------------------
kmonth = mldcycle[klon,klat,...].argmax()
print(kmonth)
acs  =scm.calc_autocorr(sstproc,lags,kmonth+1)
acs2 =scm.calc_autocorr(sstproc,lags,kmonth+1)

# Plot Result
xtk2 = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

fig,ax = plt.subplots(1,1)
title = r"$Q_{net}$ Integration SST Autocorrelation,"+ " Lag 0 = %s,\n %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in [0,1,5]:
    ax.plot(lags,acs[i],color=expcolors[i],label=expnames[i],ls=expls[i])
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s" % (locfn),dpi=200)

# --------------------------------
# %% Calculate Autocorrelation in Disjointed Chunks
# --------------------------------


m = 5
csize = 3600
cyear = int(2400/12)
numper = int(sstproc[0].shape[0]/csize)
cper = []
for i in range(int(numper)):
    per = np.arange(i*csize,(i+1)*csize)
    cper.append(per)

ssttarg = sstproc[m]

acs_per = []
sstpers = []
for i in range(numper):
    sstp = ssttarg[cper[i]]
    sstp = signal.detrend(sstp,type='linear')
    sstpers.append(sstp)

acs_per = scm.calc_autocorr(sstpers,lags,kmonth+1)


fig,ax = plt.subplots(1,1)
title = r"%s Autocorrelation by " % (expnames[m]) + "%i Year Periods Lag 0 = %s,\n %s" % (cyear,mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(numper):
    clabel = "yr %i-%i" % (int(cper[i][0]/12), int(cper[i][-1]/12))
    ax.plot(lags,acs_per[i],label=clabel)
for i in range(1):
    ax.plot(lags,acs[i],color=expcolors[i],label=expnames[i],ls=expls[i])
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend(ncol=3)
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_byperiod_chksize%i_model%s.png" % (locfn,csize,expnames[m]),dpi=200)

# --------------------------------
# %% Calculate Autocorrelation in Overlapping Chunks
# --------------------------------

plots=0
plotn =500
plotint = 100
m = 0
csize = 4800
numper = int((sstproc[0].shape[0] - csize)/12) + 1
cper = []
for i in range(int(numper)):
    
    per = np.arange(i*12,i*12+csize)
    cper.append(per)
ssttarg = sstproc[m]

acs_per = []
sstpers = []
for i in range(plotn):
    sstpers.append(ssttarg[cper[i]])

acs_per = scm.calc_autocorr(sstpers,lags,kmonth+1)

fig,ax = plt.subplots(1,1)
title = r"$Q_{net}$ Integration SST Autocorrelation by " + "%iyr Periods Lag 0 = %s,\n %s" % (csize/12,mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(plots,plotn,plotint):
    
    elab   = "yr %i - %i" % (cper[i][0]/12,cper[i][-1]/12)
    ealpha =(i+1)/(plotn)
    print(i)
    print(ealpha)
    #ealpha=1
    ax.plot(lags,acs_per[i],label=elab,color='b',alpha=ealpha)
    #ax.plot(lags,acs_per[i],label="",color='b',alpha=ealpha)
    #imax = 
    
    
    #ax.plot(lags,acs_per[i],label=cper[i][0])
for i in range(1):
    ax.plot(lags,acs[i],color=expcolors[i],label=expnames[i],ls=expls[i])
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend(ncol=2,fontsize=8)
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_byperiod_chksize%i_model%s_yr%ito%i_interval%i.png" % (locfn,csize,expnames[i],plots,plotn,plotint),dpi=200)








#
#%% Plot some Spectral Analysis
#
pct     = 0
nsmooth = 5
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95,]
axopt   = 1
clopt   = 1

specparams = []
splotparams  = []


for sst in sstproc:
    
    
    
    
    sps = ybx.yo_spec(sst,opt,nsmooth,pct)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    
    
    
#%%
pct     = 0
nsmooth = 50
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1


sst = sstproc[0]



for i in range(len(sstproc)):
    
    sst = sstproc[i]
    sps = ybx.yo_spec(sst,opt,nsmooth,pct)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    fig,ax,h,hcl,htax,hleg = pps
    
    
    def set_monthlyspec(ax,htax):
        
        # Divisions of time
        # dt  = 3600*24*30
        # fs  = dt*12
        # xtk      = np.array([1/fs/100,1/fs/50, 1/fs/25, 1/fs/10 , 1/fs/5, 1/fs])
        # xtkm    = ["%i" % np.round(i) for i in 1/xtk/dt]
        # xtklabel = ['%.1e \n (century)'%xtk[0],'%.1e \n (50yr)'%xtk[1],'%.1e \n (25yr)'%xtk[2],'%.1e \n (decade)'%xtk[3],'%.1e \n (5year)'%xtk[4],'%.2e \n (year)'%xtk[5]]
        
        # Orders of 10
        dt = 3600*24*30
        fs = dt*3
        xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
        xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
        xtkl     = ["%.1e" % s for s in xtk]
        for i,a in enumerate([ax,htax]):
            
            a.set_xticks(xtk)
            if i == 0:
                
                a.set_xticklabels(xtkl)
            else:
                a.set_xticklabels(xtkm)
        return ax,htax
    ax,htax = set_monthlyspec(ax,htax)
    
    ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
    ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
    ax.set_title("Power Spectrum for %s" % expnames[i])
    plt.tight_layout()
    plt.savefig(outfigpath+"%s_PowerSpectra.png"%expnames[i],dpi=200)
    
    

    
# # Plot raw periodogram
# fig,ax = plt.subplots(1,1)
# for i in range(4):
#     ax.plot(specparams[i][1],specparams[i][0],label=expnames[i])
# ax.legend()


#%% Load CESM Slab Autocorrelation
# cesmslabac     = np.load(datpath+"../CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
# lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
# klon360,_      = proc.find_latlon(lonf+360,latf,lon360,lat)

# slabac = cesmslabac[kmonth,lags,klat,klon360]

#%% Try detrending SST

# #sstin = sst[klon,klat,...] #- sst1.mean(2)[klon,klat,None]
# #sstdt = signal.detrend(sstin,axis=0,type='linear')
# #x,Cxx,sstdt1,yhat,ybnds = tbx.detrend_poly_LSE(sstin,1)

# # Quickly plot output sst
# fig,ax = plt.subplots(1,1,figsize=(8,4))
# ax.plot(sstin,lw=0.5,label='Raw Output',color='b',alpha=0.4)
# #ax.plot(sstdt,lw=0.5,color='r',label="Detrended")
# ax.plot(sstdt1,lw=0.5,color='orange',label="Detrended")
# ax.plot(yhat,lw=0.5,color='k',label="Removed Trend")
# #ax.plot(yhat,lw=0.5,color='k',label="Trend (y=%.2fx + %.2f)"% (x[1],x[0]))
# ax.set_xlabel("Time (Months)")
# ax.set_ylabel("SST (degC)")
# ax.grid(True,ls='dotted')
# ax.legend(fontsize=8,ncol=3)
# ax.set_ylim([-4,4])

# plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_Model_Output_detrend_10x.png" % (locfn),dpi=200)





#%%
lws = .75
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tsenso0[klon360,klat,:],label="SLAB SST",color='k',lw=lws)
ax.plot(sstn[klon,klat,:],label="Flux Integration",color='b',lw=lws)
ax.plot(ssts[klon,klat,:],label="Flux, Shifted Forcing",color='g',lw=lws)
#ax.plot(sstq[klon,klat,:],label="Flux Integration with Q Correction",color='orange',lw=lws,alpha=0.90)
#ax.set_xlim([10276,10776])
ax.set_xlabel("Time (Months)")
ax.set_ylabel("SST (degC)")
ax.grid(True,ls='dotted')
#ax.set_ylim([-2,2])
#ax.set_xlim([0,100])
ax.legend(fontsize=8,ncol=1)
#ax.set_title("SST Integration Comparison \n Location: %s" % (loctitle))
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegrationComp_Point%s_Model_Output_detrend_Quad_all.png" % (locfn),dpi=200)

#%% Do the same as above, but for annual averages


lws = 1
sstqann  = proc.ann_avg(sstq[klon,klat,:],0)
sstnann  = proc.ann_avg(sstn[klon,klat,:],0)
sstsann  = proc.ann_avg(ssts[klon,klat,:],0)
sstslab  = proc.ann_avg(tsenso0[klon360,klat,:],0)

fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(sstslab,label="SLAB SST",color='k',lw=lws)
ax.plot(sstnann,label="Flux Integration ",color='b',lw=lws,alpha=0.90)
ax.plot(sstsann,label="Flux Integration, Shifted Forcing",color='g',alpha=0.9)
#ax.plot(sstqann,label="Flux Integration with Q Correction",color='orange',lw=lws,alpha=0.90)
ax.set_xlabel("Time (Years)")
ax.set_xticks(np.arange(0,1000,50))
ax.set_ylabel("SST (degC)")
#ax.set_xlim([800,900])
ax.grid(True,ls='dotted')
#ax.legend(fontsize=8,ncol=3)
#ax.set_title("SST (Ann Average),\n Location: %s" % (loctitle))
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegrationComp_Point%s_Model_Output_detrend_Ann_quad.png" % (locfn),dpi=200)


#%% Visualize error with year

errorn = np.abs(tsenso0[klon360,klat]-sstn[klon,klat,:])
errorq = np.abs(tsenso0[klon360,klat]-sstq[klon,klat,:])


errorn = errorn.reshape(int(len(errorn)/12),12)
fig,ax = plt.subplots(1,1)
#ax.plot(mons3,errorn.T)
ax.plot(mons3,errorn.std(0))
#ax.set_xlim([0,100])
#ax.plot(errorq)

#%% Plot SST and NHFLX

flxpt = flx[klon,klat,:]
flxann = proc.ann_avg(flxpt,0)

fig,ax = plt.subplots(1,1,figsize=(4,2))
ax.plot(sstslab,color='b',lw=0.5)

ax2 = ax.twinx()

ax2.plot(flxann,color='gray',lw=0.5,alpha=0.5)


#%% Examining forcing ar particular points

of1 = -30
af1 = 50

flxr = flx.reshape(nlon,nlat,int(10776/12),12)

o,a = proc.find_latlon(of1,af1,lon,lat)
flxpt = flxr[o,a,...]
flxstd = flxpt.std(0)

fig,ax=plt.subplots(1,1)

ax.plot(mons3,flxpt.T,color='gray',alpha=0.1)
ax.plot(mons3,flxstd,color='k',ls='dashed')
ax.plot(mons3,-1*flxstd,color='k',ls='dashed')
ax.set_title("NHFLX Anomalies Lon %i, Lat %i"%(of1,af1))
ax.set_ylim([-80,80])





#%% Create Comparitive plot (testing month of forcing)
xtk2 = np.arange(0,37,2)


fig,ax = plt.subplots(1,1)
title = "Autocorrelation, Lag 0 = %s,\n Location: %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,acs[0],label="NHFLX Forcing (same month)")
ax.plot(lags,acs[1],label="NHFLX Forcing (last month)")
ax.plot(lags,slabac,label="CESM-SLAB, Actual Correlation",color='k')
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()

#%% Plot ac and slab comparison



#%% Visualize forcing at this point


flxpt = flx[klon,klat,:]
flxpt = flxpt.reshape(int(tsteps/12),12)

fig,ax = plt.subplots(1,1)

ax.plot(flxpt.T,color='gray',alpha=0.1)
ax.plot(flxpt.std(0),color='k',ls='dashed')


fig,ax= plt.subplots(1,1)
# ax.plot(testa,label="Orifile")
# ax.plot(testb,label="newfile")
ax.plot(testa-testb)
ax.legend()





#%% Try Plotting Autocorrelation of detrended SSts

sstall = [sstq[klon,klat,...],sstn[klon,klat,...],ssts[klon,klat,...]]
kmonth = mldcycle[klon,klat,...].argmax()
print(kmonth)
acs =scm.calc_autocorr(sstall,lags,kmonth+1)
xtk2 = np.arange(0,37,2)
# Plot
fig,ax = plt.subplots(1,1)
title = "Autocorrelation, Lag 0 = %s,\n Location: %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,acs[0],lw=0.85,label="Flux Integration with Q Corretion",color='orange')
ax.plot(lags,acs[1],ls='dashed',lw=0.85,label="Flux Integration",color='b')
ax.plot(lags,acs[2],ls='dashed',lw=0.85,label="Flux Integration, Shifted Forcing",color='g')
ax.plot(lags,slabac,label="CESM-SLAB, Actual Correlation",color='k')
ax.plot(lags,acnoenso,label="CESM-SLAB,ENSO Removed",color='r')
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_detrend_10x.png" % (locfn),dpi=200)

#%% Calculate Autocorrelation but for point without enso

# Get variable
tspoint = tsenso0[klon360,klat,:]
tspoint = tspoint.reshape(int(10776/12),12).T
tspoint = tspoint-tspoint.mean(1)[:,None]

# Calculate Lag correlation
acnoenso = proc.calc_lagcovar(tspoint,tspoint,lags,kmonth+1,0)



