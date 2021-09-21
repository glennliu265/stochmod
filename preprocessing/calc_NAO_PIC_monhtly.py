#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the output of calc_NAO_PIC (DJFM NAO)
to compute monthly NAO patterns 

/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_PIC_SLAB/NAO/EOF_NAO_DJFM_PIC_SLAB.npz

Created on Fri Mar 19 03:56:55 2021

@author: gliu
"""


import numpy as np
from scipy.io import loadmat
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


#%%

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210322_AMVTeleconf/"

# Mode
mconfig = 'SLAB' # "SLAB or FULL"

# PCs to calculate
N_mode = 3# User edited variable!

# Subset data for enso index calculation
bbox = [-90+360, 40, 20, 80]

# Mode:
mode = 'DJFM' # Mode is 'DJFM',or 'Monthly'
debug = False # Set to true to make figure at end


outname = "EOF_NAO_%s_PIC_SLAB.npz" % (mode)


#%%
applymask=True

# Load IN calculated nao index, pattern
#npname = "EOF_NAO_DJFM-MON_PIC_SLAB.npz"
npname = "EOF_NAO_DJFM_PIC_SLAB.npz"
ld = np.load(datpath+npname,allow_pickle=True)
pcs    = ld['pcs']
eofs   = ld['eofs']
varexp = ld['varexp']
flxpat = ld['nhflx_pattern'].reshape(192,288,3)
pslpat = ld['psl_pattern'].reshape(192,288,3)
#lon    = ld['lon']
#lat    = ld['lat']
times  = ld['times']


from scipy.io import loadmat # Load in Lat, Lon, Mask
ld = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")
lon = ld['LON'].squeeze()
lat = ld['LAT'].squeeze()
# Load Mask
msk = np.load(datpath+"landicemask_enssum.npy")



# Restrict PC (drop first 2 years due to ensolag + 3monwin, and end year due to 3monwin)
pcin  = pcs[2:-1,:]


nhflx = np.load(datpath+"ENSOREM_NHFLX_lag1_pcs2_monwin3.npy")
nmon,nlat,nlon = nhflx.shape
nyr = int(nmon/12)

if applymask:
    flxpat *= msk[:,:,None]
    nhflx *= msk[None,:,:]
    
    
# Reshape to combine spatial dimenions
flxpatr = flxpat.reshape(192*288,3)
nhflxr = nhflx.reshape(nyr,12,nlat*nlon)


if applymask:
    flxpatori = flxpatr.copy()
    nhflxrori = nhflxr.copy()
    flxpatr,knan,okpts = proc.find_nan(flxpatr,1)
    nhflxr = nhflxr[:,:,okpts]
    
#%% Step 1) Regress DJFM NHFLX Pattern to monthly nhflx

# Note: Should I use PSL Instead?
# To determine how much each month varies with NAO pattern

djfm_beta = np.zeros((nyr,12,3)) # 
for p in range(3):
    naopat = flxpatr[:,p]
    for m in tqdm(range(12)):
        
        nhflxmon = nhflxr[:,m,:] # [N x M]
        
        # Note, I'm not sure what order?
        beta,b=proc.regress_2d(naopat,nhflxmon)
        #beta1,b1 = proc.regress_2d(nhflxmon,naopat)
        
        djfm_beta[:,m,p] = beta


#%% Save forcing with fixed spatial pattern
#
#
import cmocean

lon1,flxpat1 = proc.lon360to180(lon,flxpat.transpose(1,0,2),autoreshape=True)
ftime = djfm_beta.std(0) # [month x pc]


# [lon x lat x pc x mon]
forcing = flxpat1[:,:,:,None] * ftime.T[None,None,:,:] * varexp[None,None,:,None]

# Test Plot
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
pcnames = ["NAO","EAP","EOF3"]
pcnamesf = ["NAO-DJFM","EAP-DJFM","EOF3-DJFM"]
bbox  = [260-360, 360-360, 0, 80]
cmbal = cmocean.cm.balance
pcn = 0
#cint = np.arange(-50,52,2)
cint = np.hstack([np.arange(-50,-10,10),np.arange(-10,10,2),np.arange(10,60,10)])
pcint = cint
fig,axs = plt.subplots(3,4,figsize=(16,10),subplot_kw={'projection':ccrs.PlateCarree()})
for i in tqdm(range(12)):
    
    # Initialize Axis
    ax  = axs.flatten()[i]
    ax  = viz.init_map(bbox,ax=ax)
    
    # Plot variables
    cf1 = ax.pcolormesh(lon1,lat,forcing[:,:,pcn,i].T,vmin=pcint[0],vmax=pcint[-1],cmap=cmbal)
    ax  = viz.plot_contoursign(forcing[:,:,pcn,i].T,lon1,lat,cint,ax=ax,clab=True,lw=0.5)
    
    # Set Title
    ax.set_title("%s"%(mons3[i]))
    
plt.suptitle("%s(DJFM)-NHFLX Regression Patterns, CESM SLAB" % (pcnames[pcn]),fontsize=18,y=0.92)
fig.colorbar(cf1,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.95,pad=0.015)
plt.savefig("%s%s_Pattern_Regression_Fix.png"% (outpath,pcnamesf[pcn]),dpi=150)

#%%

# Save output
outname = "%s%s_PIC_NAO_EAP_NHFLX_Forcing_DJFM-MON_Fix.npy" % (datpath,mconfig)
np.save(outname,forcing)
print("Saved to %s"%outname)
#np.save("%sForcing_Beta_%s_PIC"%(datpath,mconfig))



#%% Step 2) Regress Back to NHFLX to obtain the characterstic pattern...

npts = nlat*nlon
if applymask:
    npts = okpts.sum()

naopatmon = np.zeros((npts,12,3))

for p in range(3):
    
    for m in tqdm(range(12)):
        
        ts = djfm_beta[:,m,p]
        nhflxmon = nhflxr[:,m,:]
        beta,b=proc.regress_2d(ts,nhflxmon)
        
        naopatmon[:,m,p] = beta


if applymask:
    naopatmon1 = np.zeros((nlat*nlon,12,3))*np.nan
    naopatmon1[okpts,:,:] = naopatmon
    naopatmon=naopatmon1
    
naopatmon = naopatmon.reshape(nlat,nlon,12,3)


#%% Post Processing

# Flip longitude
naopatmon1 = naopatmon.reshape(nlat,nlon,12*3).transpose(1,0,2)
lon1,naopatmon1=proc.lon360to180(lon,naopatmon1,autoreshape=True)

naopatmon1 = naopatmon1.reshape(nlon,nlat,12,3)

plt.pcolormesh(lon1,lat,naopatmon1[:,:,0,0].T)

#%% Step 3) Plot some of the results
import cmocean
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
pcnames = ["NAO","EAP","EOF3"]
bbox  = [260-360, 360-360, 0, 80]
cmbal = cmocean.cm.balance
pcn = 2
#cint = np.arange(-50,52,2)
cint = np.hstack([np.arange(-50,-10,10),np.arange(-10,10,2),np.arange(10,60,10)])
pcint = cint
fig,axs = plt.subplots(3,4,figsize=(16,10),subplot_kw={'projection':ccrs.PlateCarree()})
for i in tqdm(range(12)):
    
    # Initialize Axis
    ax  = axs.flatten()[i]
    ax  = viz.init_map(bbox,ax=ax)
    
    # Plot variables
    cf1 = ax.pcolormesh(lon1,lat,naopatmon1[:,:,i,pcn].T,vmin=pcint[0],vmax=pcint[-1],cmap=cmbal)
    ax  = viz.plot_contoursign(naopatmon1[:,:,i,pcn].T,lon1,lat,cint,ax=ax,clab=True,lw=0.5)
    
    # Set Title
    ax.set_title("%s"%(mons3[i]))
    
plt.suptitle("%s(DJFM)-NHFLX Regression Patterns, CESM SLAB" % (pcnames[pcn]),fontsize=18,y=0.92)
fig.colorbar(cf1,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.95,pad=0.015)
plt.savefig(outpath+"SLAB_%sDJFM-MON_NHFLX_monregr.png" % (pcnames[pcn]),dpi=200) 


#%% Examine timeseries at a point



klon,klat = proc.find_latlon(-30,50,lon1,lat)
fig,ax = plt.subplots(1,1)
ax.plot(mons3,naopatmon1[klon,klat,:,0],label="NAO")
ax.plot(mons3,naopatmon1[klon,klat,:,1],label="EAP")
ax.plot(mons3,naopatmon1[klon,klat,:,2],label="EOF3")
ax.legend()
ax.set_ylabel("NHFLX (W/m2)")
ax.set_xlabel("Month")
ax.set_title("NHFLX Forcing at Lon %.2f Lat %.2f" % (-30,50))
ax.grid(True,ls='dotted')
plt.savefig(outpath+"SLAB_%sDJFM-MON_NHFLX_monregr_timeseries.png" % (pcnames[pcn]),dpi=200) 


#%% Plot Magnitude of the pattern

i = 2

pcnamesf = ["NAO-DJFM","EAP-DJFM","EOF3-DJFM"]

fig,ax = plt.subplots(1,1)
ax.plot(mons3,djfm_beta[:,:,i].T,color='gray',alpha=0.10,label="")
ax.plot(mons3,djfm_beta[0,:,i].T,color='gray',alpha=0.10,label="Indv. Year")
ax.plot(mons3,djfm_beta[:,:,i].mean(0),color='k',label="Mean")
ax.plot(mons3,djfm_beta[:,:,i].std(0),color='k',ls='dashdot',label=r"$1\sigma$")
ax.legend(fontsize=10)
ax.set_xlim([0,11])
ax.set_ylim([-10,10])
ax.grid(True,ls='dotted')
ax.set_title(r"Regression of %s Pattern onto $Q_{net}'$ in CESM-SLAB"%pcnames[i])
ax.set_ylabel("Regression Coefficient")
plt.savefig("%s%s_Pattern_Regression.png"% (outpath,pcnamesf[i]),dpi=150)


#%% from finalize_inputs.py
cint = np.arange(-50,55,5)
cmbal = cmocean.cm.balance
fig,axs = plt.subplots(2,1,subplot_kw={"projection":ccrs.PlateCarree()})

ax = axs[0]
ax = viz.init_map(bbox,ax=ax)
cf1 = ax.contourf(lon1,lat,flxpat1[:,:,0].T,levels=cint,cmap=cmbal)
ax  = viz.plot_contoursign(flxpat1[:,:,0].T,lon1,lat,cint,ax=ax,clab=True,lw=0.5)

fig.colorbar(cf1,ax=ax)
ax.set_title("$NAO_{DJFM}$ (EOF 1, Variance Explained = %.2f)"% (varexp[0]*100)+r"%")

# ax = axs[1]
# ax = viz.init_map(bbox,ax=ax)
# cf2 = ax.contourf(lon180,lat,naof_slab[:,:,1].T,levels=cint,cmap=cmbal)
# ax  = viz.plot_contoursign(naof_slab[:,:,1].T,lon180,lat,cint,ax=ax,clab=True,lw=0.5)
# fig.colorbar(cf2,ax=ax)
# ax.set_title("$EAP_{DJFM}$ (EOF 2, Variance Explained = %.2f)"% (varexp_slab[1]*100)+r"%")

#plt.suptitle("CESM1-Slab (Preindustrial Control 101-1001)",x=0.6,y=0.95)
plt.savefig(outpath+"CESM_SLAB_DJFM_Forcing.png",dpi=200,bbox_inches='tight')