#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_inputs

Script to visualize model inputs

Starts out identical to stochmod region, but visualizes inputs instead of
running the model


Created on Sat Aug 22 19:23:00 2020

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
from dask.distributed import Client,progress
import dask

import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature

import matplotlib.colors as mc
#%% User Edits -----------------------------------------------------------------           
# Point Mode
pointmode = 0 # Set to 1 to output data for the point speficied below
lonf = -30
latf = 50

# ID of the run (determines random number sequence if loading or generating)
runid = "002"

# White Noise Options. Set to 1 to load data
genrand   = 0  # Set to 1 to regenerate white noise time series, with runid above

# Forcing Type
# 0 = completely random in space time
# 1 = spatially unform forcing, temporally varying
# 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# 3 = NAO-like NHFLX Forcing, with NAO (DJFM) and NHFLX (Monthly)
# 4 = NAO-like NHFLX Forcing, with NAO (Monthly) and NHFLX (Monthly)
funiform = 6     # Forcing Mode (see options above)
fscale   = 10    # Value to scale forcing by

# Integration Options
nyr      = 1000        # Number of years to integrate over
t_end    = 12*nyr      # Calculates Integration Period
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
T0       = 0           # Initial temperature [degC]
hfix     = 50          # Fixed MLD value (meters)

# Set Constants
cp0      = 3850 # Specific Heat [J/(kg*C)]
rho      = 1025 # Density of Seawater [kg/m3]

# Set Integration Region
lonW = -100
lonE = 20
latS = -20
latN = 90

#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20200823/'

# Set input path
input_path  = datpath + 'model_input/'

# Set up some strings for labeling
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------
print("Now Running stochmod_region with the following settings: \n")
print("funiform  = " + str(funiform))
print("genrand   = " + str(genrand))
print("runid     = " + runid)
print("pointmode = " + str(pointmode))
print("fscale    = " + str(fscale))
print("nyr       = " + str(nyr))
print("Data will be saved to %s" % datpath)
allstart = time.time()
# --------------
# %% Load Variables -------------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
LON         = np.squeeze(loaddamp['LON1'])
LAT         = np.squeeze(loaddamp['LAT'])
damping     = loaddamp['ensavg']

# Load Mixed layer variables (preprocessed in prep_mld.py)
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# Save Options are here
saveforcing = 0 # Save Forcing for each point

# ------------------
# %% Restrict to region ---------------------------------------------------------
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

# Restrict MLD variables to region
hclim = mld[klon[:,None],klat[None,:],:]
kprev = kprevall[klon[:,None],klat[None,:],:]

# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]
np.save(datpath+"lat.npy",latr)
np.save(datpath+"lon.npy",lonr)

# %% Load and Prep NAO Forcing... <Move to separate script?>



if funiform > 1: # For NAO-like forcings (and EAP forcings, load in data and setup)
    # Load Longitude for processing
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    
    # Load (NAO-NHFLX)_DJFM Forcing
    if funiform == 2:
        
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1 and take ensemble average
        NAO1 = np.mean(naoforcing[0,:,:,:],0) # [Lat x Lon]
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 3:
        
        # Load NAO Forcing and take ensemble average
        naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
        NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
        
        
    elif funiform == 4:
        
        # # Load Forcing and take ensemble average
        # naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
        # NAO1 = np.nanmean(naoforcing,0)
    
          # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
        # Take ensemble average, then sum EOF 1 and EOF2
        NAO1 = naoforcing[:,:,:,:,0].mean(0)
    
    elif funiform == 5: # Apply EAP only
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC2 and take ensemble average
        NAO1 = naoforcing[1,:,:,:].mean(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 6:
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1-2 and take ensemble average + sum
        NAO1 = naoforcing[0:2,:,:,:].mean(1).sum(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 7:
        
        # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
        # Take ensemble average, then sum EOF 1 and EOF2
        NAO1 = naoforcing[:,:,:,:,:2].mean(0).sum(3)
    
    # Transpose to [Lon x Lat x Time]
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude to Degrees East
    lon180,NAO1 = proc.lon360to180(lon360,NAO1)
    
    # Test Plot
    #plt.pcolormesh(NAO1[:,:,0].T)
    
    NAO1 = NAO1[klon[:,None],klat[None,:],:]
    
    # Convert from W/m2 to C/S for the three different mld options
    NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)
    
else:
    NAOF = 1
    
    
# ----------------------------
# %% Set-up damping parameters
# ----------------------------

lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)


#%% Plot Forcing patterns (EAP, NAO, NAO+EAP)


# Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]


# Convert NAO DJFM Forcing (taken from above, funiform=2)
NAO1 = np.mean(naoforcing[0,:,:,:],0) # [Lat x Lon]
NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
NAO1 = np.transpose(NAO1,(2,1,0))
lon180,NAO1 = proc.lon360to180(lon360,NAO1)
NAODJFM = NAO1[klon[:,None],klat[None,:],:].copy()


# Convert EAP DJFM Forcing (taken from above, funiform=5)
NAO1 = naoforcing[1,:,:,:].mean(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
NAO1 = np.transpose(NAO1,(2,1,0))
lon180,NAO1 = proc.lon360to180(lon360,NAO1)
EAPDJFM = NAO1[klon[:,None],klat[None,:],:].copy()


# Conver5 EAP+NAO (funiform=6)
# Select PC1-2 and take ensemble average + sum
NAO1 = naoforcing[0:2,:,:,:].mean(1).sum(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
NAO1 = np.transpose(NAO1,(2,1,0))
lon180,NAO1 = proc.lon360to180(lon360,NAO1)
BOTHDJFM = NAO1[klon[:,None],klat[None,:],:].copy()
#%% Make input plot for Mixed layer Depth

invar = hclim.max(2) - hclim.min(2)
bbox = [-80,0,0,90]
cint = np.concatenate([np.arange(0,100,20),np.arange(100,1200,100)])
cmap = cmocean.cm.dense

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,4))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lonr,latr,invar.T,cint,cmap=cmap)
cl = ax.contour(lonr,latr,invar.T,cint,colors="k",linewidths = 0.5)     
ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='k')
ax.set_title("$MLD_{max} - MLD_{min}$" + "\n 40-member Ensemble Average",fontsize=12)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.05)
plt.savefig(outpath+"MLD_MaxDiff.png",dpi=200)

#%% Mini MLD Map (maximum)
invar = hclim.max(2)
bbox = [-90,5,-5,90]
#cint = np.concatenate([np.arange(0,100,20),np.arange(100,1200,100)])
cint = np.arange(0,1100,100)
cmap = cmocean.cm.dense

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4,3))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lonr,latr,invar.T,cint,cmap=cmap)
cl = ax.contour(lonr,latr,invar.T,cint,colors="k",linewidths = 0.5)     
#ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='w')
#Add Natl Plot
lwb = 1.5
bbox_NA    =[-80,0 ,0,65] 
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")


ax.set_title("$MLD_{max}$",fontsize=12)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.05)
plt.savefig(outpath+"MLD_Max.png",dpi=200)

#%% Make plot for EAP + NAO Forcing



bbox = [-80,0,0,90]
cint = np.arange(-50,55,5)
cmap = cmocean.cm.balance

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,4))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lonr,latr,invar.T,cint,cmap=cmap)
cl = ax.contour(lonr,latr,invar.T,cint,colors="k",linewidths = 0.5)     


ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='k')


ax.set_title("NAO+EAP Forcing Pattern (DJFM) \n (Positive into Ocean)",fontsize=12)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.05)
plt.savefig(outpath+"NAO_EAP_Forcing.png",dpi=200)

#%% Mini veresion of NAO Plots (and others)

fnames  = ["Random","Uniform","NAO (DJFM)","EAP (DJFM)","NAO+EAP (DJFM)"]
fcolors = ["teal","mediumseagreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']


# Plot NAO Countours
bbox = [-90,5,-5,90]
cint = np.arange(-50,55,5)
cmap = cmocean.cm.balance
invar = NAODJFM.squeeze()
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4,3))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lonr,latr,invar.T,cint,cmap=cmap)
cl = ax.contour(lonr,latr,invar.T,cint,colors="k",linewidths = 0.5)   

#Add Natl Plot
lwb = 1.5
bbox_NA    =[-80,0 ,0,65] 
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")

# Labels
ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='k')
ax.set_title("$F'_{NAO}$ (DJFM) \n (Positive into Ocean)",fontsize=10)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.06)
plt.savefig(outpath+"NAO_Forcing_bboxNAT_small.png",dpi=200)

#%% Sample plot but for EAP
# Plot NAO Countours
bbox = [-90,5,-5,90]
cint = np.arange(-50,55,5)
cmap = cmocean.cm.balance
invar = EAPDJFM.squeeze()
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4,3))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lonr,latr,invar.T,cint,cmap=cmap)
cl = ax.contour(lonr,latr,invar.T,cint,colors="k",linewidths = 0.5)   

#Add Natl Plot
lwb = 1.5
bbox_NA    =[-80,0 ,0,65] 
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")

# Labels
ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='k')
ax.set_title("$F'_{EAP}$ (DJFM) \n (Positive into Ocean)",fontsize=10)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.06)
plt.savefig(outpath+"EAP_Forcing_bboxNAT_small.png",dpi=200)

#%% Histogram of NAO Forcings

# Get RBG arrays
rgbfcol = [mc.to_rgba_array(x) for x in fcolors]
# Alter alpha
rgbfcol[:][3] *= 0.5
falpha=0.5

binedges = np.arange(-5,6,0.5)
fig,axs=plt.subplots(3,1,figsize=(3,3),sharey=True,sharex=True)

h1 = axs[0].hist(NAODJFM.ravel(),bins=binedges,edgecolor="k",color=fcolors[1],alpha=falpha)
axs[0].set_title("NAO DJFM, max=%03d" % h1[0].max())
h2 = axs[1].hist(EAPDJFM.ravel(),bins=binedges,edgecolor="k",color=fcolors[3],alpha=falpha)
axs[1].set_title("EAP DJFM max=%03d" % h2[0].max())
h3  = axs[2].hist(BOTHDJFM.ravel(),bins=binedges,edgecolor="k",color=fcolors[4],alpha=falpha)
axs[2].set_title("EAP+NAO DJFM max=%03d" % h3[0].max())
axs[2].set_ylim([0,5000])
axs[2].set_yticks(np.arange(0,7500,2500))
axs[2].set_xlim((-5,5))
axs[2].set_xlabel("Forcing (W/$m^{2})$")
axs[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(outpath+"Forcing_Histograms.png",dpi=200)

#%% Plot Seasonal Damping and MLD parameters, averaged over the NAtl

mons3 = np.arange(1,13,1)
damppt = proc.sel_region(dampingr,lonr,latr,bbox_NA,reg_avg=1)
hpt = proc.sel_region(hclim,lonr,latr,bbox_NA,reg_avg=1)


fig,ax1=plt.subplots(1,1,figsize=(4,3))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('$\lambda_{net} (W m^{-2} K^{-1}$)')
ln1 = ax1.plot(mons3,damppt,color='r',label=r'$\lambda_{net}$')
ax1.tick_params(axis='y',labelcolor=color)
ax1.grid(None)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mixed Layer Depth (m)',color=color)

ln2 = ax2.plot(mons3,hpt,color='b',label=r'HMXL')
ax2.tick_params(axis='y',labelcolor=color)
ax2.grid(None)
ax2.set_xticks(mons3)
# Set Legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc="upper center")

# Set Title
titlestr = "MLD and $\lambda_{a}$ (NAT Average)"
plt.title(titlestr)
plt.tight_layout()
#plt.grid(True)
plt.savefig(outpath+"Damping_MLD_NATAVG.png",dpi=200)

#%% Plot lbd/(rho*cp*h) HSEAS

mons3 = np.arange(1,13,1)
lbdplot = proc.sel_region(lbd[2],lonr,latr,bbox_NA,reg_avg=1)
FACplot = proc.sel_region(FAC[2],lonr,latr,bbox_NA,reg_avg=1)



fig,ax1=plt.subplots(1,1,figsize=(4,3))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel("degC/sec")

ln1 = ax1.plot(mons3,lbdplot,color='k',label=r'$\lambda_{a}$')
ax1.tick_params(axis='y',labelcolor=color)
ax1.set_xticks(mons3)

#plt.title(titlestr)
plt.tight_layout()
#plt.grid(True)
plt.savefig(outpath+"Damping_RhoCPH_MLD_NATAVG.png",dpi=200)
#%% Plot lbd/(rho*cp*h) HMAX

mons3 = np.arange(1,13,1)
lbdplot = proc.sel_region(lbd_entr,lonr,latr,bbox_NA,reg_avg=1)
lbdplot_noentr = proc.sel_region(lbd[2],lonr,latr,bbox_NA,reg_avg=1)


fig,ax1=plt.subplots(1,1,figsize=(4,3))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel("degC/sec")

ln1 = ax1.plot(mons3,lbdplot,color='b',label=r'$\lambda$ (with entrainment)')
ln2 = ax1.plot(mons3,lbdplot_noentr,color='k',label=r'$\lambda$ (no entrainment)')
ax1.tick_params(axis='y',labelcolor=color)
ax1.set_xticks(mons3)
plt.legend()

#plt.title(titlestr)
plt.tight_layout()
#plt.grid(True)
plt.savefig(outpath+"Damping_entr_RhoCPH_MLD_NATAVG.png",dpi=200)



#%% Visualize a point

damping_hist=damping.copy()    
damping_slab=np.load(input_path+"SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
damping_slab= damping_slab[klon[:,None],klat[None,:],:]  



nlon,nlat,nmon = damping_slab.shape
klon,klat = proc.find_latlon(-30,50,lonr,latr)

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
# Plot Heat Flux Feedback
fig,ax = plt.subplots(1,1)
ax.plot(mons3,damping_slab.reshape(nlon*nlat,12).T,color='k',alpha = 0.01,label="")

ax.plot(mons3,damping_slab[klon,klat,:],label="Lon -30, Lat 50",color='r')
ax.plot(mons3,np.nanmean(damping_slab,(0,1)),color='b',label="N Atl. Avg")
ax.legend()
ax.set_ylabel("W/m2/K")
ax.set_title("Heat Flux Feedback")
ax.set_ylim([-10,100])
plt.savefig(outpath+"HeatFluxAll.png",dpi=200)
plt.show()



# Plot MLD
fig,ax = plt.subplots(1,1)
ax.plot(mons3,hclim.reshape(nlon*nlat,12).T,color='k',alpha = 0.01,label="")

ax.plot(mons3,hclim[klon,klat,:],label="Lon -30, Lat 50",color='r')
ax.plot(mons3,np.nanmean(hclim,(0,1)),color='b',label="N Atl. Avg")
ax.legend()
ax.set_ylabel("m")
ax.set_title("Mixed Layer Depth")
#ax.set_ylim([-10,100])
plt.savefig(outpath+"MLDAll.png",dpi=200)
plt.show()



# Tiny Plot
fig,ax = plt.subplots(1,2,figsize=(8,3))
plt.style.use('seaborn')
ax[0].plot(mons3,damping_slab[klon,klat,:],label="30W,50N")
ax[0].plot(mons3,np.nanmean(damping_slab,(0,1)),label="N Atl. Avg")
ax[0].set_title("Heat Flux Feedback (W/m2/K)")
ax[0].legend()

ax[1].plot(mons3,hclim[klon,klat,:],label="30W,50N")
ax[1].plot(mons3,np.nanmean(hclim,(0,1)),label="N Atl. Avg")
ax[1].set_title("Mixed Layer Depth (m)")
ax[1].legend()

plt.show()