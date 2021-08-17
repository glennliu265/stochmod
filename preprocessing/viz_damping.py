#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

viz_damping
Created on Tue Aug 11 00:19:08 2020


Script to visualize damping parameter...
@author: gliu
"""

from scipy.io import loadmat,savemat
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import viz,proc
import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature
#%%


# Indicate Settings
flux    = "NHFLX"  # Flux Name
monwin  = 3        # 3 month smoothing or 1 month only
dof     = 82       # Degrees of Freedom for Significance Testing
p       = 0.20     # p-value
tails   = 2        # two-tailed or one-tailed test...
lags    = [1,2]      # indicate lags to use
mode    = 4      # (1) No mask (2) SST only (3) Flx only (4) Both
savevar = 1     # Option to save calulated variable
# Set Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20200823/"
# Plotting
bbox = [280, 360, 0, 90]
cmap = cmocean.cm.tempo

# Set string for lagmax
lagmax = lags[-1]
save_allens = 0 # Option to save output that has been averaged for lags, but not ens
#%% Load Necessary Data

# Load Lat Lon
mat1 = loadmat(datpath+"CESM1_LATLON.mat")
lon = np.squeeze(mat1["LON"])
lat = np.squeeze(mat1["LAT"])

# Load damping variable [lon x lat x ens x mon x lag]
mat2 = loadmat("%s%s_damping_ensorem1_monwin%i.mat" % (datpath,flux,monwin))
damping = mat2['damping'] # [lon x lat x ens x mon x lag]

# Load correlation coefficients [lon x lat x ens x mon x lag]
mat3 = loadmat("%s%s_rho_ensorem1_monwin%i.mat" % (datpath,flux,monwin))
rflx = mat3['rho']

# Load SST autoorrelation coefficients
mat4 = loadmat("%sSST_rho_ensorem1_monwin%i.mat"% (datpath,monwin))
rsst = mat4['rsst']

#%% Compute DOF stuff

ptilde    = 1-p/tails
critval   = stats.t.ppf(ptilde,dof)
corrthres = np.sqrt(1/ ((dof/np.power(critval,2))+1))

#%% Create Mask

msst = np.zeros(damping.shape)
mflx = np.zeros(damping.shape)
msst[rsst > corrthres] = 1
mflx[rflx > corrthres] = 1

if mode == 1:
    mtot = np.ones(damping.shape)     # Total Frequency of successes
    mall = np.copy(mtot)              # Mask that will be applied
    mult = 1
elif mode == 2:
    mtot = np.copy(msst)
    mall = np.copy(msst)
    mult = 1
elif mode == 3:
    mtot = np.copy(mflx)
    mall = np.copy(mflx)  
    mult = 1
elif mode == 4:
    mtot = msst + mflx
    mall = msst * mflx
    mult = 2

#%% Apply the mask

# Select # of lags and sum over all dimensions

if len(lags) > 1:
    lagidx = np.array(lags)-1
else:
    lagidx = [0]

# Apply Mask
dampmasked = damping * mall

# Select Lags
dampchoose = dampmasked[:,:,:,:,lagidx]

# Save allens data if set
if save_allens == 1:
    dampavged = np.mean(dampchoose,4)
    dampavged = np.transpose(dampavged,(1,0,3,2))
    da = xr.DataArray(dampavged,
                       coords={ 'lat':lat, 'lon':lon,  'month':np.arange(1,13,1),'ensemble':np.arange(1,43,1) },
                       dims  ={ 'lat':lat, 'lon':lon,  'month':np.arange(1,13,1),'ensemble':np.arange(1,43,1) },
                       name="NHFLX_Damping"
                       )
    da.to_netcdf(datpath+"allens_nhflxdamping_monwin3_sig020_dof082_mode4_lag1.nc")
    


# Take ensemble and lag average
if len(lags) > 1:
    dampseason  = np.mean(dampchoose,(2,4))
    dampseason1 = np.mean(dampchoose,(2,4))
else:
    dampseason = np.mean(dampchoose,(2))
    
    
# Save data if option is set
if savevar == 1:
    LON1,ensavg=proc.lon360to180(lon,dampseason)
    # Save variables, matching matlab script output
    savedict = {'ensavg':ensavg, 'LON1': LON1, 'LAT':lat }
    outname = "%sensavg_%sdamping_monwin%i_sig%03d_dof%03d_mode%i.mat" % (datpath,flux.lower(),monwin,p*100,dof,mode)
    savemat(outname,savedict)
    print("Saved data to %s" % outname)
#%% Make some frequency plots

mchoose = mtot[:,:,:,:,lagidx]

if len(lags) > 1:
    mfreq  = np.nansum(mchoose,(2,3,4))
else:
    mfreq  = np.nansum(mchoose,(2,3))


# Calculate maximum possible successes x2 (for mflx and msst)
maxscore = np.prod(mchoose.shape[2:]) * mult
freqfail = 1-mfreq/maxscore


# Make Plot
cint = np.arange(0,1.1,.1)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,4))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,mfreq.T/maxscore,cint,cmap=cmap)
cl = ax.contour(lon,lat,mfreq.T/maxscore,cint,colors="k",linewidths = 0.5)
ax.add_feature(cfeature.LAND,color='k')
#pcm = ax.contourf(lon,lat,freqfail.T,cint,cmap=cmap)
#cl = ax.contour(lon,lat,freqfail.T,cint,colors="k",linewidths = 0.5)
plt.clabel(cl,np.arange(0,1.2,0.2),fmt="%.1f")
ax.set_title("Total Number of Significant Damping Values\n"+ r"MaxFreq = %i | p = %.2f | $\rho$ > %.2f " % (maxscore,p,corrthres))
plt.colorbar(pcm,ax=ax)
plt.savefig(outpath+"%s_SigPts_mode%i_monwin%i_lags%i_sig%03d.png"%(flux,mode,monwin,lagmax,p*100),dpi=200)


#%% Make Plot of success and damping values

fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(6,4))

bbox = [-60,10,50,75]

ax = axs[0]
cint = np.arange(0,1.1,.1)
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,mfreq.T/maxscore,cint,cmap=cmap)
cl = ax.contour(lon,lat,mfreq.T/maxscore,cint,colors="k",linewidths = 0.5)
ax.clabel(cl,np.arange(0,1.2,0.2),fmt="%.1f")
ax.set_title("% of Sig. Values\n"+ r"Mode = %i, Max = %i " % (mode,maxscore))
plt.colorbar(pcm,ax=ax,orientation="horizontal")
#plt.savefig(outpath+"%s_SigPts_monwin%i_lags12_sig%03d.png"%(flux,monwin,p*100),dpi=200)



ax = axs[1]
cint = np.arange(-50,55,5)
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,np.nanmean(dampseason,2).T,cint,cmap=cmocean.cm.balance)
cl = ax.contour(lon,lat,np.nanmean(dampseason,2).T,cint,colors="k",linewidths = 0.5)
ax.clabel(cl,fmt="%i")
ax.set_title("%s Damping (Ann, Lag, Ens Avg)\n" % flux+ r"| p = %.2f | $\rho$ > %.2f " % (p,corrthres))
plt.colorbar(pcm,ax=ax,orientation="horizontal")
plt.savefig(outpath+"%s_Damping_and_SigPts_mode%i_monwin%i_lags12_sig%03d.png"%(flux,mode,monwin,p*100),dpi=200)

#%% Just plot the damping values
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,4))
cint = np.arange(-50,55,5)
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,np.mean(dampseason,2).T,cint,cmap=cmocean.cm.balance)
cl = ax.contour(lon,lat,np.mean(dampseason,2).T,cint,colors="k",linewidths = 0.5)
ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='k')
ax.set_title("$\lambda_{a,%s}$ (Ann, Lag, Ens Avg)\n" % flux+ r"p = %.2f | $\rho$ > %.2f " % (p,corrthres),fontsize=12)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040, pad=0.05)
plt.savefig(outpath+"%s_Damping__mode%i_monwin%i_lags%i_sig%03d.png"%(flux,mode,monwin,lagmax,p*100),dpi=200)


#%% Plot Damping Seasonal Cycle at a Point

lonf = -30
latf = 65
mons = np.arange(1,13,1)
locstring = "Lon%03d_Lat%03d" % (lonf,latf)

# Find point and get data
klon,klat = proc.find_latlon(lonf+360,latf,lon,lat)
damppt = dampchoose[klon,klat,:,:] # [ ens x mon ]
if len(damppt.shape) > 2:
    damppt = np.nanmean(damppt,2)

# Plot seasonal cycle
fig,ax = plt.subplots(1,1,figsize=(6,4))
for e in range(42):
    ax.plot(mons,damppt[e,:],color=[.75,.75,.75])

ln1 = ax.plot(mons,damppt[-1,:],color=[.75,.75,.75], label="Indv. Member")
ln2 = ax.plot(mons,np.nanmean(damppt,0),color='k',label="Ens. Avg.")

#ax.set_ylim([-30,90])
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0,ncol=2)
ax.set_title("Seasonal %s Damping at (%2d LON,%2d LAT) \n Mode %i | p = %.2f | monwin %i | lags %i" % (flux,lonf,latf,mode,p,monwin,lagmax))

plt.savefig(outpath+"%s_Damping_SeasonalCycle_mode%i_monwin%i_lags%i_sig%03d_%s.png"%(flux,mode,monwin,lagmax,p*100,locstring),dpi=200)

#%% Do the same, but separate out for each lag

lonf = -30
latf = 65
mons = np.arange(1,13,1)
locstring = "Lon%03d_Lat%03d" % (lonf,latf)

# Find point and get data
klon,klat = proc.find_latlon(lonf+360,latf,lon,lat)
damppt = dampchoose[klon,klat,:,:] # [ ens x mon ]

klon1,klat1 = proc.find_latlon(lonf,latf,LON1,lat)



lagcolmem = [np.array([255,134,134])/255,np.array([189,202,255])/255]
lagcolavg = ['r','b']
fig,ax = plt.subplots(1,1,figsize=(6,4))
# Plot seasonal cycle
for lag in range(2):
    
    for e in range(42):
        ax.plot(mons,damppt[e,:,lag],color=lagcolmem[lag],alpha=0.50)
    
    ax.plot(mons,damppt[-1,:,lag],color=lagcolmem[lag], label="Lag%i Member"%lag,alpha=0.30)
    #ax.plot(mons,np.nanmean(damppt[:,:,lag],0),color=lagcolavg[lag],label="Lag%i Ens. Avg."%lag)

ln1 = ax.plot(mons,np.nanmean(damppt[:,:,0],0),color=lagcolavg[0],label="Lag 1 Ens. Avg.")
ln2 = ax.plot(mons,np.nanmean(damppt[:,:,1],0),color=lagcolavg[1],label="Lag 2 Ens. Avg.")
ln3 = ax.plot(mons,np.nanmean(damppt,(0,2)) ,color='k'         ,label="Lag & Ens. Avg.")


lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0,ncol=2)
ax.set_title("Seasonal %s Damping at (%2d LON,%2d LAT) \n Mode %i | p = %.2f | monwin %i | lags %i" % (flux,lonf,latf,mode,p,monwin,lagmax))
#ax.set_ylim([-30,90])
plt.savefig(outpath+"%s_Damping_SeasonalCycle_LagSep_mode%i_monwin%i_lags%i_sig%03d_%s.png"%(flux,mode,monwin,lagmax,p*100,locstring),dpi=200)

#%% Plot for what fails at lag 1

mchoose = mtot[:,:,:,:,lagidx]

if len(lags) > 1:
    mfreq  = np.nansum(mchoose,(2,3,4))
else:
    mfreq  = np.nansum(mchoose,(2,3))


# Calculate maximum possible successes x2 (for mflx and msst)
maxscore = np.prod(mchoose.shape[2:]) * mult
freqfail = 1-mfreq/maxscore


# Make Plot
cint = np.arange(0,1.1,.1)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(6,4))
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,freqfail.T,cint,cmap=cmap)
cl = ax.contour(lon,lat,freqfail.T,cint,colors="k",linewidths = 0.5)
ax.set_title("Frequency of Failed Significance Tests\n"+ r"Max = %i | p = %.2f | $\rho$ > %.2f " % (maxscore,p,corrthres))
plt.colorbar(pcm,ax=ax)
plt.savefig(outpath+"%s_FailSigPts_mode%i_monwin%i_lag1_sig%03d.png"%(flux,mode,monwin,p*100),dpi=200)


#%% Find regions that are included in nanmean (nm) but not regular mean (mn)
nm = np.copy(dampseason)
mn = np.copy(dampseason1)

nm[np.isnan(nm)] = 0
mn[np.isnan(mn)] = 0
diff = nm - mn

maskout = np.copy(diff)
maskout[np.where(diff>0)] = 1

plt.pcolormesh(lon,lat,maskout[:,:,5].T,cmap=cmocean.cm.balance)

#%% Updated Damping Value Plot (Comparison with SLAB HF-Feedback)

lagstr = "%i-%i" % (lags[0],lags[-1])

bbox = [275-360, 360-360, 0, 65]
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,4))
cint = np.arange(-50,55,5)
ax = viz.init_map(bbox,ax=ax)
pcm = ax.contourf(lon,lat,np.mean(dampseason,2).T,cint,cmap=cmocean.cm.balance)
cl = ax.contour(lon,lat,np.mean(dampseason,2).T,cint,colors="k",linewidths = 0.5)
ax.clabel(cl,fmt="%i",fontsize=8)
ax.add_feature(cfeature.LAND,color='gray')
ax.set_title(r"CESM-Historical Annual Mean $\lambda_{a,%s}$ (Lags %s & Ens. Avg)" % (flux,lagstr)+ "\n"+r"DOF= %i | p = %.2f | R > %.2f " % (dof,p,corrthres),fontsize=12)

#ax.set_title("$\lambda_{a,%s}$ (Ann, Lag, Ens Avg)\n" % flux+ r"p = %.2f | $\rho$ > %.2f " % (p,corrthres),fontsize=12)
plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05, pad=0.05)
plt.savefig(outpath+"%s_Damping__mode%i_monwin%i_lags%i_sig%03d.png"%(flux,mode,monwin,lagmax,p*100),dpi=200)