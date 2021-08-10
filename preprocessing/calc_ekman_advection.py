#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 05:02:36 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from tqdm import tqdm

#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210726/"
   
    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx


#%% Functions

def calc_dx_dy(longitude,latitude):
    ''' This definition calculates the distance between grid points that are in
        a latitude/longitude format.
        
        Equations from:
        http://andrew.hedges.name/experiments/haversine/

        dy should be close to 55600 m
        dx at pole should be 0 m
        dx at equator should be close to 55600 m
        
        Accepts, 1D arrays for latitude and longitude
        
        Returns: dx, dy; 2D arrays of distances between grid points 
                                    in the x and y direction in meters 
    '''
    dlat = np.abs(lat[1]-lat[0])*np.pi/180
    dy   = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy   = np.ones((latitude.shape[0],longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    for i in range(latitude.shape[0]):
        a = (np.cos(latitude[i]*np.pi/180)*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
        dx[i] = c * 6371000
    dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
    return dx, dy


#%% Set constants

omega = 7.2921e-5 # rad/sec
rho   = 1026      # kg/m3
cp0   = 3996      # [J/(kg*C)]
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

#%% Load some necessary data

# Load lat/lon
lon360 = np.load(rawpath+"CESM1_lon360.npy")
lon180 = np.load(rawpath+"CESM1_lon180.npy")
lat    = np.load(rawpath+"CESM1_lat.npy") 

# Load Land/ice mask
msk = np.load(lipath)
#ds *= msk[None,:,:]

# Get distances relating to the grid
dx,dy = calc_dx_dy(lon360,lat)
xx,yy = np.meshgrid(lon360,lat)
# Set southern hemisphere distances to zero (?) since y = 0 at the equator..
#dy[yy<0] *= -1

#%% Load mixed layer depths
st = time.time()
dsmld = xr.open_dataset(rawpath+"HMXL_PIC.nc")
hmxl = dsmld.HMXL.values # [lon180 x lat x time]
print("Load MLD in %.2fs"%(time.time()-st))


# Find the climatological mean
mld     = hmxl.reshape(288,192,int(hmxl.shape[2]/12),12)
mldclim = mld.mean(2)

# Convert cm --> meters
mldclim /= 100 


#%% Load the data (temperature)

st = time.time()
ds1 = xr.open_dataset(rawpath+"../CESM_proc/TS_PIC_FULL.nc")
ts = ds1.TS.values
#ts -= 273.15 #(convert to celsius)
print("Completed in %.2fs"%(time.time()-st))

# Calculate the mean gradient for each month
ts_monmean = ts.mean(0)

# Roll longitude along axis for forward difference (gradT_x0 = T_x1 - xT0)
dTdx = (np.roll(ts_monmean,-1,axis=2)- ts_monmean) / dx[None,:,:]
dTdy = (np.roll(ts_monmean,-1,axis=1)- ts_monmean) / dy[None,:,:]
dTdy[:,-1,:] = 0 # Set top latitude to zero (since latitude is not periodic)

#dTdy = (ts_monmean[:,1:,:] - ts_monmean[:,:-1,:])  / dy[None,:-1,:] # Drop the last grid?
#dTdx = (ts_monmean[:,:,1:] - ts_monmean[:,:,0:-1]) / dx[None,:,:]
#dTdy = ts_monmean / dy[None,:,:]

# Save output...
savename = "%sFULL-PIC_Monthly_gradT_lon360.npz" % (rawpath)
# Save [mon x lat x lon360]

np.savez(savename,**{
    'dTdx':dTdx,
    'dTdy':dTdy,
    'dx':dx,
    'dy':dy,
    'lon':lon360,
    'lat':lat})


#%% Load temperature gradient and plot

# Load output
savename = "%sFULL-PIC_Monthly_gradT_lon360.npz" % (rawpath)
ld       = np.load(savename)
dTdx     = ld['dTdx']
dTdy     = ld['dTdy']
dx       = ld['dx']
dy       = ld['dy']
lon360   = ld['lon']
lat      = ld['lat']

# Set the month
im = 0


# Plot zonal temperature difference
fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=[-180,180,-90,90])
pcm = ax.pcolormesh(lon360,lat,(np.roll(ts_monmean,-1,axis=2)- ts_monmean)[im,:,:],vmin=-0.5,vmax=0.5,cmap="RdBu_r")
fig.colorbar(pcm,ax=ax)
ax.set_title(r"Zonal differece ($T_1 - T0$)" + " for %s (degC/meter)" % (mons3[im]) )


# Plot zonal temperature gradient
fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=[-180,180,-90,90])
pcm = ax.pcolormesh(lon360,lat,(dTdx)[im,:,:],vmin=-0.5e-5,vmax=0.5e-5,cmap="RdBu_r")
fig.colorbar(pcm,ax=ax)
ax.set_title(r"Zonal gradient ($\frac{\partial T}{\partial x}$)" + " for %s (degC/meter)" % (mons3[im]) )

# Plot meridional temperature gradient
fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=[-180,180,-90,90])
pcm = ax.pcolormesh(lon360,lat,(dTdy)[im,:,:],vmin=-0.5e-4,vmax=0.5e-4,cmap="RdBu_r")
fig.colorbar(pcm,ax=ax)
ax.set_title(r"Meridional gradient ($\frac{\partial T}{\partial x}$)" + " for %s (degC/meter)" % (mons3[im]) )

#%% Load the wind stress and the PCs to prepare for regression
N_mode = 100


# Load the PCs
ld = np.load("%sNHFLX_FULL-PIC_100EOFsPCs_lon260to20_lat0to65.npz" % (rawpath),allow_pickle=True)
pcall = ld['pcall'] # [PC x MON x TIME]
eofall = ld['eofall']
eofslp = ld['eofslp']

# Load each wind stress component [yr mon lat lon]
st = time.time()
dsx = xr.open_dataset(rawpath+"../CESM_proc/TAUX_PIC_FULL.nc")
taux = dsx.TAUX.values
dsx = xr.open_dataset(rawpath+"../CESM_proc/TAUY_PIC_FULL.nc")
tauy = dsx.TAUY.values
print("Loaded wind stress data in %.2fs"%(time.time()-st))


# Convert stress from stress on OCN on ATM --> ATM on OCN
taux*= -1
tauy*= -1

#%% Flip EOFs

spgbox = [-60,20,40,80]
N_modeplot = 5

for N in tqdm(range(N_modeplot)):
    for m in range(12):
        
        sumflx = proc.sel_region(eofall[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
        #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
        
        if sumflx > 0:
            
            print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
            
            print("Check before, pc is %f" % (pcall[N,m,22]))
            
            eofall[:,:,m,N] *=-1
            pcall[N,m,:]    *= -1
            eofslp[:,:,m,N] *=-1

            print("Check after, pc is %f" % (pcall[N,m,22]))


#%% Preprocess Wind Stress Variables
takeanom = False 

fullx = taux.copy()
fully = tauy.copy()
    
nyr,_,nlat,nlon = taux.shape
taux = taux.reshape(nyr,12,nlat*nlon) # Combine space
tauy = tauy.reshape(nyr,12,nlat*nlon)

if takeanom:
    taux = taux - taux.mean(1)[:,None,:]
    tauy = tauy - tauy.mean(1)[:,None,:]



#%% Regress wind stress components
taux_pat = np.zeros((nlat*nlon,12,N_mode))
tauy_pat = np.zeros((nlat*nlon,12,N_mode))
for m in tqdm(range(12)):
    
    
    tx_in = taux[:,m,:]
    ty_in = tauy[:,m,:]
    pc_in = pcall[:,m,:]
    pcstd = pc_in / pc_in.std(1)[:,None] # Standardize in time dimension
    
    eof_x,_ = proc.regress_2d(pcstd,tx_in)
    eof_y,_ = proc.regress_2d(pcstd,ty_in)
    
    taux_pat[:,m,:] = eof_x.T
    tauy_pat[:,m,:] = eof_y.T
    
# Reshape, postprocess
procvars = [taux_pat,tauy_pat]
fin    = []
for invar in procvars:
    
    # Reshape things for more processing
    invar = invar.reshape(nlat,nlon,12*N_mode) # Make 3D
    invar = invar.transpose(1,0,2) # [Lon x lat x otherdims]
    
    # Flip to degreeseast/west
    _,outvar = proc.lon360to180(lon360,invar)

    # Reshape variable
    outvar = outvar.reshape(nlon,nlat,12,N_mode)
    fin.append(outvar)


# Unflipped variable
taux_pat = taux_pat.reshape(nlat,nlon,12,N_mode)
tauy_pat = tauy_pat.reshape(nlat,nlon,12,N_mode)

taux_pat_fin,tauy_pat_fin = fin


#%% Flip EOFs (again)

spgbox = [-60,20,40,80]
N_modeplot = 5

for N in tqdm(range(N_modeplot)):
    for m in range(12):
        
        sumflx = proc.sel_region(eofall[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
        #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
        
        if sumflx > 0:
            
            print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
            
            eofall[:,:,m,N] *=-1
            pcall[N,m,:]    *= -1
            eofslp[:,:,m,N] *=-1
            taux_pat_fin[:,:,m,N] *=-1
            tauy_pat_fin[:,:,m,N] *=-1

#%% Save the files

# Save output...
savename = "%sFULL-PIC_Monthly_NHFLXEOF_TAUX_TAUY.npz" % (rawpath)
np.savez(savename,**{
    'taux':taux_pat_fin,
    'tauy':tauy_pat_fin,
    'lon':lon180,
    'lat':lat})

#%% Individual monthly wind stress plots for EOF N

im = 9
N  = 0
scaler =    .75 # # of data units per arrow
bboxplot = [-100,20,0,80]
labeltau = 0.10
slplevs = np.arange(-400,500,100)
flxlim = [-30,30]
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

oint = 5
aint = 5

for im in tqdm(range(12)):
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
    pcm = ax.pcolormesh(lon180,lat,eofall[:,:,im,N].T,vmin=flxlim[0],vmax=flxlim[-1],cmap="RdBu_r")
    cl  = ax.contour(lon180,lat,eofslp[:,:,im,N].T,levels=slplevs,colors='k',linewidth=0.75)
    
    qv = ax.quiver(lon180[::oint],lat[::aint],
                   taux_pat_fin[::oint,::aint,im,N].T,
                   tauy_pat_fin[::oint,::aint,im,N].T,
                   scale=scaler,color='gray',width=.008,
                   headlength=5,headwidth=2,zorder=9)
    ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
    
    fig.colorbar(pcm,ax=ax,fraction=0.035)
    ax.set_title("%s Wind Stress Associated with NHFLX EOF %i \n CESM-FULL"%(mons3[im],N+1))
    savename = "%sCESM_FULL-PIC_WindStressMap_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
    plt.savefig(savename,dpi=150,bbox_tight='inches')

#%% Seasonal NHFLX-SLP-Windstress plots

season_idx  = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
season_name = ["DJF","MAM","JJA","SON"]
scaler = 0.5
fig,axs = plt.subplots(1,4,figsize=(16,3),subplot_kw={'projection':ccrs.PlateCarree()})
for i in range(4):
    
    sid   = season_idx[i]
    sname = season_name[i]
    ax    = axs.flatten()[i]
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.pcolormesh(lon180,lat,eofall[:,:,sid,N].mean(2).T,vmin=flxlim[0],vmax=flxlim[-1],cmap="RdBu_r")
    cl  = ax.contour(lon180,lat,eofslp[:,:,sid,N].mean(2).T,levels=slplevs,colors='k',linewidth=0.95)
    
    qv = ax.quiver(lon180[::oint],lat[::aint],
               taux_pat_fin[::oint,::aint,sid,N].mean(2).T,
               tauy_pat_fin[::oint,::aint,sid,N].mean(2).T,
               scale=scaler,color='gray',width=.008,
               headlength=5,headwidth=2,zorder=9)
    #ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
    
    ax.set_title(sname)

fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.35,pad=0.01)
plt.suptitle("Seasonal Averages for EOF %i of $Q_{net}$ (colors), $SLP$ (contours), and Wind Stress (quivers)" % (N+1),y=0.74)
savename = "%sCESM_FULL-PIC_WindStressMap_EOF%02i_seasonal.png" %(figpath,N+1)   
plt.savefig(savename,dpi=150,bbox_tight='inches')




#%% Test visualzation of the wind stress variable

fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=[-180,180,-90,90])

oint     = 7
aint     = 7
t        = 555
labeltau = 0.1
scaler   = 2

#Contour the meridional wind
pcm = ax.pcolormesh(lon360,lat,np.mean(fully,(0,1)),vmin=-.2,vmax=.2,cmap="RdBu_r")
fig.colorbar(pcm,ax=ax,fraction = 0.025)
qv  = ax.quiver(lon360[::oint],lat[::aint],
               np.mean(fullx[:,:,::aint,::oint],(0,1)),
               np.mean(fully[:,:,::aint,::oint],(0,1)),
               scale=scaler,color='gray',width=.008,
               headlength=5,headwidth=2,zorder=9)
ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
ax.set_title("Meridional Wind Stress (colors) and the wind stress vectors (arrows)")
# End Result [lon x lat x mon x mode]


#%% Calculate the ekman velocity

# First, lets deal with the coriolis parameter
f       = 2*omega*np.sin(np.radians(yy))
dividef = 1/f 

# Remove values around equator
dividef[np.abs(yy)<=6] = np.nan

# Remove coastal points 
xroll = msk * np.roll(msk,-1,axis=1) * np.roll(msk,1,axis=1)
yroll = msk * np.roll(msk,-1,axis=0) * np.roll(msk,1,axis=0)
mskcoastal = msk * xroll * yroll

# Scrap plot to examine values near the equator
#plt.pcolormesh(lon360,lat,dividef),plt.colorbar(),plt.ylim([-20,20])

_,mld360 = proc.lon180to360(lon180,mldclim)
mld360 = mld360.transpose(1,0,2) # lat x lon x time

# Calculate the anomalous ekman current
u_ek = dividef[:,:,None,None] * tauy_pat  / (rho*mld360[:,:,:,None])
v_ek = dividef[:,:,None,None] * -taux_pat  / (rho*mld360[:,:,:,None])

# Transpose to from [mon x lat x lon] to [lat x lon x mon]
dSSTdx = dTdx.transpose(1,2,0)
dSSTdy = dTdy.transpose(1,2,0)

# Calculate ekman heat flux #[lat x lon x mon x N]
q_ek = cp0 * dividef[:,:,None,None] * (-tauy_pat*dSSTdx[:,:,:,None] + taux_pat*dSSTdy[:,:,:,None])

q_ek = -1* cp0 *(rho*mld360[:,:,:,None]) * (u_ek*dSSTdx[:,:,:,None] + v_ek*dSSTdy[:,:,:,None])
q_ek_msk = q_ek * mskcoastal[:,:,None,None]


#%%

fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax)
pcm = ax.pcolormesh(lon360,lat,dividef)
fig.colorbar(pcm,ax=ax)

#%% Visualize ekman advection
im = 0
N  = 0


scaler   = 0.1 
labeltau = 0.01
viz_tau  = False

clevs =np.arange(-30,40,10)
lablevs = np.arange(-30,35,5)

# Projection
for im in range(12):
    
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.pcolormesh(lon360,lat,(q_ek_msk)[:,:,im,N],vmin=-25,vmax=25,cmap="RdBu_r")
    cl = ax.contour(lon360,lat,(q_ek_msk)[:,:,im,N],levels=clevs,colors='k',linewidths=0.75)
    fig.colorbar(pcm,ax=ax,fraction=0.035)
    
    qv = ax.quiver(lon360[::oint],lat[::aint],
                   u_ek[::aint,::oint,im,N],
                   v_ek[::aint,::oint,im,N],
                   scale=scaler,color='gray',width=.008,
                   headlength=5,headwidth=2,zorder=9)
    ax.quiverkey(qv,1.1,1.035,labeltau,"%.3f $m/s$" % (labeltau))
    if viz_tau:
        qv2 = ax.quiver(lon180[::oint],lat[::aint],
                   taux_pat_fin[::oint,::aint,im,N].T,
                   tauy_pat_fin[::oint,::aint,im,N].T,
                   scale=0.5,color='blue',width=.008,
                   headlength=5,headwidth=2,zorder=9)
    ax.set_title(r"%s $Q_{ek}$ (Contour Interval: 10 $\frac{W}{m^{2}}$)" % (mons3[im]))
    
    savename = "%sCESM_FULL-PIC_Qek-Map_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
    plt.savefig(savename,dpi=150,bbox_tight='inches')
    
    

#%% Save the output

invars  = [q_ek_msk,u_ek,v_ek]
outvars = []
for i in tqdm(range(len(invars))):
    
    # Get variable
    invar = invars[i]
    
    # Change to lon x lat x otherdims
    invar = invar.reshape(nlat,nlon,12*N_mode).transpose(1,0,2)
    
    # Flip longitude
    _,invar = proc.lon360to180(lon360,invar)
    
    # Uncombine mon x N_mode
    invar = invar.reshape(nlon,nlat,12,N_mode)
    outvars.append(invar)
    
q_ek180,u_ek180,v_ek180 = outvars

# Save output...
savename = "%sFULL-PIC_Monthly_NHFLXEOF_Qek.npz" % (rawpath)
np.savez(savename,**{
    'q_ek':q_ek180,
    'u_ek':u_ek180,
    'v_ek':v_ek180,
    'lat':lat,
    'lon':lon180})

#%% Combine output with net heat flux

q_ek180add = q_ek180.copy()
q_ek180add[np.isnan(q_ek180)] = 0

# Combine Heat Fluxes and save
q_comb = eofall + q_ek180add


#%% Save a selected # of EOFS
mcname = "SLAB-PIC"
N_mode_choose = 20
eofforce      = q_comb.copy()
eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
eofforce      = eofforce[:,:,:N_mode_choose,:]
savenamefrc   = "%sflxeof_qek_%ieofs_%s.npy" % (rawpath,N_mode_choose,mcname)
np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)


#%% Load data again (optional) and save just the EOFs for a given season

loadagain       = True
N_mode_choose   = 25
mcname          = "SLAB-PIC"
saveid          = [11,1,2] # Indices of months to average over
savenamenew     = "%sflxeof_qek_%ieofs_%s_DJF.npy" % (rawpath,N_mode_choose,mcname)

if loadagain:
    savenamefrc   = "%sflxeof_qek_%ieofs_%s.npy" % (rawpath,N_mode_choose,mcname)
    eofforce = np.load(savenamefrc)

eofforceseas = np.mean(eofforce[:,:,:,saveid],-1,keepdims=True) # Take mean along month axis
eofforceseas = np.tile(eofforceseas,12) # Tile along last dimension
np.save(savenamenew,eofforceseas)
print("Saved data to "+savenamenew)


# Check plot
N = 0
bboxplot = [-100,20,0,80]
fig,axs = plt.subplots(3,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})
for im in range(12):
    ax = axs.flatten()[im]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    ax.pcolormesh(lon180,lat,eofforceseas[:,:,0,im].T,vmin=-30,vmax=30,cmap="RdBu_r")
    ax.set_title("Mon %i"%(im+1))
    


#%%
plotvars  = [eofall,q_ek180add,q_comb]
plotlabs  = ["$Q_{net}$ ($Wm^{-2}$)","$Q_{ek}$ ($Wm^{-2}$)","$Q_{total}$ ($Wm^{-2}$)"]

N = 30


for im in tqdm(range(12)):
    fig,axs = plt.subplots(1,3,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})
    for i in range(3):
        ax = axs.flatten()[i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        pcm = ax.pcolormesh(lon180,lat,plotvars[i][:,:,im,N].T,vmin=-5,vmax=5,cmap="RdBu_r")
        fig.colorbar(pcm,ax=ax,fraction=0.035)
        ax.set_title(plotlabs[i])
    plt.suptitle("EOF %i (%s)" % (N+1,mons3[im] ))
    
    savename = "%sCESM_FULL-PIC_AddQek_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
    plt.savefig(savename,dpi=150,bbox_tight='inches')



