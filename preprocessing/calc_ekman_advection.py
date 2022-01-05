#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Ekman Advection for the corresponding EOFs and save

Uses variables processed by investigate_forcing.ipynb

Created on Wed Aug 4 05:02:36 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from   tqdm import tqdm

#%% Import modules

stormtrack = 0

if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210104/"
   
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

proc.makedir(figpath)

#%% Additional Functions

def calc_dx_dy(longitude,latitude,centered=False):
    ''' 
        This definition calculates the distance between grid points that are in
        a latitude/longitude format.
        
        Function from: https://github.com/Unidata/MetPy/issues/288
        added "centered" option to double the distance for centered-difference
        
        Equations from:
        http://andrew.hedges.name/experiments/haversine/

        dy should be close to 55600 m
        dx at pole should be 0 m
        dx at equator should be close to 55600 m
        
        Accepts, 1D arrays for latitude and longitude
        
        Returns: dx, dy; 2D arrays of distances between grid points 
                                    in the x and y direction in meters 
    '''
    dlat = np.abs(latitude[1]-latitude[0])*np.pi/180
    if centered:
        dlat *= 2
    dy   = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy   = np.ones((latitude.shape[0],longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    if centered:
        dlon *= 2
    for i in range(latitude.shape[0]):
        # Apply cos^2 latitude weight
        a = (np.cos(latitude[i]*np.pi/180)*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
        dx[i] = c * 6371000
    dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
    return dx, dy

#%% Set Constants

omega = 7.2921e-5 # rad/sec
rho   = 1026      # kg/m3
cp0   = 3996      # [J/(kg*C)]
mons3 = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

centered  = True # Set to True to load centered-difference temperature
calc_dT   = False # Set to True to recalculate temperature gradients (Part 1)
calc_dtau = False # Set to True to perform wind-stress regressions to PCs (Part 2)
calc_qek  = False # set to True to calculate ekman forcing 
debug     = True # Set to True to visualize for debugging

# -------------------
#%% Part 1: LOAD DATA
# -------------------
# Load lat/lon
lon360  = np.load(rawpath+"CESM1_lon360.npy")
lon180  = np.load(rawpath+"CESM1_lon180.npy")
lat     = np.load(rawpath+"CESM1_lat.npy") 

# Load Land/ice mask
msk     = np.load(lipath)
#ds *= msk[None,:,:]

# Get distances relating to the grid for gradient calculations
dx,dy   = calc_dx_dy(lon360,lat)
dx2,dy2 = calc_dx_dy(lon360,lat,centered=True) # Additional for centered diff
xx,yy   = np.meshgrid(lon360,lat)

# -----------------------------------------
#%% Part 1: CALCULATE TEMPERATURE GRADIENTS
# -----------------------------------------
if calc_dT:
    #% Load the data (temperature)
    st   = time.time()
    ds1  = xr.open_dataset(rawpath+"../CESM_proc/TS_PIC_FULL.nc")
    ts   = ds1.TS.values
    #ts -= 273.15 #(convert to celsius)
    print("Completed in %.2fs"%(time.time()-st))
    
    # Calculate the mean temperature for each month
    ts_monmean = ts.mean(0) # [month x lat x lon]
    
    #% Temperature Gradient (Forward Difference)
    
    # Roll longitude along axis for <FORWARD> difference (gradT_x0 = T_x1 - xT0)
    dTdx = (np.roll(ts_monmean,-1,axis=2)- ts_monmean) / dx[None,:,:]
    dTdy = (np.roll(ts_monmean,-1,axis=1)- ts_monmean) / dy[None,:,:]
    dTdy[:,-1,:] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    # Save output [mon x lat x lon360]
    savename = "%sFULL-PIC_Monthly_gradT_lon360.npz" % (rawpath)
    np.savez(savename,**{
        'ts_monmean':ts_monmean,
        'dTdx':dTdx,
        'dTdy':dTdy,
        'dx':dx,
        'dy':dy,
        'lon':lon360,
        'lat':lat})
    print("Saved forward difference output to: %s" % savename)
    
    #% Temperature Gradient (Centered Difference)
    
    # Try Centered Difference (compare with Numpy's Gradient Function)
    if debug:
        dTdx2   = (np.roll(ts_monmean,-1,axis=2) - np.roll(ts_monmean,1,axis=2))/2
        dx2test = np.gradient(ts_monmean,axis=2) # Using gradient function
        
        # Try visualizing difference
        #plt.pcolormesh(dx2test[0,...],vmin=-10,vmax=10),plt.colorbar()
        #plt.pcolormesh(dTdx2[0,...],vmin=-10,vmax=10),plt.colorbar()
        plt.pcolormesh(dTdx2[0,...]-dx2test[0,...]),plt.colorbar(),plt.title("dTdx (manual - np.gradient)")
        plt.show()
    
    # Calculate <CENTERED> difference
    dTx2          = np.roll(ts_monmean,-1,axis=2) - np.roll(ts_monmean,1,axis=2)
    dTy2          = np.roll(ts_monmean,-1,axis=1) - np.roll(ts_monmean,1,axis=1)
    dTdx2         = dTx2 / dx2[None,...]
    dTdy2         = dTy2 / dy2[None,...]
    dTdy2[:,-1,:] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    # Visualize differences
    if debug:
        plt.pcolormesh(dTdx2[0,:,:]- dTdx[0,:,:],vmin=-0.6e-5,vmax=0.6e-5,cmap='RdBu_r'),plt.colorbar()
        plt.title("dTdx (centered-forward)")
        plt.show()
    
    # Save  output [mon x lat x lon360]
    savename = "%sFULL-PIC_Monthly_gradT2_lon360.npz" % (rawpath)
    np.savez(savename,**{
        'ts_monmean':ts_monmean,
        'dTdx':dTdx2,
        'dTdy':dTdy2,
        'dx':dx2,
        'dy':dy2,
        'lon':lon360,
        'lat':lat})
    print("Saved centered difference output to: %s" % savename)

    #%% Experimenting with other 
    
    # https://gradsaddict.blogspot.com/2019/11/python-tutorial-temperature-advection.html?m=0
    
    proj      =  ccrs.Miller()#UTM(zone=1)#ccrs.PlateCarree(central_longitude=360)
    mlon,mlat = np.meshgrid(lon360,lat)
    output    = proj.transform_points(ccrs.PlateCarree(),mlon,mlat)
    
    x,y=output[:,:,0],output[:,:,1]
    gradx=np.gradient(x,axis=1)
    grady=np.gradient(y,axis=0)


#%% Load temperature gradient and plot

# Load output
if centered:
    savename = "%sFULL-PIC_Monthly_gradT2_lon360.npz" % (rawpath)
else:
    savename = "%sFULL-PIC_Monthly_gradT_lon360.npz" % (rawpath)
print("Loading the following: %s"% savename)
print("Centered Difference: %s" % centered)
ld         = np.load(savename)
ts_monmean = ld['ts_monmean']
dTdx       = ld['dTdx']
dTdy       = ld['dTdy']
dx         = ld['dx']
dy         = ld['dy']
lon360     = ld['lon']
lat        = ld['lat']

# Set the month
if debug:
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


# --------------------------------
#%% Part 2: WIND STRESS REGRESSION
# --------------------------------

#% Load the wind stress and the PCs to prepare for regression
# -----------------------------------------------------------
# This was calculated in NHFLX_EOF_monthly.py
N_mode = 200

# Load the PCs
ld        = np.load("%sNHFLX_FULL-PIC_%sEOFsPCs_lon260to20_lat0to65.npz" % (rawpath,N_mode),allow_pickle=True)
pcall     = ld['pcall'] # [PC x MON x TIME]
eofall    = ld['eofall']
eofslp    = ld['eofslp']
varexpall = ld['varexpall']
lon = ld['lon']
lat = ld['lat']

# Flip signs
spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP
N_modeplot = 5              # Just flip the first few
for N in tqdm(range(N_modeplot)):
    if N == 1:
        chkbox = eapbox # Shift coordinates west
    else:
        chkbox = spgbox
    for m in range(12):
        
        sumflx = proc.sel_region(eofall[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
        sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
        
        if sumflx > 0:
            print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
            eofall[:,:,m,N]*=-1
            pcall[N,m,:] *= -1
        if sumslp > 0:
            print("Flipping sign for SLP, mode %i month %i" % (N+1,m+1))
            eofslp[:,:,m,N]*=-1

if calc_dtau:
    
    # Load each wind stress component [yr mon lat lon]
    st   = time.time()
    dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUX_PIC_FULL.nc")
    taux = dsx.TAUX.values
    dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUY_PIC_FULL.nc")
    tauy = dsx.TAUY.values
    print("Loaded wind stress data in %.2fs"%(time.time()-st))

    # Convert stress from stress on OCN on ATM --> ATM on OCN
    taux*= -1
    tauy*= -1
    
    #% Preprocess Wind Stress Variables
    # ---------------------------------
    takeanom = False 
    
    fullx = taux.copy()
    fully = tauy.copy()
        
    nyr,_,nlat,nlon = taux.shape
    taux = taux.reshape(nyr,12,nlat*nlon) # Combine space
    tauy = tauy.reshape(nyr,12,nlat*nlon)
    
    if takeanom:
        taux = taux - taux.mean(1)[:,None,:]
        tauy = tauy - tauy.mean(1)[:,None,:]



    #% Regress wind stress components to NHFLX PCs
    # ---------------------------------
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


    # #%% Flip EOFs (again)
    
    # spgbox = [-60,20,40,80]
    # N_modeplot = 5
    
    # for N in tqdm(range(N_modeplot)):
    #     for m in range(12):
            
    #         sumflx = proc.sel_region(eofall[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
    #         #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
    #         if sumflx > 0:
                
    #             print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                
    #             eofall[:,:,m,N] *=-1
    #             pcall[N,m,:]    *= -1
    #             eofslp[:,:,m,N] *=-1
    #             taux_pat_fin[:,:,m,N] *=-1
    #             tauy_pat_fin[:,:,m,N] *=-1

    #% Save the files
    # ---------------------------------
    
    # Save output...
    savename = "%sFULL-PIC_Monthly_NHFLXEOF_TAUX_TAUY_centered%i.npz" % (rawpath,centered)
    np.savez(savename,**{
        'taux':taux_pat_fin,
        'tauy':tauy_pat_fin,
        'lon':lon180,
        'lat':lat})
    print("Saved wind-stress regression output to: %s" % savename)


#% Load the files otherwise
else:
    savename = "%sFULL-PIC_Monthly_NHFLXEOF_TAUX_TAUY_centered%i.npz" % (rawpath,centered)
    ld = np.load(savename)
    taux_pat_fin = ld['taux']
    tauy_pat_fin = ld['tauy']
    lon180 = ld['lon']
    lat = ld['lat']
    print("Loading wind-stress regression output from: %s" % savename)
    print("Centering is: %s" %centered)
    

# ------------------------------------------
#%% Visualization for Wind Stress Regression
# ------------------------------------------

#% Individual monthly wind stress plots for EOF N
im = 9
N  = 1
scaler   =    .75 # # of data units per arrow
bboxplot = [-100,20,0,80]
labeltau = 0.10
slplevs  = np.arange(-400,500,100)
flxlim   = [-30,30]
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
    plt.savefig(savename,dpi=150,bbox_inches='tight')

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
plt.suptitle("Seasonal Averages for EOF %i of $Q_{net}$ (colors), $SLP$ (contours), and Wind Stress (quivers)" % (N+1),y=0.94)
savename = "%sCESM_FULL-PIC_WindStressMap_EOF%02i_seasonal.png" %(figpath,N+1)   
plt.savefig(savename,dpi=150,bbox_inches='tight')

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

# --------------------------------------
#%% Part 3: Calculate the ekman velocity
# --------------------------------------
if calc_qek:
    
    #% Load mixed layer depths
    # ------------------------
    st    = time.time()
    dsmld = xr.open_dataset(rawpath+"HMXL_PIC.nc")
    hmxl  = dsmld.HMXL.values # [lon180 x lat x time]
    print("Load MLD in %.2fs"%(time.time()-st))

    # Find the climatological mean
    mld     = hmxl.reshape(288,192,int(hmxl.shape[2]/12),12)
    mldclim = mld.mean(2)

    # Convert cm --> meters
    mldclim /= 100 
    
    # First, lets deal with the coriolis parameter
    # --------------------------------------------
    f       = 2*omega*np.sin(np.radians(yy))
    dividef = 1/f 
    
    # Remove values around equator
    dividef[np.abs(yy)<=6] = np.nan
    if debug: # Test plot 1/f
        fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
        ax = viz.add_coast_grid(ax)
        pcm = ax.pcolormesh(lon360,lat,dividef)
        fig.colorbar(pcm,ax=ax)
    
    # Remove coastal points 
    xroll = msk * np.roll(msk,-1,axis=1) * np.roll(msk,1,axis=1)
    yroll = msk * np.roll(msk,-1,axis=0) * np.roll(msk,1,axis=0)
    mskcoastal = msk * xroll * yroll
    
    # Scrap plot to examine values near the equator
    #plt.pcolormesh(lon360,lat,dividef),plt.colorbar(),plt.ylim([-20,20])
    _,mld360 = proc.lon180to360(lon180,mldclim)
    mld360 = mld360.transpose(1,0,2) # lat x lon x time
    
    # Calculate the anomalous ekman current
    # -------------------------------------
    u_ek = dividef[:,:,None,None] * tauy_pat  / (rho*mld360[:,:,:,None])
    v_ek = dividef[:,:,None,None] * -taux_pat  / (rho*mld360[:,:,:,None])
    
    # Transpose to from [mon x lat x lon] to [lat x lon x mon]
    dSSTdx = dTdx.transpose(1,2,0)
    dSSTdy = dTdy.transpose(1,2,0)
    
    # Calculate ekman heat flux #[lat x lon x mon x N]
    # ------------------------------------------------
    #q_ek = cp0 * dividef[:,:,None,None] * (-tauy_pat*dSSTdx[:,:,:,None] + taux_pat*dSSTdy[:,:,:,None])
    q_ek = -1* cp0 *(rho*mld360[:,:,:,None]) * (u_ek*dSSTdx[:,:,:,None] + v_ek*dSSTdy[:,:,:,None])
    q_ek_msk = q_ek * mskcoastal[:,:,None,None] # Apply coastal mask
    
    #% Save the output
    # ----------------
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

    # Save output output
    savename = "%sFULL-PIC_Monthly_NHFLXEOF_Qek_centered%i.npz" % (rawpath,centered)
    np.savez(savename,**{
        'q_ek':q_ek180,
        'u_ek':u_ek180,
        'v_ek':v_ek180,
        'lat':lat,
        'lon':lon180})
    print("Saving Ekman Forcing to: %s" % savename)
else:
    
    savename = "%sFULL-PIC_Monthly_NHFLXEOF_Qek_centered%i.npz" % (rawpath,centered)
    ld = np.load(savename)
    q_ek180 = ld['q_ek']
    u_ek180 = ld['u_ek']
    v_ek180 = ld['v_ek']
    print("Loading Ekman Forcing from: %s" % savename)
    print("Centering is: %s" %centered)


#%% Visualizations for Q-ek calculations
# --------------------------------------

#% %est plot temperature 
im = 0
fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax)
pcm = ax.pcolormesh(lon360,lat,ts_monmean[im,:,:]*msk)
fig.colorbar(pcm,ax=ax)

#%% Visualize ekman advection


# Option
#im = 0 # Month Index (for debugging)
N  = 0 # Mode Index
viz_tau      = False # True: Include wind stress quivers
contour_temp = True # True: contour SST ... False: contour q_ek


# Silly things: flip back to degrees east just for plotting :(....
_,q_ek_msk = proc.lon180to360(lon180,q_ek180)
_,u_ek = proc.lon180to360(lon180,u_ek180)
_,v_ek = proc.lon180to360(lon180,v_ek180)
q_ek_msk = q_ek_msk.transpose(1,0,2,3)
u_ek = u_ek.transpose(1,0,2,3)
v_ek = v_ek.transpose(1,0,2,3)


# U_ek quiver options
scaler   = 0.1 
labeltau = 0.01

# Q_ek contour levels
clevs =np.arange(-30,40,10)
lablevs = np.arange(-30,35,5)

# Temperature contour levels
tlm = [275,310] 
tlevs = np.arange(tlm[0],tlm[1]+1,1)
tlab  = np.arange(tlm[0],tlm[1]+5,5)

# Projection
for im in range(12):
    
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.pcolormesh(lon360,lat,(q_ek_msk)[:,:,im,N],vmin=-25,vmax=25,cmap="RdBu_r")
    
    if contour_temp:
        cl = ax.contour(lon360,lat,ts_monmean[im,:,:]*msk,levels=tlevs,colors='k',linewidths=0.75)
        ax.clabel(cl,levels=tlab)
    else:
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
    if contour_temp:
        ax.set_title(r"%s $Q_{ek}$ (Contour Interval: 10 $\frac{W}{m^{2}}$; 1 $^{\circ}C$)" % (mons3[im]))
    else:
        
        ax.set_title(r"%s $Q_{ek}$ (Contour Interval: 10 $\frac{W}{m^{2}}$)" % (mons3[im]))
    
    savename = "%sCESM_FULL-PIC_Qek-Map_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

# -------------------------------------------------------------
# %% PART 4: Save selected number (based on selected threshold)
# -------------------------------------------------------------

# First, obtain index for selected threshold for variance explained
# Below section is from NHFLX_EOF_monthly, copied 2021.01.04

# Calculate cumulative variance at each EOF
cvarall = np.zeros(varexpall.shape)
for i in range(N_mode):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Select threshold based on variance explained
vthres  = 0.90
thresid = np.argmax(cvarall>vthres,axis=0)
thresperc = []
for i in range(12):
    
    print("Before")
    print(cvarall[thresid[i]-1,i])
    print("After")
    print(cvarall[thresid[i],i])
    
    # Append percentage
    thresperc.append(cvarall[thresid[i],i])
thresperc = np.array(thresperc)
if debug:
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    ax.bar(mons3,thresid,color=[0.56,0.90,0.70],alpha=0.80)
    ax.set_title("Number of EOFs required \n to explain %i"%(vthres*100)+"% of the $Q_{net}$ variance")
    #ax.set_yticks(ytk)
    ax.set_ylabel("# EOFs")
    ax.grid(True,ls='dotted')
    
    rects = ax.patches
    labels = thresid
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + -5, label, ha="center", va="bottom"
        )
        
#%% Take threshold from variable (NOTE: no correction is applied!)
eofcorr = 0

qekforce = q_ek180.copy() # [lon x lat x month x pc]
cvartest = cvarall.copy()
for i in range(12):
    # Set all points after crossing the variance threshold to zero
    stop_id = thresid[i]
    print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
    qekforce[:,:,i,stop_id+1:] = 0
    cvartest[stop_id+1:,i] = 0
qekforce= qekforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

# Cut to maximum EOF
nmax = thresid.max()
qekforce = qekforce[:,:,:nmax+1,:]
savenamefrc = "%sQek_eof_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,"FULL_PIC",eofcorr)
np.save(savenamefrc,qekforce)

print("Saved postprocessed Q-ek forcing to %s" % (savenamefrc))






#%%% Old Scripts Below
#%% Combine output with net heat flux

q_ek180add = q_ek180.copy()
q_ek180add[np.isnan(q_ek180)] = 0

# Combine Heat Fluxes and save
q_comb = eofall + q_ek180add


#%% Save a selected # of EOFS
mcname = "SLAB-PIC"
N_mode_choose = 50
eofforce      = q_comb.copy()
eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
eofforce      = eofforce[:,:,:N_mode_choose,:]
savenamefrc   = "%sflxeof_qek_%ieofs_%s.npy" % (rawpath,N_mode_choose,mcname)
np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)



#%%

# Calculate correction factor
eofcorr  = True
if eofcorr:
    ampfactor = 1/thresperc
else:
    ampfactor = 1

eofforce = q_comb.copy() # [lon x lat x month x pc]
cvartest = cvarall.copy()
for i in range(12):
    # Set all points after crossing the variance threshold to zero
    stop_id = thresid[i]
    print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
    eofforce[:,:,i,stop_id+1:] = 0
    cvartest[stop_id+1:,i] = 0
eofforce = eofforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

if eofcorr:
    eofforce *= ampfactor[None,None,None,:]

# Cut to maximum EOF
nmax = thresid.max()
eofforce = eofforce[:,:,:nmax+1,:]

savenamefrc = "%sflxeof_q-ek_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,"SLAB-PIC",eofcorr)
np.save(savenamefrc,eofforce)


#%% Load data again (optional) and save just the EOFs for a given season

loadagain       = True
N_mode_choose   = 2
mcname          = "SLAB-PIC"
saveid          = [5,6,7] # Indices of months to average over
savenamenew     = "%sflxeof_qek_%ieofs_%s_JJA.npy" % (rawpath,N_mode_choose,mcname)

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



