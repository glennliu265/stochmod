#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:13:44 2020

@author: gliu
"""

import xarray as xr
from scipy import signal
import numpy as np
import time

import matplotlib.pyplot as plt
from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point


def eof_simple(pattern,N_mode,remove_timemean):
    pattern1 = pattern.copy()
    nt = pattern1.shape[1] # Get time dimension size
    ns = pattern1.shape[0] # Get space dimension size
    
    # Preallocate
    eofs = np.zeros((ns,N_mode))
    pcs  = np.zeros((nt,N_mode))
    varexp = np.zeros((N_mode))
    
    # Remove time mean if option is set
    if remove_timemean == 1:
        pattern1 = pattern1 - pattern1.mean(axis=1)[:,None] # Note, the None adds another dimension and helps with broadcasting
    
    # Compute SVD
    [U, sigma, V] = np.linalg.svd(pattern1, full_matrices=False)
    
    # Compute variance (total?)
    norm_sq_S = (sigma**2).sum()
    
    for II in range(N_mode):
        
        # Calculate explained variance
        varexp[II] = sigma[II]**2/norm_sq_S
        
        # Calculate PCs
        pcs[:,II] = np.squeeze(V[II,:]*np.sqrt(nt-1))
        
        # Calculate EOFs and normalize
        eofs[:,II] = np.squeeze(U[:,II]*sigma[II]/np.sqrt(nt-1))
    return eofs, pcs, varexp

    
def regress_2d(A,B):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    
    # 2D Matrix is in A [MxN]
    if len(A.shape) > len(B.shape):
        
        # Tranpose A so that A = [MxN]
        if A.shape[1] != B.shape[0]:
            A = A.T
        
        
        # Set axis for summing/averaging
        a_axis = 1
        b_axis = 0
        
        # Compute anomalies along appropriate axis
        Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
        Banom = B - np.nanmean(B,axis=b_axis)
        

        
    # 2D matrix is B [N x M]
    elif len(A.shape) < len(B.shape):
        
        # Tranpose B so that it is [N x M]
        if B.shape[0] != A.shape[0]:
            B = B.T
        
        # Set axis for summing/averaging
        a_axis = 0
        b_axis = 0
        
        # Compute anomalies along appropriate axis        
        Aanom = A - np.nanmean(A,axis=a_axis)
        Banom = B - np.nanmean(B,axis=b_axis)[None,:]
    
    # Calculate denominator, summing over N
    Aanom2 = np.power(Aanom,2)
    denom = np.sum(Aanom2,axis=a_axis)    
    
    # Calculate Beta
    beta = Aanom @ Banom / denom
    
        
    return beta

    



# Path to data
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath  = projpath + '01_Data/'
ncother = datpath+"PSL_NAOproc.nc"
ds = xr.open_dataset(ncother)
outpath = projpath+'/02_Figures/20200709/'

# Read to numpy arrays
slp = ds.PSL.values # ens x year x lat x lon
lat = ds.lat.values
lon = ds.lon.values
year = ds.year.values

# Get respective dimension sizes
lonsize = len(lon)
latsize = len(lat)
tsize   = len(year)
latpos  = slp.shape.index(latsize)
lonpos  = slp.shape.index(lonsize)

# EOF options
N_mode = 3
station = "Reykjavik"
flipeof = 1

lonLisbon = -9.1398 + 360
latLisbon = 38.7223
lonReykjavik = -21.9426 + 360
latReykjavik = 64.146621
tol = 5 # in degrees

# Regression method
method = 3 # 1 = pt by pt loop with polyfit, #2 = just take raw eof, #3 = regress2d function (vectorized)


# Find box indices
oidr = np.abs((lon - lonReykjavik)).argmin() 
aidr = np.abs((lat - latReykjavik)).argmin()
oidl = np.abs((lon - lonLisbon)).argmin() 
aidl = np.abs((lat - latLisbon)).argmin()

# Remove ensemble mean
slp = slp - np.mean(slp,axis=0)

# Compute area weights and apply to vvariable
wgt = np.sqrt(np.cos(np.radians(lat)))
plt.plot(wgt)

slp = slp * wgt[None,None,:,None]

# Preallocate
pcall  = np.zeros((slp.shape[0],tsize,N_mode)) * np.nan
eofall = np.zeros((slp.shape[0],latsize,lonsize,N_mode)) * np.nan
varexpall = np.zeros((slp.shape[0],N_mode)) * np.nan


# Loop by ensemble...
for e in range(slp.shape[0]):
    startloop = time.time()
    # Get data for ensemble member
    data = slp[e,...] # year x lat x lon
    
    # Combine spatial dimensions
    data = np.reshape(data,(tsize,latsize*lonsize))
    
    #%% Get non-nan indices ....
    # Find indices of nan pts and non-nan (ok) pts
    nanpts = np.isnan(data)
    okpts  = np.invert(nanpts)

    # Drop nan pts and reshape again to separate space and time dimensions
    data_ok = data[okpts]

    # Reshape from linear indices back to [time x space]
    data_ok = np.reshape(data_ok,(tsize,int(len(data_ok)/tsize)))


    # Get indices of points that are not nan (in space)
    inotnan = np.prod(okpts,axis=0) # take product along time dimension
    inotnan = np.where(inotnan==True)
    
    #%% Detrend data
    data_dt = signal.detrend(data_ok,axis=0)
    
    # Prepare data
    datain = np.transpose(data_dt,(1,0))
    
    #%% Perform EOF
    eofs,pcs,varexpall[e,:]=eof_simple(datain,N_mode,1)
    
    #%% Standardize Pcs
    #pcall[e,:,:] = pcs / np.std(pcs,axis=0)
    pcraw = pcs / np.std(pcs,axis=0)
    
    #%% Reshape EOFs
    
    #varfill = np.zeros((data.shape[1],N_mode))
    #varfill[inotnan,:] = eofs
    
    # if lonpos > latpos:
    #     eofall[e,:,:,:] = np.reshape(eofrs,(latsize,lonsize,N_mode))
    # elif lonpos < latpos:
    #     eofall[e,:,:,:] = np.reshape(eofrs,(lonsize,latsize,N_mode))
        
    #%% Regress Pcs back to variable...
    
    # Explicit Regression step
    if method == 1:
        eofrs = np.zeros((data.shape[1],N_mode))
        for pc in range(pcs.shape[1]):
            
            # Get corresponding PC times series
            ts = pcs[:,pc]
            i = 0
            
            # Loop thru each pt
            for pt in range(data.shape[1]):
                
                vartime = data[:,pt]
                
                # Skip nan points
                if np.any(np.isnan(vartime)):
                    eofrs[pt,pc]=np.nan
                    
                    i += 1
                    continue
                
                # Perform regrssion
                r = np.polyfit(ts,vartime,deg=1)
                
                eofrs[pt,pc] = r[0]
                i += 1
                msg = '\rCompleted Regression for %i of %i points for PC %i' % (i,data.shape[1],pc+1)
                print(msg,end="\r",flush=True)
            #print('\n')
    elif method == 2:
                
        eofrs = np.zeros((data.shape[1],N_mode))
        eofrs[inotnan,:] = eofs
    elif method == 3:
        
        eofrs = np.zeros((data.shape[1],N_mode))
        for pc in range(pcs.shape[1]):
            eofrs[:,pc] = regress_2d(pcs[:,pc],data) 
        
        
    #%% Flip EOFs based on boxes...
    if flipeof == 1:
        # EOF 1 only (need to figure out the other EOFs..)
        # First check using reykjavik
        rboxlon = np.where((lon >= lonReykjavik-tol) & (lon <= lonReykjavik+tol))[0]
        rboxlat = np.where((lat >= latReykjavik-tol) & (lat <= latReykjavik+tol))[0]
        
        
        eofraw = np.reshape(eofrs,(latsize,lonsize,N_mode))
        chksum = np.sum(eofraw[rboxlat[:,None],rboxlon[None,:],0],(0,1))
        if chksum > 0:
            print("\t Flipping sign based on Reykjavik, Ens%i" % (e+1))
            eofraw *= -1
            pcraw *= -1
            
        
    
        # Double Check with Lisbon
        lboxlon = np.where((lon >= lonLisbon-tol) & (lon <= lonLisbon+tol))[0]
        lboxlat = np.where((lat >= latLisbon-tol) & (lat <= latLisbon+tol))[0]
        
        
        eofraw = np.reshape(eofrs,(latsize,lonsize,N_mode))
        chksum = np.sum(eofraw[lboxlat[:,None],lboxlon[None,:],0],(0,1))
        if chksum < 0:
            print("\t Flipping sign based on Lisbon, Ens%i" % (e+1))
            eofraw *= -1
            pcraw *= -1
        #%% Assign to outside variable
        eofall[e,:,:,:] = eofraw
        pcall[e,:,:] = pcraw
            
        print("Calculated EOF for ens %i of %i in %.2fs\n" % (e+1,slp.shape[0],time.time()-startloop))
    else:
        eofall[e,:,:,:] = np.reshape(eofrs,(latsize,lonsize,N_mode))
        pcall[e,:,:] = pcraw
    

#%% 
np.savez(datpath+"Manual_EOF_Calc_NAO.npz",eofall=eofall,pcall=pcall,varexpall=varexpall)


#%% Load Data
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO.npz")
eofall    = npzdata['eofall']
pcall     = npzdata['pcall']
varexpall = npzdata['varexpall'] 



# %% Try some plots

# Linearly detrend the data
# plot the results
# Try visualizing some data (EOF1)
N = 0
e = 'avg'

def plot_nao_eof(N,e):
    if e == "avg":
        pexplained = np.nanmean(varexpall,0)[N]*100
        var = np.nanmean(eofall[:,:,:,N],0)/100
        estring = e
        tsplot = np.nanmean(pcall,0)[:,N]
    else:
        pexplained = varexpall[e-1,N]*100  
        var = np.copy(eofall[e-1,:,:,N])/100 # Convert to milibars
        estring = str(e)
        tsplot = pcall[e,:,N]
    cmap = cmocean.cm.balance
    # NAO Calculation Settings
    lonW = -90
    lonE = 40
    latS = 20
    latN = 80
    
    #cint = np.arange(-0.5,0.55,0.05)
    if N == 0:
        cint = np.arange(-3,3.5,0.5)
    elif N == 1:
        cint = np.arange(-1,1.2,0.2)
    elif N == 2:
        cint = np.arange(-2,2.1,0.1)
    
    
    # Plot the EOF
    #var1,lon1 = add_cyclic_point(var,lon) 
    plt.style.use('ggplot')
    fig,ax= plt.subplots(1,1,figsize=(6,4))
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lonW,lonE,latS,latN])
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE)
    
    # Add contours
    cs = ax.contourf(lon,lat,var,cint,cmap=cmap,transform=ccrs.PlateCarree())
    # Negative contours
    cln = ax.contour(lon,lat,var,
                cint[cint<0],
                linestyles='dashed',
                colors='k',
                linewidths=0.5,
                transform=ccrs.PlateCarree())
    
    # Positive Contours
    clp = ax.contour(lon,lat,var,
                cint[cint>=0],
                colors='k',
                linewidths=0.5,
                transform=ccrs.PlateCarree())    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='black',linestyle=':')
    
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    bc = plt.colorbar(cs,orientation='horizontal')
    
    ax.set_title("EOF %i, Variance Explained: %.02f%% " %(N+1,pexplained),fontsize=16)
    plt.savefig(outpath+"NAO_EOF%i_Ens%s.png"%(N+1,estring), bbox_inches="tight",dpi=200)
    return fig

plot_nao_eof(0,"avg")


for N in range(2):
    for e in range(42):
        plot_nao_eof(N,e)

#%% Plot the PCs

plt.style.use('ggplot')
fig,ax = plt.subplots(3,1,figsize=(10,6),sharex=True)

for N in range(3):
    
    plt.subplot(3,1,N+1)
    
    # Plot for each ensemble member
    for e in range(42):
        
        varplot = np.squeeze(pcall[e,:,N])
        plt.plot(year,varplot,color=[0.65,0.65,0.65])    
    l1 = plt.plot(year,pcall[-1,:,N],color=[0.65,0.65,0.65],label="Indv. Member")
    varensavg = np.nanmean(pcall[:,:,N],axis=0)
    l2 = plt.plot(year,varensavg,color='k',label='Ensemble Average')
    
    
    explained = np.nanmean(varexpall,0)[N]*100
    plt.title("PC %i, Variance Explained: %.02f%% " %(N+1,explained),fontsize=16)
    if N == 1:
        plt.ylabel("NAO Index")
    plt.ylim([-3,3])
    
plt.tight_layout()
plt.xlabel("Years")    
plt.savefig(outpath+"NAO_PCAll_EnsAll.png", bbox_inches="tight",dpi=200)






plt.style.use('ggplot')
fig,ax= plt.subplots(1,1,figsize=(6,4))
ax.plot(year,tsplot)
ax.set_ylabel("NAO Index")
ax.set_xlabel("Years")
ax.set_ylim([-0.5,0.5])
ax.set_title("PC %i, Variance Explained: %.02f%% " %(N+1,pexplained),fontsize=16)
plt.savefig(outpath+"NAO_PC%i_Ens%s.png"%(N+1,estring), bbox_inches="tight",dpi=200)


