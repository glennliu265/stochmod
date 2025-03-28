#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate EOFs for Surface Heat Flux Anomalies and prepare forcing for the
stochastic model output

(**) Indicates section needs to be run if output has not been generated,
i.e. is optional if the script has been run before output saved elsewhere

(@@) Indicates a plotting section

Created on Tue Jul 20 11:49:09 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm

import cmocean
#%%
stormtrack = 0

if stormtrack == 1:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
elif stormtrack == 0:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    #datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220113/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"

from amv import proc,viz
import scm

def sel_regionxr(ds,bbox):
    """
    Select region from xr.dataset with 'lon' (degrees East), 'lat'.
    Current supports crossing the prime meridian

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    bbox : TYPE
        DESCRIPTION.

    Returns
    -------
    dsreg : TYPE
        DESCRIPTION.

    """
    # Select lon (lon360)
    if bbox[0] > bbox[1]: # Crossing Prime Meridian
        print("Crossing Prime Meridian!")
        dsreg = ds.isel(lon =(ds.lon < bbox[1]) + (ds.lon>bbox[0]))
        dsreg = dsreg.sel(lat=slice(bbox[2],bbox[3]))
    else:
        dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    return dsreg

#%% User Edits

# Set your configuration
mconfig = "PIC_FULL"

# Set up names (need to change this at some point, silly coding...)
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
elif mconfig == "PIC_FULL":
    mcname = "FULL-PIC"

# Bounding Box of Plotting and EOF Analysis
bbox    = [260,20,0,65]
bboxeof = [280,20,0,65]

# Debuggin params
debug   = True
lonf    = -30
latf    = 50

# EOF parameters
N_mode = 200

# Plotting params
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
blabels=[0,0,0,0]

# Correction Mode
correction     = True # Set to True to use [Qnet + lambda*T], rather than [Qnet]
rolln          = 0

if rolln != 1: # Add rolln to end of file name
    print("Adding extra string for indexing")
    rollnstr       = "_rolln%s" % str(rolln)
    correction_str = "_Fprime_rolln0"
else: # No rolln for case where it is T(t-1)
    rollnstr       = ""
    correction_str = "_Fprime"

#%% Open the dataset ~63.26s

"""
Output for this section
-----------------------
flxglob  - Global Net Heat Flux anomalies      [time x lat x lon360]
slpglob  - Global sea level pressure anomalies [time x lat x lon360]
msk      - Land-ice mask                       [lat x lon360]
lat,lon,lon180 (1-d latitude and longitude arrays)
"""

# Open the dataset (Net Heat Flux)
st      = time.time()


    
lon360flag = True # Flip Qnet)

if correction: # Load Fprime (=Qnet + lambda*T)
    
    #Open Dataset
    ds = xr.open_dataset("%s../Fprime_%s%s.nc" % (datpath,mconfig,rollnstr))
    
    if np.any(ds.lon > 180): # Correct for cases where lon is degrees west
        lon360flag = False
        print("Flipping lon")
        
        
    # Select Region
    dsreg = sel_regionxr(ds,bboxeof)
    #flxreg = dsreg.Fprime.values
    lon180  = ds.lon.values
    lon     = np.load(datpath+"CESM1_lon360.npy") # Load 360 data
    lat     = ds.lat.values
    flxglob = ds.Fprime.values
    
    # Apply land/ice mask
    msk = np.load(lipath)
    
else: # Load Qnet Data
    
    if mconfig == "PIC_SLAB":
        # Open dataset
        ds = xr.open_dataset("%sNHFLX_PIC_SLAB_raw.nc" % datpath) # time x lat x lon
    
        # Apply land/ice mask
        msk = np.load(lipath)
        ds *= msk[None,:,:]
        
        # Select Region
        dsreg = sel_regionxr(ds,bboxeof)
        
        # Read out data
        flxglob = ds.NHFLX.values
        #flxreg = dsreg.NHFLX.values
        lon     = ds.lon.values
        lat     = ds.lat.values
        
    elif mconfig == "PIC_FULL":
        
        # Load Data
        flxglob = np.load("%s../NHFLX_PIC_FULL.npy"%(datpath)) # yr x mon x lat x lon
        lon = np.load(datpath+"CESM1_lon360.npy")
        lon180 = np.load(datpath+"CESM1_lon180.npy")
        lat    = np.load(datpath+"CESM1_lat.npy") 
        
        # Combine time
        nmon,_,nlat,nlon = flxglob.shape
        flxglob = flxglob.reshape(nmon*12,nlat,nlon)
        
        # Apply a mask
        msk = np.load(lipath)
        flxglob *= msk[None,:,:]
        

# Load in the SLP:
if mconfig == "PIC_SLAB":
    slpglob = np.load(datpath + "../CESM_proc/PSL_PIC_SLAB.npy")
else:
    ds = xr.open_dataset("%s../CESM_proc/PSL_PIC_FULL.nc" % datpath)
    slpglob = ds.PSL.values
    
print("Loaded data in %.2fs"%(time.time()-st))

#%% Preprocess, EOF Analysis ~304.38s
st = time.time()
ntime,nlat,nlon = flxglob.shape
nyr = int(ntime/12)

#% SLP reshape and apply mask ------------------------------------------------
slpglob = slpglob.reshape(ntime,nlat,nlon) # [yr x mon x lat x lon] to [time lat lon]
#slpglob *= msk[None,...]

# Detrend --------------------------------------------------------------------
sttemp = time.time()
flxa,linmod,beta,intercept = proc.detrend_dim(flxglob,0)
print("Detrended Qnet in %.2fs" % (time.time()-sttemp))

sttemp = time.time() # ~212s
slpa,_,_,_                 = proc.detrend_dim(slpglob,0)
print("Detrended SLP in %.2fs" % (time.time()-sttemp))

# Plot Spatial Map
if debug: # Appears to be positive into the ocean
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.pcolormesh(lon,lat,flxglob[0,:,:],vmin=-500,vmax=500,cmap="RdBu_r")
    fig.colorbar(pcm,ax=ax)
# Plot Detrending
klon,klat = proc.find_latlon(lonf+360,latf,lon,lat)
if debug: 
    fig,ax = plt.subplots(1,1)
    t = np.arange(0,ntime)
    ax.scatter(t,flxglob[:,latf,lonf],label="Raw")
    ax.plot(t,linmod[:,latf,lonf],label="Model")
    ax.scatter(t,flxa[:,latf,lonf],label="Detrended")
    ax.legend()
print("Completed process in %.2fs"%(time.time()-st))

#%% (**) Apply Area Weight (to region) ----------------------------------------------
# ~1m5s

wgt = np.sqrt(np.cos(np.radians(lat)))

#plt.plot(wgt)

flxwgt = flxa * wgt[None,:,None]
#slpwgt = slpa * wgt[None,:,None] # Don't apply area-weight to regressed variable

# Select region --------------------------------------------------------------
flxreg,lonr,latr = proc.sel_region(flxwgt.transpose(2,1,0),lon,lat,bboxeof)
nlonr,nlatr,_ = flxreg.shape
flxreg = flxreg.transpose(2,1,0) # Back to time x lat x lon

# Remove NaN Points [time x npts] --------------------------------------------
flxwgt = flxa.reshape((ntime,nlat*nlon)) # Dont use weighted variable
okdata,knan,okpts = proc.find_nan(flxwgt,0)
npts = okdata.shape[1]

flxreg = flxreg.reshape((ntime,nlatr*nlonr)) # Use lat weights for EOF region
okdatar,knanr,okptsr = proc.find_nan(flxreg,0)
nptsr = okdatar.shape[1]

nptsall = nlat*nlon
#slpwgt = slpwgt.reshape(ntime,nptsall) # Repeat for slp 
slpwgt = slpa.reshape(ntime,nptsall) # Repeat for slp 
okslp  = slpwgt#[:,okpts]
#
# Calculate Monthly Anomalies, change to [yr x mon x npts] -------------------
okdata = okdata.reshape((nyr,12,npts))
okdata = okdata - okdata.mean(0)[None,:,:]
okdatar = okdatar.reshape((nyr,12,nptsr)) # Repeat for region
okdatar = okdatar - okdatar.mean(0)[None,:,:]
okslp = okslp.reshape((nyr,12,nptsall))

# Prepare for eof anaylsis ---------------------------------------------------
eofall    = np.zeros((N_mode,12,nlat*nlon)) * np.nan
eofslp    = eofall.copy()
pcall     = np.zeros((N_mode,12,nyr)) * np.nan
varexpall = np.zeros((N_mode,12)) * np.nan
# Looping for each month
for m in tqdm(range(12)):
    
    # Calculate EOF
    datain = okdatar[:,m,:].T # [space x time]
    regrin = okdata[:,m,:].T
    slpin  = okslp[:,m,:].T
    
    eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
    
    # Standardize PCs
    pcstd = pcs / pcs.std(0)[None,:]
    
    # Regress back to dataset
    eof,b = proc.regress_2d(pcstd.T,regrin.T)
    eof_s,_ = proc.regress_2d(pcstd.T,slpin.T)
    
    # if debug:
    #     # Check to make sure both regress_2d methods are the same
    #     # (comparing looping by PC, and using A= [P x N])
    #     eof1 = np.zeros((N_mode,npts))
    #     b1  = np.zeros(eof1.shape)
    #     # Regress back to the dataset
    #     for n in range(N_mode):
    #         eof1[n,:],b1[n,:] = proc.regress_2d(pcstd[:,n],regrin)
    #     print("max diff for eof (matrix vs loop) is %f"%(np.nanmax(np.abs(eof-eof1))))
    #     print("max diff for b (matrix vs loop) is %f"%(np.nanmax(np.abs(b-b1))))
        
    # Save the data
    eofall[:,m,okpts] = eof
    eofslp[:,m,:] = eof_s
    pcall[:,m,:] = pcs.T
    varexpall[:,m] = varexp

# Flip longitude ------------------------------------------------------------
eofall = eofall.reshape(N_mode,12,nlat,nlon)
eofall = eofall.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofall = proc.lon360to180(lon,eofall.reshape(nlon,nlat,N_mode*12))
eofall = eofall.reshape(nlon,nlat,12,N_mode)
# Repeat for SLP eofs
eofslp = eofslp.reshape(N_mode,12,nlat,nlon)
eofslp = eofslp.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofslp = proc.lon360to180(lon,eofslp.reshape(nlon,nlat,N_mode*12))
eofslp = eofslp.reshape(nlon,nlat,12,N_mode)

#%% F
#%% (**) Save the results
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)

np.savez(savename,**{
    "eofall":eofall,
    "eofslp":eofslp,
    "pcall":pcall,
    "varexpall":varexpall,
    'lon':lon180,
    'lat':lat},allow_pickle=True)

#%% -------Analysis Section Below

#%% Load the data
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
elif mconfig == "PIC_FULL":
    mcname = "FULL-PIC"

bboxtext  = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr   = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename  = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)
ld        = np.load(savename,allow_pickle=True)

eofall    = ld['eofall']
eofslp    = ld['eofslp']
pcall     = ld['pcall']
varexpall = ld['varexpall']

lon = ld['lon']
lat = ld['lat']

lon360        = np.load(datpath+"../CESM_lon360.npy")
lon180,_        = scm.load_latlon()
#%% Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG)

spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP

N_modeplot = 5
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
#%% Preprocess net heat flux variable from above ~45sec
#% Apply landice mask to flx
# Flip flx anomalies to degrees west

st = time.time()
flxa          *= msk[None,:,:]
lon360        = np.load(datpath+"../CESM_lon360.npy")
lon180,flx180 = proc.lon360to180(lon360,flxa.transpose(2,1,0))
flx180        = flx180.reshape(nlon,nlat,nyr,12).transpose(0,1,3,2) # lon x lat x mon x year
print("Flipped SLP and FLX in %.2fs" % (time.time()-st))

#%% (**) Now calculate the variance explained using sum of squares
#% 1m7s

debug = False
test_mode_index = np.arange(0,N_mode,1)
varflx_EOF = np.zeros((nlon,nlat,12,N_mode)) # [lon x lat x mon x #_of_modes_cumulative]
for i in tqdm(range(N_mode)):
    eofN = (eofall[:,:,:,:(i+1)]**2).sum(-1)# Sum of squares along mode dimension
    varflx_EOF[:,:,:,i] = eofN.copy()
    
    if debug:
        if (i+1)%50 == 0:
            print(test_mode_index[:(i+1)])

# Calculate the ratio at each point
varflx_ori       = np.var(flx180,axis=-1,keepdims=True)
varflx_ratio     =  varflx_EOF / varflx_ori

# Compute variance ratio (non-cumulative) (at each point)
EOF_var = eofall.copy() **2
vratio_alone = (EOF_var)/varflx_ori # Calculate the ratio
eof_corr = eofall.copy()
eof_corr *= (1/vratio_alone)

# Save all the files
savename = datpath + "../NHFLX_EOF_Ratios_%s.npz" % mcname
if correction:
    savename = proc.addstrtoext(savename,correction_str)
np.savez(savename,**{
    'varflx_EOF':varflx_EOF,
    'varflx_ori':varflx_ori,
    'varflx_ratio':varflx_ratio,
    'varflx_ratio_alone':vratio_alone,
    'lon':lon180,
    'lat':lat
    })
#%% Corrected Flux with SLP (Save it all together, EOF and variance explained)

savename = "%sNHFLX_%s_%iEOFsPCs_%s_eofcorr2.npz" % (datpath,mcname,N_mode,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)
np.savez(savename,**{
    "varflx_ratio_alone":vratio_alone,
    "eofall":eof_corr,
    "eofslp":eofslp,
    "pcall":pcall,
    "varexpall":varexpall,  
    'lon':lon180,
    'lat':lat},allow_pickle=True)

#%% Load the file from above
savename     = datpath + "../NHFLX_EOF_Ratios_%s.npz" % mcname
if correction:
    savename = proc.addstrtoext(savename,correction_str)
ld           = np.load(savename)
varflx_EOF   = ld['varflx_EOF']
varflx_ori   = ld['varflx_ori']
varflx_ratio = ld['varflx_ratio']
vratio_alone = ld['varflx_ratio_alone']

eof_corr     = eofall * 1/vratio_alone

# # -----------------
# #%% (**) Save as netcdf
# # ------------------
# mode_number = np.arange(1,N_mode+1,1)
# months = np.arange(1,13,1)

# # [Mode x Month x Lat x Lon]
# varflx_EOF = varflx_EOF.transpose(3,2,1,0)
# varflx_ori   = varflx_ori.transpose(3,2,1,0) 
# varflx_ratio = varflx_ratio.transpose(3,2,1,0)

# # Make into data array
# vnames = ["var_NHFLX_by_EOF","var_NHFLX","var_NHFLX_ratio"]
# das = []
# for v,invar in enumerate([varflx_EOF,varflx_ori,varflx_ratio]):
    
#     if v == 1:
#         mode_in = np.array([1])
#     else:
#         mode_in = mode_number
#     da = xr.DataArray(invar,
#                 dims={'mode_number':mode_in,'months':months,'lat':lat,'lon':lon},
#                 coords={'mode_number':mode_in,'months':months,'lat':lat,'lon':lon},
#                 name = vnames[v]
#                 )
    
#     savename = datpath + "../NHFLX_200EOF_SLAB-PIC_%s.nc" % (vnames[v])
#     da.to_netcdf(savename,
#          encoding={vnames[v]: {'zlib': True,'_FillValue':-2147483647,"scale_factor":0.0001,'dtype':int}})
    
#     das.append(da)

# #%% (@@) Make some plots (compatible only if it the above section has been run)
# N_mode_plot = 4
# month       = 0
# vlm         = [0.5,1]

# fig,ax =plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
# ax = viz.add_coast_grid(ax,bbox=bbox)
# pcm = ax.pcolormesh(lon180,lat,varflx_ratio[N_mode_plot,month,...],vmin=vlm[0],vmax=vlm[1],cmap="magma")
# fig.colorbar(pcm,ax=ax)
# ax.set_title("Ratio: $var(NHFLX_{EOF})/var(NHFLX_{CESM})$, \n N=%i; Month=%i"%(N_mode_plot+1,month+1))
# savename = "%sNHFLX_Variance_Ratio_Map_Nmode%03i_month%02i.png" % (outpath,N_mode_plot+1,month+1)
# plt.savefig(savename,dpi=100)



# ------------------------------------------------
#%% Calculate/plot cumulative variance explained
# ------------------------------------------------
# mode number (x-axis) vs. %-variance explained (y-axis)
# separate lines used for each month

# Calculate cumulative variance at each EOF
cvarall = np.zeros(varexpall.shape)
for i in range(N_mode):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Plot Params
N_modeplot = 50
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)

for m in range(12):
    plt.plot(modes,cvarall[:,m]*100,label="Month %i"% (m+1),marker="o",markersize=4)
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
savename = "%s%s_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,mcname,N_modeplot,bboxtext)
if correction:
    savename = proc.addstrtoext(savename,correction_str)
plt.savefig(savename,dpi=150)

# ------------------------------------------------
# %% Make the Forcings...
# ------------------------------------------------

# -----------------------------
#% Save a select number of EOFs
# -----------------------------
eofcorr       = 0
N_mode_choose = 2

# Select the mode
eofforce      = eofall.copy()
eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
eofforce      = eofforce[:,:,:N_mode_choose,:]

# Prepare correction based on variance explained
if eofcorr == 1: # Correct based on EOF "basinwide" variance explained
    perc_in   = varexpall[:N_mode_choose,:].sum(0) # Sum variance for selected modes
    ampfactor = 1/perc_in # [Months]
    eofforce  *= ampfactor
elif eofcorr == 2: # Correct based on the LOCAL variance explained at each point...
    perc_in   = np.zeros([nlon,nlat,12]) * np.nan
    for im in range(12): # Loop for each nonth...
        perc_in[:,:,im] =  varflx_ratio[:,:,im,N_mode_choose]
    ampfactor = 1/perc_in
    eofforce  *= ampfactor[:,:,None,:] 

savenamefrc   = "%sflxeof_%ieofs_%s_eofcorr%i.npy" % (datpath,N_mode_choose,mcname,eofcorr)
if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)

# -----------------------------
#%% Save a given EOF
# -----------------------------

eofcorr       = 0
N_mode_choose = 0

# Select the mode
eofforce = eofall.copy()
eofforce = eofforce.transpose(0,1,3,2)   # lon x lat x pc x mon
eofforce = eofforce[:,:,N_mode_choose,:] # lon x lat x mon

# Prepare correction based on variance explained
if eofcorr == 1: # Correct based on EOF "basinwide" variance explained
    perc_in   = varexpall[N_mode_choose,:] # Sum variance for selected modes
    ampfactor = 1/perc_in # [Months]
    eofforce  *= ampfactor
elif eofcorr == 2: # Correct based on the LOCAL variance explained at each point...

    EOF_var = eofforce[:,:,:].copy() **2 # Variance of that EOF mode at each point
    perc_in = EOF_var/varflx_ori.squeeze() # Percentage of original variance
    ampfactor = 1/perc_in
    eofforce *= ampfactor

eofforce = eofforce[:,:,None,:] # lon x lat x pc x mon

savenamefrc   = "%sflxeof_EOF%i_%s_eofcorr%i.npy" % (datpath,N_mode_choose+1,mcname,eofcorr)
if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)

np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)



#%% (@@) Plot the above EOFs
# Note this is also the bottom of figure from SM Paper Outline

darkmode = True
if darkmode:
    plt.style.use('dark_background')
    dfcol = "black"
else:
    plt.style.use('default')
    dfcol = 'k'

notitle = False

ids = [11,0,1]
clvl = np.arange(-60,65,5)
plvl = np.arange(-800,900,100)/100

titles = ["EOF 1 (NAO, %.2f" % (varexpall[0,ids].mean()*100) + "% Variance)" ,
          "EOF 2 (EAP, %.2f" % (varexpall[1,ids].mean()*100) + "% Variance)" ]
#fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,7))
fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4),
                       constrained_layout=True)

spid = 0
for k in range(2):
    blabel = [1,0,0,1]
    
    if k == 1:
        blabel = [0,0,0,1]
    ax = axs[k]
    ax.set_facecolor('white')
    ax  = viz.add_coast_grid(ax,bbox=bbox,line_color="gray",fill_color='gray',blabels=blabel)
    pcm = ax.contourf(lon180,lat,eofall[:,:,ids,k].mean(-1).T,levels=clvl,cmap='RdBu_r')
    cl  = ax.contour(lon180,lat,eofslp[:,:,ids,k].mean(-1).T/100,levels=plvl,colors=dfcol,linewidths=.7)
    ax.clabel(cl,fontsize=10)
    if notitle is not True:
        ax.set_title(titles[k])
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
    spid += 1


cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.015,pad=0.01)
#cb.set_label("$Q_{net}$ ,Contour = 5 $Wm^{-2} \,\sigma_{PC}^{-1}$")
cb.set_label("$Q_{net}$ ($Wm^{-2}$ $\sigma_{PC}^{-1}$)")
#plt.suptitle("SLP (Contours), Interval = 100 hPa",x=0.6, y=.955,fontsize=14)
if notitle is False:
    plt.suptitle("$Q_{net}$ Forcing Pattern (Colors) and SLP (Contours, Interval = 100 hPa), DJF Mean",fontsize=12)

#fig.text(0.07, 0.5, mcname, va='center', rotation='vertical',fontsize=12)

plt.savefig("%sEOF1_EOF2_uncorrected_%s.png"%(outpath,mcname),dpi=150,bbox_inches='tight')

#%% Plot Select EOF and PC
k   = 1
ids = [0,]

replace_wn = False

fig = plt.figure(constrained_layout=False, facecolor='w',figsize=(12,4))

gs = fig.add_gridspec(nrows=1, ncols=3, left=.02, right=1,
                      hspace=.075, wspace=0.25)

# Plot the Map
ax0 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax  = ax0
ax  = viz.add_coast_grid(ax,bbox=bbox,line_color=dfcol)
pcm = ax.contourf(lon180,lat,eofall[:,:,ids,k].mean(-1).T,levels=clvl,cmap='RdBu_r',extend='both')
cl  = ax.contour(lon180,lat,eofslp[:,:,ids,k].mean(-1).T,levels=plvl,colors=dfcol,linewidths=.7)
ax.clabel(cl,fontsize=10)
ax.set_title("EOF %i Pattern (SLP, Contour = 100 hPa)" % (k+1))
cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')
cb.set_label("$Q_{net}$ ,Contour = 5 $Wm^{-2}$")

#annotate_axes(ax0, 'ax0')
ax1 = fig.add_subplot(gs[0,1:3])
if replace_wn:
    ax1.plot(np.random.normal(0,1,pcall.shape[-1]),color='gray')
else:
    ax1.plot(pcall[k,ids,:].mean(0))
ystd = np.std(pcall[k,ids,:]*3)
ax1.set_ylim([-ystd,ystd])
ax1.set_title("%s PC %i (Percent Variance Explained: %.2f" % (mons3[ids[0]],k+1,varexpall[1,ids].mean()*100) + "%)" )
ax1.set_xlim([0,pcall.shape[-1]])
ax1.grid(True,ls='dotted')
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("PC %i"%(k+1))
ax1.axhline([0],ls='dashed',color='k')

#annotate_axes(ax1, 'ax1')
#annotate_axes(ax2, 'ax2')
#fig.suptitle('Manual gridspec with right=0.75')
#plt.show()
if replace_wn:
    savename = "%s%s_EOF%i_Pattern_WhiteNoise.png" % (outpath,mcname,k+1)
else:
    savename="%s%s_EOF%i_Pattern_PC.png" % (outpath,mcname,k+1)

plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Quickly check what is going on with EOF SLP

N = 0
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4))

for k in range(3):
    
    im = ids[k]
    ax = axs[k]
    ax  = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.contourf(lon180,lat,eofall[:,:,im,N].T,levels=clvl,cmap='PRGn')
    cl  = ax.contour(lon180,lat,eofslp[:,:,im,N].T,levels=plvl,colors='k',linewidths=.7)
    ax.clabel(cl,fontsize=10)
    ax.set_title("Mode " + str(N+1)+"; Month"+str(im))



#%% Plot correction ratio

ids = [11,0,1]
clvl = np.arange(0,1.5,.1)

titles = ["EOF 1 (NAO, %.2f" % (varexpall[0,ids].mean()*100) + "% Variance)" ,
          "EOF 2 (EAP, %.2f" % (varexpall[1,ids].mean()*100) + "% Variance)" ]
fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,7))

for k in range(2):
    ax = axs[k]
    ax  = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.contourf(lon180,lat,vratio_alone[:,:,ids,k].mean(-1).T,levels=clvl,cmap=cmocean.cm.balance)
    cl  = ax.contour(lon180,lat,vratio_alone[:,:,ids,k].mean(-1).T,levels=clvl,colors='k',linewidths=.7)
    ax.clabel(cl,fontsize=10) 
    ax.set_title(titles[k])


cb = fig.colorbar(pcm,ax=axs.flatten())
cb.set_label("$Q_{net}$ ,Interval = 5 $W/m^2$ per $\sigma_{PC}$")
#plt.suptitle("SLP (Contours), Interval = 100 hPa",x=0.6, y=.955,fontsize=14)

plt.savefig("%sEOF1_EOF2_vratio_%s.png"%(outpath,mcname),dpi=150,bbox_inches='tight')


#%%

ids = [11,0,1]
clvl = np.arange(-600,650,50)
plvl = np.arange(-800,900,100)

titles = ["EOF 1 (NAO, %.2f" % (varexpall[0,ids].mean()*100) + "% Variance)" ,
          "EOF 2 (EAP, %.2f" % (varexpall[1,ids].mean()*100) + "% Variance)" ]
fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,7))

for k in range(2):
    ax = axs[k]
    ax  = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.contourf(lon180,lat,eof_corr[:,:,ids,k].mean(-1).T,levels=clvl,cmap=cmocean.cm.balance)
    cl  = ax.contour(lon180,lat,eofslp[:,:,ids,k].mean(-1).T,levels=plvl,colors='k',linewidths=.7)
    ax.clabel(cl,fontsize=10)
    ax.set_title(titles[k])


cb = fig.colorbar(pcm,ax=axs.flatten())
#cb.set_label("$Q_{net}$ ,Interval = 5 $W/m^2$ per $\sigma_{PC}$")
cb.set_label("$Q_{net}$ ($W/m^2$ per $\sigma_{PC}$)")
plt.suptitle("SLP (Contours), Interval = 100 hPa",x=0.6, y=.955,fontsize=14)

plt.savefig("%sEOF1_EOF2_corrected_%s.png"%(outpath,mcname),dpi=150,bbox_inches='tight')

# ------------------------------------
#%% Find indices of variance threshold
# ------------------------------------
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


#ytk = np.arange(0,37,2)
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


plt.savefig("%s%s_NHFLX_EOFs%i_%s_NumEOFs_%ipctvar_bymon.png"%(outpath,mcname,N_mode,bboxtext,vthres*100),dpi=150,bbox_inches='tight')

# -------------------------------------------------------------------------
#%% Save outptut as forcing for stochastic model, variance based threshold
# -------------------------------------------------------------------------

# Calculate correction factor
eofcorr  = 2

nlon,nlat,_,_ = varflx_ratio.shape
if eofcorr == 1: # Flat variance-based correct
    ampfactor = 1/vthres#thresperc
elif eofcorr ==2: # Point[wise variance-based correction
    thresperc = np.zeros([nlon,nlat,12])*np.nan
    for im in range(12):
        thresidx = thresid[im]
        thresperc[:,:,im] = varflx_ratio[:,:,im,thresidx] # Index for each month
    ampfactor = 1/thresperc
else:
    ampfactor = 1

eofforce = eofall.copy() # [lon x lat x month x pc]
cvartest = cvarall.copy()
for i in range(12):
    # Set all points after crossing the variance threshold to zero
    stop_id = thresid[i]
    print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
    eofforce[:,:,i,stop_id+1:] = 0
    cvartest[stop_id+1:,i] = 0
eofforce = eofforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

if eofcorr == 1:
    eofforce *= ampfactor
elif eofcorr == 2:
    ampfactor = ampfactor[:,:,None,:] # []
    eofforce *= ampfactor # Broadcast along EOF dimension

# Cut to maximum EOF
nmax = thresid.max()
eofforce = eofforce[:,:,:nmax+1,:]
savenamefrc = "%sflxeof_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,mcname,eofcorr)

if correction:
    savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
np.save(savenamefrc,eofforce)

#%% Test Plot (Summed EOF)

eofsum = np.nansum(eofforce,2)


clvl = np.linspace(-75,75,100)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bbox)
pcm = ax.contourf(lon,lat,np.nanmean(eofsum,2).T,levels=clvl,cmap=cmocean.cm.balance)

fig.colorbar(pcm,ax=ax)

#%% Test Plot (Given Mode)

eofchoose=eofforce[:,:,0,:]

clvl = np.linspace(-75,75,100)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bbox)
pcm = ax.contourf(lon,lat,np.nanmean(eofchoose,2).T,levels=clvl,cmap=cmocean.cm.balance)

fig.colorbar(pcm,ax=ax)
#%% Test plot maps
i = 0
fig,ax = plt.subplots(2,1)
ax[0].pcolormesh(eofforce[:,:,stop_id,i].T),plt.colorbar()
ax[1].pcolormesh(eofforce[:,:,stop_id+1,i].T),plt.colorbar()

#%% Test Plot Cumulative % for indexing chec
N_modeplot = 20
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)
for m in range(12):
    plt.plot(modes,cvartest[:,m]*100,label="Month %i"% (m+1))
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
#plt.savefig("%sSLAB-PIC_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,N_modeplot,bboxtext),dpi=150)

#%% Plot Amp factor (Paper Outline)

# Plot for the paper
bboxplot = [-100,20,0,80]
bbox_NA  = [-80,0 ,10,65]
clvls    = np.arange(1,1.55,.05)
#clvls    = np.arange(1,1.55,.05)
fig,ax   = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4))
ax       = viz.add_coast_grid(ax=ax,bbox=bboxplot)
pcm      = ax.contourf(lon180,lat,ampfactor.mean(-1).squeeze().T,levels=clvls,cmap="Oranges")
p        = ax.pcolormesh(lon180,lat,ampfactor.mean(-1).squeeze().T,vmin=clvls[0],vmax=clvls[-1],cmap="Oranges",zorder=-1)
cb       = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.055)
cl       = ax.contour(lon180,lat,ampfactor.mean(-1).squeeze().T,levels=clvls,colors='k',linewidths=.2)
#ax.axhline(10,ls='dashed',color='k')
ax,ll = viz.plot_box(bbox_NA,ax=ax,leglab="AMV",
                     color="k",linestyle="dashed",linewidth=2,return_line=True)
ax.clabel(cl,levels=np.arange(1,1.5,.1),fontsize=8)
#cb.set_label(r"$\frac{q}{Q_{net}}$")
ax.set_title(r"Annual Mean Variance Correction ($Q_{net}/Q_{EOF}$)"+ "\nContour Interval = 0.05")
plt.savefig("%sRatio_varcorr_local_%s.png"% (outpath,mcname),dpi=150,bbox_inches='tight')


# ------------------------------------------------------------
# %% Reload and save a monthly version of one of the outputs
# ------------------------------------------------------------

# Trying to do something fancy, ignore this for now lets hard code it...
# mons1   = [i[0] for i in mons3]
# monsave = [11,0,1]
# monname = np.array(mons1)[monsave].tolist()
# monname = [''.join(i) for i in monname]
monids   = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
monnames = ("DJF","MAM","JJA","SON")
for s in tqdm(range(4)):
    # Calculate seasonal average
    eofseas = np.mean(eofforce[:,:,:,monids[s]],-1,keepdims=True)
    
    # Save the output
    savenamefrc = "%sflxeof_%03ipct_%s_eofcorr%i_%s.npy" % (datpath,vthres*100,mcname,eofcorr,monnames[s])
    if correction:
        savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
    print("Saving to %s"%savenamefrc)
    np.save(savenamefrc,eofseas)
#%%Do some analysis/visualization
# Plot the map of ratio  NHFLX(n-percent EOF)/NHFLX_SLAB
plot_contours = True

bboxplot = [-90,0,0,75]
vlm = [0.75,1.0]
interval = 0.025
levels = np.arange(vlm[0],vlm[-1]+interval,interval)

fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(16,18))
for im in range(12):
    ax = axs.flatten()[im]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    if plot_contours:
        pcm1 = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
                            cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
        pcm = ax.contourf(lon180,lat,thresperc[:,:,im].T,
                          levels=levels,cmap='inferno')
        cl = ax.contour(lon180,lat,thresperc[:,:,im].T,
                          levels=levels,colors='k',linewidths=0.50)
        ax.clabel(cl)

    else:
        pcm = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
                            cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
    ax.set_title("%s" % (mons3[im]))

fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.01,anchor=(1.5,0.7))
plt.savefig("%s%s_NHFLX_EOFs_vratio_%03ipercEOFs_allmon.png"%(outpath,mcname,vthres*100),dpi=150,bbox_inches='tight')

#%% Plot Annual average of above
vlm = [0.75,1.0]
interval = 0.025
levels = np.arange(vlm[0],vlm[-1]+interval,interval)

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4))

ax = viz.add_coast_grid(ax,bbox=bboxplot)
if plot_contours:
    pcm1 = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
                        cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
    pcm = ax.contourf(lon180,lat,thresperc.mean(2).T,
                      levels=levels,cmap='inferno')
    cl = ax.contour(lon180,lat,thresperc.mean(2).T,
                      levels=levels,colors='k',linewidths=0.50)
    ax.clabel(cl)

else:
    pcm = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
                        cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
ax.set_title("Ratio of $var(NHFLX_{EOF})/var(NHFLX_{SLAB})$,\n %03i perc. variance explained., Ann. Mean" % (vthres*100))
fig.colorbar(pcm,ax=ax,pad=0.01)
plt.savefig("%s%s_NHFLX_EOFs_vratio_%03ipercEOFs_annmean.png"%(outpath,mcname,vthres*100),dpi=200,bbox_inches='tight')


#%%  Plot Mode vs. % Variance Exp (for each Month)
xtk = np.arange(1,N_modeplot+1,1)
fig,ax = plt.subplots(1,1)
for m in range(12):
    plt.plot(xtk,varexpall[:N_modeplot,m]*100,label="Month %i"% (m+1),marker="o")
ax.legend()
ax.set_ylabel("% Variance Explained")
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Perc. Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xticks(xtk)
plt.savefig("%s%s_NHFLX_EOFs_%s_ModevVariance_bymon.png"%(outpath,mcname,bboxtext),dpi=150)


# Same as above, but cumulative plot
#cvarallplot = np.zeros(varexpall.shape)
#for i in range(N_modeplot):
#    cvarallplot[i,:] = varexpall[:i+1,:].sum(0)

#%% Plot Month vs. Total % Variance Exp for 10 EOFs)
N_modeplot=5
fig,ax = plt.subplots(1,1)
ax.bar(mons3,varexpall[:N_modeplot,:].sum(0)*100,color='cornflowerblue',alpha=0.7)
ax.set_title("Total Percent Variance Explained by first %i EOFs" % N_modeplot)
ax.set_ylabel("% Variance Explained")
ax.set_ylim([0,100])
ax.set_yticks(np.arange(0,110,10))
ax.grid(True,ls='dotted')
plt.savefig("%s%s_NHFLX_EOFs_%s_TotalVariance_First%iEOFs_bymon.png"%(outpath,mcname,bboxtext,N_modeplot),dpi=150)

#%% Plot Net Heat Flux EOF Patterns

vlim = [-30,30]
#slplevels = np.arange()
blabels=[1,0,0,1]
N_modeplot = 5
slp_int = np.arange(-400,450,50)
slp_lab = np.arange(-400,500,100)

bboxplot = bbox.copy()
for i in range(2):
    if bbox[i] > 180:
        bboxplot[i] -= 360

for n in tqdm(range(N_modeplot)):
    plotord = np.roll(np.arange(0,12,1),1)
    fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,12))
    for p,m in enumerate(plotord):
        ax = axs.flatten()[p] # Plot Index
        ax.set_title("%s (%.1f" % (mons3[m],varexpall[n,m]*100) + "%)",fontsize=10)
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabels)
        
        # Plot NHFLX
        pcm = ax.pcolormesh(lon,lat,eofall[:,:,m,n].T,vmin=vlim[0],vmax=vlim[-1],cmap="RdBu_r")
        
        # Plot SLP and Labels
        cl = ax.contour(lon,lat,eofslp[:,:,m,n].T,levels=slp_int,colors='k',linewidths=1)
        ax.clabel(cl,levels=slp_lab,fontsize=10)
        
        # Scrap
        #pcm = ax.contourf(lon,lat,eofslp[:,:,m,n].T,levels=slp_int,cmap=cmocean.cm.balance,linewidths=1.)
        #fig.colorbar(pcm,ax=ax,fraction=0.035)
    
    fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.05,anchor=(1.75,0.7))
    plt.suptitle("NHFLX EOF %i (CESM1-SLAB) (W/$m^2$ per $\sigma_{PC}$) \n SLP Contour Interval: 50 mb" % (n+1),fontsize=14)
    fig.subplots_adjust(top=0.90)
    plt.savefig("%s%s_NHFLX_EOFs_%s_EOF%iPattern_bymon.png"%(outpath,mcname,bboxtext,n+1),dpi=150)

#%% Seasonally Averaged Plots


sid = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
snm = ["DJF","MAM","JJA","SON"]
#blabels=[0,]

n = 0
fig,axs = plt.subplots(3,4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(14,8))
for n in range(3):
    for p in range(4):
        ks = sid[p]
        ax = axs[n,p]
        ax.set_title("EOF %i: %s (%.1f" % (n+1,snm[p],varexpall[n,ks].mean()*100) + "%)",fontsize=10)
        if p == 0:
            blabels=[1,0,0,0]
        elif n == 2:
            blabels=[0,0,0,1]
        elif (n == 2) and (p ==0):
            blabels=[1,0,0,1]
        else:
            blabels=[0,0,0,0]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabels)
        
        # Plot NHFLX
        pcm = ax.pcolormesh(lon,lat,eofall[:,:,ks,n].mean(2).T,vmin=vlim[0],vmax=vlim[-1],cmap="RdBu_r")
        
        # Plot SLP and Labels
        cl = ax.contour(lon,lat,eofslp[:,:,ks,n].mean(2).T,levels=slp_int,colors='k',linewidths=1)
        ax.clabel(cl,levels=slp_lab,fontsize=10)
plt.savefig("%s%s_NHFLX_EOFs1-3_EOFPattern_%s_seasavg.png"%(outpath,mcname,bboxtext),dpi=150)




#%% Plot PCs and Spectra

dt  = 365*24*3600
pct = 0.10
nsmooth = 50

tsin = []
for n in range(N_mode):
    for m in range(12):
        tsin.append(pcall[n,m,:])
        
freqs = []
specs = []

specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(tsin,nsmooth,pct,opt=1,dt=dt,clvl=[.95])

#%% Calculate contribution of each EOF to AMV

# Regional Analysis Settings
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")        # Region Names
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

rids = [0,1,3]

aavg_eof = np.zeros((N_mode,12,len(rids)))*np.nan
for r in range(len(rids)):
    rid = rids[r]
    rbbox = bboxes[rid]
    for m in range(12):
        ineof = eofall[:,:,m,:]
        
        eofaa = proc.sel_region(ineof,lon,lat,rbbox,reg_avg=1,awgt=1)
        aavg_eof[:,m,r] = eofaa.copy()

step = 1
ylm = [-30,30]
xlm = [0,20]
xtk = np.arange(xlm[0],xlm[-1]+step,step)

m = 0
for m in range(12):
    fig,ax = plt.subplots(1,1,figsize=(6,3))
    for r in range(len(rids)):
        ax.plot(np.arange(1,N_mode+1),aavg_eof[:,m,r],label="%s AMV Index"%(regions[r]),marker="o")
    ax.legend()
    ax.set_xlim(xlm)
    ax.set_ylim(ylm)
    ax.set_xticks(xtk)
    ax.grid(True)
    
    ax.set_ylabel("EOF Area Average (W/m2/$\sigma_{PC}$)")
    ax.set_xlabel("Mode")
    ax.set_title("%s, Area Average of EOF)"% (mons3[m]))
    plt.savefig("%sEOF_NHFLX_Area_Avg_Mon%02d.png"%(outpath,m+1),dpi=200)
# -------------------------
#%% Plot Spectra of the PCs
# -------------------------

im      = 0
N_mode  = 0
nsmooth = 60
pct     = 0.10

