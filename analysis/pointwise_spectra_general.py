#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generalized version of pointwise_spectra (applies to variables beyond SST).
Works for CESM1 Preindustrial Control, SLAB and FULL

Created on Fri Jul 15 14:35:52 2022

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import numpy as np
import xarray as xr
from tqdm import tqdm 
import time

import cartopy.crs as ccrs
#%%

# Spectra Parametsrs
pct        = 0.10
nsmooth    = 30
smoothname = "smooth%03i-taper%03i" % (nsmooth,pct*100)



figpath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220305/'
proc.makedir(figpath)


# Postprocess Continuous SM  Run
# ------------------------------
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
fnames     = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
snames     = ["spectra_%s_Fprime_rolln0_ampq0_method5_dmp0_run2%02d.nc" % (smoothname,i) for i in range(10)]
mnames     = ["constant h","vary h","entraining"] 

# Postproess Continuous CESM Run
# ------------------------------
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
#fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FULL","SLAB"] 
vname      = "NHFLX"
bbox       = [-80,0,0,65] # Region Crop
snames     = ["%s_spectra_%s_PIC-%s.nc" % (vname,smoothname,mnames[i]) for i in range(2)]






# Other Params
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

# Save Variable
save_varreg = True

#%% Compute the spectra

for mc,mconfig in enumerate(mnames):
    # Load and Preprocess Variable
    # Input: Raw Variable in DataSet       [Mon x Yr x Lat x Lon] 
    # Output: Anomalized, Masked Variable :[Model x Lon x Lat x Time]
    # ----------------------------
    print("Loading netCDF")
    # Open the dataset
    sst_fn = "%s%s_PIC_%s.nc" % (datpath,vname,mconfig)
    ds  = xr.open_dataset(sst_fn)
    
    if ~np.any(ds.lon.values < 0): # No negative values (degrees East)
        # Check longitude, flip if necessary
        ds = proc.lon360to180_ds(ds,lonname='lon')
    else:
        print("Warning, no land-ice mask applied because lon is degrees W")
        
    # Select region
    dsreg  = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    
    # Read out values
    lon    = dsreg.lon.values
    lat    = dsreg.lat.values
    outvar = dsreg[vname].values
    
    # Preprocess (Remove Seasonal Cycle)
    if len(outvar.shape) < 4:
        print("Dimensions Less than 4. Separating month and year")
        clim,outvar =proc.calc_clim(outvar,0,returnts=1)
    else: # Already 4D (yr x mon x lat x lon)
        clim = outvar.mean(0)
    anom = outvar - clim[None,:,:,:]
        
    
    # Apply land/ice mask
    limaskname = datpath + "../CESM-%s_landicemask360.npy" % (mconfig)
    limask = np.load(limaskname)
    latg    = ds.lat.values
    lon360 = np.load(datpath+ "../CESM_lon360.npy")
    lon180,limask = proc.lon360to180(lon360,limask.T)
    limask_reg,_,_ = proc.sel_region(limask,lon180,latg,bbox)
    anom *= limask_reg.T[None,None,:,:]
    
    # Recombine Mon x Year
    if len(anom.shape) > 3:
        nyr,_,nlat,nlon = anom.shape
        anom = anom.reshape(nyr*12,nlat,nlon) # ]time x lat x lon
    
    # Check dimensions (Should be model x lon x lat x time)
    anom = anom.transpose(2,1,0)
    if len(anom.shape) < 4:
        print("Dimensions Less than 4. Adding Extra at Start")
        anom = anom[None,...]
    nmod,nlon,nlat,ntime = anom.shape
    
    
    # Calculate Spectra
    # -----------------
    # Compute Size of Frequency (take from yo_spec)
    fmax  = 0.5
    df    = 1/ntime
    freq  = np.arange(1/ntime,fmax+df,df)
    nfreq = len(freq)
    
    # Preallocate 
    specs = np.zeros((nmod,nlon,nlat,nfreq))*np.nan 
    dofs  = np.zeros((nmod,nlon,nlat,nfreq))*np.nan # Not sure if I want to save all of this
    r1s   = np.zeros((nmod,nlon,nlat,nfreq))*np.nan 
    
    # Compute the spectra (Pointwise)
    for mid in range(nmod):
        for o in tqdm(range(nlon)): # Loop for longitude
            
            for a in range(nlat): # Loop for latitude
                
                # Get Input
                inssts   = [anom[mid,o,a,:],]
                nsmooths = [nsmooth,]
                
                # Skip if NaNs are Present 
                if np.any(np.isnan(anom[mid,o,a,:])):
                    continue
                
                # Compute Spectra and save
                spec,freqs,_,dof,r1 = scm.quick_spectrum(inssts,nsmooths,pct)
                specs[mid,o,a,:] = spec[0].copy()
                dofs[mid,o,a,:]  = dof[0].copy()
                r1s[mid,o,a,:]   = r1[0].copy()
                
    # Now work on saving it
    if "CESM" in sst_fn:
        mname_in = [mnames[mc],]
    else:
        mname_in  = mnames
    dims = {'models'          : mname_in,
            'lon'             : lon,
            'lat'             : lat,
            'frequency'       : freqs[0]
            }
    
    das = []
    varnames = ("spectra","dofs","r1s")
    outvars  = [specs,dofs,r1s]
    for v,name in enumerate(varnames):
        da = xr.DataArray(outvars[v],
                    dims=dims,
                    coords=dims,
                    name = name,
                    )
        if v == 0:
            ds = da.to_dataset() # Convert to dataset
        else:
            ds = ds.merge(da) # Merge other datasets
            
        # Append to list if I want to save separate dataarrays
        das.append(ds)
    
    #% Save as netCDF
    # ---------------
    #st = time.time()
    encoding_dict = {name : {'zlib': True} for name in varnames} 
    savename = outpath+snames[mc]
    print("Saving as " + savename)
    ds.to_netcdf(savename,
             encoding=encoding_dict)
    #print("Saved in %.2fs" % (time.time()-st))
# ------------------
#%% Analysis Section
# ------------------
# Currently supports analysis of continuous data
debug = True

# Find Frequency (lowerthres <= f <= thresval)
thresval   = 1/20 # 20 years (Multidecadal)
lowerthres = 0 # 
dtplot     = 3600*24*365


# Load the data (Just the Spectra)
snames_full = [outpath+sname for sname in snames]
if "PIC" in snames_full[0]:
    dsall = []
    for i in range(2):
        ds       = xr.open_mfdataset(snames_full[i])
        dsall.append(ds)
        
    
    # Load Spectra
    st = time.time()
    cfreqs       = []
    cspecs       = []
    modelnames   = []
    for i in range(2):
        cspecs.append(dsall[i].spectra.values)
        cfreqs.append(dsall[i].frequency.values)
        modelnames.append(dsall[i].models.values)
    #specs       = ds.spectra.values # [model x lon x lat x freq]
    
    #specsds     = dsmean.spectra
    print("Loaded data in %.2fs" % (st-time.time()))
        
else:
    dsall       = xr.open_mfdataset(snames_full
                                    ,combine="nested",concat_dim="run")



    # Take mean of all runs
    st = time.time()
    dsmean      = dsall.mean('run')
    freq        = dsmean.frequency.values
    specs       = dsmean.spectra.values # [model x lon x lat x freq]
    modelnames  = dsmean.models.values
    #specsds     = dsmean.spectra
    print("Loaded data in %.2fs" % (st-time.time()))


if debug:
    tids = proc.calc_specvar(freq,specs,thresval,dtplot,
                             return_thresids=True,lowerthres=lowerthres)
    plt.plot(freq*dtplot,tids),plt.xlim([0,1])


# Compute Spectra sum under selected frequencies
specsum = proc.calc_specvar(freq,specs,thresval,dtplot,lowerthres=lowerthres)

#%% Plot Spectra for a point


klon,klat = proc.find_latlon(-30,50,lon,lat)

fig,ax = plt.subplots(1,1)
for im in range(3):
    ax.plot(freq*dtplot,specs[im,klon,klat,:]/dtplot,label=modelnames[im])

ax.set_xlim([0,0.5])
ax.axvline(thresval,color="k")

#ax.axhline(0)
ax.legend()



#%% Part 1, Lets see the role of entrainment


# Find Frequency (lowerthres <= f <= thresval)
#thresval   = 1/5 # 20 years (Multidecadal)

periods    = [2,3,4,5,6,7,8,9,10,15,20,30,40,50]
thresvals  = [1/x for x in periods]
lowerthres = 0 # 
dtplot     = 3600*24*365

plotentrain  = True # False to examine effect of adding MLD variations


for thresval in tqdm(thresvals):
    
    # Compute Spectra sum under selected frequencies
    specsum = proc.calc_specvar(freq,specs,thresval,dtplot,lowerthres=lowerthres)
    
    # Difference in variance < 1/20 years
    if plotentrain:
        entraindiff = specsum[2,...] - specsum[1,...]
        plotvar     = entraindiff
        vname       = "Entrain_VaryH"
        vname_title = "$\sigma^2_{entrain}$ - $\sigma^2_{vary \, h}$"
    else:
        mlddiff     = specsum[1,...] - specsum[0,...]
        plotvar     = mlddiff
        vname       = "VaryH_ConstantH"
        vname_title = "$\sigma^2_{vary \, h}$ - $\sigma^2_{constant \, h}$"
    
    
    cint        = np.arange(-.1,.11,0.01)
    fig,ax      = plt.subplots(1,1,constrained_layout=True,
                               subplot_kw={'projection':ccrs.PlateCarree()})
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray')
    #pcm        = ax.pcolormesh(lon,lat,entraindiff.T,cmap='cmo.balance',vmin=-.15,vmax=.15)
    pcm         = ax.contourf(lon,lat,plotvar.T,cmap='cmo.balance',levels=cint,extend='both')
    cl          = ax.contour(lon,lat,plotvar.T,levels=[0,1],colors="k",lw=1.5)
    ax.clabel(cl)
    cb = fig.colorbar(pcm,ax=ax)
    cb.set_label("SST Variance Difference ($K^2$)")
    ax.set_title("%s @ Frequencies < %i years$^{-1}$"% (vname_title,1/thresval))
    
    savename = "%s%s_Difference_period%02d.png" %  (figpath,vname,1/thresval)
    plt.savefig(savename,dpi=150)

#%% Load Stochastic Model Inputs

# Use the function used for sm_rewrite.py
frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
input_path = datpath + "../model_input/"
lagstr     = "lag1"
method     = 5 # Damping Correction Method
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
long,latg,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt   = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt   = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]

# Compute Specific Region
reg_sel  = [lon[0],lon[-1],lat[0],lat[-1]]
#reg_sel   = [-80,0,30,65]
inputs = [h,kprevall,dampingslab,dampingfull,alpha,alpha_full,hblt]
outputs,lonr2,latr2 = scm.cut_regions(inputs,long,latg,reg_sel,0)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs

npts = len(lon) * len(lat)
#%% Next, Try to Plot a given threshold

bboxinset = [-80,0,30,60]
# Compute Spectra sum under selected frequencies
thresval    = 1/20 # 1/year
lowerthres  = 0 # 
dtplot      = 3600*24*365
specsum     = proc.calc_specvar(freq,specs,thresval,dtplot,lowerthres=lowerthres)
entraindiff = specsum[2,...] - specsum[1,...]
plotvar = entraindiff

vlm     = [-0.1,0.1]

vthreses = np.arange(0.02,0.09,0.01) # [0.08,0.10,0.12,0.14,0.16,0.18,0.20]
ylabs = ("MLD (m)",
         "Damping ($Wm^{-2}K^{-1}$)",
         "Forcing ($Wm^{-2}$)")

palpha = 0.05
salpha = 0.3 # Seleccted Alpha
# Just Plot Everything
xlims  = [0,11]

for vthres_sel in tqdm(vthreses):

    kthres     = plotvar > vthres_sel
    kthres     = kthres.flatten()
    
    fig,axs = plt.subplots(4,1,figsize=(8,8),constrained_layout=True,sharex=False)
    
    # Plot MLD
    ax = axs[0]
    mlins = ax.plot(mons3,h.reshape(npts,12).T,alpha=palpha,color="k")
    ax.set_ylabel(ylabs[0])
    mlins_sel = ax.plot(mons3,h.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")
    ax.set_xlim(xlims)
    ax.grid(True,ls='dotted',alpha=0.25)
    
    ax = axs[1]
    dlins = ax.plot(mons3,damping.reshape(npts,12).T,alpha=palpha,color="r")
    dlins2 = ax.plot(mons3,dampingfull.reshape(npts,12).T,alpha=palpha,color="b")
    ax.set_ylabel(ylabs[1])
    dlins2_sel = ax.plot(mons3,dampingfull.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")
    ax.set_xlim(xlims)
    ax.grid(True,ls='dotted',alpha=0.25)
    
    ax = axs[2]
    flins = ax.plot(mons3,np.linalg.norm(alpha,axis=2).reshape(npts,12).T,alpha=palpha,color="r")
    flins_2 = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12).T,alpha=palpha,color="b")
    ax.set_xlim(xlims)
    ax.set_ylabel(ylabs[2])
    
    flins_sel = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12)[kthres,:].T,
                        alpha=salpha,color="yellow")
    ax.grid(True,ls='dotted',alpha=0.25)
    
    ax = axs[3]
    ax.axis('off')
    
    # Add Locator
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    ax2 = axs[3]
    left, bottom, width, height = [0.18, -0.45, 0.70, 1]
    ax2 = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
    ax2 = viz.add_coast_grid(ax2,bbox=bboxinset,fill_color='gray',ignore_error=True)
    pcm = ax2.pcolormesh(lon,lat,plotvar.T,
                    cmap='cmo.balance',vmin=vlm[0],vmax=vlm[1])
    cl  = ax2.contour(lon,lat,plotvar.T,levels=[vthres_sel,],colors="k")
    ax2.clabel(cl)
    #viz.plot_mask(lon,lat,dmsks[mid],ax=ax2,markersize=2,color='k',marker="o")
    fig.colorbar(pcm,ax=ax2,fraction=0.02)
    
    plt.suptitle("Red (SLAB), Blue (FULL), Yellow ($\sigma^2_{SST}$ >  %.2f $K^{2}$)"% (vthres_sel))
    plt.savefig("%sEntrain_Diff_Params_thres%iyrs_thresval%.2f_wmap.png"% (figpath,int(1/thresval),vthres_sel),dpi=150,bbox_inches='tight')
# add spectra?


# ----------------------------------------------------------
#%% Plot all the spectra/autocorrelation over a given region
# ----------------------------------------------------------

cid     = 0
bbsel = [-80, -40, 20, 40]

bbstr = "lon%ito%i_lat%ito%i" % (bbsel[0],bbsel[1],bbsel[2],bbsel[3])

# Visualize Spectra in this region
dsreg = dsall[cid].sel(lon=slice(bbsel[0],bbsel[1]),lat=slice(bbsel[2],bbsel[3]))
rspec = dsreg.spectra.values
rlon  = dsreg.lon.values
rlat  = dsreg.lat.values
rfreq = dsreg.frequency.values

#%

# Spaghetti Plot

dtplot = 3600*24*365


xtks = [1/100,1/50,1/25,1/10,1/5,1/2,1/1]
xper = [int(1/x) for x in xtks]


nmod,nlon,nlat,ntime = rspec.shape
specsplot = rspec.reshape(nmod*nlon*nlat,ntime)

fig,ax = plt.subplots(1,1,figsize=(10,3))
ln1    = ax.plot(rfreq*dtplot,specsplot.T/dtplot,label="",lw=1,alpha=0.05,color='k')

ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xlabel("Period (Years)")
ax.set_ylabel("Power ($K^2 cpy^{-1}$)")
ax.grid(True)

ax2 = ax.twiny()
ax2.set_xlim([xtks[0],xtks[-1]])
ax2.set_xticks(xtks)
ax2.set_xticklabels(xper,fontsize=8,rotation=45)
ax2.grid(True,ls='dotted',color='gray')
ax2.set_xlabel("Period (Years)")
#ln1_se = ax.plot(freqs[0]*dtplot,specsplot[kthres,:].T/dtplot,label="",lw=1,alpha=0.5,color='yellow')

left, bottom, width, height = [0.65, 0.50, 0.30, 0.35]
ax3 = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
ax3 = viz.add_coast_grid(ax3,bbox=bboxplot,fill_color='gray',ignore_error=True)
ax3 = viz.plot_box(bbsel,ax=ax3,linestyle='solid',color='k')


savename = "%sCESM-%s_Spectra_bbox_%s_%s.png" % (figpath,mnames[cid],bbstr,smoothname)

plt.savefig(savename,dpi=200)
#ax3.plot(lonr2[klon],latr2[klat],marker="x",color="k",markersize=10)