 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot Regional Spectra, Ratio

Plot spectral ratio over different regions

Created on Tue Mar 15 22:05:37 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr
import time
#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220407/"
   
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

#%%

# Spectra Parametsrs
pct        = 0.10
nsmooth    = 30
smoothname = "smooth%03i-taper%03i" % (nsmooth,pct*100)

outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
figpath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
proc.makedir(figpath)

#sst_fn = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run200_ampq3_method5_dmp0.npz"
#sstname = "CESM_FULL_PIC"
#sstname  = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run200_ampq3_method5_dmp0"


# Postprocess Continuous SM  Run
# ------------------------------
fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
snames      = ["spectra_%s_Fprime_rolln0_ampq0_method5_dmp0_run2%02d.nc" % (smoothname,i) for i in range(10)]
mnames      = ["constant h","vary h","entraining"] 
snames_full = [outpath+sname for sname in snames]

# Postproess CESM Run
# ------------------------------
cesm_fnames      = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
cesm_mnames      = ["FULL","SLAB"] 
cesm_snames      = ["spectra_%s_PIC-%s.nc" % (smoothname,cesm_mnames[i]) for i in range(2)]
cesm_snames_full = [outpath+sname for sname in cesm_snames]

# Other Params
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

#%% Load in the spectra for each

debug = True

# Find Frequency (lowerthres <= f <= thresval)
thresval   = 1/20 # 20 years (Multidecadal)
lowerthres = 0 # 
dtplot     = 3600*24*365

# Load CESM
# ---------
cesm_dsall = []
for i in range(2):
    ds       = xr.open_mfdataset(cesm_snames_full[i])
    cesm_dsall.append(ds)
    

# Load Spectra
st = time.time()
cfreqs       = []
cspecs       = []
cmodelnames   = []
for i in range(2):
    cspecs.append(cesm_dsall[i].spectra.values)
    cfreqs.append(cesm_dsall[i].frequency.values)
    cmodelnames.append(cesm_dsall[i].models.values)

    print("Loaded data in %.2fs" % (time.time()-st))
            
# Load Stochastic Model
# ---------------------------
dsall       = xr.open_mfdataset(snames_full
                                ,combine="nested",concat_dim="run")

# Take mean of all runs
st = time.time()
dsmean      = dsall.mean('run')
freq        = dsmean.frequency.values
specs       = dsmean.spectra.values # [model x lon x lat x freq]
modelnames  = dsmean.models.values
#specsds     = dsmean.spectra
print("Loaded data in %.2fs" % (time.time()-st))


#%% Plot variance thresholds over specific regions


#threses = ([0,1/20],[1/20,1/10],[1/10,1/2]) # [lower freq, upper freq]
threses = ([0,1/10],[1/10,1/2])
dtplot  = 3600*24*365

inspecs = (specs[0,...],
           specs[1,...],
           specs[2,...],
           cspecs[0].squeeze(),
           cspecs[1].squeeze())

infreqs = (freq,freq,freq,cfreqs[0],cfreqs[1])
innames = mnames+cesm_mnames


nmods,nlon,nlat,nfreqsm = specs.shape

sumvals = np.zeros((nlon,nlat,len(inspecs),len(threses))) # [lon, lat x model x threshold]

sumvals_vp = sumvals.copy() # Variance preserving form
for t in range(len(threses)):
    
    
    lowerthres,thresval = threses[t]
    
    for mid in range(len(inspecs)):
        inspec = inspecs[mid]
        infreq = infreqs[mid]
        
        # Non variance preserving
        outfreq = proc.calc_specvar(infreq,inspec,thresval,dtplot,
                                 return_thresids=False,lowerthres=lowerthres)
        sumvals[:,:,mid,t] = outfreq.copy()
        
        # Variance-preserving
        outfreq_vp = proc.calc_specvar(infreq,inspec*infreq,thresval,dtplot,
                                 return_thresids=False,lowerthres=lowerthres)
        sumvals_vp[:,:,mid,t] = outfreq_vp.copy()
        
    
#%% Now Lets Plot the Ratios

#sumvals_in = sumvals_vp


"""
vp = 0 Linear-Linear
vp = 1 Frequency x Power (Now this is incorrect b/c we need log(freq))
"""
f_s   = [] # [vp][lon, lat x threshold]
e_hv  = []
hv_hc = []
e_f   = []
hc_s  = []
for sumvals_in in [sumvals,sumvals_vp]:
    full_slab     = np.log(sumvals_in[:,:,3,:]/sumvals_in[:,:,4,:])
    entrain_hvary = np.log(sumvals_in[:,:,2,:]/sumvals_in[:,:,1,:])
    hvary_hconst  = np.log(sumvals_in[:,:,1,:]/sumvals_in[:,:,0,:])
    entrain_full  = np.log(sumvals_in[:,:,2,:]/sumvals_in[:,:,3,:])
    hconst_slab   = np.log(sumvals_in[:,:,0,:]/sumvals_in[:,:,4,:])
    
    f_s.append(full_slab) 
    e_hv.append(entrain_hvary)
    hv_hc.append(hvary_hconst)
    e_f.append(entrain_full)
    hc_s.append(hconst_slab)
    


#%%

# ratiosel  = entrain_hvary
# rationame = "log($\sigma^2_{entrain}$/$\sigma^2_{h vary}$)"
# rationame_fn = "log_entrain_hvary"

ratiosel     = full_slab
rationame    = "log($\sigma^2_{full}$/$\sigma^2_{slab}$)"
rationame_fn = "log_full_slab"

ratiosel     = hvary_hconst
rationame    = "log($\sigma^2_{h vary}$/$\sigma^2_{h const}$)"
rationame_fn = "log_hvary_hconst"

ratiosel     = e_f[0]#entrain_full
rationame    = "log($\sigma^2_{entrain}$/$\sigma^2_{full}$)"
rationame_fn = "log_entrain_full"

# ratiosel     = hconst_slab
# rationame    = "log($\sigma^2_{h const}$/$\sigma^2_{slab}$)"
# rationame_fn = "log_hconst_slab"


use_contours = True
cints       = np.arange(-1.5,1.55,0.05)
cl_ints     = np.arange(-1.5,1.6,0.1)
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4)) 
for t in range(3):
    blabel = [0,0,0,1]
    if t == 0:
        blabel[0] = 1
    
    ax = axs.flatten()[t]
    print(t)
    if t == 0:
        ptitle = r"> %03d Years" % (1/threses[t][1])
    else:
        ptitle = "%03d to %03d Years" % (1/threses[t][1],1/threses[t][0])
    ax.set_title(ptitle)
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,fill_color='gray')
    if use_contours:
        pcm = ax.contourf(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,extend='both',cmap='cmo.balance')
        cl  = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5)
        ax.clabel(cl,cl_ints[::2],fmt="%.1f")
    else:
        pcm = ax.pcolormesh(ds.lon,ds.lat,ratiosel[:,:,t].T,vmin=-1.5,vmax=1.5,cmap='cmo.balance')

fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
ax = axs[0]
ax.text(-0.15, 0.55, rationame, va='bottom', ha='center',rotation='vertical',
        rotation_mode='anchor',transform=ax.transAxes)
plt.savefig("%sSpectra_Ratio_%s_%s.png"% (figpath,rationame_fn,smoothname),dpi=150)


#%% Randomly grab and plot barotropic streamfunction


# BSF
bsf_ds = xr.open_dataset(datpath+"../CESM_proc/BSF_FULL_PIC_bilinear.nc")
ds_reg = bsf_ds.BSF.sel(lon=slice(bboxplot[0],bboxplot[1]),lat=slice(bboxplot[2],bboxplot[-1]))

ds_reg   = ds_reg.mean('time')
bsf      = ds_reg.values
bsf_mean = bsf.T
#bsf_mean = bsf.mean(-1)


# Do the same with SSH
ssh_ds = xr.open_dataset(datpath+"../CESM_proc/SSH_FULL_PIC_bilinear.nc")
ssh_reg = ssh_ds.SSH.sel(lon=slice(bboxplot[0],bboxplot[1]),lat=slice(bboxplot[2],bboxplot[-1]))

ssh_reg   = ssh_reg.mean('time')
ssh      = ssh_reg.values
ssh_mean = ssh.T
#%% PLot BSF
ratiosel     = e_f[0]
rationame    = "log($\sigma^2_{entrain}$/$\sigma^2_{full}$)"
rationame_fn = "log_entrain_full"



plotcontour = "BSF"

use_contours = True
cints       = np.arange(-1.5,1.55,0.05)
cl_ints     = np.arange(-1.5,1.6,0.1)
sshcint     = np.arange(-150,155,5)
bsfcint     = np.arange(-30,35,5)
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4)) 
for t in range(3):
    blabel = [0,0,0,1]
    if t == 0:
        blabel[0] = 1
    
    ax = axs.flatten()[t]
    print(t)
    if t == 0:
        ptitle = r"> %03d Years" % (1/threses[t][1])
    else:
        ptitle = "%03d to %03d Years" % (1/threses[t][1],1/threses[t][0])
    ax.set_title(ptitle)
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,fill_color='gray')
    if use_contours:
        pcm = ax.contourf(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,extend='both',cmap='cmo.balance')
        #cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5)
        #ax.clabel(cl,cl_ints[::2],fmt="%.1f")
    else:
        pcm = ax.pcolormesh(ds.lon,ds.lat,ratiosel[:,:,t].T,vmin=-1.5,vmax=1.5,cmap='cmo.balance')
        
    if plotcontour=="BSF":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,bsf_mean.T,colors='k',levels=bsfcint,linewidths=0.5)
        ax.clabel(cl,levels=bsfcint)
    elif plotcontour=="SSH":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,ssh_mean.T,colors='k',levels=sshcint,linewidths=0.5)
        ax.clabel(cl,levels=sshcint)
    
fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
ax = axs[0]
ax.text(-0.15, 0.55, rationame, va='bottom', ha='center',rotation='vertical',
        rotation_mode='anchor',transform=ax.transAxes)
plt.savefig("%sSpectra_Ratio_%s_%s%s.png"% (figpath,rationame_fn,smoothname,plotcontour),dpi=150)
#%% Updated version with just 2 plots
ratiosel     = e_f[0]
rationame    = "log($\sigma^2_{entrain}$/$\sigma^2_{full}$)"
rationame_fn = "log_entrain_full"



plotcontour = None # "BSF","SSH", or None

use_contours = True

# Full ersion
cints       = np.arange(-1.5,1.55,0.05)
cl_ints     = np.arange(-1.5,1.6,0.1)

cints       = np.arange(-1.5,1.75,0.25)
clints      = cints

sshcint     = np.arange(-150,155,5)
bsfcint     = np.arange(-30,35,5)

fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4)) 
for t in range(2):
    
    blabel = [0,0,0,1]
    if t == 0:
        blabel[0] = 1
    
    ax = axs.flatten()[t]
    print(t)
    if t == 0:
        ptitle = r"> %03d Years" % (1/threses[t][1])
    else:
        ptitle = "%03d to %03d Years" % (1/threses[t][1],1/threses[t][0])
    ax.set_title(ptitle)
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,fill_color='gray')
    if use_contours:
        pcm = ax.contourf(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,extend='both',cmap='cmo.balance')
        #cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5)
        #ax.clabel(cl,cl_ints[::2],fmt="%.1f")
    else:
        pcm = ax.pcolormesh(ds.lon,ds.lat,ratiosel[:,:,t].T,vmin=-1.5,vmax=1.5,cmap='cmo.balance')
    if plotcontour=="BSF":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,bsf_mean.T,colors='k',levels=bsfcint,linewidths=0.5)
        ax.clabel(cl,levels=bsfcint)
    elif plotcontour=="SSH":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,ssh_mean.T,colors='k',levels=sshcint,linewidths=0.5)
        ax.clabel(cl,levels=sshcint)
    else:
        cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5)
        ax.clabel(cl,cl_ints,fmt="%.1f")
    
fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
ax = axs[0]
ax.text(-0.15, 0.55, rationame, va='bottom', ha='center',rotation='vertical',
        rotation_mode='anchor',transform=ax.transAxes)
plt.savefig("%sSpectra_Ratio_%s_%s%s_2only.png"% (figpath,rationame_fn,smoothname,plotcontour),dpi=150)

#%% Focus on a region (to answer Claude's question)

ratiosel     = e_f[0] # select 0, gotta erase 1 because its incorrect (variance preserving already...)
rationame    = "log($\sigma^2_{entrain}$/$\sigma^2_{full}$)"
rationame_fn = "log_entrain_full"


viz_bbox = [-60,-15,40,65]#[-80,-40,20,40]

plotcontour  = None # "BSF","SSH", or None
use_contours = True

# Full ersion
cints       = np.arange(-1.5,1.55,0.05)
cl_ints     = np.arange(-1.5,1.6,0.1)

cints       = np.arange(-1.5,1.6,0.1)
clints      = cints

sshcint     = np.arange(-150,155,5)
bsfcint     = np.arange(-30,35,5)

fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4)) 
for t in range(2):
    
    blabel = [0,0,0,1]
    if t == 0:
        blabel[0] = 1
    
    ax = axs.flatten()[t]
    print(t)
    if t == 0:
        ptitle = r"> %03d Years" % (1/threses[t][1])
    else:
        ptitle = "%03d to %03d Years" % (1/threses[t][1],1/threses[t][0])
    ax.set_title(ptitle)
    ax = viz.add_coast_grid(ax,bbox=viz_bbox,blabels=blabel,fill_color='gray')
    if use_contours:
        pcm = ax.contourf(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,extend='both',cmap='cmo.balance')
        #cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5)
        #ax.clabel(cl,cl_ints[::2],fmt="%.1f")
    else:
        pcm = ax.pcolormesh(ds.lon,ds.lat,ratiosel[:,:,t].T,vmin=-1.5,vmax=1.5,cmap='cmo.balance')
    if plotcontour=="BSF":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,bsf_mean.T,colors='k',levels=bsfcint,linewidths=0.5)
        ax.clabel(cl,levels=bsfcint)
    elif plotcontour=="SSH":
        cl = ax.contour(ds_reg.lon,ds_reg.lat,ssh_mean.T,colors='k',levels=sshcint,linewidths=0.5)
        ax.clabel(cl,levels=sshcint)
    else:
        cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,colors='k',linewidths=0.5)
        ax.clabel(cl,clints,fmt="%.1f")
    
fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
ax = axs[0]
ax.text(-0.15, 0.55, rationame, va='bottom', ha='center',rotation='vertical',
        rotation_mode='anchor',transform=ax.transAxes)
plt.savefig("%sSpectra_Ratio_%s_%s%s_2only_region.png"% (figpath,rationame_fn,smoothname,plotcontour),dpi=150)

#%% SM Draft Plot
use_contours = False
cints       = np.arange(-1.5,1.6,0.1)
cl_ints     = cints#np.arange(-1.5,1.8,0.3)
cl_alpha     = 1
fig,axs = plt.subplots(3,3,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,10))

for row in range(3):
    
    if row == 0:
        ratiosel     = full_slab
        rationame    =  "FULL/SLAB"# "log($\sigma^2_{full}$/$\sigma^2_{slab}$)"
        rationame_fn = "log_full_slab"
    elif row == 1:
        ratiosel  = entrain_hvary
        rationame = "Entrain/Non-Entraining"#"log($\sigma^2_{entrain}$/$\sigma^2_{h vary}$)"
        rationame_fn = "log_entrain_hvary"
    elif row == 2:
        ratiosel     = entrain_full
        rationame    = "Entrain/FULL"#"log($\sigma^2_{entrain}$/$\sigma^2_{full}$)"
        rationame_fn = "log_entrain_full"
    
    for t in range(3):
        ax = axs[row,t]
        
        blabel = [0,0,0,0]
        if t == 0:
            blabel[0] = 1
        if row == 2:
            blabel[-1] = 1
        
        if row == 0:
            if t == 0:
                ptitle = r"> %d Years" % (1/threses[t][1])
            else:
                ptitle = "%d to %d Years" % (1/threses[t][1],1/threses[t][0])
            
            ax.set_title(ptitle)
        
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,fill_color='gray',ignore_error=True)
        if use_contours:
            pcm = ax.contourf(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cints,extend='both',cmap='cmo.balance')
            cl = ax.contour(ds.lon,ds.lat,ratiosel[:,:,t].T,levels=cl_ints,colors='k',linewidths=0.5,alpha=cl_alpha)
            ax.clabel(cl,cl_ints[::2],fmt="%.1f")
        else:
            pcm = ax.pcolormesh(ds.lon,ds.lat,ratiosel[:,:,t].T,vmin=-1.5,vmax=1.5,cmap='cmo.balance')

    #fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
    ax = axs[row,0]
    ax.text(-0.15, 0.55, rationame, va='bottom', ha='center',rotation='vertical',
            rotation_mode='anchor',transform=ax.transAxes)
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
cb.set_label("SST Log Ratio")

plt.savefig("%sSpectra_Ratio_combine_%sBSF.png"% (figpath,smoothname),dpi=150)


#%% scrap to figure out some stuff

klon,klat = proc.find_lon

#%% References for myself for log ratio (NTS: Make some interactive version)

rawratio = np.arange(0,8,0.05)

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(rawratio,np.log(rawratio),lw=1.5,color="yellowgreen",label="$log_{e}(r)$")
ax.plot(rawratio,np.log10(rawratio),lw=1.5,color="mediumpurple",label="$log_{10}(r)$")
ax.minorticks_on()
ax.set_xlabel("Raw Ratio ($r$)")
ax.set_ylabel("Log Ratio")

ax.grid(True,ls='dashed',which='major',lw=1,alpha=0.5)
ax.grid(True,ls='dotted',which='minor',lw=1,alpha=0.2)

ax.axhline(0,color='k')
ax.axvline(1,color='k')
ax.legend()

ax.set_xlim([0,8])
ax.set_ylim([-2.5,2])
ax.set_title("Log Ratio Reference (base $e$)")
plt.savefig("%sLogRatioReference.png"%figpath,dpi=150)



