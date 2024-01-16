#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Monthly Variance over a particular region

Copied upper section from pointwise autocorrelation script\
Copied experiment loading sections from compare_Tddamp

Copied sections from viz_inputs_point


Created on Tue Nov 28 21:04:54 2023

@author: gliu

"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cartopy.crs as ccrs

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20240119/"
   
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
#%% Import Parameters for Stochmod

cwd = os.getcwd()
sys.path.append(cwd+"/../")
import stochmod_params as sparams

#%% User Edits

# Get the list of files (Stochastic Model Output)
expname      = "default" # Indicate Experiment name (see stochmod_params)
flist        = sparams.rundicts[expname]
print(flist)
continuous   = True

# Bounding Box to Select Data Over
bbox_sel = [-45,-10,50,60]
#bbox_sel = [-31,-30,49,50]

#%% Load the forcing/damping and MLD files

# rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
# # Load Fprime 
# fname   = "Fprime_PIC_SLAB_rolln0.nc"
# dsf     = xr.open_dataset(rawpath+"../"+fname)
# Fprime  = ds.
# Fprime  = dsf.sel(lon=lonf+360,lat=latf,method='nearest').Fprime.values # Time
# ntime   = len(Fprime)
# #Fpt     = Fprime.reshape(int(ntime/12),12).std(0) 

# Load data that was saved for bokeh testing
bkpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/bokeh_test/"
invars_name = ("h","lbd_slab","lbd_full","alpha2_slab","alpha2_full")


ds_all = []

for v,varname in tqdm(enumerate(invars_name)):
    savename = "%s%s_sm_input.nc" % (bkpath,varname)
    ds = xr.open_dataset(savename)
    ds_all.append(ds)
    
    
damping_slab = proc.sel_region_xr(ds_all[1],bbox_sel).lbd_slab.values.transpose(2,1,0)
forcing_slab = proc.sel_region_xr(ds_all[3],bbox_sel).alpha2_slab.values.transpose(2,1,0)



#%% Load the Stochastic Model Output files, processing by runid

st = time.time()

debug    = False
f        = 0
ssts     = []
#ssts_reg = []
for f in range(len(flist)):
    # Load Information for the run
    expid   = flist[f]
    ld      = np.load(datpath+"stoch_output_%s.npz"%(expid),allow_pickle=True)
    sst     = ld['sst'] # [hconfig x lon x lat x time]
    lon     = ld['lon']
    lat     = ld['lat']
    
    # Select region
    sst      = sst.transpose(1,2,3,0) # [lon x lat x time x hconfig]
    nlon,nlat,ntime,nmodels=sst.shape
    sst      = sst.reshape(nlon,nlat,ntime*nmodels) 
    sst_ravg = proc.sel_region(sst,lon,lat,bbox_sel,reg_avg=True,awgt=1)
    sst_ravg = sst_ravg.reshape(ntime,nmodels)
    ssts.append(sst_ravg)
    
    # Also Select the region
    sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
    if f == 0:
        nlons,nlats,_=sstr.shape
        ssts_reg = np.ones((nlons,nlats,ntime,3,len(flist)))*np.nan
    ssts_reg[:,:,:,:,f] = sstr.reshape(nlons,nlats,ntime,nmodels)
        #ssts_reg.append(sstr.reshape())
    
    
    
    #sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
    #ssts_reg.append(sstr)
    
    # Check 
    if debug:
        #sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
        ax = viz.add_coast_grid(ax,fill_color="k")
        ax.set_extent([-80,0,0,65])
        pcm = ax.pcolormesh(lonr,latr,sstr[:,:,0].T)



ssts     = np.concatenate(ssts,axis=0) # [time x hconfig]


print("Completed Loading SM files in %.2fs" % (time.time()-st))
#ssts_reg = [sst[:,:,:,ii,:].reshape(nlons,nlats,ntime*len(flist)) for ii in range(3)]


# Concatenate and rearragen sstreg
#ssts_reg = [sst.reshape(nlons,nlats,int(ntimes/12),12) for sst in ssts_reg]
#ssts_reg = np.concatenate(ssts_reg,axis=2) # [Lon x Lat x Hconfig]

#%% Load SOM and FCM (use version processed by viz_AMV_CESM.py)

fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FCM","SOM"] 
mcolors    = ["k","gray"]

#datpathc    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
sstscesm      = []
ssts_reg_cesm = []
for f in range(2):
    varname = "SST"
    sst_fn = fnames[f]
    ds   = xr.open_dataset(datpath+"../"+sst_fn)
    ds   = ds.sel(lon=slice(bbox_sel[0],bbox_sel[1]),lat=slice(bbox_sel[2],bbox_sel[3]))
    lonc = ds.lon.values
    latc = ds.lat.values
    sstc = ds[varname].values # [lon x lat x time]
    sstscesm.append(sstc)
    ssts_reg_cesm.append(sstc.copy())


# Compute the mean
sstscesm      = [proc.area_avg(s,bbox_sel,lonc,latc,1) for s in sstscesm]
#ssts_reg_cesm = 

# Reshape
nlonc,nlatc,_ = ssts_reg_cesm[0].shape
ntimes_cesm = [s.shape[0] for s in sstscesm]
nyrs_cesm   = [int(t/12) for t in ntimes_cesm]
sstscesm = [sstscesm[ii].reshape(nyrs_cesm[ii],12) for ii in range(2)]


ssts_reg_cesm = [ssts_reg_cesm[ii].reshape(nlonc,nlatc,nyrs_cesm[ii],12) for ii in range(2)] # # [nlon, nlat nyrs, 12]
#%% Load non-processed version of CESM1 data (no land ice mask)

# Load Raw SST
sstsc     = []
mconfigs = ["FULL","SLAB"]
for i in range(2):
    # Open the file
    ds = xr.open_dataset(datpath+"../CESM_proc/"+"TS_anom_PIC_%s.nc"%(mconfigs[i]))
    if i == 0:
        ds = ds.sel(time=slice('0800-02-01','2201-01-01'))
        
        
    # Select the region
    ds = proc.format_ds(ds,)
    ds = proc.sel_region_xr(ds,bbox_sel)
    sst = ds.TS.values
    
    sstsc.append(sst)


# Do preproc
sstsc_proc = []
for sst in sstsc:
    
    # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
    sst = sst.transpose(2,1,0)
    
    # Flip longitude
    # st = time.time()
    # lon180,sst = proc.lon360to180(lon360,sst)
    # print("Flipped Longitude in %.2fs"%(time.time()-st))
    
    # Remove monthly anomalies
    st = time.time()
    nlon,nlat,ntime = sst.shape
    sst = sst.reshape(nlon,nlat,int(ntime/12),12)
    ssta = sst - sst.mean(2)[:,:,None,:]
    print("Deseasoned in %.2fs"%(time.time()-st))
    print("Mean was %e" % (np.nanmax(ssta.mean(2))))
    ssta = ssta.reshape(nlon,nlat,int(ntime/12)*12)
    
    
    sstsc_proc.append(ssta)

lonrc = ds.lon.values
latrc = ds.lat.values

# Remove Points if option is set
remove_points=True
if remove_points:
    
    sstsc_proc_in = []
    for mc in range(2):
        
        #sstr,lonrc,latrc = proc.sel_region(sstsc_proc[mc],lon180,latglob,bbox_sel,)
        sst_rem = sstsc_proc[mc]
        # for o in range(2):
        #     sst_rem[o,-1,:] = np.nan
        sstsc_proc_in.append(sst_rem)
        
    # Compute the mean over the rgion
    # Compute the mean
    sstscesm = [proc.area_avg(s,bbox_sel,lonrc,latrc,0) for s in sstsc_proc_in]
    
    
else:
    
    sstsc_proc_in = sstsc_proc


    # Compute the mean over the rgion
    # Compute the mean
    sstscesm = [proc.area_avg(s,bbox_sel,lon180,latglob,1) for s in sstsc_proc_in]



if debug:
    # Check Region
    #sstr,lonrc,latrc = proc.sel_region(sstsc_proc[0],lonrc,latrc,bbox_sel,)
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                          constrained_layout=True,figsize=(12,4))
    ax = viz.add_coast_grid(ax,fill_color="k")
    ax.set_extent([-80,0,40,65])
    pcm = ax.pcolormesh(lonrc,latrc,np.std(sstsc_proc[0],2).T,cmap="inferno")
    fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01)
    
    #viz.plot_box(bbox_sel,color="y")
    
    for xx in range(4):
        ax.scatter(lonrc[xx],latrc[-1],marker="x",color="k")


ntimes_cesm = [s.shape[0] for s in sstscesm]
nyrs_cesm   = [int(t/12) for t in ntimes_cesm]
sstscesm = [sstscesm[ii].reshape(nyrs_cesm[ii],12) for ii in range(2)]

#%% Compute the monthly variance

# Remove Seasonal Cycle
scycle,ssts = proc.calc_clim(ssts,0,returnts=1) # [Year x Mon x Hconfig]
sstsa       = ssts - scycle[None,:,:]

# Remove Seasonal Cycle for CESM1

# nlonr,nlatr,_ = ssts_reg.shape
# ssts_re
# scycler,ssts_reg = proc.calc_clim(ssts_reg,2,returnts=1)# [lon x lat x time x 12]

#%% Plot the monthly variance!

mons3  = proc.get_monstr(3)
fig,ax = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)

# Plot CESM
cesmvars = []
for cc in range(2):
    plotvar = np.var(sstscesm[cc],0)
    ax.plot(mons3,plotvar,c=mcolors[cc],label=mnames[cc],marker="o")
    cesmvars.append(plotvar)
    
sstvars = []
for hc in range(nmodels):
    
    plotvar = np.nanvar(ssts,0)[:,hc]
    ax.plot(mons3,plotvar,c=sparams.mcolors[hc],label=sparams.modelnames[hc],marker="o")
    sstvars.append(plotvar)
ax.legend()
ax.set_ylabel("Monthly Variance ($\degree C^2$)")

ax.set_title("Seasonal Distribution of Variance (Stochastic Model)",fontsize=14)
#ax = viz.add_ticks()
ax.grid(True,ls='dotted')
#ax.set_ylim([0,0.8])


savename = "%sStochastic_Model_Hierarchy_Monthly_Variance.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')


#%% Look At timeseries

fig,ax = plt.subplots(1,1)

ax.plot(sstscesm[:,0])
ax

#%% Save the monthly variance

sstvars = np.array(sstvars)


coords_dict = {'hierarchy_level': list(sparams.modelnames),
               'mon'            : mons3}


da_monvar = xr.DataArray(sstvars,coords=coords_dict,dims=coords_dict,name="monthly_variance")
#edict     = proc.make_encoding_dict(da_monvar)
savename  = "%sMonthly_Variance_Stochastic_Model.nc" % figpath
da_monvar.to_netcdf(savename)

#%% Investigate Forcing/Damping/MLD Terms over this same region
input_path  = datpath + '../model_input/'
frcname     = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
method      = 4
lagstr      = 'lag1'

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs



#%% Save for when plot is a single point ()


vars_pt = np.concatenate([np.array(cesmvars),np.array(sstvars)],axis=0) # [model x 12]
names   = np.hstack([mnames,sparams.modelnames])
savename_pt = "%sSPG_Point_Monthly_Variance_PiC.npz" % (figpath)
    np.savez(savename_pt,**{
        'monvar':vars_pt,
        'names' : names,
        'mons3' : mons3,
        'lon'   : lonr,
        'lat'   : latr
        },allow_pickle=True)



#%% Restrict to Region and Take the Regional Average

invars      = [h,dampingslab,dampingfull,np.sum(alpha**2,2),np.sum(alpha_full**2,2)]

invars_ravg = [proc.sel_region(v,lon,lat,bbox_sel,reg_avg=True,awgt=1) for v in invars]


#%% Plot the monthly variance for each of the terms

fig,axs = plt.subplots(3,1,figsize=(6,8),constrained_layout=True)

for v in range(3):
    
    ax = axs[v]
    
    if v == 0:
        ax.plot(mons3,invars_ravg[0],color="magenta")
        ylab = "Mixed layer Depth \n (m)"
        
    elif v == 1:
        ax.plot(mons3,invars_ravg[1],label="Damping (Slab)",color='blue')
        ax.plot(mons3,invars_ravg[2],label="Damping (Full)",color='red')
        ylab = "Damping \n ($W m^{-2}  \degree C^{-1}$)"
        
    elif v == 2:
        ax.plot(mons3,invars_ravg[3],label="Forcing (Slab)",color='blue')
        ax.plot(mons3,invars_ravg[4],label="Forcing (Full)",color='red')
        ylab = "Forcing Amplitude \n $(W m^{-2})^2$"
    ax.legend()
    ax.set_ylabel(ylab)
    ax.grid(True,ls='dotted')

savename = "%sStochastic_Model_Hierarchy_Monthly_Variance_Inputs_Separate.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Do Pointwise Monthly Variance COmputation (Just run this ONCE!)


# Focust on SLAB vs Constant H
monvar_cesm  = [np.nanvar(sst,2) for sst in ssts_reg_cesm]


# Do some reshaping/dimension reorganization
ssts_reg_new = ssts_reg.transpose(0,1,4,2,3) # [lon x lat x file x time x hconfig]
ssts_reg_new = ssts_reg_new.reshape(nlons,nlats,len(flist)*ntime,3) # [lon x lat x all_time x hconfig]
ssts_reg_new = ssts_reg_new.reshape(nlons,nlats,int(len(flist)*ntime/12),12,3) # [lon x lat x yr x mon x hconfig]

# Calclate monthly variance for SM
monvar_sm    = [np.nanvar(ssts_reg_new[:,:,:,:,ii],2) for ii in range(3)]

#%% Visualize the difference
mons3 = proc.get_monstr(nletters=3)
bbox_plot = [-50,-5,45,65]
fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,8),
                       constrained_layout=True)

for im in range(12):
    
    ax =axs.flatten()[im]
    ax.set_title(mons3[im])
    
    blabel=[0,0,0,0]
    if im%3 ==0:
        blabel[0] = 1
    if im>8:
        blabel[-1] = 1
    ax =viz.add_coast_grid(ax,bbox=bbox_plot,blabels=blabel,fill_color="k")
    ax.plot(-30,50,marker="x",color="gray",markersize=25)
    
    plotvar = monvar_sm[0][:,:,im] - monvar_cesm[1][:,:,im]
    pcm = ax.pcolormesh(lonr,latr,plotvar.T,cmap="RdBu_r",vmin=-.4,vmax=.4)
    #fig.colorbar(pcm,ax=ax)
plt.suptitle("Constant h (SM) - SLAB (CESM)")
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.05)

savename = "%sSlab_v_SM_MonVar.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Do Monthly variance plot again
# Using data for whole region, selecting 1 point

lonf        = -40
latf        = 49

plotspread  = False # Set to True to plot spread across 10 model runs
klons,klats = proc.find_latlon(lonf, latf,lonr,latr)


alphas      = np.linspace(0.1,.8,len(flist))

fig,ax      = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)

for cc in range(2):
    plotvar = monvar_cesm[cc][klonc,klatc,:]
    ax.plot(mons3,plotvar,c=mcolors[cc],label=mnames[cc],marker="o")

for hc in range(nmodels):
    if plotspread:
        for f in range(len(flist)):
            plotvar = ssts_reg[klons,klats,:,hc,f].reshape(int(ntime/12),12).var(0)
            ax.plot(mons3,plotvar,c=sparams.mcolors[hc],label=sparams.modelnames[hc],marker="o",alpha=alphas[f])
    else:
        plotvar = monvar_sm[hc][klons,klats,:]
        #plotvar=ssts_reg[klons,klats,:,hc,:].T.flatten().reshape(int(ntime*len(flist)/12),12).var(0) # Old Version
        ax.plot(mons3,plotvar,c=sparams.mcolors[hc],label=sparams.modelnames[hc],marker="o",)

#ax.legend()
ax.set_ylabel("Monthly Variance ($\degree C^2$)")

ax.set_title("Seasonal Distribution of Variance (Stochastic Model)",fontsize=14)
ax.grid(True,ls='dotted')

if plotspread:
    savename = "%sStochastic_Model_Hierarchy_Monthly_Variance_Redo_plotspread.png" % figpath
else:
    savename = "%sStochastic_Model_Hierarchy_Monthly_Variance_Redo.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')


#%% Look at where the monthly variance difference is maximum
monvardiff = plotvar = monvar_sm[0][:,:,:] - monvar_cesm[1][:,:,:] # [Lon x Lat x 12]



fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,3.4),
                       constrained_layout=True)

ax =viz.add_coast_grid(ax,bbox=bbox_plot,blabels=blabel,fill_color="k")

plotvar = np.abs(monvardiff).sum(-1)

idmax = np.nanargmax(plotvar.flatten())

idlon,idlat = np.unravel_index(idmax,plotvar.shape)
print("Found Max = %f, Max is %f" % (plotvar[idlon,idlat],np.nanmax(plotvar.flatten())))


pcm = ax.pcolormesh(lonr,latr,plotvar.T,cmap="cmo.dense",vmin=0,vmax=2)

ax.plot(lonr[idlon],latr[idlat],color="yellow",marker="x",markersize=12)
fig.colorbar(pcm,ax=ax)

#%% Organize data by where the difference in monthly variance values


monvardiff # [lon x lat x month]
nlonr,nlatr,_ = monvardiff.shape

inarrs = [monvardiff,damping_slab,forcing_slab,monvar_sm[0],monvar_cesm[1]]
inarrs = [(arrs**2).sum(2).flatten() for arrs in inarrs] # Sum Monthly Values




# Prepare Inputs
sortarr       = inarrs[0]  # Indices to sort by (smallest to largest)
targarrs      = [arrs.reshape(nlonr*nlatr,12) for arrs in [damping_slab,forcing_slab,monvar_sm[0],monvar_cesm[1],monvardiff]]

targarrs_name = [r"Damping ($\lambda ^a$, $[ \degree C/ W/m^2])$","Forcing [W/m^2])","Monthly Variance (Stochastic Model)","Monthly Variance (SOM)","Variance Diff"]
targarrs_fid  = ["Damping","Forcing","StochModelMonVar","SOMMonVar","VarDiff"]


sortid,sorttarg,coords,coords_val=proc.sort_by_axis(sortarr,targarrs,axis=0,lon=lonr,lat=latr)

# # Get sorting indices and sort data
# sortid   = np.argsort(sortarr,axis=0)
# sortid   = [sid for sid in sortid if ~np.isnan(sortarr[sid])]# Drop NaN values
# sorttarg = [np.take(arrs,sortid,axis=0) for arrs in targarrs]

# # Get Coordinate Pairs
# xx,yy = np.unravel_index(sortid,(nlonr,nlatr))
# coords = [[ "%.2f, %.2f" % (lonr[xx[i]], latr[yy[i]])] for i in range(len(sortid))]
# coords_val = [[lonr[xx[i]], latr[yy[i]]] for i in range(len(sortid))]

#%% Make the above into a function

def sort_by_axis(sortarr,targarrs,axis=0,lon=None,lat=None):
    """
    Sort a list of arrays [targarrs] given values from [sortarr] along [axis] from smallest to largest
    
    Parameters
    ----------
    sortarr : np.array
        Array containing values to sort by along [axis]
    targarrs : list of np.arrays
        List containing target arrays to sort (same axis)
    axis : INT, optional
        Axis along which to sort. The default is 0.
    lon : np.array, optional
        Longitude Values. The default is None.
    lat : np.array, optional
        Latitude values. The default is None.
        
    Returns
    -------
    sortid : np.array
        Array containing indices that would sort array from smallest to largest
    sorttarg : list of np.arrays
        Sorted list of arrays
    coords_str : list of STR ["lon,lat"] (%.2f) for corresponding points
    coords_val : list of lists [lon,lat] in float for corresponding points
    """
    
    
    sortid   = np.argsort(sortarr,axis=axis)
    sortid   = [sid for sid in sortid if ~np.isnan(sortarr[sid])]# Drop NaN values
    sorttarg = [np.take(arrs,sortid,axis=axis) for arrs in targarrs]
    if (lon is None) and (lat is None):
        nlon,nlat=len(lon),len(lat)
        xx,yy  = np.unravel_index(sortid,(nlon,nlat))
        coords_str = [[ "%.2f, %.2f" % (lon[xx[i]], lat[yy[i]])] for i in range(len(sortid))] # String Formatted Version
        coords_val = [[lon[xx[i]], lat[yy[i]]] for i in range(len(sortid))] # Actual values
        return sortid,sorttarg,coords_str,coords_val
    return sortid,sorttarg
    
    
        
    
    



#%% Examine Charactersitics of Top/Bottom 10 Points

idtarg     = 4
varin      = sorttarg[idtarg]
varname_in = targarrs_name[idtarg]#"Damping ($\lambda ^a$)"

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(6,4.5),sharey=True)

for aa in range(2):
    
    ax = axs[aa]
    
    if aa == 0:
        title = "Top 10 (Largest Error)"
        plotvar = varin[-10:]
        coordin = coords[-10:]
        
    else:
        
        title = "Bot 10 (Smallest Error)"
        
        plotvar = varin[:10]
        coordin = coords[:10]
        
    ax.set_title(title)
    
    
    for ii in range(len(sortid)):
        ax.plot(mons3,varin[ii,:],label="",color='gray',alpha=0.01)
        
        
    for ii in range(10):
        
        ax.plot(mons3,plotvar[ii,:],label=coordin[ii])
    

    
    #ax.legend(ncol=5)
    #ax.plot()
    ax.grid(True,ls='dotted')
    ax.set_xlim([0,11])
    
plt.suptitle("Seasonal Cycle of %s" % varname_in)
savename = "%sTopBot10_SOM_v_SM_%s.png" % (figpath,targarrs_fid[idtarg])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Locations with Largest and Smallest Difference

fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,3.4),
                       constrained_layout=True)

for aa in range(2):
    ax = axs[aa]
    blabel=[1,0,0,1]
    ax =viz.add_coast_grid(ax,bbox=bbox_plot,blabels=blabel,fill_color="k")
    
    plotvar = np.abs(monvardiff).sum(-1)
    
    # idmax = np.nanargmax(plotvar.flatten())
    # idlon,idlat = np.unravel_index(idmax,plotvar.shape)
    # print("Found Max = %f, Max is %f" % (plotvar[idlon,idlat],np.nanmax(plotvar.flatten())))
    
    
    pcm = ax.pcolormesh(lonr,latr,plotvar.T,cmap="cmo.dense",vmin=0,vmax=2)
    
    if aa == 0:
        title = "Top 10 (Largest Error)"
        plotvar = varin[-10:]
        coordin = coords_val[-10:]
        
    else:
        
        title = "Bot 10 (Smallest Error)"
        
        plotvar = varin[:10]
        coordin = coords_val[:10]
        
    for ii in range(10):
        ax.plot(coordin[ii][0],coordin[ii][1],color="yellow",marker="x",markersize=5)
    ax.set_title(title)
    
fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.04)

savename = "%sTopBot10_SOM_v_SM_Locations.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight')



