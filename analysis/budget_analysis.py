#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget Analysis

Created on Mon Mar 28 16:58:22 2022

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from scipy import signal
from tqdm import tqdm
import sys

#%% Import Modules

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

sys.path.append("/Users/gliu/Downloads/06_School/06_Fall2021/12860/materials_2020/CVD_Tutorials/")
import cvd_utils as cvd
#%% User Edits

amvbbox = [-80,0,0,65]
def load_sm_terms(ld,vnames=None,multFAC=True):
    
    # Get list of variables
    if vnames is None:
        vnames = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]
        
    # Prepare Integration Factor 
    if multFAC:
        FAC = ld['FAC']
    
    # Load in the data, and take the annual averages
    for v,vname in enumerate(vnames):
        
        vld    = ld[vname].squeeze()
        
        
        if v == 0:
            # Get Lat/Lon/Dimensions
            lon          = ld['lon']
            lat          = ld['lat']
            nlon,nlat,ntime = vld.shape
            nyr          = int(ntime/12)
            
            # Tile integration factor
            if multFAC:
                FACyr = np.tile(FAC,nyr)
            
            # Preallocate
            sm_vars = np.zeros((nlon,nlat,nyr,len(vnames)))*np.nan
            
        if multFAC:
            if vname in ["forcing_term","ekman_term",]:
                print("Multiplying %s by FAC"%(vname))
                vld  = vld * FACyr
        
        vld_ann = proc.ann_avg(vld,2) 
        sm_vars[:,:,:,v] = vld_ann.copy()
        
        if vname == "sst":
            # Save and load separate copy
            T_ann = vld_ann.copy()
            
    return sm_vars,T_ann,lon,lat


def check_term_diff(ld,vnames=None,multFAC=True,return_allvar=False):
    # Get list of variables
    if vnames is None:
        vnames = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]
        
    # Prepare Integration Factor 
    if multFAC:
        FAC = ld['FAC']
    
    # Load in the data, and take the annual averages
    for v,vname in enumerate(vnames):
        
        vld    = ld[vname].squeeze()
        
        
        if v == 0:
            # Get Lat/Lon/Dimensions
            lon          = ld['lon']
            lat          = ld['lat']
            nlon,nlat,ntime = vld.shape
            nyr          = int(ntime/12)
            
            # Tile integration factor
            if multFAC:
                FACyr = np.tile(FAC,nyr)
            
            # Preallocate
            sm_vars = np.zeros((nlon,nlat,ntime,len(vnames)))*np.nan
        
        
        if multFAC:
            if vname in ["forcing_term","ekman_term",]:
                print("Multiplying %s by FAC"%(vname))
                vld  = vld * FACyr
        
        # Compute differences
        sm_vars[:,:,:,v] = vld.copy()
        
        if vname == "sst":
            # Save and load separate copy
            T = vld.copy()
    
    if return_allvar:
        return T,sm_vars
    # Now compute the differences
    diff = np.abs(T - sm_vars[...,1:].sum(-1))
    return diff


def checkpoint(checkpoints,invar,debug=True):
    """
    Groups values of invar between values specified in checkpoints

    Parameters
    ----------
    checkpoints : ARRAY
        1-D Array of checkpoint/threshold values. Checks (z-1 < x <= z)
    invar : ARRAY
        1-D Array of values to check
    debug : TYPE, optional
        True to print messages (default)

    Returns
    -------
    ids_all : TYPE
        Indices of invar for each group

    """
    
    ids_all = []
    for z in range(len(checkpoints)+1):
        if z == 0: # <= First value
            if debug:
                print("Looking for indices <= %i"% (checkpoints[z]))
            ids = np.where(invar <= checkpoints[z])[0]
            
        elif z == len(checkpoints): # > Last value
            if debug:
                print("Looking for indices > %i"% (checkpoints[z-1]))
            ids = np.where(invar > checkpoints[z-1])[0]
            if len(ids)==0:
                continue
            else:
                print("Found %s"% str(np.array(invar)[ids]))
                ids_all.append(ids)
            return ids_all # Exit on last value
        else: # Check values between z-1, z
            ids = np.where((invar > checkpoints[z-1]) & (invar <= checkpoints[z]))[0]
            if debug and (z%100 == 0) or (z < 10) or (z>len(checkpoints)-10):
                print("Looking for indices %i < x <= %i" % (checkpoints[z-1],checkpoints[z]))
        
        
        if len(ids)==0:
            continue
        else:
            if debug and (z%100 == 0) or (z < 10) or (z>len(checkpoints)-10):
                print("Found %s"% str(np.array(invar)[ids]))
            ids_all.append(ids)
    return ids_all


#%% User Edits

# Set variable names and order in axis
vnames        = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]

# Set the path to the data
dp            = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"

# Set filenames
fns = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02i_ampq0_method5_dmp0_budget_Qek.npz" % (s) for s in np.arange(0,10,1)]

# Figure output path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220407/"
proc.makedir(figpath)

# Plotting Parameters
plotbbox      = [-80,0,10,60]
proj          = ccrs.PlateCarree()
# -------------------------
#%% (00) Load for each file
# -------------------------

sm_vars_all = []
T_all       = []
diffs_all   = []
for f,fn in tqdm(enumerate(fns)):
    
    ld            = np.load(dp+fn,allow_pickle=True) 
    sm_vars,T_ann,lon,lat = load_sm_terms(ld,vnames=None)
    sm_vars_all.append(sm_vars)
    T_all.append(T_ann)

# Convert to 1-D
T_all       = np.array(T_all)       # [run x lon x lat x time]
sm_vars_all = np.array(sm_vars_all) # [run x lon x lat x time x variable]

nrun,nlon,nlat,nyr,nvar = sm_vars_all.shape

#% Transpose to proper dimensions and unravel time [lon x lat x RUN x TIME x var]
T_all       = T_all.transpose(1,2,0,3).reshape(nlon,nlat,nrun*nyr)
sm_vars_all = sm_vars_all.transpose(1,2,0,3,4).reshape(nlon,nlat,nrun*nyr,nvar)

"""
T_all = [lon x lat x year]
sm_vars_all = [lon x lat x year x run]
"""
# ----------------------------
#%% (01) Compute the AMV Index
# ----------------------------

"""
5) awgt:  number to indicate weight type
            0 = no weighting
            1 = cos(lat)
            2 = sqrt(cos(lat))
"""

# Compute the AMV Index
amvid,amvpat = proc.calc_AMVquick(T_all.squeeze(),lon,lat,amvbbox,dropedge=5,anndata=True)

# Now get indices of positive.negative points
pks,_ = signal.find_peaks(np.abs(amvid))

# Separate into positive and negative
idneg = [pk for pk in pks if amvid[pk] <= 0]
idpos = [pk for pk in pks if amvid[pk] > 0]
zerocross = np.where(np.diff(np.sign(amvid)))[0] # Final index BEFORE zero crossing

# Group values by zero crossings
chkpos     = checkpoint(zerocross,idpos,debug=True) # Indices of idpos
chkneg     = checkpoint(zerocross,idneg,debug=True) # Indices of idneg

# Cull values
idpos_cull = []
for cp_group in chkpos: # For each group
    cids  = np.array(idpos)[cp_group] # Get relevant indices
    vals  = amvid[cids] # Get corresponding valuesvalues
    idpos_cull.append(cids[np.argmax(cids)]) # Save local maxima
idneg_cull = []
for cp_group in chkneg: # For each group
    cids  = np.array(idneg)[cp_group] # Get relevant indices
    vals  = amvid[cids] # Get corresponding valuesvalues
    idneg_cull.append(cids[np.argmin(cids)]) # Save local maxima

#%% Plot the AMV Index

# Set some parameters
xlm    = [0,1000]
t      = np.arange(0,len(amvid))
fig,ax = plt.subplots(1,1,figsize=(16,4))

# Plot timeseries, peaks, and zero-crossings
ax     = cvd.plot_anomaly(t,amvid,ax=ax)
ax.scatter(idneg,amvid[idneg],marker="d",color="darkblue",label="Maxima")
ax.scatter(idpos,amvid[idpos],marker="x",color="darkred",label="Minima")
ax.scatter(zerocross,amvid[zerocross],marker="o",color='yellow',label="Zero Crossing")

# Labeling and Adjustments
ax.set_xlabel("Years")
ax.set_ylabel("AMV Index")
ax.set_title("AMV Index for Years %i to %i" % (xlm[0],xlm[1]))
ax.legend()
ax.set_xlim(xlm)
ax.grid(True,ls='dotted')

#%% Check culled values

# Set some parameters
chkmax = 500
xlm    = [0,chkmax]
t      = np.arange(0,len(amvid))
fig,ax = plt.subplots(1,1,figsize=(16,4))


zerocross_plot = [z for z in zerocross if z < chkmax]
cpos_plot = [c for c in chkpos if np.all(c < chkmax)]
cneg_plot = [c for c in chkneg if np.all(c < chkmax)]

# Plot timeseries, peaks, and zero-crossings
ax     = cvd.plot_anomaly(t,amvid,ax=ax)
ax.set_xlim(xlm)

# Plot all peaks
ax.scatter(idneg,amvid[idneg],marker="d",color="darkblue",label="Min")
ax.scatter(idpos,amvid[idpos],marker="d",color="darkred",label="Max")

# Plot local max/min
ax.scatter(idneg_cull,amvid[idneg_cull],s=120,marker="x",color="darkblue",label="KeepMin")
ax.scatter(idpos_cull,amvid[idpos_cull],s=120,marker="x",color="darkred",label="KeepMax")

# Plot zero crossings
[ax.axvline(_x,color='gray',label="") for _x in zerocross_plot]
#ax.vlines(zerocross_plot,ymin=-.2,ymax=.2,color='gray',label="Zero Crossing")

# Labeling and Adjustments
ax.set_xlabel("Years")
ax.set_ylabel("AMV Index")
ax.set_title("AMV Index for Years %i to %i" % (xlm[0],xlm[1]))
ax.legend()

ax.grid(True,ls='dotted')

#%% Make AMV+ and AMV- Composites

thresholds = [0,]
y_class = proc.make_classes_nd(amvid,thresholds,debug=True)
in_str     = "AMV Index"

thpats = []
thname = []
for th in range(len(thresholds)+1):
    
    id_sel = np.where(y_class == th)[0]
    thpats.append(T_all[:,:,id_sel].mean(-1)) # lon x lat
    
    if th == 0:
        thname.append("%s < %i" % (in_str,thresholds[th]))
    elif th == len(thresholds):
        thname.append("%s > %i" % (in_str,thresholds[th-1]))
    else:
        thname.append("%i < %s < %i" % (thresholds[th-1],in_str,thresholds[th]))
    
    
#%%
use_pcm = False
vm    = 0.2
vstep = 0.025
cint = np.arange(-vm,vm+vstep,vstep)
fig,axs = plt.subplots(1,3,subplot_kw={'projection':proj},figsize=(18,6))

for th in range(3):
    
    ax = axs.flatten()[th]
    blabel = [0,0,0,1]
    if th == 0:
        blabel[0] = 1
    
    # Add Grid + Backdrop
    ax = viz.add_coast_grid(ax,bbox=plotbbox,fill_color='gray',
                            ignore_error=True,blabels=blabel)
    
    if th < 2:
        plotpat = thpats[th].T
        ptitle = "Composite (%s)" % thname[th]
    else:
        plotpat = amvpat.T
        ptitle = "AMV Pattern ($\degree C \sigma_{AMV}^{-1}$)"
        
    
    # Make Plot
    if use_pcm:
        pcm = ax.pcolormesh(lon,lat,plotpat,vmin=-vm,vmax=vm,cmap='cmo.balance')
    else:
        pcm = ax.contourf(lon,lat,plotpat,levels=cint,cmap='cmo.balance',extend="both")
    
    cl  = ax.contour(lon,lat,plotpat,levels=cint,colors="k",linewidths=0.75)
    ax.clabel(cl)
    
    # Add colorbar
    ax.set_title(ptitle)
    
    
    
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.05)
cb.set_label("SST' ($\degree C$)")
#plt.suptitle("AMV Composites")

plt.savefig("%sAMV_Composites.png"%figpath,dpi=150,bbox_inches='tight')

# ------------------------------------------
#%% (02) Find Increasing/Decreasing Segments
# ------------------------------------------

localmaxmin = True # Set to True to use local max/minima

decr = np.zeros((nlon,nlat))*np.nan # Decrease
incr = np.zeros((nlon,nlat))*np.nan # Inrease

decr_ids = []
incr_ids = []
# Loop through each zero crossing
for zero in range(len(zerocross)):
    idzero = zerocross[zero]
    
    # Find nearest pos/neg peaks
    if localmaxmin:
        kpos   = idpos[np.abs(idpos_cull-idzero).argmin()]
        kneg   = idneg[np.abs(idneg_cull-idzero).argmin()]
    else:
        kpos   = idpos[np.abs(idpos-idzero).argmin()]
        kneg   = idneg[np.abs(idneg-idzero).argmin()]
    
    
    
    #print("Nearest crossings to %i are Neg: %i, Pos: %i"%(idzero,kneg,kpos))
    # Record intevals to corresponding array (Note: need to check if this is indexing properly)
    if kpos > kneg: # Increasing
        incr_ids.append(np.arange(kneg,kpos+1,1))
    elif kpos < kneg: # Decreasing
        decr_ids.append(np.arange(kpos,kneg+1,1))
        
n_incr = len(incr_ids)
n_decr = len(decr_ids)
print("Found %i increasing sections, %i decreasing sections."% (n_incr,n_decr))

#%% Visualize identified sections

xlm = [0,200]

t      = np.arange(0,len(amvid))
fig,ax = plt.subplots(1,1,figsize=(16,4))

ax     = cvd.plot_anomaly(t,amvid,ax=ax,xlabfreq=10)

# Plot the increasing/decreasing segments
for n in tqdm(range(n_incr)):
    plotids = incr_ids[n]
    ax.scatter(plotids,amvid[plotids],marker="x",color="r",label="Increasing")
for n in tqdm(range(n_decr)):
    plotids = decr_ids[n]
    ax.scatter(plotids,amvid[plotids],marker="+",color="b",label="Decreasing")

# Labeling and Axis Adjustments
ax.set_xlabel("Years")
ax.set_ylabel("AMV Index")
ax.set_title("AMV Index for Years %i to %i" % (xlm[0],xlm[1]))
ax.set_xlim(xlm)
ax.grid(True,ls='dotted')
plt.savefig("%sAMV_Integr_Example_yr%ito%i.png"%(figpath,xlm[0],xlm[1]),bbox_inches='tight',dpi=150)

# # --------------------------------
#%% (03) Integrate for each variable
# # --------------------------------

# Preallocate
decr = np.zeros((nlon,nlat,n_decr,nvar))*np.nan # Decrease
incr = np.zeros((nlon,nlat,n_incr,nvar))*np.nan # Inrease

# Loop (variable)
for v in tqdm(range(nvar)):
    
    invar = sm_vars_all[:,:,:,v]
    
    # Integrate Decreasing values, scaled by # of months
    for d in range(n_decr):
        
        ids_in      = decr_ids[d]
        decr[:,:,d,v] = invar[:,:,ids_in].sum(-1) / len(ids_in)
        
        
    # Integrate increasing values, scaled by # of months
    for d in range(n_incr):
        ids_in      = incr_ids[d]
        incr[:,:,d,v] = invar[:,:,ids_in].sum(-1) / len(ids_in)
        
    # End Variable Loop

# # ---------------------
#%% (04) Plot mean values
# # ---------------------


specify_vlims = True 

# Label and vlim settings
vnames_fancy = (
    "Total: $\int \, T' \, dt$",
    "Damping: $\int \, \lambda T' \, dt$",
    "Forcing: $\int \, F' \, dt$",
    r"Entrain: $\int \, \frac{w_e}{h} T_d' \, dt$",
    "Ekman: $\int \, F_{ek}' \, dt$",
    )
plotvars  = [decr,incr]
plotnames = ["Decreasing AMV (%i events)"%n_decr,"Increasing AMV (%i events)"%n_incr]
plotlims  = (0.04,0.04,0.01,0.01,0.01)

proj = ccrs.PlateCarree()
fig,axs = plt.subplots(nvar,2,figsize=(8,12),constrained_layout=True,
                      subplot_kw={'projection':proj})

for v in range(nvar):
    
    for i in range(2):
        
        ax      = axs[v,i]
        
        plotvar = plotvars[i][...,v].mean(-1).T # [incr/decr][lon x lat x event x variable]
        
        if specify_vlims: # Used specified max/min
            vm = plotlims[v]
        else: # Automatically set axis max/min
            vm = 4 * np.nanstd(plotvar.flatten())
        
        # Do some Labeling
        blabel = [0,0,0,0]
        if i == 0:
            ax.text(-0.15, 0.55, vnames_fancy[v], va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes,fontsize=14)
            blabel[0] = 1
        if v == nvar-1:
            blabel[-1]=1
        if v == 0:
            ax.set_title(plotnames[i])
            
        # Add Grid + Backdrop
        ax = viz.add_coast_grid(ax,bbox=plotbbox,fill_color='gray',
                                ignore_error=True,blabels=blabel)
        
        # Make Plot
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-vm,vmax=vm,cmap='cmo.balance')
        if i == 1:
            fig.colorbar(pcm,ax=ax,fraction=0.035)
        
savename = "%sSM_Avg_Integrated_Terms.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Check the differences


diffs_all   = []
for f,fn in tqdm(enumerate(fns)):
    
    ld            = np.load(dp+fn,allow_pickle=True) 
    diff = check_term_diff(ld,vnames=None)
    diffs_all.append(diff)
diffs_all = np.array(diffs_all)


# Check differences by segment
diffs_byseg = diffs_all.max(-1)

for i in range(10):
    plt.pcolormesh(lon,lat,diffs_byseg[i,...].T),plt.colorbar(),plt.show()

#diffs
#%% Check if things are adding up


sm_vars_add = sm_vars_all[...,1:].sum(-1)

diff = sm_vars_add - sm_vars_all[...,0]

plt.pcolormesh(lon,lat,diff.mean(-1).T),plt.colorbar()

plt.pcolormesh(lon,lat,diff.argmax(-1).T),plt.colorbar()


#%% Try to figure out why the values are different...
"""
After Running the interior section of check_term_diff, run the following....
"""

T,sm_vars   = check_term_diff(ld,vnames=None,return_allvar=True,multFAC=True)
sm_vars_sum = sm_vars[...,1:].sum(-1)
#%% Visualize errors/differences
tstart = 850
tend   = 900
klon   = 32
klat   = 55

fig,ax = plt.subplots(1,1)

for v in range(5):
    
    ax.plot(sm_vars[klon,klat,:,v],label=vnames[v])

ax.plot(sm_vars_sum[klon,klat,:],label="Sum Terms",color="k",ls='dotted')

# Find and compute maximum difference in the period. Add to the plot
maxdiff = np.max(np.abs(sm_vars[klon,klat,tstart:tend,0]-sm_vars_sum[klon,klat,tstart:tend]))
maxid   = tstart + np.argmax(np.abs(sm_vars[klon,klat,tstart:tend,0]-sm_vars_sum[klon,klat,tstart:tend]))
print("Maximum difference in period was %f" % (maxdiff))
print("This occured at t=%i"%maxid)
ax.axvline(maxid,ls='dashed',color="k",zorder=-1)

ax.legend()
ax.set_xlim([tstart,tend])
ax.grid(True,ls='dotted')
#ax.set_ylim([-1,1])

#%% Check which month the errors are occuring in...

sm_vars_mon = sm_vars.reshape(nlon,nlat,nyr,12,nvar)

sm_diff = np.abs(sm_vars_mon[...,0] - sm_vars_mon[...,1:].sum(-1)) # [lon x lat x time x mon]



fig,ax = plt.subplots(1,1)
for y in range(nyr):
    ax.plot(sm_diff[klon,klat,y,:],alpha=0.1,color="gray")
ax.plot(sm_diff[klon,klat,...].mean(0),alpha=1,color="k")

#ax.plot(FAC[klon,klat])


