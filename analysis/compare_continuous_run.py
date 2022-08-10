#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare multiple continuous runs from the stochastic model.

This includes
(1) Load in SST, Ann. Avg, then compute AMV



Copies sections from the following script:
    - calc_AMV_continuous.py
    - Analyze_SM_Output.ipynb

Created on Tue Jul 19 09:14:20 2022

@author: gliu
"""



import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr
import sys
import glob
import matplotlib.patheffects as PathEffects

# %% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath = projpath + '01_Data/model_output/'
    rawpath = projpath + '01_Data/model_input/'
    outpathdat = datpath + '/proc/'
    figpath = projpath + "02_Figures/20220808/"

    sys.path.append(
        "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append(
        "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat = datpath + '/proc/'

    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append(
        "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc, viz
import tbx
import scm
#%% User Edits

# Indicate Experiment Information
# -------------------------------

# Name of experiments (for plotting, counting)


"""
Experiment 1: SLAB vs FULL HFF
"""
expset   = "SLABvFULLHFF"
expnames = ("BOTH_HFF","SLAB_HFF","FULL_HFF")
# Search Strings for the experiment files
expstrs  = (
            "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2*_ampq0_method5_dmp0.npz",
            "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2*_ampq0_method5_useslab2.npz",
            "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2*_ampq0_method5_useslab4.npz"
            )


"""
Experiment 2: Comparing HFF with and without ENSO removal
"""
expset   = "ENSOremoval"
expnames = ("ENSO_removed","ENSO_present")
expstrs = (
    "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2*_ampq0_method5_dmp0.npz",
    "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2*_ampq0_method5_useslab0_ensorem0.npz"
    )



# Indicate AMV Calculation Settings
calc_AMV  = False # Take annual average for AMV calculation
amvbbox   = [-80, 0, 20, 60] # AMV Index Bounding Area
applymask = False # Set to True to Mask out Insignificant Points

# Indicate area average spectra calculation settings
calc_rspec = True # Take regional average for spectra calculation
bboxreg    = [-40,-10,20,40]
bboxname   = "STGe"
nsmooth    = np.concatenate([np.ones(3)*300,[100,100,]])
pct        = 0.10
dtplot     = 3600*24*365 

# Other Plotting Settings
modelnames = ("Constant h","Vary h","Entraining")

# -------------------------
#%% First, build file names
# -------------------------
# Script searches datpath for: datpath + expstrs, where expstrs has wildcards.
nexp = len(expnames)

f_exps   = []
sst_exps = [] 

sst_regs = []

for e in range(nexp):
    
    # Get List of Files
    # -----------------
    searchstring = "%s%s" % (datpath,expstrs[e])
    fnames       = glob.glob(searchstring)
    print("Found %i files for Experiment %i!" % (len(fnames),e))
    #print(*nclist,sep="\n") # Uncomment to print all found files
    f_exps.append(fnames)
    
    # Load in data and ...
    # ------------------------------------
    sst_all = []
    sst_reg = []
    for f, fname in tqdm(enumerate(fnames)):
        ld = np.load(fname, allow_pickle=True)
        ssts = ld['sst']
        if f == 0:
            lonr = ld['lon']
            latr = ld['lat']
        
        # Take annual average (for AMV calculation)
        if calc_AMV:
            ssts_ann = proc.ann_avg(ssts, 3)
            sst_all.append(ssts_ann) # 
        
        # Take regional average (for spectra calculation)
        if calc_rspec:
            
            rssts = []
            for m in range(3):
                sstreg = proc.sel_region(ssts[m,:,:,:],lonr,latr,bboxreg,reg_avg=True,awgt=1)
                rssts.append(sstreg) # [model][time]
            #rssts = np.array(rssts) # [model x time]
            sst_reg.append(rssts) # [run][model][time]
        
    if calc_AMV:
        sst_all = np.concatenate(sst_all, axis=3)  # [model x lon x lat x year]
        sst_exps.append(sst_all)
    if calc_rspec:
        sst_reg = np.concatenate(sst_reg,axis=1)   # [model x time (mons)]
        sst_regs.append(sst_reg)
    
    #
    # ------------------------------------
    
# Load Lat/Lon
lonr = ld['lon']
latr = ld['lat']

#%% Load Data for CESM

# Copied from calc_AMV_continuous.py (2022.07.19)
mconfigs = ("SLAB", "FULL")
cdatpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
bbox = [np.floor(lonr[0]), np.ceil(lonr[-1]),
        np.floor(latr[0]), np.ceil(latr[-1])]
print("Simulation bounding Box is %s " % (str(bbox)))

sst_cesm = []
for mconfig in mconfigs:
    fname = "%sCESM1_%s_postprocessed_NAtl.nc" % (cdatpath, mconfig)
    ds = xr.open_dataset(fname)
    dsreg = ds.sel(lon=slice(bbox[0], bbox[1]), lat=slice(bbox[2], bbox[3]))
    sst_cesm.append(dsreg.SST.values)


#%% 

# Load dmsks
dmsks = scm.load_dmasks(bbox=[lonr[0], lonr[-1], latr[0], latr[-1]])
dmsks.append(dmsks[-1])

# Preallocate
amvpats = [] # [exp][model x lon x lat]
amvids  = [] # [exp][model x time]

# Also compute regionally-averaged spectra
rspectra = [] 
rfreqs   = []
for e in tqdm(range(nexp)):
    
    # Compute AMV
    # -----------
    if calc_AMV:
        sst_all = sst_exps[e]
    
        # Preallocate and loop
        nmod,nlon,nlat,nyr = sst_all.shape
    
        pats = np.zeros((nmod, nlon, nlat)) * \
            np.nan  # [model x lon x lat]
        ids = np.zeros((nmod, nyr)) * np.nan #[model x time]
        
        # Loop for each stochastic model
        for mid in range(nmod):
            
            if applymask:
                inmask = dmsks[mid]
            else:
                inmask = None
            sst_in = sst_all[mid, ...]
            amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, amvbbox, anndata=True,
                                               runmean=False, dropedge=5, mask=inmask)
            pats[mid, ...] = amvpat.copy()
            ids[mid, ...] = amvid.copy()
        
        amvpats.append(pats)
        amvids.append(ids)
        
        # End AMV calculation Loop
    
    # Compute regional spectra
    # ------------------------
    if calc_rspec:
        
        sst_reg = sst_regs[e] # [model x time]
        
        especs = []
        efreqs = []
        
        inssts = [sst for sst in sst_reg]
        specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooth[:3],pct)
        
        rspectra.append(specs)
        rfreqs.append(freqs)
        
        # End Spectra calculation loop
        
        
# Do for CESM
cpats  = []
cids   = []
cspecs = []
cfreqs = []
crssts = []
for i in range(2):
    
    sst_in = sst_cesm[i]
    
    if calc_AMV:
        
        amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, amvbbox, anndata=False,
                                           runmean=False, dropedge=5, mask=None)
        cpats.append(amvpat)
        cids.append(amvid)
        
    if calc_rspec:
        
        creg = proc.sel_region(sst_in,lonr,latr,bboxreg,reg_avg=1,awgt=1)
        specs,freqs,CCs,dofs,r1s = scm.quick_spectrum([creg,],[nsmooth[3+i],],pct)
        cspecs.append(specs[0])
        cfreqs.append(freqs[0])
        crssts.append(creg)

#%% Observe Differences in the AMV Pattern
# Compare Non-entraining and entraining

# Copied/adapted from calc_AMV_continuous.py

# Plot settings
# -------------
notitle      = True
darkmode     = False
cmax         = 0.5
cstep        = 0.025
lstep        = 0.05
cint, cl_int = viz.return_clevels(cmax, cstep, lstep)
clb          = ["%.2f" % i for i in cint[::4]]
cl_int       = cint
sel_rid      = 1
plotbbox     = False
useC         = True
bbin         = amvbbox
bbplot       = [-80,0,10,60]


"""
Experiment 1: SLAB vs FULL HFF

"""
# # Set AMVs to Plot
# plotamvs = (
#             amvpats[0][0,:,:],
#             amvpats[1][0,:,:],
#             amvpats[1][2,:,:],
#             amvpats[0][2,:,:]
#             )

# plotnames = ("Non-entraining (SLAB HFF)","Non-entraining(FULL HFF)",
#              "Entraining (SLAB HFF)","Entraining (FULL HFF)")
# plotids  = (
#             amvids[0][0,:],
#             amvids[2][0,:],
#             amvids[1][2,:],
#             amvids[0][2,:]
#             )
# mids     = (0,2,0,2) # Heat Flux Feedback mask model id
# hffs     = ("SLAB","FULL","SLAB","FULL")
# mids_lab = (0,0,2,2) # Model Id Label



"""
Experiment 2: Comparing ENSO removal
"""

plotamvs = (
            amvpats[0][0,:,:],
            amvpats[1][0,:,:],
            amvpats[0][2,:,:],
            amvpats[1][2,:,:]
            )

plotnames = ("Non-entraining (ENSO removed)","Non-entraining(ENSO present)",
             "Entraining (ENSO removed)","Entraining (ENSO present)")
plotids  = (
            amvids[0][0,:],
            amvids[1][0,:],
            amvids[0][2,:],
            amvids[1][2,:]
            )

mids     = (0,2,0,2) # Heat Flux Feedback mask model id
hffs     = ("SLAB","SLAB","FULL","FULL")
mids_lab = (0,0,2,2) # Model Id Label




print("Plotting AMV for bbox: %s" % (bbin))
bbstr = "lon%ito%i_lat%ito%i" % (bbin[0], bbin[1], bbin[2], bbin[3])

spid = 0
proj = ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': proj}, figsize=(11, 9),
                        constrained_layout=True)

if darkmode:
    plt.style.use('dark_background')

    savename = "%sSST_AMVPattern_Comparison_%s_region%s_mask%i_dark.png" % (
        figpath, fnames[f], bbstr, applymask)
    fig.patch.set_facecolor('black')
    dfcol = 'k'
else:
    plt.style.use('default')
    savename = "%sSST_AMVPattern_Comparison_%s_region%s_mask%i.png" % (
        figpath, fnames[f], bbstr, applymask)
    fig.patch.set_facecolor('white')
    dfcol = 'k'


for aid in range(4):
    ax = axs.flatten()[aid]
    
    # Set Labels, Axis, Coastline
    blabel = [0,0,0,0]
    
    if aid%2==0:
        blabel[0] = 1
        ax.text(-0.23, 0.45, '%s'% (modelnames[mids_lab[aid]]), va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes,fontsize=14)
    
    if aid > 1:
        blabel[-1] = 1
    else:
        ax.text(0.5, 1.1, '%s HFF'% (hffs[aid]), va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes,fontsize=14)

    # Make the Plot
    ax = viz.add_coast_grid(ax, bbplot, blabels=blabel, line_color=dfcol,
                            fill_color='gray')
    pcm = ax.contourf(
        lonr, latr, plotamvs[aid].T, levels=cint, cmap='cmo.balance')
    # ax.pcolormesh(lon,lat,amvpats[mid,:,:,rid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(
        lonr, latr, plotamvs[aid].T, levels=cl_int, colors="k", linewidths=0.5)
    
    ax.clabel(cl, levels=cl_int[::2], fontsize=8, fmt="%.02f")

    if useC:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)" % (
            plotnames[aid], np.var(plotids[aid]))
    else:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f $K^2$)" % (
            plotnames[aid], np.var(plotids[aid]))
    ax.set_title(ptitle)
    if plotbbox:
        ax, ll = viz.plot_box(bbin, ax=ax, leglab="AMV",
                              color=dfcol, linestyle="dashed", linewidth=2, return_line=True)
    
    viz.plot_mask(lonr, latr, dmsks[mids[aid]], ax=ax, markersize=0.3)
    
    if mids[aid] == 2:
        # Not sure why logical or works///
        viz.plot_mask(lonr, latr, np.logical_or(dmsks[0],dmsks[-1]), ax=ax, markersize=2,marker="x")
    
    ax.set_facecolor = dfcol
    ax = viz.label_sp(spid, case='lower', ax=ax, labelstyle="(%s)",
                      fontsize=16, alpha=0.7, fontcolor=dfcol)
    spid += 1
    
cb=fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
cb.set_label("AMV Pattern ($\degree C \, \sigma_{AMV}^{-1}$)")

savename = "%sAMV_Patterns_HFF_Ablation_%s.png" % (figpath,expset)
plt.savefig(savename, dpi=150, bbox_inches='tight')

#%% Experiment 2, plot spectra for the selected region



# rspectra : [experiment][modelnumber]

fig,ax = plt.subplots(1,1,figsize=(8,5))

plotspecs = (rspectra[0][1],rspectra[1][1],
             rspectra[0][2],rspectra[1][2],
             cspecs[1]
             )
plotssts       = (sst_regs[0][1,:],sst_regs[1][1,:],
              sst_regs[0][2,:],sst_regs[1][2,:],
              crssts[1]
              )

specnames = ("Vary h (ENSO removed)","Vary h",
             "Entraining (ENSO removed)","Entraining",
             "CESM-FULL")
speccolors = ("magenta","magenta",
              "orange","orange",
              "k")
specls     = ("solid","dashed",
              "solid","dashed",
              "solid"
              )

lws         = (1,1.5,1,1.5,1,)


smspec     = rfreqs[0][0]
cspec      = cfreqs[1]


nspecs = len(plotspecs)
for n in range(nspecs):
    if n < nspecs-1:
        
        plotfreq = smspec * dtplot
    else:
        plotfreq = cspec * dtplot
    
    speclab = specnames[n]#"%s (%.4f $^{\circ}C$)" % (specnames[n],np.var(plotssts[n]))
    ax.plot(plotfreq,plotspecs[n]/dtplot,
            color=speccolors[n],
            linestyle=specls[n],
            label=speclab,lw=lws[n])
ax.legend()

ax.set_xlim([1/100,1/2])
ax.grid(True,ls='dotted')

xtks = np.array([1/100,1/20,1/10,1/5])
xper = (1/xtks).astype(int)
ax2 = ax.twiny()
ax2.set_xlim([1/100,1/2])

ax2.set_xticks(xtks)
ax2.set_xticklabels(xper)
ax2.grid(True,ls='dotted',color='gray')
ax2.set_xlabel("Period (Years)")

ax.set_title("ENSO Removal Effect from Heat Flux Feedback in %s" % (bboxname))
plt.savefig("%sENSO_comparison_%s.png" % (figpath,bboxname),dpi=150)







