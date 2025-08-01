#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV for a continuous simulation

Upper section taken from viz_continuous.py

- Includes Stochastic Model Paper Plots (Draft 4-5)
- Includes US AMOC 2022 Meeting Poster Plots

Created on Wed Mar  2 15:11:39 2022

@author: gliu
"""
from amv import proc, viz
import tbx
import scm
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr

import matplotlib.patheffects as PathEffects
# %% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath = projpath + '01_Data/model_output/'
    rawpath = projpath + '01_Data/model_input/'
    outpathdat = datpath + '/proc/'
    figpath = projpath + "02_Figures/20241124/"

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


proc.makedir(figpath)
# %% User Edits

# Analysis Options
lags = np.arange(0, 37, 1)

# Options to determine the experiment ID
mconfig = "SLAB_PIC"
nyrs = 1000        # Number of years to integrate over

# Visualize Continuous run 200, Fprime
fnames = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0" %
          i for i in range(10)]
frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
exname = "Fprime_amq0_method5_cont"

# # Visualize Continuous run 200, Qnet
# fnames =["forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run2%02d_ampq3_method5_dmp0"%i for i in range(10)]
# frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
# exname ="Qnet_amq3_method5_cont



# Plotting Params
darkmode = True
debug    = False
pubready = False
# %% Functions
def calc_conflag(ac, conf, tails, n):
    cflags = np.zeros((len(ac), 2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin, conf, tails, n)
        cflags[l, :] = cfout
    return cflags

# %% User Edits


# Regional Analysis Settings (OLD)
bbox_SP = [-60, -15, 40, 65]
bbox_ST = [-80, -10, 20, 40]
bbox_TR = [-75, -15, 10, 20]
bbox_NA = [-80, 0, 0, 65]
bbox_NNA = [-80, 0, 10, 65]
regions = ("SPG", "STG", "TRO", "NAT", "NAT")  # ,"NNAT")        # Region Names
regionlong = ("Subpolar", "Subtropical", "Tropical",
              "North Atlantic", "North Atlantic (10N-65N)")
bboxes = (bbox_SP, bbox_ST, bbox_TR, bbox_NA, bbox_NNA)  # Bounding Boxes
bbcol = ["Blue", "Red", "Yellow", "Black", "Black"]
bbsty = ["solid", "dashed", "solid", "dotted", "dotted"]

# # Regional Analysis Setting (NEW, STG SPLOIT)
# Regional Analysis Settings
bbox_SP = [-60, -15, 40, 65]
bbox_ST = [-80, -10, 20, 40]
bbox_TR = [-75, -15, 10, 20]
bbox_NA = [-80, 0, 0, 65]
bbox_NA_new = [-80, 0, 10, 65]
bbox_ST_w = [-80, -40, 20, 40]
bbox_ST_e = [-40, -10, 20, 40]
bbox_NAextr = [-80, 0, 20, 60]

regions = ("SPG", "STG", "TRO", "NAT", "NNAT",
           "STGe", "STGw")        # Region Names
bboxes = (bbox_SP, bbox_ST, bbox_TR, bbox_NA, bbox_NA_new,
          bbox_ST_e, bbox_ST_w)  # Bounding Boxes
regionlong = ("Subpolar", "Subtropical", "Tropical", "North Atlantic",
              "North Atlantic", "Subtropical (East)", "Subtropical (West)",)
bbcol = ["cornflowerblue", "Red", "Yellow",
         "Black", "Black", "limegreen", "indigo"]
bbsty = ["solid", "dashed", "solid", "dotted", "dotted", "dashed", "dotted"]


# AMV Pattern Contours
cint = np.arange(-0.45, 0.50, 0.05)  # Used this for 7/26/2021 Meeting
cl_int = np.arange(-0.45, 0.50, 0.05)
cmax = 0.5
cstep = 0.025
lstep = 0.05
cint, cl_int = viz.return_clevels(cmax, cstep, lstep)
clb = ["%.2f" % i for i in cint[::4]]
bboxplot = [-80, 0, 9, 62]

modelnames = ("Constant h (Level 3)", "Vary h (Level 4)", "Entraining (Level 5)")
mcolors = ["red", "magenta", "orange"]

# CESM Names
cesmname = ["CESM-FULL", "CESM-SLAB"]
cesmcolor = ["k", "gray"]
cesmline = ["dashed", "dotted"]

# Autocorrelation PLots
xtk2 = np.arange(0, 37, 2)
mons3 = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
conf = 0.95
tails = 2

proj = ccrs.PlateCarree()
dfcol = "k"

# Linear-Power Spectra, < 2-yr
xlm = [1e-2, 5e0]
# xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100, 50, 20, 10, 5, 2])
xtks = 1/xper
xlm = [xtks[0], xtks[-1]]


# %% load some additional data

# Load lat/lon regional
# Get one of the outputs
ldname = "%sstoch_output_%s.npz" % (datpath, fnames[0])
if exname == "numEOFs":
    ldname = ldname.replace("50", "2")


ld = np.load(ldname, allow_pickle=True)
lon = ld['lon']
lat = ld['lat']

#lon = np.load(datpath+"lon.npy")
#lat = np.load(datpath+"lat.npy")

# Load global lat/lon
lon180g, latg = scm.load_latlon(rawpath)

# Load dmsks
dmsks = scm.load_dmasks(bbox=[lon[0], lon[-1], lat[0], lat[-1]])
dmsks.append(dmsks[-1])
# %% For each model read in the data

if debug:
    for f in range(10):
        print("Range is %i to %i" % (f*1000, (f+1)*1000-1))
    f = 0
    fname = fnames[f]

# Load in data and take annual average
sst_all = []
for f, fname in tqdm(enumerate(fnames)):
    ld = np.load(datpath+"stoch_output_%s.npz" % fname, allow_pickle=True)
    ssts = ld['sst']
    if f == 0:
        lonr = ld['lon']
        latr = ld['lat']
    ssts_ann = proc.ann_avg(ssts, 3)
    sst_all.append(ssts_ann)
sst_all = np.concatenate(sst_all, axis=3)  # [model x lon x lat x year]

# %% Load data for CESM

# Copied from Explore_Regional_Properties.ipynb, 03/11/2022
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


# %% Now Compute the AMV Pattern

applymask = False
amvbboxes = ([-80, 0, 0, 65], [-80, 0, 20, 60],
             [-80, 0, 40, 60], [-80, 0, 10, 65])
nboxes = len(amvbboxes)
nmod, nlon, nlat, nyr = sst_all.shape

# Compute the AMV Pattern (for the stochastic model)
amvpats = np.zeros((nmod, nlon, nlat, nboxes)) * \
    np.nan  # [model x lon x lat x region]
amvids = np.zeros((nmod, nyr, nboxes)) * np.nan
camvpats = []  # [bbox][cesm-config]
camvids = []
for b, bbin in tqdm(enumerate(amvbboxes)):

    # Do for Stochastic Models
    for mid in range(nmod):
        
        if applymask:
            inmask = dmsks[mid]
        else:
            inmask = None
        sst_in = sst_all[mid, ...]
        amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, bbin, anndata=True,
                                           runmean=False, dropedge=5, mask=inmask)
        amvpats[mid, ..., b] = amvpat.copy()
        amvids[mid, ..., b] = amvid.copy()
        
    # Do for CESM
    cpats = []
    cids = []
    for i in range(2):
        sst_in = sst_cesm[i]
        amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, bbin, anndata=False,
                                           runmean=False, dropedge=5, mask=None)
        cpats.append(amvpat)
        cids.append(amvid)
    camvpats.append(cpats)
    camvids.append(cids)

# %% Plot Traditional AMV Pattern (3 Panel) for each AMV bbox

b = 0
bbox_plot = [-85, 5, 0, 60]
fig, axs = plt.subplots(1, 3, figsize=(12, 6),
                        subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
#cint = np.arange(-.5,0.525,0.025)
for mid in range(3):
    blabel = [0, 0, 0, 1]
    if b == 0:
        blabel[0] = 1

    ax = axs.flatten()[mid]
    ax = viz.add_coast_grid(ax, bbox_plot, fill_color='gray', blabels=blabel)

    cf = ax.contourf(
        lonr, latr, amvpats[mid, :, :, b].T, levels=cint, cmap='cmo.balance')
    cl = ax.contour(
        lonr, latr, amvpats[mid, :, :, b].T, levels=cl_int, colors='k', linewidths=0.55)
    ax.clabel(cl)
    ax.set_title(modelnames[mid])

    ax = viz.plot_box(bbin, ax=ax, linewidth=1.5, linestyle='dashed')
    #plt.setp(axs[mopt, :], ylabel=maskopt[mopt])
cb = fig.colorbar(cf, ax=axs.flatten(), fraction=0.0156)
cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Comparison_bbox_allmodels.png" % (figpath), dpi=150)

# %% Compare BBOX for a selected model

mid = 2
bbox_plot = [-85, 5, 0, 60]
fig, axs = plt.subplots(1, 3, figsize=(12, 6),
                        subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
#cint = np.arange(-.5,0.525,0.025)
for b, bbin in enumerate(amvbboxes):
    blabel = [0, 0, 0, 1]
    if b == 0:
        blabel[0] = 1

    ax = axs.flatten()[b]
    ax = viz.add_coast_grid(ax, bbox_plot, fill_color='gray', blabels=blabel)

    cf = ax.contourf(
        lonr, latr, amvpats[mid, :, :, b].T, levels=cint, cmap='cmo.balance')
    cl = ax.contour(
        lonr, latr, amvpats[mid, :, :, b].T, levels=cl_int, colors='k', linewidths=0.55)
    ax.clabel(cl)
    ax.set_title(amvbboxes[b])

    ax = viz.plot_box(bbin, ax=ax, linewidth=1.5, linestyle='dashed')
    #plt.setp(axs[mopt, :], ylabel=maskopt[mopt])
cb = fig.colorbar(cf, ax=axs.flatten(), fraction=0.0156)
cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Comparison_bboxes.png" % (figpath), dpi=150)

# %% Compare/Quantify 2 bboxes

bbd = 1  # d = denominator
bbn = 3  # n = numerator
print("Comparing bboxes %s / %s" % (amvbboxes[bbn], amvbboxes[bbd]))

bbox_plot = [-85, 5, 0, 60]

fig, axs = plt.subplots(1, 3, figsize=(12, 6),
                        subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
#cint = np.arange(-.5,0.525,0.025)
for i in range(3):
    ax = axs.flatten()[i]
    blabel = [0, 0, 0, 1]
    if i == 0:
        blabel[0] = 1
        b = bbn
    elif i == 1:
        b = bbd

    if i < 2:

        plotamv = amvpats[mid, :, :, b].T
        ptitle = "%s" % amvbboxes[b]
        ax = viz.plot_box(amvbboxes[b], ax=ax,
                          linewidth=1.5, linestyle='dashed',)
        cint_in = cint
        cl_int_in = cl_int

    else:
        plotamv = (amvpats[mid, :, :, bbn]/amvpats[mid, :, :, bbd]).T
        ptitle = "%s / %s" % (amvbboxes[bbn], amvbboxes[bbd])
        cint_in = np.arange(0, 2.1, .1)
        cl_int_in = cint_in

    ax = viz.add_coast_grid(ax, bbox_plot, fill_color='gray', blabels=blabel)

    cf = ax.contourf(lonr, latr, plotamv, levels=cint_in, cmap='cmo.balance')
    cl = ax.contour(lonr, latr, plotamv, levels=cl_int_in,
                    colors='k', linewidths=0.55)
    # ax.clabel(cl)
    ax.set_title(ptitle)

    #plt.setp(axs[mopt, :], ylabel=maskopt[mopt])
cb = fig.colorbar(cf, ax=axs.flatten(), fraction=0.0156)
cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Comparison_2bboxes.png" % (figpath), dpi=150)

# %% Focus comparison on the tropics


tropbbox = [-80, 0, 0, 20]


plotamv = np.log((amvpats[mid, :, :, bbn]/amvpats[mid, :, :, bbd]).T)

#cint_in   = np.arange(0,2.1,.1)
#cl_int_in = cint_in


# Select Region

varr, lonrr, latrr = proc.sel_region(plotamv.T, lon, lat, tropbbox)

xx, yy = np.meshgrid(lonrr, latrr,)
idmax = np.nanargmax(varr)
lonmax = xx.T.flatten()[idmax]
latmax = yy.T.flatten()[idmax]

klon, klat = proc.find_latlon(lonmax, latmax, lonr, latr)


# Make the plot

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})

ptitle = "%s / %s" % (amvbboxes[bbn], amvbboxes[bbd])
blabel = [1, 0, 0, 1]
ax = viz.add_coast_grid(ax, tropbbox, fill_color='gray', blabels=blabel)
#cf  = ax.contourf(lonr,latr,plotamv,levels=cint_in,cmap='cmo.balance')
cf = ax.pcolormesh(lonr, latr, plotamv, vmin=-2.5,
                   vmax=2.5, cmap="cmo.balance")
cl = ax.contour(lonr, latr, plotamv, levels=cl_int_in,
                colors='k', linewidths=0.55)

ax.plot(lonmax, latmax, marker="x", markersize=20, color="yellow")
cb = fig.colorbar(cf, ax=axs.flatten(), fraction=0.0156)

cb = fig.colorbar(cf, ax=ax, fraction=0.0156)
ax.set_title(ptitle)
cb.set_label("Log Ratio")

# Ok so largest value is not reliable because some values are near-zero
print("Maximum log-ratio has value of %f" % (varr.flatten()[idmax]))
print("AMV Amplitide is %f $K^2$ for the numerator" %
      (amvpats[mid, klon, klat, bbn]))
print("AMV Amplitide is %f $K^2$ for the denominator" %
      (amvpats[mid, klon, klat, bbd]))

# %% Let's instead find the max value in the tropics for each case
locmax = []
valmax = []
for b in [bbn, bbd]:

    varr, lonrr, latrr = proc.sel_region(
        amvpats[mid, :, :, b].T, lon, lat, tropbbox)

    xx, yy = np.meshgrid(lonrr, latrr,)
    idmax = np.nanargmax(varr)
    lonmax = xx.T.flatten()[idmax]
    latmax = yy.T.flatten()[idmax]

    locmax.append([lonmax, latmax])
    valmax.append(varr.flatten()[idmax])

print("AMV Max in Tropics is %f in Numerator" % (valmax[0]))
print("AMV Max in Tropics is %f in Denominator" % (valmax[1]))
print("Thus, the ratio is %f" % (valmax[0]/valmax[1]))
# %% Redo Stochastic Model Paper Plot
"""
# Copied from viz_AMV_comparison.py (03/11/2022)
# Updated 08/24/2022 for Revision 1
"""

# Plot settings
notitle      = True
darkmode     = True
cmax         = 0.5
cstep        = 0.025
lstep        = 0.05
cint, cl_int = viz.return_clevels(cmax, cstep, lstep)
clb          = ["%.2f" % i for i in cint[::4]]

thesisdef_ver = True # Omit SP Label and Title
 

bboxplot = [-80,0,9,62]
cl_int   = cint
sel_rid  = 1
plotbbox = False
useC     = True


# Begin Plotting
# ----------------
rid = sel_rid
bbin = amvbboxes[rid]
print("Plotting AMV for bbox: %s" % (bbin))
bbstr = "lon%ito%i_lat%ito%i" % (bbin[0], bbin[1], bbin[2], bbin[3])

spid = 0
proj = ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': proj}, figsize=(10, 8.5),
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


# figsize=(12,6)
# ncol = 3
# fig,axs = viz.init_2rowodd(ncol,proj,figsize=figsize,oddtop=False,debug=True)

# Plot Stochastic Model Output
for aid, mid in enumerate([0, 2]):
    if not thesisdef_ver:
        ax = axs.flatten()[aid]
        
        # Set Labels, Axis, Coastline
        if mid == 0:
            blabel = [1, 0, 0, 0]
        elif mid == 1:
            blabel = [0, 0, 0, 1]
        else:
            blabel = [0, 0, 0, 0]
            
    else:
        ax = axs[1,aid]
        
        if mid == 0:
            blabel = [1,0,0,1]
        else:
            blabel = [0,0,0,1]
        #blabel   = [1,0,0,1]
    
    # Make the Plot
    ax = viz.add_coast_grid(ax, bboxplot, blabels=blabel, line_color=dfcol,
                            fill_color='gray')
    pcm = ax.contourf(
        lon, lat, amvpats[mid, :, :, rid].T, levels=cint, cmap='cmo.balance')
    # ax.pcolormesh(lon,lat,amvpats[mid,:,:,rid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(
        lon, lat, amvpats[mid, :, :, rid].T, levels=cl_int, colors="k", linewidths=0.5)
    ax.clabel(cl, levels=cl_int[::2], fontsize=8, fmt="%.02f")

    if useC:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)" % (
            modelnames[mid], np.var(amvids[mid, :, rid]))
    else:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f $K^2$)" % (
            modelnames[mid], np.var(amvids[mid, :, rid]))
    
    if plotbbox:
        ax, ll = viz.plot_box(amvbboxes[rid], ax=ax, leglab="AMV",
                              color=dfcol, linestyle="dashed", linewidth=2, return_line=True)

    viz.plot_mask(lon, lat, dmsks[mid], ax=ax, markersize=0.3)
    ax.set_facecolor = dfcol
    
    if not thesisdef_ver:
        ax = viz.label_sp(spid, case='lower', ax=ax, labelstyle="(%s)",
                          fontsize=16, alpha=0.7, fontcolor=dfcol)
        ax.set_title(ptitle)
        
    spid += 1

# Plot CESM1
# axs[1,1].axis('off')

for cid in range(2):
    if not thesisdef_ver:
        ax = axs[1, cid]
    else:
        ax = axs[0, cid]
    if cid == 0:
        #ax = axs[1,0]
        blabel = [1, 0, 0, 1]
        spid = 2

        # spid = 3 # Flipped order
    else:
        blabel = [0, 0, 0, 1]
        #ax = axs[1,0]

        spid = 3

    # Make the Plot
    ax = viz.add_coast_grid(ax, bboxplot, blabels=blabel, line_color=dfcol,
                            fill_color='gray')
    pcm = ax.contourf(
        lon, lat, camvpats[rid][cid].T, levels=cint, cmap='cmo.balance')
    ax.pcolormesh(lon, lat, camvpats[rid][cid].T, vmin=cint[0],
                  vmax=cint[-1], cmap='cmo.balance', zorder=-1)
    cl = ax.contour(lon, lat, camvpats[rid][cid].T,
                    levels=cl_int, colors="k", linewidths=0.5)
    ax.clabel(cl, levels=cl_int[::2], fontsize=8, fmt="%.02f")
    if useC:
        ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)" % (
            mconfigs[cid], np.var(camvids[rid][cid]))
    else:
        ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f $K^2$)" % (
            mconfigs[cid], np.var(camvids[rid][cid]))
    
    if plotbbox:
        ax, ll = viz.plot_box(amvbboxes[rid], ax=ax, leglab="",
                              color=dfcol, linestyle="dashed", linewidth=2, return_line=True)
        
    if not thesisdef_ver:
        ax = viz.label_sp(spid, case='lower', ax=ax, labelstyle="(%s)",
                          fontsize=16, alpha=0.7, fontcolor=dfcol)
        ax.set_title(ptitle)

if not thesisdef_ver: # Add Horizontal Colorbar

    cb = fig.colorbar(pcm, ax=axs.flatten(),
                      orientation='horizontal', fraction=0.030, pad=0.05)
else:
    cb = fig.colorbar(pcm, ax=axs.flatten(),
                      orientation='horizontal', fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=12)
cb.set_ticks(cint[::4])
cb.ax.set_xticklabels(clb, rotation=45)
if useC:
    cb.set_label("SST ($\degree C \, \sigma_{AMV}^{-1}$)",fontsize=14)
else:
    cb.set_label("SST ($\degree C \, \sigma_{AMV}^{-1}$)",fontsize=14)
# cb.ax.set_xticklabels(cint[::2],rotation=90)
#tick_start = np.argmin(abs(cint-cint[0]))
# cb.ax.set_xticklabels(cint[tick_start::2],rotation=90)
if notitle is False:
    plt.suptitle("%s AMV Pattern and Index Variance [Forcing = %s]" % (
        regionlong[rid], frcnamelong[f]), fontsize=14)

if pubready:
    plt.savefig("%sFig08_AMV_Pattern_Comparison.png" % (figpath), dpi=900, bbox_inches='tight')
else:
    plt.savefig(savename, dpi=150, bbox_inches='tight',transparent=True)
    
    
    
#%% Remake above plot, but for thesis overview
# (Focus on the Comparison between CESM-FULL and Entraining)
# Just copied above on 2024.11.24

# Plot settings
notitle      = True
darkmode     = True
cmax         = 0.5
cstep        = 0.025
lstep        = 0.05
cint, cl_int = viz.return_clevels(cmax, cstep, lstep)
clb          = ["%.2f" % i for i in cint[::4]]

bboxplot = [-80,0,9,62]
cl_int   = cint
sel_rid  = 1
plotbbox = False
useC     = True


fsz_tick = 12



# Begin Plotting
# ----------------
rid = sel_rid
bbin = amvbboxes[rid]
print("Plotting AMV for bbox: %s" % (bbin))
bbstr = "lon%ito%i_lat%ito%i" % (bbin[0], bbin[1], bbin[2], bbin[3])

spid = 0
proj = ccrs.PlateCarree()
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': proj}, figsize=(10, 8.5),
                        constrained_layout=True)

if darkmode:
    plt.style.use('dark_background')

    savename = "%sSST_AMVPattern_ThesisOverview_dark.png" % (
        figpath)
    fig.patch.set_facecolor('black')
    dfcol = 'k'
    bgcol = np.array([15,15,15])/256
else:
    plt.style.use('default')
    savename = "%sSST_AMVPattern_ThesisOverview.png" % (
        figpath)
    fig.patch.set_facecolor('white')
    bgcol = np.array([15,15,15])/256

# Plot Stochastic Model ----------

ax      = axs[1]
mid     = 2
blabel  = [0,0,0,1]

# # figsize=(12,6)
# # ncol = 3
# # fig,axs = viz.init_2rowodd(ncol,proj,figsize=figsize,oddtop=False,debug=True)

# # Plot Stochastic Model Output
# for aid, mid in enumerate([0, 2]):
#     ax = axs.flatten()[aid]

# Make the Plot
ax = viz.add_coast_grid(ax, bboxplot, blabels=blabel, line_color=dfcol,
                        fill_color='gray')
pcm = ax.contourf(
    lon, lat, amvpats[mid, :, :, rid].T, levels=cint, cmap='cmo.balance')
# ax.pcolormesh(lon,lat,amvpats[mid,:,:,rid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
cl = ax.contour(
    lon, lat, amvpats[mid, :, :, rid].T, levels=cl_int, colors="k", linewidths=0.5)
ax.clabel(cl, levels=cl_int[::2], fontsize=fsz_tick, fmt="%.02f")

if useC:
    ptitle = "%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)" % (
        modelnames[mid], np.var(amvids[mid, :, rid]))
else:
    ptitle = "%s ($\sigma^2_{AMV}$ = %.04f $K^2$)" % (
        modelnames[mid], np.var(amvids[mid, :, rid]))
ax.set_title(ptitle)
if plotbbox:
    ax, ll = viz.plot_box(amvbboxes[rid], ax=ax, leglab="AMV",
                          color=dfcol, linestyle="dashed", linewidth=2, return_line=True)

viz.plot_mask(lon, lat, dmsks[mid], ax=ax, markersize=0.3)
ax.set_facecolor = bgcol
# ax = viz.label_sp(spid, case='lower', ax=ax, labelstyle="(%s)",
#                   fontsize=16, alpha=0.7, fontcolor=dfcol)
# spid += 1

# Plot CESM1 -----------------------
ax = axs[0]
cid = 1

blabel = [1,0,0,1]

# Make the Plot
ax = viz.add_coast_grid(ax, bboxplot, blabels=blabel, line_color=dfcol,
                        fill_color='gray')
pcm = ax.contourf(
    lon, lat, camvpats[rid][cid].T, levels=cint, cmap='cmo.balance')
ax.pcolormesh(lon, lat, camvpats[rid][cid].T, vmin=cint[0],
              vmax=cint[-1], cmap='cmo.balance', zorder=-1)
cl = ax.contour(lon, lat, camvpats[rid][cid].T,
                levels=cl_int, colors="k", linewidths=0.5)
ax.clabel(cl, levels=cl_int[::2], fontsize=fsz_tick, fmt="%.02f")
if useC:
    ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)" % (
        mconfigs[cid], np.var(camvids[rid][cid]))
else:
    ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f $K^2$)" % (
        mconfigs[cid], np.var(camvids[rid][cid]))
ax.set_title(ptitle)
if plotbbox:
    ax, ll = viz.plot_box(amvbboxes[rid], ax=ax, leglab="",
                          color=dfcol, linestyle="dashed", linewidth=2, return_line=True)

# ax = viz.label_sp(spid, case='lower', ax=ax, labelstyle="(%s)",
#                   fontsize=16, alpha=0.7, fontcolor=dfcol)

# -----------------------

cb = fig.colorbar(pcm, ax=axs.flatten(),
                  orientation='horizontal', fraction=0.030, pad=0.05)
cb.set_ticks(cint[::4])
cb.ax.set_xticklabels(clb, rotation=45)
cb.ax.tick_params(labelsize=fsz_tick)

cb.set_label("SST ($\degree C \, \sigma_{AMV}^{-1}$)",fontsize=14)


plt.savefig(savename, dpi=150, bbox_inches='tight',transparent=True)





# %% Save the parameters needed for the plot
"""
This is used in make_US_AMOC_figs.py
"""

# Select a Region
rid = 1
bbin = amvbboxes[rid]
print("Saving AMV for bbox: %s" % (bbin))

# Make dict in preparation
bbstr = "lon%ito%i_lat%ito%i" % (bbin[0], bbin[1], bbin[2], bbin[3])
savedict = {'bbox_in': amvbboxes[rid],
            'bbox_str': bbstr,
            'amv_pats_sm': amvpats[:, :, :, rid],  # [Model x Lon x Lat]
            'amv_pats_cesm': camvpats[rid],       # [Model][Lon x Lat]
            'amv_ids_sm': amvids[:, :, rid],     # [Model x Time]
            'amv_ids_cesm': camvids[rid],        # [Model][Time]
            'lon': lon,
            'lat': lat,
            'fnames': fnames,
            'origin_script': "calc_AMV_continuous.py",
            'mask_insig_hf': applymask,
            }

# Save it!
savename = "%sUSAMOC_AMV_Patterns_bbox%s_mask%i.npz" % (
    figpath, bbstr, applymask)
print("Saving output to %s" % savename)
np.savez(savename, **savedict)

# %% Lets analyze conditions at a particular point
input_path = datpath+"model_input/"
frcname = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
inputs = scm.load_inputs('SLAB_PIC', frcname, input_path, load_both=True)
lon, lat, h, kprevall, dampingslab, dampingfull, alpha, alpha_full = inputs

#%% Load Mean currents (albeit from CESM1 HTR)

from amv import loaders as dl
ds_uvel,ds_vvel = dl.load_current()

# %% SM Paper Draft 3 (CESM AMV Inset for Seasonal Cycle Figure)

# Calculate the AMV over bounding box bbin
cid = 1  # Set the CESM model
bbin = [-80, 0, 20, 60]
bboxplot = [-80, 0, 10, 60]
useC = True
sst_in = sst_cesm[cid]
amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, bbin, anndata=False,
                                   runmean=False, dropedge=5)

# Prepare Tick Labels
cl_int = np.arange(-0.45, 0.50, 0.05)
cb_lab = np.arange(-.5, .6, .1)

# Make the Plot
fig, ax = plt.subplots(
    1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 4),constrained_layout=True)
#ax = viz.add_coast_grid(ax, bbox=bboxplot)
ax = viz.add_coast_grid(ax, bboxplot, blabels=[1, 0, 0, 1], line_color=dfcol,
                        fill_color='gray')
pcm = ax.contourf(lon, lat, amvpat.T, levels=cint,
                  cmap='cmo.balance', extend='both')
# ax.pcolormesh(lon,lat,amvpat.T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
cl = ax.contour(lon, lat, amvpat.T, levels=cl_int, colors="k", linewidths=0.5)
ax.scatter(-30, 50, 200, marker="*", facecolor='yellow',
           zorder=9, edgecolor='k', linewidth=.5)
ax.clabel(cl, levels=cl_int, fontsize=10)

# Add Colorbar, Reduce tick labels
cb = fig.colorbar(pcm, ax=ax, orientation="vertical",
                  fraction=0.030, pad=0.1)
if useC:
    cb.set_label("SST ($\degree C \sigma_{AMV}^{-1}$)")
else:
    cb.set_label("SST ($K \sigma_{AMV}^{-1}$)")
cb.set_ticks(cb_lab)

# # Add Currents
# # Plot Currents
# qint = 2
# tlon = ds_uvel.TLONG.data.mean(0)
# tlat = ds_uvel.TLAT.data.mean(0)
# plotu = ds_uvel.UVEL.mean('ens').mean('month').values
# plotv = ds_vvel.VVEL.mean('ens').mean('month').values
# ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
#           color='navy',alpha=0.75)


plt.savefig("%sFig04b_SPG_Locator_%s.png" %
            (figpath, mconfigs[cid]), dpi=900, bbox_inches='tight',transparent=True)

# %% Plot of regional AMV with bounding Boxes (moved from plot_temporal_region)

"""
Old Param Combinations that worked...

Having the bounding box and legend box right below it
bboxtemp = [-90,5,15,68]
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4.5,3))
ax.text(-69,69,"Bounding Boxes",ha='center',bbox=props,fontsize=12) # (works for SPG Only)
ax.legend(ncol=1,fontsize=8,loc=6,bbox_to_anchor=(0, .75))
"""

cid      = 0
rids     = [0, 6, 5, ]
bboxtemp = [-85, -5, 15, 68]
cint     = np.arange(-0.45, 0.50, 0.05)
plotamv  = True  # Add AMV Plot as backdrop (False=WhiteBackdrop)
plotamvpat = camvpats[1][0]

# Select Which Lon/Lat to Plot
fix_lon = [-80, -40, 0]
fix_lat = [20, 40, 65]

# Start the PLot
fig, ax = plt.subplots(
    1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(3, 2))
ax = viz.add_coast_grid(ax, bboxtemp, fill_color='gray',
                        ignore_error=True, fix_lon=fix_lon, fix_lat=fix_lat)
if plotamv:
    pcm = ax.contourf(lon, lat, plotamvpat.T, cmap='cmo.balance', levels=cint)
fig.patch.set_alpha(1)  # solution

# # Plot the amv pattern
props = dict(boxstyle='square', facecolor='white', alpha=0.8)

# Add text
txtspg = ax.text(-38, 50, "SPG", ha='center', fontsize=15, weight='bold')
txtstgw = ax.text(-60, 27, "STGw", ha='center', fontsize=15, weight='bold')
txtstge = ax.text(-25, 27, "STGe", ha='center', fontsize=15, weight='bold')
for txt in [txtspg, txtstgw, txtstge]:
    txt.set_path_effects(
        [PathEffects.withStroke(linewidth=2.5, foreground='w')])

# First PLot Solid lines below
for bb in rids:
    ax, ll = viz.plot_box(bboxes[bb], ax=ax, leglab=regions[bb],
                          color=bbcol[bb], linestyle="solid", linewidth=3, return_line=True)

# Then plot dashed lines above
ls = []
for bb in rids:

    ax, ll = viz.plot_box(bboxes[bb], ax=ax, leglab=regions[bb],
                          color=bbcol[bb], linestyle="dotted", linewidth=3, return_line=True)
    ls.append(ll)

if pubready:
    plt.savefig("%sFig7b_Regional_BBOX_Locator_wamv.png" %
                figpath, dpi=900, bbox_inches='tight', transparent=True)
else:
    plt.savefig("%sRegional_BBOX_Locator_wamv.png" %
                figpath, dpi=100, bbox_inches='tight', transparent=True)
# %% Compare averaged and non-averaged AMV Pattern

bbin = [-80, 0, 20, 60]
nrun = 10
# Compute the AMV Pattern (for the stochastic model)
amvpats_m = np.zeros((nrun, nlon, nlat))*np.nan  # [model x lon x lat x region]
amvids_m = np.zeros((nrun, 1000)) * np.nan
apply_mask = False
mid = 0

# Do for Stochastic Models
for chunk in tqdm(range(nrun)):

    sst_in = sst_all[mid, ..., chunk*1000:(chunk+1)*1000]
    amvid, amvpat = proc.calc_AMVquick(sst_in, lonr, latr, bbin, anndata=True,
                                       runmean=False, dropedge=5, mask=None)

    amvpats_m[chunk, :, :] = amvpat.copy()
    amvids_m[chunk, :] = amvid.copy()

# %%

fig, axs = plt.subplots(2, 5, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6),
                        constrained_layout=True)
for c in range(10):
    ax = axs.flatten()[c]
    ax = viz.add_coast_grid(ax, bbox=bboxplot)
    #amvpat = amvpats_m[c,...]
   # amvpat = patup[c,7,0,:,:]
    pcm = ax.contourf(lon, lat, amvpat.T, levels=cint,
                      cmap='cmo.balance', extend='both')
    # ax.pcolormesh(lon,lat,amvpat.T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(lon, lat, amvpat.T, levels=cl_int,
                    colors="k", linewidths=0.5)
    ax.clabel(cl, levels=cl_int, fontsize=8)

# %% Load the weird stuff and compare
# Unpack and load AMV Patterns

amvpats = []
amvids = []
q = 1
for f in range(len(fnames)):
    if q == 0:
        expid = fnames[f]
        mid = 2
    else:
        expid = fnames[f]
        mid = 0

    rsst_fn = "%sproc/AMV_Region_%s.npz" % (datpath, expid)
    print("Loading %s" % rsst_fn)
    ld = np.load(rsst_fn, allow_pickle=True)  # .item()

    amvidx = ld['amvidx_region'].item()
    amvpat = ld['amvpat_region'].item()

    amvpats.append(amvpat)
    amvids.append(amvidx)

# Unpack dicts into array [nrun,nreg,nmod,nlon,nlat]
patup = scm.unpack_smdict(amvpats)
idxup = scm.unpack_smdict(amvids)


# %% Compare the NNAT Spectra (set-up)

use_lp = False  # Set to True to use LP

# Prepare the SSTs
insst = []
for i in range(3):  # Each of the Stochastic models
    varr = proc.sel_region(sst_all[i, ...], lon,
                           lat, [-80, 0, 20, 60], reg_avg=1, awgt=1)
    print(varr.shape)
    insst.append(varr)


# CESM [SLAB,FULL] --> [FULL,SLAB]
for i in range(2):
    varr = proc.sel_region(
        sst_cesm[i], lon, lat, [-80, 0, 20, 60], reg_avg=1, awgt=1)
    varr = proc.ann_avg(varr, 0)[1:]
    print(varr.shape)
    insst.append(varr)


# Combined plots
innames = modelnames + tuple(np.flip(cesmname))
incolors = tuple(mcolors) + tuple(np.flip(cesmcolor))

# Pass Lowpass Filter
if use_lp:
    insst = [proc.lp_butter(sst, 10, 6) for sst in insst]

# %% Compute the spectra

pct = 0.10
ssmooth = 500
cnsmooths = [100, 100]  # CESM1 Smoothing
pct = 0.10
alpha = 0.05      # For CI Calculatuions
sdof = 1000       # Degrees of freedom
cdofs = [900, 1800]
dt = 3600*24*365

lw = 3

if use_lp:
    xlm = [0, 0.15]
else:

    xlm = [0, 0.5]


# Additional Setup
nsmooths = np.concatenate([np.ones(3)*ssmooth, np.ones(2)*cnsmooths])
dofs = np.concatenate([np.ones(3)*sdof, cdofs],)
smoothname = "smth-obs%03i-full%02i-slab%02i" % (
    ssmooth, cnsmooths[0], cnsmooths[1])

# Perform Computations
specs, freqs, CCs, dofs, r1s = scm.quick_spectrum(insst, nsmooths, pct, dt=dt)


# Plot Result
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
speclabels = ["%s (%.3f $^{\circ}C^2$)" % (
    innames[i], np.var(insst[i])) for i in range(len(innames))]
ax, ax2 = viz.plot_freqlin(specs, freqs, speclabels, incolors, lw=lw,
                           ax=ax, plottitle="Spectral Estimate for NASST Index",
                           xlm=xlm, xtick=xtks, return_ax2=True)

ax.set_ylim([0, 0.25])
ax2.set_xlim(xlm)
ax.set_xlim(xlm)


plt.savefig("%sNASST_Index_Spectra_lpfilter%i_%s.png" %
            (figpath, use_lp, smoothname), dpi=100)
