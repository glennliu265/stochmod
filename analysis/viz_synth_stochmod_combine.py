#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Autocorrelation Plots for

synth_stochmod_spectra 
and
constant_v_variable


Contains Plots for
 - Stochastic Model Plots (ACs, Spectra)
 - Presentations, successively adding lines for lower/upper hierarchy
 - AGU 2021 Poster (Vertically stacked AC and Spectra)
 - OSM 2022 Plots
 - Cloud Locked Simulation Spectra + AC Comparison

Created on Wed Oct  6 22:17:26 2021

@author: gliu

"""
import matplotlib.ticker as tck
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import scipy as sp

#%% Settings

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20240119/'
proc.makedir(outpath)

pubready   = True

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
#fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")
fullauto = np.load(datpath+"CESM_clim/TS_FULL_Autocorrelation.npy")

# Set some Labels
mons3    = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# From SM Stylesheet -------------------------------------------------
# SM Lower Hierarchy (05/25/2021)
ecol_lower       = ["blue",'cyan','gold','red']
els_lower        = ["dotted","dashdot","dashed","solid"]
labels_lower     = ["All Constant (Level 1)",
                     r"Vary $F'$ (Level 2b)",
                     r"Vary $\lambda_a$ (Level 2a)",
                     "Vary $F'$ and $\lambda_a$ (Level 3)"] 

# SM Upper Hierarchy (05/25/2021)
labels_upper = ["h=50m",
                 "Vary $F'$ and $\lambda_a$ (Level 3)",
                 "Vary $F'$, $h$, and $\lambda_a$ (Level 4)",
                 "Entraining (Level 5)"]
ecol_upper = ('mediumorchid','red','magenta','orange')
els_upper = ["dashdot","solid","dotted","dashed"]


# Confidence Level Calculations
conf  = 0.95
tails = 2

# End SM Stylesheet --------------------------------------------------

#hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

# Set up Configuration
config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn009"   # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy"   #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at 
config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = outpath # Note need to fix this
config['smooth_forcing'] = False
config['method']      = 0 # Correction Factor (q-corr, lambda-based)
config['favg' ]       = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)



# Plotting Mode
darkmode = False
if darkmode:
    plt.style.use("dark_background")
    dfcol = "w"
else:
    plt.style.use("default")
    dfcol = "k"
#% ------------
#%% Clean Run
#% ------------

#% Load some data into the local workspace for plotting
query   = config['query']
mconfig = config['mconfig']
lags    = config['lags']
ftype   = config['ftype']
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])

# Run Model
#config['Fpt'] = np.roll(Fpt,1)
ac,sst,dmp,frc,ent,Td,kmonth,params = scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt        = params
darkmode = False

# Read in CESM autocorrelation for all points'
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = scm.load_data(mconfig,ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

# Plot some differences
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=params[2],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='gray')
ax.plot(lags,cesmautofull,color=dfcol,label='CESM Full',ls='dashdot')

for i in range(1,4):
    ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i])

ax.legend()
ax3.set_ylabel("Mixed Layer Depth (m)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"Default_Autocorrelation.png",dpi=200)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

sstall = sst

#%% Load Constant_v_vary experiments
# This was saved from constant_v_variable.py

savenames = "%sconst_v_vary_%s_runid%s_%s.npz" % (output_path,config['mconfig'],config['runid'],config['fname'])
print("Loading %s" % savenames)
ld        = np.load(savenames,allow_pickle=True)
c_acs     = ld['acs']
c_ssts    = ld['ssts']
expids    = ld['expids']
confs     = ld['confs']

# print("Saving clean run to %s" % savenames)
# np.savez(savenames,**{
#     'expids':expids,
#     'acs':acs,
#     'ssts':ssts,
#     'damps':damps,
#     'mlds':mlds,
#     'forces':forces,
#     'explongs':explongs
#     },allow_pickle=True)

# --------------------
#%% Calculate Confidence Intervals
# --------------------
def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags

plt.style.use('default')

nlags   = len(lags)
cfstoch = np.zeros([4,nlags,2])
for m in range(4):
    inac = ac[m]
    
    n = int(len(sst[m])/12)
    print(n)
    cfs = calc_conflag(inac,conf,tails,n)
    cfstoch[m,:,:] = cfs
cfslab = calc_conflag(cesmauto2,conf,tails,898)
cffull = calc_conflag(cesmautofull,conf,tails,1798)

#%% Plot SST Autocorrelation at the test point (SM Paper)
"""
Updated for Revision 01 08/24/2022
"""
notitle    = True  # Remove Title for publications
sepfig     = False # Plot figures separately, for presentaiton, or together)
sepentrain = False  # Separate entrain/non-entraining models
usegrid    = True
tickfreq   = 2
custom_order = np.flip([1,3,2,4,0]) # Set custom order
custom_order_upper = [0,2,1]
#reducetick = 3  # Set to true to label ticks more sparsely



# Indicate (more) Plotting Options
lw         = 3
markersize = 0
plotacs = c_acs
model   = 1

# Option to add tiled variable
addvar  = False
plotvar = Fpt
ylab    = "Forcing ($W/m^{2}$)"
#plotvar = Fpt/np.roll(Fpt,1) # Plot Ratio Current Month/Prev Month
#ylab    = "Forcing Ratio (Current/Previous)"
#plotvar = Fpt - np.roll(Fpt,1)
plotvar = damppt
ylab =  "Atmospheric Damping ($W/m^{2}$)"

# Figure Separation <1> Figure Creation
if sepfig:
    fig,ax = plt.subplots(1,1,figsize=(6,4))
else:
    fig,axs     = plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True,
                               constrained_layout=False)
    
    # Plot Lower Hierarchy
    ax = axs[0]
if notitle:
    title = ""
else:
    title = r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)"

# Initialize PLot
xtk2       = np.arange(0,37,2)
if addvar:
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,
                                 loopvar=plotvar,tickfreq=tickfreq,usegrid=usegrid)
    ax3.set_ylabel(ylab)
    ax3.yaxis.label.set_color('gray')
else:
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,usegrid=usegrid,
                             tickfreq=tickfreq)
    
# Plot CESM
ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=markersize,lw=lw)
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)

# Plot for each model in lower hierarchy
for i,e in enumerate([0,1,2,3]):
    title=""
    ax.set_ylabel("")
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label=labels_lower[i],color=ecol_lower[i],ls=els_lower[i],marker="o",markersize=markersize,lw=lw)
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color=ecol_lower[i],alpha=0.2)
    
if sepentrain:
    i = 2
    ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=markersize,lw=lw)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)

# Set labels, legend

if custom_order is not None:
    #get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    order = custom_order
    #add legend to plot
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
else:
    ax.legend(fontsize=12,ncol=1) # fontsize 10, ncol=3 was the old one

ax.set_ylabel("Correlation")

# --------------------------------------------
# Plot Upper Hierarchy

# Figure Separation <2> Figure Save, Fig 2 Creation
if sepfig:
    plt.savefig(outpath+"Autocorrelation_lower-hierarchy_%s.png"%locstring,dpi=200,bbox_inches='tight')
    fig,ax = plt.subplots(1,1,figsize=(6,4))
else:
    ax = viz.label_sp(0,case='lower', ax=ax, fontsize=16, labelstyle="(%s)",x=0.010,alpha=0.25)
    ax = axs[1]
if notitle:
    title = ""
else:
    title = "Adding Varying Mixed Layer Depth ($h$) and Entrainment"

#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)

# Plot CESM Data
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,
                        tickfreq=tickfreq,usegrid=usegrid)
#ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=3)
#ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=markersize,lw=lw)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

if sepentrain:
    plotrange = [3] # Just Entrain
else:
    plotrange = [2,3] # Entrain + Vary h

for i in plotrange:
    ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=markersize,lw=lw)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)

if custom_order is not None:
    #get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    order = custom_order_upper
    #add legend to plot
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
else:
    ax.legend(fontsize=12,ncol=1) # fontsize 10, ncol=3 was the old one
plt.tight_layout()
ax.set_ylabel("")

ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

if notitle is False:
    plt.suptitle("Monthly SST Autocorrelation at 50N, 30W (Lag 0 = February)",fontsize=12,y=1.01)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

# Figure Separation <3> Figure 2 Save
if pubready:
    ax = viz.label_sp(1,case='lower', ax=ax, fontsize=16, labelstyle="(%s)",x=0.010,alpha=0.25)
    plt.savefig(outpath+"Fig05_SPG_Autocorrelation.png",dpi=1200,bbox_inches='tight',format='png')
else:
    if sepfig:
        ax.set_ylabel("Correlation")
        plt.savefig(outpath+"Autocorrelation_upper-hierarchy_%s.png"%locstring,dpi=200,bbox_inches='tight')
    else:
        ax = viz.label_sp(1,case='lower', ax=ax, fontsize=16, labelstyle="(%s)",x=0.010,alpha=0.25)
        plt.savefig(outpath+"Autocorrelation_2-PANEL_%s.png"%locstring,dpi=200,bbox_inches='tight')

# Unified ylabel
for ax in axs:
    ax.set_xlabel("")
fig.text(0.5, 0.01, 'Lag (Months)', ha='center',fontsize=12)

#%% Load and calculate CESM Spectra
cssts = scm.load_cesm_pt(datpath,loadname='both',grabpoint=[-30,50])
#%% Calculate Spectra


debug    = False
notitle  = True

# Some edits for Draft 4
use_ann     = True # Option to take ann avg before calcualtion
useC        = True # Swap to label as degree celsius

# Smoothing Params
nsmooth    = 350
cnsmooths  = [100,100]
pct        = 0.00

smoothname = "smth-obs%03i-full%02i-slab%02i" % (nsmooth,cnsmooths[0],cnsmooths[1])

# Spectra Plotting Params
plottype = "freqlin"
xlm  = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0,3.0]
plotids = [[0,1,2,3,8],
           [5,6,7]
           ]
detrend_cesm = True

# Combine lower and upper hierarchy
inssts   = [c_ssts[0][1],c_ssts[1][1],c_ssts[2][1],c_ssts[3][1],sst[1],sst[2],sst[3],cssts[0],cssts[1]]
nsmooths = np.concatenate([np.ones(len(inssts)-2)*nsmooth,cnsmooths])
labels   = np.concatenate([labels_lower,labels_upper[1:],['CESM-FULL','CESM-SLAB']])

if detrend_cesm:
    for cexp in ['CESM-FULL','CESM-SLAB']:
        c_index = list(labels).index(cexp)
        sst_in = inssts[c_index]
        #sst_dt = sp.signal.detrend(sst_in,)
        output,tsmodel,residual = proc.polyfit_1d(np.arange(sst_in.shape[0]),sst_in,0)
        sst_dt = sst_in - tsmodel
        inssts[c_index] = sst_dt
        print("Detrended %s" % cexp)
        if debug:
            fig,ax=plt.subplots(1,1)
            ax.plot(sst_in,label="Undetreded SSTs")
            ax.plot(sst_dt,label="Detrended SSTs")
            ax.legend()
        

if useC:
    speclabels = ["%s (%.2f$ \, \degree C^{2}$)" % (labels[i],np.var(inssts[i])) for i in range(len(inssts))]

else:
    speclabels = ["%s (%.2f$ \, K^{2}$)" % (labels[i],np.var(inssts[i])) for i in range(len(inssts))]
allcols  = np.concatenate([ecol_lower,ecol_upper[1:],[dfcol,"gray"]])

# Calculate Autocorrelation (?)
allacs,allconfs = scm.calc_autocorr(inssts,lags,kmonth+1,calc_conf=True)

# Convert Dict --> Array
oac=[]
ocf=[]
for i in range(len(allacs)):
    oac.append(allacs[i])
    ocf.append(allconfs[i])
allacs=oac
allconfs=ocf

if debug: # Check if variables were properly concatenated using ACs
    fig,axs = plt.subplots(1,2,figsize=(16,4))
    ax = axs[0]
    plotac = allacs[:4]
    for i in range(4):
        ax.plot(lags,plotac[i],label=labels_lower[i],color=ecol_lower[i],)
    ax.legend()
    ax = axs[1]
    plotac = allacs[4:]
    for i in range(3):
        ax.plot(lags,plotac[i],label=labels_upper[i+1],color=ecol_upper[i+1],)
    ax.legend()

# Do spectral Analysis
if use_ann:
    inssts = [proc.ann_avg(sst,0) for sst in inssts]
    inssts[-2] = inssts[-2][1:] # Drop 1st year for even yrs
    inssts[-1] = inssts[-1][1:] # Drop 1st year for even yrs
    dt = 3600*24*365
else:
    dt = None
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooths,pct,dt=dt)
#cspecs,cfreqs,cCCs,cdofs,cr1s = scm.quick_spectrum(cssts,cnsmooths,pct)

# Convert to list for indexing NumPy style
convert = [specs,freqs,speclabels,allcols]
for i in range(len(convert)):
    convert[i] = np.array(convert[i])
specs,freqs,speclabels,allcols = convert

# Compute the variance of each ts
sstvars     = [np.var(insst) for insst in inssts]
sstvars_lp  = [np.var(proc.lp_butter(insst,120,6)) for insst in inssts]
sststds = [np.std(insst) for insst in inssts]

if useC:
    sstvars_str = ["%s (%.2f $\degree C$)" % (labels[sv],sstvars[sv]) for sv in range(len(sstvars))]
else:
    sstvars_str = ["%s (%.2f $K^2$)" % (labels[sv],sstvars[sv]) for sv in range(len(sstvars))]

#%% # Plot the spectra

"""
Updated for Revision 01 08/24/2022
"""

plottype    = 'freqlin'#'freqlin'
sepentrain  = False  # Separate entrain/non-entraining models
sepfig      = False
#includevar = False # Include Variance in Legend
lower_focus = False # Set to true to include specific lines for this particular plot 
periodx     = False # Set to true to have just 1 x-axis, with periods
linearx     = 1 # Keep frequency axis linear, period axis marked 
lw          = 3
incl_legend = True 

usegrid     = False

axis_fsz    = 14
tick_fsz    = 14
legend_fsz  = 12

xtks = [0.01, 0.05, 0.1 , 0.2 , 0.5 ]

if sepentrain:
    plotids = [[0,1,2,3,8,5],
               [6,7]
               ]
    plotids = [[0,1,2,3,8],
               [3,5,6,7]
               ]
else:
    if lower_focus:
        plotids = [[0,3,8],
                   [3,5,6,7]
                   ]
    else:
        plotids = [[0,2,1,3,8],
                   [5,6,7]
                   ]

if notitle:
    titles = ["",""]
else:
    titles = (r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",
              "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
              )
sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n" + \
"Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1])

#% Plot the spectra
if sepfig is False:
    fig,axs = plt.subplots(1,2,figsize=(16,4))

for i in range(2):
    
    if sepfig is True:
        fig,ax = plt.subplots(1,1,figsize=(8,4))
    else:
        ax = axs[i]
    
    plotid = plotids[i]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,
                             lw=lw,linearx=linearx,usegrid=usegrid)
        if useC:
            ylabel = "Power ($\degree C^2/cpy$)"
        else:
            ylabel = "Power ($K^2/cpy$)"
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=lw,
                             semilogx=True)
        #ax.set_ylim([1e-1,1e1])
        if useC:
            ylabel="Variance ($\degree C^2$)"
        else:
            ylabel = "Variance ($K^2$)"
        
    #ax2.set_xlabel("Period (Years)")
    if i == 1:
        ax.set_ylabel("")
    ax.set_xlabel("")
        
    plt.setp(ax2.get_xticklabels(), rotation=0,fontsize=tick_fsz)
    plt.setp(ax.get_xticklabels(), rotation=0,fontsize=tick_fsz)
    
    # if plottype is not 'freqlog':
    ax.set_ylim(ylm)
    
    if periodx: # Switch Frequency with Period for x-axis.
        ax2.set_xlabel("")
        xtk2 = ax2.get_xticklabels()
        xtk2new = np.repeat("",len(xtk2))
        ax2.set_xticklabels(xtk2new,fontsize=tick_fsz)
        ax.set_xticklabels(1/xtks,fontsize=tick_fsz)
    
    if incl_legend:
        lgd = viz.reorder_legend(ax)
    if sepfig is True: # Save separate figures
        if periodx:
            ax.set_xlabel('Period (Years)',fontsize=axis_fsz)
        else:
            ax.set_xlabel('Frequency (cycles/year)',fontsize=axis_fsz)
        if useC:
            ax.set_ylabel("Power ($\degree C^2/cpy$)",fontsize=axis_fsz)
        else:
            ax.set_ylabel("Power ($K^2/cpy$)",fontsize=axis_fsz)
        savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i_part%i.png" % (outpath,plottype,smoothname,pct*100,i)
        plt.savefig(savename,dpi=200,bbox_inches='tight')
    else:
        if i == 0:
             ax.set_xlabel("")
             if useC:
                 ax.set_ylabel("Power ($\degree C^2/cpy$)",fontsize=axis_fsz)
             else:
                 ax.set_ylabel("Power ($K^2/cpy$)",fontsize=axis_fsz)
        #if i == 1:
           # ax.set_xlabel("Period (Years)")
        ax = viz.label_sp(i,case='lower', ax=ax, fontsize=16, labelstyle="(%s)")
    
    # Y-axis Adjust
    plt.setp(ax.get_yticklabels(),fontsize=tick_fsz)
    # Turn on minor label
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())


    

if sepfig is False:
    #fig.text(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
    if periodx:
        fig.text(0.5, -0.05, 'Period (Years)', ha='center',fontsize=axis_fsz)
    else:
        fig.text(0.5, 1.02, 'Period (Years)', ha='center',fontsize=axis_fsz)
        fig.text(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=axis_fsz)
    #plt.suptitle("SST Power Spectra at 50$\degree$N, 30$\degree$W",y=1.15,fontsize=14)
    if notitle is False:
        plt.suptitle(sharetitle,y=1.05,fontsize=14)
    savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i.png" % (outpath,plottype,smoothname,pct*100)
    
    if pubready:
        plt.savefig("%sFig06_SPG_Spectra.eps"% (outpath),dpi=1200,bbox_inches='tight',format='eps')
    else:
        plt.savefig(savename,dpi=200,bbox_inches='tight')


#%% Load and compare with the spectra from the cloud-locked simnulations
# Copied figure from above

clpath            = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM1_Cloud_Locking/proc/"
nsmooth_select_cl = 25
cl_ld = np.load("%sCloudLocking_%s_Spectra_nsmooth%i.npz" % (clpath,"TS",nsmooth_select_cl))
kloncl,klatcl = proc.find_latlon(-30,50,cl_ld['lon'],cl_ld['lat'])

cloudlock_spec = cl_ld['specs'][:,klatcl,kloncl]
cloudlock_freq = cl_ld['freq']



plottype    = 'freqlin'#'freqlin'
sepentrain  = False  # Separate entrain/non-entraining models
sepfig      = False
#includevar = False # Include Variance in Legend
lower_focus = False # Set to true to include specific lines for this particular plot 
periodx     = False # Set to true to have just 1 x-axis, with periods
linearx     = 1 # Keep frequency axis linear, period axis marked 
lw          = 3
incl_legend = True 

usegrid     = False

axis_fsz    = 14
tick_fsz    = 14
legend_fsz  = 12

xtks = [0.01, 0.05, 0.1 , 0.2 , 0.5 ]

if sepentrain:
    plotids = [[0,1,2,3,8,5],
               [6,7]
               ]
    plotids = [[0,1,2,3,8],
               [3,5,6,7]
               ]
else:
    if lower_focus:
        plotids = [[0,3,8],
                   [3,5,6,7]
                   ]
    else:
        plotids = [[0,2,1,3,8],
                   [5,6,7,8]
                   ]

if notitle:
    titles = ["",""]
else:
    titles = (r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",
              "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
              )
sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n" + \
"Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1])

#% Plot the spectra


fig,ax = plt.subplots(1,1,figsize=(8,4))

plotid = plotids[i]


    
if plottype == "freqxpower":
    ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                         ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
elif plottype == "freqlin":
    ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                         ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,
                         lw=lw,linearx=linearx,usegrid=usegrid)
    if useC:
        ylabel = "Power ($\degree C^2/cpy$)"
    else:
        ylabel = "Power ($K^2/cpy$)"
elif plottype == "freqlog":
    ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                         ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=lw,
                         semilogx=True)
    #ax.set_ylim([1e-1,1e1])
    if useC:
        ylabel="Variance ($\degree C^2$)"
    else:
        ylabel = "Variance ($K^2$)"
        
    #ax2.set_xlabel("Period (Years)")
if i == 1:
    ax.set_ylabel("")
ax.set_xlabel("")
    
plt.setp(ax2.get_xticklabels(), rotation=0,fontsize=tick_fsz)
plt.setp(ax.get_xticklabels(), rotation=0,fontsize=tick_fsz)

# if plottype is not 'freqlog':
ax.set_ylim(ylm)
if periodx: # Switch Frequency with Period for x-axis.
    ax2.set_xlabel("")
    xtk2 = ax2.get_xticklabels()
    xtk2new = np.repeat("",len(xtk2))
    ax2.set_xticklabels(xtk2new,fontsize=tick_fsz)
    ax.set_xticklabels(1/xtks,fontsize=tick_fsz)

if incl_legend:
    lgd = viz.reorder_legend(ax)
if sepfig is True: # Save separate figures
    if periodx:
        ax.set_xlabel('Period (Years)',fontsize=axis_fsz)
    else:
        ax.set_xlabel('Frequency (cycles/year)',fontsize=axis_fsz)
    if useC:
        ax.set_ylabel("Power ($\degree C^2/cpy$)",fontsize=axis_fsz)
    else:
        ax.set_ylabel("Power ($K^2/cpy$)",fontsize=axis_fsz)
    savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i_part%i.png" % (outpath,plottype,smoothname,pct*100,i)
    plt.savefig(savename,dpi=200,bbox_inches='tight')
else:
    if i == 0:
         ax.set_xlabel("")
         if useC:
             ax.set_ylabel("Power ($\degree C^2/cpy$)",fontsize=axis_fsz)
         else:
             ax.set_ylabel("Power ($K^2/cpy$)",fontsize=axis_fsz)
    #if i == 1:
       # ax.set_xlabel("Period (Years)")
    #ax = viz.label_sp(i,case='lower', ax=ax, fontsize=16, labelstyle="(%s)")


# Add cloud-locking spectra
ax.plot(cloudlock_freq[1:]*dt,cloudlock_spec/dt,color="cyan")
#ax.legend()

# Y-axis Adjust
plt.setp(ax.get_yticklabels(),fontsize=tick_fsz)
# Turn on minor label
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

#%% Barplots

fig,axs = plt.subplots(1,2,constrained_layout=True,sharey=True,figsize=(8,4))

for i in range(2):
    
    # Get Axis and plotids
    ax = axs.flatten()[i]
    plotid = plotids[i]
    
    # Plot the bars
    bar   = ax.bar(np.array(labels)[plotid],np.array(sstvars)[plotid],
                   color=np.array(allcols)[plotid],
                   alpha=1)
    
    #viz.label_barplots(np.array(sstvars)[plotid],ax=ax)
    
    barlp = ax.bar(np.array(labels)[plotid],np.array(sstvars_lp)[plotid],
                   color=np.array(allcols)[plotid],width=0.8,hatch="//",alpha=0.3)
    
    #viz.label_barplots(np.array(sstvars_lp)[plotid],ax=ax)
    
    # Label the bars
    
    
    # Set Labels, Grids
    if i == 0:
        ax.set_ylabel("SST Variance ($K^2$)")
        fig.text(0.5, -0.05, 'Model Configuration', ha='center',fontsize=12)
        
    ax.set_xticklabels(np.array(labels)[plotid],rotation=45)
    ax.grid(True,ls='dashed')

#%%


# fig, ax= plt.subplots(1,1)
# dtplot = 3600*24*365

# ax.semilogx(freqs[0]*dtplot,specs[0]/dtplot)
# ax.set_xticks(xtks)
# ax.set_xlim([xtks[0],xtks[-1]])

"""
Presentation Plots (Successively Adding Lines)
"""
# ------------------------------
#%%  Sucessively Add Each Line 
# This is for the LOWER hierarchy
# ------------------------------
"""
Note this is the section used to generate figures for the following
presentations
    - Generals Exam Presentation
    - Presentation to ECCO/CSU Group
    - OSM 2022 Presentation
"""

lw         = 3
markersize = 6
addvar     = False
darkmode   = True
incl_slab  = False
add_hvar   = True
ylims      = [-0.05,1.05]

if darkmode:
    plt.style.use('dark_background')
    dfcol      = 'w'
    dfalph     = 0.30
    dfalph_col = 0.40
else:
    plt.style.use('default')
    dfcol      = 'k'
    dfalph     = 0.1
    dfalph_col = 0.25 

#plotlags = np.arange(0,24)
lags    = np.arange(0,25,1)
xtk2    = np.arange(0,25,2)
for es in range(4+add_hvar):
    loopis = np.arange(0,es+1)
    if es > 3:
        loopis = np.arange(0,es)
    print(loopis)
    
    figs,ax = plt.subplots(1,1,figsize=(6,4))
    if addvar:
        ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=plotvar)
        ax3.set_ylabel(ylab)
        ax3.yaxis.label.set_color('gray')
    else:
        ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
    
    if incl_slab is True:
        # Plot CESm1-SLAB
        ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=markersize,lw=lw)
        ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=dfalph)
    
    # Plot stochastic models
    for i,e in enumerate(loopis):
        title=""
        ax.set_ylabel("")
        cfs = confs[e,model,:,:]
        ax.plot(lags,plotacs[e][model][lags],
                label=labels_lower[i],color=ecol_lower[i],
                ls=els_lower[i],marker="o",markersize=markersize,
                lw=lw)
        ax.fill_between(lags,cfs[lags,0],cfs[lags,1],color=ecol_lower[i],alpha=dfalph_col)
        
    
    if es == 4: # Plot hvary if option is set
        ax.plot(lags,ac[2][lags],label=labels_upper[2],color=ecol_upper[2],ls=els_upper[2],marker="o",markersize=markersize,lw=lw)
        ax.fill_between(lags,cfstoch[2,lags,0],cfstoch[2,lags,1],color=ecol_upper[2],alpha=0.25)
        
    ax.legend(fontsize=8,ncol=3)
    
    ax.set_ylim(ylims)
    ax.set_ylabel("Correlation")
    plt.suptitle("SST Autocorrelation: Non-Entraining Stochastic Model \n Adding Varying Damping and Forcing",fontsize=12)
    plt.tight_layout()
    plt.savefig("%sAutocorrelation_LowerHierarchy_%i.png"% (outpath,es),dpi=150)
    print("Done With %i"% es)

# ---------------------------------------------
#%% Upper Hierarchy (succesively Add each line)
# ---------------------------------------------
lw         = 3
markersize = 3#4
addvar     = False

#plotlags = np.arange(0,24)

lags    = np.arange(0,37,1)
xtk2    = np.arange(0,37,2)

loopis = [2,3] # Which is to look through

for es in range(2):
    print(loopis)
    
    figs,ax = plt.subplots(1,1,figsize=(6,4))
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
    
    # Plot CESM1-FULL
    ax.plot(lags,cesmautofull[lags],color=dfcol,label='CESM1 Full',ls='dashdot',marker="o",markersize=markersize,lw=lw)
    ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color=dfcol,alpha=dfalph)
    
    # plot for the stochastic model
    
    for count in range(es+1):
        i = loopis[count]
        
        ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=markersize,lw=lw)
        ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)
        
        ax.legend(fontsize=8,ncol=3)
        
    ax.set_ylabel("Correlation")
    plt.suptitle("SST Autocorrelation: Non-Entraining Stochastic Model \n Adding Varying Damping and Forcing",fontsize=12)
    plt.tight_layout()
    plt.savefig("%sAutocorrelation_UpperHierarchy_%i.png"% (outpath,es),dpi=150)
    print("Done With %i"% es)


"""
Conference Plots
"""
# ----------------------
#%% OSM, Separate Plots
# ----------------------

plottype   = 'freqlin'#'freqlin'
sepentrain = False  # Separate entrain/non-entraining models
sepfig     = True
#includevar = False # Include Variance in Legend
lower_focus = True # Set to true to include specific lines for this particular plot 
plotslab    = True # Set to true to plot slab

if sepentrain:
    plotids = [[0,1,2,3,8,5],
               [6,7]
               ]
    plotids = [[0,1,2,3,8],
               [3,5,6,7]
               ]
else:
    if lower_focus:
        plotids = [[0,3,8],
                   [3,5,6,7]
                   ]
    else:
        plotids = [[0,1,2,3,8],
                   [5,6,7]
                   ]

if plotslab:
    plotids = [[7,8],[7,8,3],[7,8,3,5],[7,8,5,6]]
else:
    plotids = [[7,3],[7,5],[7,5,6]]

if notitle:
    titles = ["",""]
else:
    titles = (r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",
              "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
              )
sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n" + \
"Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1])


for i in range(len(plotids)):

    fig,ax = plt.subplots(1,1,figsize=(8,4))
    plotid = plotids[i]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle="",xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle="",xtick=xtks,xlm=xlm,return_ax2=True,lw=lw)
        ylabel = "Power ($K^2/cpy$)"
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle="",xtick=xtks,xlm=xlm,return_ax2=True,lw=lw,
                             semilogx=True)
        ylabel = "Variance ($K^2$)"
    ax2.set_xlabel("Period (Years)")
    
    if i == 1:
        ax.set_ylabel("")
    ax.set_xlabel("")
    
    plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    
    # if plottype is not 'freqlog':
    ax.set_ylim(ylm)
    
    ax2.set_xlabel("")
    xtk2 = ax2.get_xticklabels()
    xtk2new = np.repeat("",len(xtk2))
    ax2.set_xticklabels(xtk2new)
    
    ax.set_xticklabels(1/np.array(xtks))
    
    ax.set_xlabel('Period (Years)',fontsize=12)
    ax.set_ylabel("Power ($K^2/cpy$)")
    
    savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i_part%i.png" % (outpath,plottype,smoothname,pct*100,i)
    plt.savefig(savename,dpi=200,bbox_inches='tight')
# -------------------------------------------------------
#%% Autocorrelation AGU Style Plot (Vertically Stacked)
# -------------------------------------------------------
fig,axs     = plt.subplots(2,1,figsize=(6,8),sharex=True,sharey=True)


# Plot Lower Hierarchy
ax = axs[0]

title = r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)"
title = ""
plotacs = c_acs
model   = 1

# Option to add tiled variable
addvar  = False
plotvar = Fpt
ylab    = "Forcing ($W/m^{2}$)"
#plotvar = Fpt/np.roll(Fpt,1) # Plot Ratio Current Month/Prev Month
#ylab    = "Forcing Ratio (Current/Previous)"
#plotvar = Fpt - np.roll(Fpt,1)
plotvar = damppt
ylab =  "Atmospheric Damping ($W/m^{2}$)"

xtk2       = np.arange(0,37,2)
if addvar:
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=plotvar)
    ax3.set_ylabel(ylab)
    ax3.yaxis.label.set_color('gray')
else:
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=4)
#ax.scatter(lags,cesmauto2[lags],10,label="",color='k')
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)
#ax.grid(minor=True)

for i,e in enumerate([0,1,2,3]):
    
    title=""
    ax.set_ylabel("")
    
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label=labels_lower[i],color=ecol_lower[i],ls=els_lower[i],marker="o",markersize=4)
    #ax.scatter(lags,plotacs[e][model],10,label="",color=ecol[i])
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color=ecol_lower[i],alpha=0.2)
    
    ax.legend(fontsize=10,ncol=3)
ax.set_ylabel("")
fig.text(0.04,0.5,'Correlation',va='center', rotation='vertical')


# --------------------------------------------

# Plot Upper Hierarchy
ax = axs[1]
#title = "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
title = ""
#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)

# Plot CESM Data
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
#ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=3)
#ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

for i in range(2,4):
    ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=3)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
ax.set_ylabel("")

plt.suptitle("Monthly SST Autocorrelation at 50$\degree$N, 30$\degree$W",fontsize=12,y=1.01)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

plt.savefig(outpath+"Autocorrelation_2-PANEL_%s_vertical.png"%locstring,dpi=200,bbox_inches='tight')
# -------------------------------------------------------
#%% Spectra AGU Version (Vertically Stacked)
# -------------------------------------------------------
debug = False

# Smoothing Params
nsmooth    = 500
cnsmooths = [120,100]
pct        = 0.10

smoothname = "smth-obs%03i-full%02i-slab%02i" % (nsmooth,cnsmooths[0],cnsmooths[1])

# Spectra Plotting Params
xlm = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]

ylm  = [0,3.0]

plotids = [[0,1,2,3,8],
           [4,5,6,7]
           ]
plottype = "freqlin"


# Convert Dict --> Array
oac=[]
ocf=[]
for i in range(len(allacs)):
    oac.append(allacs[i])
    ocf.append(allconfs[i])
allacs=oac
allconfs=ocf
    
    
# Do spectral Analysis
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooths,pct)
#cspecs,cfreqs,cCCs,cdofs,cr1s = scm.quick_spectrum(cssts,cnsmooths,pct)

# Convert to list for indexing NumPy style
convert = [specs,freqs,speclabels,allcols]
for i in range(len(convert)):
    convert[i] = np.array(convert[i])
specs,freqs,speclabels,allcols = convert



titles = (r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",
          "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
          )

titles = ("","")
sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n" 

#+ \
#"Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1])

#% Plot the spectra
fig,axs     = plt.subplots(2,1,figsize=(7.5,10),sharex=True,sharey=True)
for i in range(2):
    ax = axs[i]
    plotid = plotids[i]
    if i == 1: # Drop constant h from second plot
        plotid = plotid[1:]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2= viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=2.5)
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    ax2.set_xlabel("")
    xtk2 = ax2.get_xticklabels()
    xtk2new = np.repeat("",len(xtk2))
    ax2.set_xticklabels(xtk2new)
    
    ax.set_xticklabels(1/xtks)
    ax.set_ylabel("")
    if i == 0:
         ax.set_xlabel("")
    if i == 1:
        ax.set_xlabel("Period (Years)")
    
        
    plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    
    ax.set_ylim(ylm)
    
    #ax.set_xlabel(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
#ax.set_xlabel(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
fig.text(0.04,0.5,'Power ($\degree C^2/cpy$)',va='center', rotation='vertical')
#plt.suptitle("SST Power Spectra at 50$\degree$N, 30$\degree$W",y=1.15,fontsize=14)
plt.suptitle(sharetitle,fontsize=14,y=.925)
savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i_vertical.png" % (outpath,plottype,smoothname,pct*100)
plt.savefig(savename,dpi=200,bbox_inches='tight')

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

#% Plot the spectra
fig,axs = plt.subplots(1,2,figsize=(16,4))

inax1 = plt.axes([0,0,1,1])
inax2 = plt.axes([0,1,0,1])

for i in range(2):
    ax = axs[i]
    plotid = plotids[i]
    
    
    
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=1.5)
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    ax2.set_xlabel("Period (Years)")
    if i == 1:
        ax.set_ylabel("")
        
        
    
plt.suptitle("SST Power Spectra at 50N, 30W",y=1.05)
savename = "%sNASST_Spectra_Stochmod_%s_nsmooth%i_pct%03i.png" % (outpath,plottype,nsmooth,pct*100)
plt.savefig(savename,dpi=200,bbox_inches='tight')


# ------------------------------------------------------
#%% Plot Specific ranges similar to Patrizio et al. 2021
# ------------------------------------------------------

plottype = 'freqlog'#'freqlin'
# Shorter Timescales
# xper = np.array([25,10,5,2.5,1,0.5,0.2])
# xtks = 1/xper
# xlm  = [xtks[0],xtks[-1]]
# Clement et al. 2015 range
xper = np.array([50,25,10,5,2.5,1.0])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0.5e-1,5e1]

sepfig   = True
if notitle:
    titles = ["",""]
else:
    titles = (r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",
              "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
              )
sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n" + \
"Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1])

#% Plot the spectra
if sepfig is False:
    fig,axs = plt.subplots(1,2,figsize=(16,4))


for i in range(2):
    
    if sepfig is True:
        fig,ax = plt.subplots(1,1,figsize=(8,4))
    else:
        ax = axs[i]
    plotid = plotids[i]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
        
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=lw)
        ylabel = "Power ($K^2/cpy$)"
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,lw=lw)
        #ax.set_ylim([1e-1,1e1])
        #ax.set_ylim([1e-2,1e2])
        ylabel = "Variance ($K^2$)"
    ax2.set_xlabel("Period (Years)")
    if i == 1:
        ax.set_ylabel("")
    ax.set_xlabel("")
        
    plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    
    ax.set_ylim(ylm)
    
    ax2.set_xlabel("")
    xtk2 = ax2.get_xticklabels()
    xtk2new = np.repeat("",len(xtk2))
    ax2.set_xticklabels(xtk2new)
    
    ax.set_xticklabels(1/xtks)
    
    if sepfig is True: # Save separate figures
        ax.set_xlabel('Period (Years)',fontsize=12)
        ax.set_ylabel("Power ($K^2/cpy$)")
        
        savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i_part%i.png" % (outpath,plottype,smoothname,pct*100,i)
        plt.savefig(savename,dpi=200,bbox_inches='tight')
    else:
        if i == 0:
             ax.set_xlabel("")
             ax.set_ylabel("Power ($K^2/cpy$)")
        #if i == 1:
           # ax.set_xlabel("Period (Years)")
        ax = viz.label_sp(i,case='lower', ax=ax, fontsize=16, labelstyle="(%s)")
        

if sepfig is False:
    #fig.text(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
    fig.text(0.5, -0.05, 'Period (Years)', ha='center',fontsize=12)
    #plt.suptitle("SST Power Spectra at 50$\degree$N, 30$\degree$W",y=1.15,fontsize=14)
    if notitle is False:
        plt.suptitle(sharetitle,y=1.05,fontsize=14)
    savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i.png" % (outpath,plottype,smoothname,pct*100)
    plt.savefig(savename,dpi=200,bbox_inches='tight')


"""
Incomplete Section (wip)
"""
# ----------------------------
#%% Make a plot with an inset
# ----------------------------

"""
# Run with useann=True
yrfreq = freqs[0]
yrspec = specs[0]
# Run with useann=False
monfreq = freqs[0]
monspec = specs[0]
"""


fig,ax = plt.subplots(1,1)
ax.plot(yrfreq*dtplot , yrspec/dtplot,label="yearly")
ax.plot(monfreq*dtplot, monspec/dtplot,label="monthly")
ax.set_xlim([0,1])
ax.set_ylabel("Frequency (1/year)")
#ax.plot(yrfreqs*dtplot,yrspec/dtplot)



# -----------------------------------------------------------------------------
# %% Monthly Variance
# -----------------------------------------------------------------------------
"""
Note: Copied lines from "calculate spectra cell around l.371

"""

# Make Monthly Variance Plots
inssts   = [c_ssts[0][1],c_ssts[1][1],c_ssts[2][1],c_ssts[3][1],sst[1],sst[2],sst[3],cssts[0],cssts[1]]
nsmooths = np.concatenate([np.ones(len(inssts)-2)*nsmooth,cnsmooths])
labels   = np.concatenate([labels_lower,labels_upper[1:],['CESM-FULL','CESM-SLAB']])
allcols  = np.concatenate([ecol_lower,ecol_upper[1:],[dfcol,"gray"]])

# Check and compute monthly variance
[print(sst.shape) for sst in inssts]
monvars = [proc.calc_monvar(sst) for sst in inssts]

# Load/Set some plotting params
plotids = [[0,1,2,4,7,8],[4,5,6,7,8]] # Top/Bottom Hierarchy Levels


# Save the Output for debugging
outdir = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%ssynth_stochmod_combine_output.npz" % outdir
np.savez(savename,**{
    'ssts':inssts,
    'labels' : labels,
    'colors' : allcols,
    'monvars': monvars,
    'acs' : allacs,
    'specs': specs,
    'freqs': freqs,
    })

#%% Plot Monthly Variance for Each level of the hiErarchy
fsz_axis = 14
fsz_sp   = 16
fsz_ticks = 14
labelssp = ["Levels 1-3","Levels (3-5)"]
# Visualize Output
fig,axs = plt.subplots(2,1,sharey=True,constrained_layout=True,figsize=(8,8))

for a in range(2):
    ax = axs[a]
    
    plotid = plotids[a]
    
    # Plot Lower hierarchy
    for pid in plotid:
        if pid > 6:
            ls='dashed'
            marker="d"
        else:
            ls="solid"
            marker="o"
        ax.plot(mons3,monvars[pid],label=labels[pid],color=allcols[pid],lw=2.5,marker=marker,ls=ls)
    
    # Axis Formatting
    ax.set_xlim([0,11])
    ax.legend(ncol=2,fontsize=fsz_axis-2)
    ax.grid(True,ls='dashed')
    ax.set_ylim([0.1,1.6])
    ax = viz.label_sp(labelssp[a],labelstyle="%s",usenumber=True,ax=ax,alpha=0.75,fontsize=fsz_sp)
    ax.set_ylabel(r"SST Variance ($\degree C^2$)",fontsize=  fsz_axis)
    ax.tick_params(axis='both', labelsize=fsz_ticks)

savename = "%sMonthlyVariance_50N30W.png" % outpath
plt.savefig(savename,dpi=150,bbox_inches="tight")
