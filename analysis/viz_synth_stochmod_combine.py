#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Autocorrelation Plots for

synth_stochmod_spectra 
and
constant_v_variable

Created on Wed Oct  6 22:17:26 2021

@author: gliu
"""

import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs

from scipy import signal


# #%%
# output_path = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/'

# # Experimental Configurations
# config = {}
# config['mconfig'] = 'SLAB_PIC'
# config['fname']   = 'flxeof_090pct_SLAB-PIC_eofcorr2.npy'
# config['runid']   = 'syn007'

# #%% Load CESM Data
# #%% Load Constant_v_vary experiments
# savenames = "%sconst_v_vary_%s_runid%s_%s.npz" % (output_path,config['mconfig'],config['runid'],config['fname'])
# print("Loading %s" % savenames)
# ld = np.load(savenames,allow_pickle=True)

# c_acs  = ld['acs'].item
# c_ssts = ld['ssts'].item
# # print("Saving clean run to %s" % savenames)
# # np.savez(savenames,**{
# #     'expids':expids,
# #     'acs':acs,
# #     'ssts':ssts,
# #     'damps':damps,
# #     'mlds':mlds,
# #     'forces':forces,
# #     'explongs':explongs
# #     },allow_pickle=True)

# #%% Load Synthetic Experiments
# savenames = "%scleanrun_%s_runid%s_%s.npz" % (output_path,config['mconfig'],config['runid'],config['fname'])
# print("Loading clean run %s" % savenames)
# ld = np.load(savenames,allow_pickle=True)

# s_acs = ld['ac'].item
# s_ssts = ld['sst'].item
# # np.savez(savenames,**{
# #     'ac':ac,
# #     'sst': sst,
# #     'dmp': dmp,
# #     'frc': frc,
# #     'entr' : ent,
# #     'Td': Td,
# #     'kmonth':kmonth,
# #     'params':params
# #     }, allow_pickle=True)

# #%%



import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs

from scipy import signal

#%% Settings

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20211011/'
proc.makedir(outpath)

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
#fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")
fullauto = np.load(datpath+"CESM_clim/TS_FULL_Autocorrelation.npy")

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
labelsnew = ["h=50m","Constant $h$","Vary $h$","Entraining"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
#expcolors = ('blue','red','magenta','orange')
expcolors = ('mediumorchid','red','magenta','orange')
els = ["dashdot","solid","dotted","dashed"]
#hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

# Set up Configuration
config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn007"   # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2.npy"   #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at 
config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = outpath # Note need to fix this
config['smooth_forcing'] = False
config['method']      = 3 
config['favg' ]       = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)

# Confidence Level Calculations
conf  = 0.95
tails = 2
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
ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt       =params

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
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.plot(lags,cesmautofull,color='k',label='CESM Full',ls='dashdot')

for i in range(1,4):
    ax.plot(lags,ac[i],label=labels[i],color=expcolors[i])

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

#%%

fig,axs     = plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)

# UPDATED Colors and names for generals (5/25/2021)
#ecol = ["blue",'cyan','gold','red']
ecol = ["blue",'cyan','gold','red']
els  = ["dotted","dashdot","dashed","solid"]
ename = ["All Constant",
         r"Vary $\alpha$",
         r"Vary $\lambda_a$",
         "All Varying"]

# Plot Lower Hierarchy
ax = axs[0]

title = r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)"

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
    ax.plot(lags,plotacs[e][model],label=ename[i],color=ecol[i],ls=els[i],marker="o",markersize=4)
    #ax.scatter(lags,plotacs[e][model],10,label="",color=ecol[i])
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color=ecol[i],alpha=0.2)
    
    ax.legend(fontsize=10,ncol=3)
    
ax.set_ylabel("Correlation")


# --------------------------------------------

# Plot Upper Hierarchy
ax = axs[1]
title = "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)

# Plot CESM Data
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
#ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=3)
#ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

for i in range(1,4):
    ax.plot(lags,ac[i],label=labelsnew[i],color=expcolors[i],ls=els[i],marker="o",markersize=3)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=expcolors[i],alpha=0.25)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
ax.set_ylabel("")

plt.suptitle("Monthly SST Autocorrelation at 50N, 30W (Lag 0 = February)",fontsize=12,y=1.01)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

plt.savefig(outpath+"Autocorrelation_2-PANEL_%s.png"%locstring,dpi=200,bbox_inches='tight')

#%% Load and calculate CESM Spectra

cssts = scm.load_cesm_pt(datpath,loadname='both',grabpoint=[-30,50])



#%% Calculate Spectra

debug = False

# Smoothing Params
nsmooth = 250
cnsmooths = [45,37]
pct        = 0.10

smoothname = "smth-obs%03i-full%02i-slab%02i" % (nsmooth,cnsmooths[0],cnsmooths[1])

# Spectra Plotting Params
xlm = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]

ylm  = [0,3.1]

plotids = [[0,1,2,3,8],
           [4,5,6,7]
           ]

plottype = "freqlin"

# First, pull out the needed SSTs
inssts   = [c_ssts[0][1],c_ssts[1][1],c_ssts[2][1],c_ssts[3][1],sst[1],sst[2],sst[3],cssts[0],cssts[1]]
nsmooths = np.concatenate([np.ones(len(inssts)-2)*nsmooth,cnsmooths])
labels   = np.concatenate([ename,labelsnew[1:],['CESM-FULL','CESM-SLAB']])
speclabels = ["%s (%.2f$\degree \, C^{2}$)" % (labels[i],np.var(inssts[i])) for i in range(len(inssts))]
allcols  = np.concatenate([ecol,expcolors[1:],["k","gray"]])

# Calculate Autocorrelation
allacs,allconfs = scm.calc_autocorr(inssts,lags,kmonth+1,calc_conf=True)

# Convert Dict --> Array
oac=[]
ocf=[]
for i in range(len(allacs)):
    oac.append(allacs[i])
    ocf.append(allconfs[i])
allacs=oac
allconfs=ocf
    


if debug: # Check its alright
    fig,axs = plt.subplots(1,2,figsize=(16,4))
    ax = axs[0]
    plotac = allacs[:4]
    for i in range(4):
        ax.plot(lags,plotac[i],label=ename[i],color=ecol[i],)
    ax.legend()
    ax = axs[1]
    plotac = allacs[4:]
    for i in range(3):
        ax.plot(lags,plotac[i],label=labelsnew[i+1],color=expcolors[i+1],)
    ax.legend()
    #
    
    
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

sharetitle = "SST Spectra (50$\degree$N, 30$\degree$W) \n " + 
    "Smoothing (# bands): Stochastic Model (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (nsmooth,cnsmooths[0],cnsmooths[1]))

#% Plot the spectra
fig,axs = plt.subplots(1,2,figsize=(16,4))
for i in range(2):
    ax = axs[i]
    plotid = plotids[i]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    ax2.set_xlabel("Period (Years)")
    if i == 1:
        ax.set_ylabel("")
    ax.set_xlabel("")
        
    plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    
    ax.set_ylim(ylm)
    
fig.text(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
#plt.suptitle("SST Power Spectra at 50$\degree$N, 30$\degree$W",y=1.15,fontsize=14)
savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i.png" % (outpath,plottype,smoothname,pct*100)
plt.savefig(savename,dpi=200,bbox_inches='tight')


# plotid = 
# if plottype == "freqxpower":
#     ax = viz.plot_freqxpower(specs[:4],freqs[:4],speclabels[:4],allcols[:4],
#                          ax=ax,plottitle=r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",xtick=xtks,xlm=xlm)
# elif plottype == "freqlin":
#     ax = viz.plot_freqlin(specs[:4],freqs[:4],speclabels[:4],allcols[:4],
#                          ax=ax,plottitle=r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)",xtick=xtks,xlm=xlm)
# elif plottype == "freqlog":
#     ax = viz.plot_freqlog(specs[:4],freqs[:4],speclabels[:4],allcols[:4],
#                          ax=ax,plottitle="Adding Varying Mixed Layer Depth ($h$) and Entrainment",xtick=xtks,xlm=xlm)


# ax = axs[1]
# if plottype == "freqxpower":
#     ax = viz.plot_freqxpower(specs[4:],freqs[4:],speclabels[4:],allcols[4:],
#                          ax=ax,plottitle="Adding Varying Mixed Layer Depth ($h$) and Entrainment",xtick=xtks,xlm=xlm)
# elif plottype == "freqlin":
#     ax = viz.plot_freqlin(specs[4:],freqs[4:],speclabels[4:],allcols[4:],
#                          ax=ax,plottitle="Adding Varying Mixed Layer Depth ($h$) and Entrainment",xtick=xtks,xlm=xlm)
# elif plottype == "freqlog":
#     ax = viz.plot_freqlog(specs[4:],freqs[4:],speclabels[4:],allcols[4:],
#                          ax=ax,plottitle="Adding Varying Mixed Layer Depth ($h$) and Entrainment",xtick=xtks,xlm=xlm)
# ax.set_ylabel("")

# #plt.suptitle("Regional AMV Index Spectra (unsmoothed, Forcing=%s)"%(frcnamelong[f]))




# (all const, varyforce, vary damp)
# inssts.insert(sst[1]) # Append all vary/constanth
# inssts.append(sst[2]) # Append hvary
# inssts.append(sst[3]) # Append entrain
# #np.hstack([c_ssts[:3],])



# Get the spectra
#specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(sstin,nsmooth,pct)

#%% Make a plot with an inset
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
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    ax2.set_xlabel("Period (Years)")
    if i == 1:
        ax.set_ylabel("")
        
        
    
plt.suptitle("SST Power Spectra at 50N, 30W",y=1.05)
savename = "%sNASST_Spectra_Stochmod_%s_nsmooth%i_pct%03i.png" % (outpath,plottype,nsmooth,pct*100)
plt.savefig(savename,dpi=200,bbox_inches='tight')

