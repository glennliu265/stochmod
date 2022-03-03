#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Vary something, and compare the output

Compied from synth_spectra_analysis.py on:
    
Created on Mon Feb 14 19:46:02 2022

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

#%% Settings

# Set Paths
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20220305/'
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
expcolors = ('blue','red','magenta','orange')
els = ["dashdot","solid","dotted","dashed"]
#hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

# Set up Configuration
config = {}
config['mconfig']     = "FULL_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 1          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn009"   # White Noise ID
config['fname']       = "flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0.npy"   #['NAO','EAP,'EOF3','FLXSTD']
#config['fname']       = 'FLXSTD'

config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at (SPG Test)
#config['query']       = [-77,28]   # Point to run model at  (GS)
#config['query']       = [-36,58]  # SE Greenland
config['query']       = [-29,15.5] # Problem (eastern) Tropic Point

config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']           = np.arange(0,37,1)
config['output_path']    = outpath # Note need to fix this
config['smooth_forcing'] = False
config['method']         = 0 #3 Refers to Forcing Correction Method

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)

# Confidence Level Calculations
conf  = 0.95
tails = 2

darkmode=False
#%% Functions

def interp_quad(ts):
    
    # Interpolate qdp as well
    tsr = np.roll(ts,1)
    tsquad = (ts+tsr)/2
    
    fig,ax = plt.subplots(1,1)
    ax.set_title("Interpolation")
    ax.plot(np.arange(1.5,13.5,1),ts,label="ori",color='k',marker='d')
    ax.plot(np.arange(1,13,1),tsquad,label="shift",marker="o",color='red')
    ax.set_xticks(np.arange(1,13,1))
    ax.grid(True,ls="dotted")
    
    return tsquad
    #ax.set_xticklabels()
    
def adjust_axis(ax,htax,dt,multiple):
    
    # Divisions of time
    # dt  = 3600*24*30
    # fs  = dt*12
    # xtk      = np.array([1/fs/100,1/fs/50, 1/fs/25, 1/fs/10 , 1/fs/5, 1/fs])
    # xtkm    = ["%i" % np.round(i) for i in 1/xtk/dt]
    # xtklabel = ['%.1e \n (century)'%xtk[0],'%.1e \n (50yr)'%xtk[1],'%.1e \n (25yr)'%xtk[2],'%.1e \n (decade)'%xtk[3],'%.1e \n (5year)'%xtk[4],'%.2e \n (year)'%xtk[5]]
    
    fs = dt*multiple
    xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
    xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
    xtkl     = ["%.1e" % s for s in xtk]
    for i,a in enumerate([ax,htax]):
        
        a.set_xticks(xtk)
        if i == 0:
            
            a.set_xticklabels(xtkl)
        else:
            a.set_xticklabels(xtkm)
    return ax,htax

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


#%% Test different inputs (forcing, damping, mixed-layer depth)

"""
This section should load in needed inputs, and exit with
mlds   = [] List of ARRAY[12]
alphas = [] List of ARRAY[12,N_mode]
lbds   = [] List of ARRAY[12]
"""

# ---------------------------------------
#%% Experiment 1, Using Amplified Forcing
# ---------------------------------------
exname = "Fprime-v-Qnet"

# Load forcing for a single point
#config_custom = config.copy()
#config['fname'] = "flxeof_090pct_FULL-PIC_eofcorr2_Fprime.npy"


# Get indices of target point
lonf,latf = config['query']
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring = "Lon %.f, Lat %.f" % (lon[klon],lat[klat])
locfn     = "lon%i_lat%i" % (lon[klon],lat[klat])

# Get the points
Fload     = np.load(input_path+"flxeof_090pct_FULL-PIC_eofcorr2_Fprime.npy")
Fload2    = np.load(input_path+"flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0.npy")
Fptload   = Fload[klon,klat,:,:] 
Fptload2  = Fload2[klon,klat,:,:] 

# -----------------
# Set up the inputs
# -----------------
mid     = 3 # Model id (50m,Const,Vary,Entrain)
mnames  = ["$Q_{net}$ (Corrected)","$F'$ ($T_{t-1}$)","$F'$ ($T_{t}$)"]
alphas  = [Fptdef,Fptload,Fptload2]
lbds    = [dampdef,dampdef,dampdef]
mlds    = [mlddef ,mlddef,mlddef]
nexps   = len(mnames)
mcolors = ["mediumblue","firebrick","magenta"] 

#for v in enumerate([alphas,lbds,mlds]):
fig,ax = plt.subplots(1,1)
for i in range(nexps):
    if i == 0:
        ls='solid'
    else:
        ls='dashed'
    ts_plot = np.linalg.norm(alphas[i],axis=0)
    ax.plot(mons3,ts_plot,label=mnames[i],ls=ls)
    ax.set_title("Forcing Amplitude")
ax.legend()


# ---------------------------------------
#%% Experiment 2, Testing MLD Effects in Tropical Point
#  LON: 29 W  |  LAT: 15.5 N
# ---------------------------------------
exname = "TropMLD_lon29_lat16"

# Load forcing for a single point
#config_custom = config.copy()
#config['fname'] = "flxeof_090pct_FULL-PIC_eofcorr2_Fprime.npy"


# Get indices of target point
lonf,latf = config['query']
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring = "Lon %.f, Lat %.f" % (lon[klon],lat[klat])
locfn     = "lon%i_lat%i" % (lon[klon],lat[klat])


# Set different mixed layers
mldorig    = mlddef.copy()
mld_springdrop = mlddef.copy()
mld_springdrop[3:9] = mlddef[5:11]
mld_springdrop[9:11] = mlddef[3:5]
# plot mixed layer
fig,ax = plt.subplots(1,1)
ax.plot(mons3,mldorig,label="Original MLD",marker="+",color='mediumblue')
ax.plot(mons3,mld_springdrop,label="No Spring Persistence",marker="x",color='firebrick')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_xlim([0,11])
ax.set_ylabel("MLD (m)")
plt.savefig("%sCustom_MLD_%s.png"%(outpath,exname))


mldmax = np.ones(12)*mlddef.max()
mldmin = np.ones(12)*mlddef.min()

# Get the points
Fload     = np.load(input_path+"flxeof_090pct_FULL-PIC_eofcorr2_Fprime.npy")
Fload2    = np.load(input_path+"flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0.npy")

Fptload   = Fload[klon,klat,:,:] 
Fptload2  = Fload2[klon,klat,:,:] 

# -----------------
# Set up the inputs
# -----------------
mid     = 3 # Model id (50m,Const,Vary,Entrain)
mnames  = ["Default MLD","No Spring Persistence","MLD max","MLD min"]
alphas  = [Fptdef,Fptdef,Fptdef,Fptdef]
lbds    = [dampdef,dampdef,dampdef,dampdef]
mlds    = [mlddef ,mld_springdrop,mldmax,mldmin]
nexps   = len(mnames)
mcolors = ["mediumblue","firebrick","magenta","orange"] 


    
"""
End Loading Section
"""
print("Found %i experiments"% (nexps))

#%% Now run the stochastic model n times

ssts = []
for e in tqdm(range(nexps)):
    # Modify the config
    config['Fpt']    = alphas[e]
    config['damppt'] = lbds[e]
    config['mldpt']  = mlds[e]
    
    if e == 0:
        config['method'] = 3
    else:
        config['method'] = 0
    
    # Run the model, append the results
    _,sst,_,_,_,_,_,_=scm.synth_stochmod(config,projpath=projpath)
    ssts.append(sst[mid])
    
    # Clean up the config
    config.pop('Fpt',None)
    config.pop('damppt',None)
    config.pop('mldpt',None)

#%% Load CESM Data

# Load SST for selected point [FULL,SLAB]
cssts     = scm.load_cesm_pt(datpath,loadname='both',grabpoint=config['query'])
cnames    = ["CESM-FULL","CESM-SLAB"]
ccolors   = ["k",'gray']

#%% Merge everything

inssts   = ssts + cssts[:1]
incolors = mcolors + ccolors[:1]
innames  = mnames  + cnames[:1] 
#%% Check Autocorrelation

# Calculate the Autocorrelation
allacs,allconfs = scm.calc_autocorr(inssts,lags,kmonth+1,calc_conf=True)

#%% Plot the Autocorrelation

fig,ax = plt.subplots(1,1,figsize=(6,4))
lw      = 3
plotacs = allacs
title   = "%s SST Autocorrelation @ %s" % (viz.return_mon_label(kmonth+1),locstring)

xtk2       = np.arange(0,37,2)
ax,ax2     = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)


for m in range(len(inssts)):
    
    

    ax.plot(lags,plotacs[m],label=innames[m],color=incolors[m],marker="o",markersize=4,lw=lw)
    ax.fill_between(lags,allconfs[m][lags,0],allconfs[m][lags,1],color=incolors[m],alpha=0.1)
ax.legend(fontsize=10,ncol=3)
    
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")
plt.savefig("%sAutocorrelation_Point_%s_%s.png"% (outpath,exname,locfn),dpi=150)

#%% Now Calculate and Plot the spectra

# Set Parameters
ssmooth    = 250
cnsmooths  = [50,50]
pct        = 0.10
nsmooth    = np.concatenate([np.ones(nexps)*ssmooth,np.ones(2)*cnsmooths])
smoothname = "smth-obs%03i-full%02i-slab%02i" % (ssmooth,cnsmooths[0],cnsmooths[1])
pct        = 0.10
dofs       = [1000,1000,898,1798] # In number of years
plottype   = 'freqlin'
lw         = 3

# From sm stylesheet
xlm  = [1e-2,5e0]
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
#ylm  = [0,0.5]
ylm   = [0,2.5]

# Do spectral analysis
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooth,pct)

# Plot the results
title=""


fig,ax = plt.subplots(1,1,figsize=(8,4))

if plottype == "freqxpower":
    ax,ax2 = viz.plot_freqxpower(specs,freqs,innames,incolors,
                         ax=ax,plottitle="",xtick=xtks,xlm=xlm,return_ax2=True)
elif plottype == "freqlin":
    ax,ax2 = viz.plot_freqlin(specs,freqs,innames,incolors,
                         ax=ax,plottitle="",xtick=xtks,xlm=xlm,return_ax2=True,lw=lw)
    ylabel = "Power ($K^2/cpy$)"
elif plottype == "freqlog":
    ax,ax2 = viz.plot_freqlog(specs,freqs,innames,incolors,
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

ax.set_xticklabels(1/xtks)

ax.set_xlabel('Period (Years)',fontsize=12)
ax.set_ylabel("Power ($K^2/cpy$)")

savename = "%sSpectra_Stochmod_%s_%s_%s_%s.png" % (outpath,exname,plottype,locstring,smoothname)
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Just Plot the SST Itself

custom_xlim = [0,1000]
fig,axs = plt.subplots(3,1,sharey=True)
for i in range(3):
    ax = axs[i]
    ax.plot(inssts[i],color=incolors[i],label=innames[i]+" (Var = %.3f)" % (np.var(inssts[i])),lw=0.5)
    ax.axhline(0,color="gray",ls='dotted')
    
    if custom_xlim:
        ax.set_xlim(custom_xlim)
    else:
        ax.set_xlim([0,len(inssts[i])])
    ax.legend()
    
#%% Barplot of frequency variance

sstvars    = [np.var(sst) for sst in inssts]
sstvars_lp = [np.var(proc.lp_butter(sst,120,6)) for sst in inssts]
labels1    = ["%.2f"%sst for sst in sstvars]
labels2    = ["%.2f"%sst for sst in sstvars_lp]

# Make Plot
fig,ax = plt.subplots(1,1,figsize=(8,4))

# Plot Variance
bars1 = ax.bar(innames,sstvars,color='seagreen',label="Raw")
ax,rects = viz.label_barplots(labels1 ,ax=ax,rects=bars1,fontsize=12,fontcolor='seagreen')

# Plot LP Variance
bars2 = ax.bar(innames,sstvars_lp,color='thistle',label="10-year Low Pass Filtered")
ax,rects = viz.label_barplots(labels2 ,ax=ax,rects=bars2,fontsize=12,fontcolor='thistle')

# Adjust labels, grids, axes
ax.set_xticklabels(innames,rotation=10)
ax.grid(True,ls='dotted')
ax.set_ylabel("SST Variance ($K^2$)")
ax.set_xlabel("Simulation/Experiment Name")
ax.set_ylim([0,0.90])
ax.legend()

savename = "%sSpectra_Stochmod_%s_Barplot_%s.png" % (outpath,exname,locstring)
plt.savefig(savename,dpi=200,bbox_inches='tight')


