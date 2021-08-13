#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Spectral Estimates for Generals Exam

Created on Sun Jun  6 13:38:43 2021

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
from scipy.ndimage.filters import uniform_filter1d

#%% Settings

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
datpathgen  = projpath + '01_Data/Generals_Report/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath = projpath + '02_Figures/20210610/'
proc.makedir(outpath)

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','orange','magenta','red')
hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab



# UPDATED Colors and names for generals (5/25/2021)
ecol = ["blue",'cyan','gold','red']
els  = ["dotted","dashdot","dashed","solid"]
ename = ["All Constant",
         r"Vary $\alpha$",
         r"Vary $\lambda_a$",
         "All Varying"]


config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0
config['fstd']        = 1
config['t_end']       = 120000    # Number of months in simulation
config['runid']       = "syn001"  # White Noise ID
config['fname']       = "FLXSTD" #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1
config['query']       = [-30,50]
config['applyfac']    = 2 # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = projpath + '02_Figures/20210223/'
config['smooth_forcing'] = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)

#% ----------------------
#%% Load PiC Data
#% ----------------------
# Streamline this using the function you wrote

# Load CESM 
st = time.time()
sstfull,sstslab = scm.load_cesm_pt(datpath,loadname='both')

# # Load full sst data from model
# ld  = np.load(datpath+"FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
# sstfull = ld['TS']
# ld2 = np.load(datpath+"SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
# sstslab = ld2['TS']

# Load lat/lon
lon360,lat  = scm.load_latlon(lon360=True)

print("Loaded PiC Data in %.2fs"%(time.time()-st))

#% ----------------------
# %% Calculate PiC Spectrum
#% ----------------------

#%%
# # Parameters
# pct     = 0.10
# nsmooth = 1
# opt     = 1
# dt      = 3600*24*30
# tunit   = "Months"
# clvl    = [0.95]
# axopt   = 3
# clopt   = 1
# specnames = "nsmooth%i_taper%i" % (nsmooth,pct*100)
# Key Params
plotcesm = True
cnames  = ["CESM1 FULL","CESM1 SLAB"]
nsmooths = [500,250] # Set Smothing

# Other Params
pct     = 0.10
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1

# Retrieve point
lonf,latf = config['query']
if lonf < 0:
    lonf += 360
klon360,klat = proc.find_latlon(lonf,latf,lon360,lat)
fullpt = sstfull[:,klat,klon360]
slabpt = sstslab[:,klat,klon360]

# Calculate spectra
freq1s,P1s,CLs = [],[],[]
for i,sstin in enumerate([fullpt,slabpt]):
    
    # Calculate and Plot
    sps = ybx.yo_spec(sstin,opt,nsmooths[i],pct,debug=False)
    P,freq,dof,r1=sps
    
    # Plot if option is set
    if plotcesm:
        pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
        fig,ax,h,hcl,htax,hleg = pps
        #ax,htax = viz.make_axtime(ax,htax)
        ax = viz.add_yrlines(ax)
        ax.set_title("%s Spectral Estimate \n nsmooth=%i, taper = %.2f" % (cnames[i],nsmooths[i],pct*100) +r"%")
        ax.grid(True,which='both',ls='dotted')
        ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
        plt.tight_layout()
        plt.savefig("%sSpectralEstimate_%s_nsmooth%i_taper%i.png"%(outpath,cnames[i],nsmooths[i],pct*100),dpi=200)
    CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
    P    = P*dt
    freq = freq/dt
    CC   = CC*dt
    P1s.append(P)
    freq1s.append(freq)
    CLs.append(CC)

# Read outvariables
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s
clfull,clslab = CLs

# #%% Load Constant_v_Variable (Lower Hierarchy) Results
# nsmooth  = 1000

# loadname =datpathgen + "/lower_hierarchy_data_nsmooth%i.npz" % (nsmooth)

# #ld1 = np.load(loadname)

def load_spectra(loadname):
    ld1 = np.load(loadname,allow_pickle=True)
    freqs = ld1['freqs']
    specs = ld1['specs']
    CCs   = ld1['CCs']
    ecols = ld1['ecolors']
    enames = ld1['enames']
    return freqs,specs,CCs,ecols,enames

# freqs1,specs1,CCs1,ecols1,enames1 = load_spectra(loadname)


# #%% Load Upper Hierarchy
# loadname =datpathgen + "/upper_hierarchy_data_nsmooth%i.npz" % (nsmooth)
# freqs2,specs2,CCs2,ecols2,enames2 = load_spectra(loadname)

#%% Load lower and upper hierarchy

def load_spectra(loadname):
    ld1 = np.load(loadname,allow_pickle=True)
    freqs = ld1['freqs']
    specs = ld1['specs']
    ssts  = ld1['sst'][None][0]
    CCs   = ld1['CCs']
    ecols = ld1['ecolors']
    enames = ld1['enames']
    return freqs,specs,CCs,ecols,enames,ssts

# Get File Names
nsmooth  = 1000
loadname1 = datpathgen + "/lower_hierarchy_data_nsmooth%i.npz" % (nsmooth)
loadname2 = datpathgen + "/upper_hierarchy_data_nsmooth%i.npz" % (nsmooth)
loadnames = [loadname1,loadname2]

# Load Everything
fall    = [] 
sall    = [] 
ccall    = [] 
ecall = []
enall = []
sstall = []
for i in range(2):
    loadname = loadnames[i]
    f,s,cc,ec,en,sst = load_spectra(loadname)
    fall.append(f)
    sall.append(s)
    ccall.append(cc)
    ecall.append(ec)
    enall.append(en)
    sstall.append(sst)
    
    
#%% Plot Everything

# Plotting Params
plotdt = 3600*24*365
xtick  = [float(10)**(x) for x in np.arange(-4,2)]
ylm    = [-.01,.5]
xlm    = [5e-4,10]
titles = ("Varying Damping and Forcing (Levels 1-3)",
          "Varying Mixed-Layer Depth (Levels 3-5)")
plotconf=False


fig,axs = plt.subplots(1,2,figsize=(10,4))

for i in range(2):
    
    # Get corresponding variables
    ax      = axs[i]
    specs   = sall[i]
    freqs   = fall[i]
    CCs     = ccall[i]
    ecolors = ecall[i]
    enames  = enall[i]
    sstin   = sstall[i]
    
    # Plot Stochastic Model Spectra
    if i == 0:
        nplot = np.arange(0,4)
    else:
        nplot = [1,2,3]
    
    for n in nplot:
        ax.semilogx(freqs[n]*plotdt,specs[n]*freqs[n],color=ecolors[n],label=enames[n]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(sstin[n])))
        
        if plotconf:
            ax.semilogx(freqs[n]*plotdt,CCs[n][:,1]*freqs[n],color=ecolors[n],alpha=0.5,ls='dashed')
            ax.semilogx(freqs[n]*plotdt,CCs[n][:,0]*freqs[n],color=ecolors[n],alpha=0.5,ls='dotted')
        
    # Plot CESM Spectra
    if i == 0:
        ax.semilogx(freqcesmslab*plotdt,freqcesmslab*Pcesmslab,color='gray',label="CESM1-SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(slabpt)))
    elif i == 1:
        ax.semilogx(freqcesmfull*plotdt,freqcesmfull*Pcesmfull,color='black',label="CESM1-FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(fullpt)))
        
    # Set Labels
    if i == 0:
        ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)
    ax.set_ylim(ylm)
    htax.set_ylim(ylm)
    
    ax.legend(fontsize=10)
    ax.set_title(titles[i])
    
    

locstring= "30W, 50N"
plt.suptitle("SST Spectral Estimates (10,000 year Integration at %s)"%locstring)
plt.tight_layout()
plt.savefig("%sSpectral_Estimate_Comparisons_GeneralsReport.png"%outpath,dpi=150,bbox_tight='inches')
#plt.savefig("%sSpectra_Comparison_2panel_nsmooth%i_taper%i.png"%(outpath,nsmooth,pct*100),dpi=150)
#%% Lin-Lin Plots

# Plotting Params
plotdt = 3600*24*365
xtick  = np.arange(0,1.4,.2)
ylm    = [0,2.5]
xlm    = [0,1.2]
titles = ("Varying Damping and Forcing (Levels 1-3)",
          "Varying Mixed-Layer Depth (Levels 3-5)")
plotconf=False


fig,axs = plt.subplots(1,2,figsize=(10,4))

for i in range(2):
    
    # Get corresponding variables
    ax      = axs[i]
    specs   = sall[i]
    freqs   = fall[i]
    CCs     = ccall[i]
    ecolors = ecall[i]
    enames  = enall[i]
    sstin   = sstall[i]
    
    # Plot Stochastic Model Spectra
    if i == 0:
        nplot = np.arange(0,4)
    else:
        nplot = [1,2,3]
    
    for n in nplot:
        ax.plot(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(sstin[n])))
        
        if plotconf:
            ax.plot(freqs[n]*plotdt,CCs[n][:,1]/plotdt,color=ecolors[n],alpha=0.5,ls='dashed')
            ax.plot(freqs[n]*plotdt,CCs[n][:,0]/plotdt,color=ecolors[n],alpha=0.5,ls='dotted')
        
    # Plot CESM Spectra
    if i == 0:
        ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1-SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(slabpt)))
    elif i == 1:
        ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1-FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(fullpt)))
        
    # Set Labels
    if i == 0:
        ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    
    
    #ax = ax.set_xticks(xtick)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)
    ax.set_ylim(ylm)
    htax.set_ylim(ylm)
    
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    ax.set_title(titles[i])
    
    

locstring= "30W, 50N"
plt.suptitle("SST Spectral Estimates (10,000 year Integration at %s)"%locstring)
plt.tight_layout()
plt.savefig("%sSpectral_Estimate_Comparisons_GeneralsReport_LinLin.png"%outpath,dpi=150,bbox_tight='inches')
#plt.savefig("%sSpectra_Comparison_2panel_nsmooth%i_taper



#%% LOGLOG

# Plotting Params
plotdt = 3600*24*365
xtick  = [float(10)**(x) for x in np.arange(-3,3)]
ylm    = [1e-3,3]
xlm    = [1e-3,5]
titles = ("Varying Damping and Forcing (Levels 1-3)",
          "Varying Mixed-Layer Depth (Levels 3-5)")
plotconf=False


fig,axs = plt.subplots(1,2,figsize=(10,4))

for i in range(2):
    
    # Get corresponding variables
    ax      = axs[i]
    specs   = sall[i]
    freqs   = fall[i]
    CCs     = ccall[i]
    ecolors = ecall[i]
    enames  = enall[i]
    sstin   = sstall[i]
    
    # Plot Stochastic Model Spectra
    if i == 0:
        nplot = np.arange(0,4)
    else:
        nplot = [1,2,3]
    
    for n in nplot:
        ax.loglog(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(sstin[n])))
        
        if plotconf:
            ax.loglog(freqs[n]*plotdt,CCs[n][:,1]/plotdt,color=ecolors[n],alpha=0.5,ls='dashed')
            ax.loglog(freqs[n]*plotdt,CCs[n][:,0]/plotdt,color=ecolors[n],alpha=0.5,ls='dotted')
        
    # Plot CESM Spectra
    if i == 0:
        ax.loglog(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1-SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(slabpt)))
    elif i == 1:
        ax.loglog(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1-FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(fullpt)))
        
    # Set Labels
    if i == 0:
        ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    
    
    #ax = ax.set_xticks(xtick)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-log',xtick=xtick)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)
    ax.set_ylim(ylm)
    htax.set_ylim(ylm)
    
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    ax.set_title(titles[i])
    
    

locstring= "30W, 50N"
plt.suptitle("SST Spectral Estimates (10,000 year Integration at %s)"%locstring)
plt.tight_layout()
#plt.savefig("%sSpectral_Estimate_Comparisons_GeneralsReport.png"%outpath,dpi=150,bbox_tight='inches')
#plt.savefig("%sSpectra_Comparison_2panel_nsmooth%i_taper



