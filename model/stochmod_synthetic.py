#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test Synthetic Stochastic Model

Created on Tue Jan 12 03:46:45 2021

@author: gliu
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs

#%% Set Options
#bboxsim  = [-100,20,-20,90] # Simulation Box
query     = [-30,50]
pointmode = 1
mconfig   = "SLAB_PIC"
t_end     = 120000
hfix      = 50
dt        = 3600*24*30
multFAC   = 0
T0        = 0
lags      = np.arange(0,37,1)
fstd      = 1
multFAC   = 1
ftype     = "DJFM-MON"
genrand   = 0
runid     = "syn001"

#pcnames = ["NAO","EAP","NAO+EAP"]
#exps = 

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath = projpath + '02_Figures/20210112/'

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

#%%


def load_data(verbose=False):
    
    

def synth_stochmod(config):
    """
    

    Parameters
    ----------
    config : DICT
        

    Returns
    -------
    None.

    """    
    
    # Load data
    
    
    

#%%
# Load Data (MLD and kprev, damping)
mld            = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall       = np.load(input_path+"HMXL_kprev.npy") # Entraining Month
dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp       = loadmat(input_path+dampmat)
lon            = np.squeeze(loaddamp['LON1'])
lat            = np.squeeze(loaddamp['LAT'])
cesmslabac     = np.load(datpath+"CESM_clim/TS_SLAB_Autocorrelation.npy")
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])

if mconfig == "SLAB_PIC":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
elif mconfig=="FULL_HTR":
    damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")

# Load Forcing
forcing = np.load(input_path+mconfig+ "_NAO_EAP_NHFLX_Forcing_%s.npy" % ftype)#[:,:,0,:]
forcing = forcing[:,:,0,:]
if genrand:
    randts = np.random.normal(0,fstd,t_end)
    #randts = np.abs(randts)
    np.save(output_path+"Forcing_fstd%.2f_%s.npy"% (fstd,runid),randts) 
else:
    randts = np.load(output_path+"Forcing_fstd%.2f_%s.npy"% (fstd,runid))

# Restrict to point
params = scm.get_data(pointmode,query,lat,lon,damping,mld,kprevall,forcing)
[o,a],damppt,hclim,kprev,Fpt = params

#Visualize points (raw)
#fig,ax = viz.summarize_params(lat,lon,params)
#plt.tight_layout()

# Synthetic Parameters Prep **************************************************
# Set synthetic parameters if any

# 3 = max in march
xtks = np.arange(1,13,1)
#Fpt = np.sin(-1*np.pi*xtks/6+10)*0.3*20*-1

#Fpt = np.ones(12)

# Make up forcing for now
#F = np.random.normal(0,1,size=mld.shape)
#F = np.ones(mld.shape)
# F = np.ones(mld.shape)* np.array([0,0,211,
#                                    212,0,0,
#                                    0,0,0,
#                                    0,0,0])[None,None,:]

#Fpt[10]=15
##Fpt[11]=25
#Fpt[9] =10
#Fpt[0]=15
#Fh     = Fh[None,None,:] * np.ones(288,192,1)

# Synthetic Damping
#damppt = damppt*2
#damppt = np.ones(12) * np.mean(damppt)
#damppt = np.sin(-1*np.pi*xtks/6+11)*-1*10+15
# damppt = np.array([13,13,13,
#                     13,13,13,
#                     13,13,13,
#                     13,26,26])

#Visualize model inputs
# Synthetic Forcing
#mldpt = np.ones(12)*params[2].max() # Indicate the Mixed layer depth that is used
mldpt = np.ones(12)*hclim.mean()
#mldpt = hclim

synth = [damppt,mldpt,Fpt] #[damping,mld,forcing]

fig,ax = viz.summarize_params(lat,lon,params,synth=synth)
#ax.set_title(i)
plt.tight_layout()
#plt.savefig(outpath+"AC_Slab_Stoch_Comparison_NAOForce_SeasDamp_Params.png",dpi=200)

# ****************************************************************************

# Prepare Forcing
#randts = np.abs(randts)


Fh = randts * np.tile(Fpt,int(t_end/12)) * (dt/(3850*1025*mldpt[0]))*100
#plt.plot(randts,lw=0.5)

# Convert Parameters
lbd,lbd_entr,FAC,beta = scm.set_stochparams(mldpt,damppt,dt,ND=False,hfix=hfix)

Fh = {}
for i in range(3):
    
    if i == 0:
        Fh[i] = randts * np.tile(Fpt,int(t_end/12)) * (dt/(3850*1025*50))
    elif i == 1:
        Fh[i] = randts * np.tile(Fpt,int(t_end/12)) * (dt/(3850*1025*mldpt.max()))
    elif i == 2:
        Fh[i] = randts * np.tile(Fpt*mldpt,int(t_end/12)) * (dt/(3850*1025))
    
    
    
    
# Run the stochastic model
sst = {}
for i in range(3):
    if i > 1:
        multFAC =1
    else:
        multFAC = 0
    sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh[i],FAC[i],multFAC=multFAC,debug=False)
    
    if i == 1:
        
        # Additional Pure White Noise Time Series
        sst[4] = scm.noentrain(t_end,lbd[i],T0,randts,FAC[i],multFAC=multFAC,debug=False)
        
    
    
sst[3]=scm.entrain(t_end,lbd[3],T0,Fh[2],beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False,debugprint=False)



# Calculate Autocorrelation
kmonth = hclim.argmax()
autocorr = scm.calc_autocorr(sst,lags,kmonth+1)

# Load inautocorrelation values from CESM Slab
#cesmauto = np.load(projpath + "01_Data/CESM-SLAB_PIC_autocorrelation_pointlon330_lat50.npy")

# Read in CESM autocorrelation for all points
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto = cesmauto2[lags]
# fig,ax = plt.subplots(1,1)
# ax.plot(cesmauto,label="old")
# ax.plot(cesmauto2[lags],label="new")
# ax.legend()

# Section below was scrap used to write the init_acplot function
# mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
# mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
# mons3tile = np.concatenate([np.roll(mons3tile,-kmonth),[mons3[kmonth]]])
# xtk2 = np.arange(0,37,2)

# fig,ax = plt.subplots(1,1)

# # Set up second axis
# ax2 = ax.twiny()
# ax2.set_xticks(xtk2)
# ax2.set_xticklabels(mons3tile[xtk2], rotation = 45)
# ax2.set_axisbelow(True)
# ax2.grid(zorder=0,alpha=0)

# for i in range(4):
#     ax.plot(autocorr[i])

# ax.set_xticks(xtk2)
# ax.set_xlim([xtk2[0],xtk2[-1]])
# ax.set_title("SST Autocorrelation, Lag 0 = %s" % (mons3[kmonth]))
# ax.set_xlabel("Lags (Months)")
# ax.set_ylabel("Correlation")
# ax.grid(True,linestyle='dotted')
# plt.tight_layout()

#%% Plot Autocorrelatiom (Just Slab)

labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,figsize=(6,4))
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
#ax.plot(lags,autocorr[1],label=labels[i])
#ax.plot(lags,exps[0],label="Sinusoidal Forcing",color='b',ls='solid')
ax.plot(lags,autocorr[4],label="No Seasonal Cycle",color='k',ls='dotted')
#ax.plot(lags,exps[1],label="NAO Forcing",color='magenta',ls='solid')
ax3.set_ylabel("Forcing (W/m2)")
ax.plot(lags,autocorr[1],label="NAO Forcing, Mean MLD",color='magenta')

ax.plot(lags,exps[0],label="Sinusoidal Forcing, Max MLD",color='gold')
#ax.plot(lags,autocorr[1],label="Sin. Forcing + Seasonal Damping",color='r')
ax3.yaxis.label.set_color('gray')
#ax3.set_ylim([-30,30])
#ax3.axhline(y=0,color='gray',linestyle="dotted",lw=0.75)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_Slab_Stoch_Comparison_Sinusoid_MaxMLD.png",dpi=200)

#%% Init exp parameter
exps = {}
expnames = ["Sinusoid","NAO"]
expcolors = ['b']

expf = {}
#%% Save Experiment
exps[0] = autocorr[1].copy()
expf[0] = Fpt.copy()

np.save(outpath+"Exps_saved.npy",exps)

#%% Plot Autocorrelation (All Models)
labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]

#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
colors=["red","orange","green","blue"]

title = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[hclim.argmax()])


xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i in [0,1,2,3]:
    ax.plot(autocorr[i],label=labels[i],color=colors[i])
#ax.legend(ncol=3,fontsize=10)
#ax3.set_ylabel("MLD (m)")
#ax3.set_ylabel("Damping (W/m2)")
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_testpositive.png"%(locstring),dpi=200)


#%% Just Plot a few autocorrelation curves

queries = [[-30,50],[-70,30],[]]
acplots = []
for q in queries:
    kmonth = 
    ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
    acplots.append()


ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]


# **********************************
#%% Experiment Set 1 Varying Forcing (Month of Maximum)
# **********************************

# Seasonally varying forcing, shifting around
Fs   = []
mmax = []
ploti    = 1
for m in tqdm(range(12)):
    # Make Forcing
    Fpt = np.sin(-1*np.pi*(xtks)/6+m)*0.3*20*-1
    #Fpt = np.sin(-1*np.pi*(xtks)/6+m)+1
    Fs.append(Fpt)
    mmax.append(Fpt.argmax()+1)
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    
    # Run the stochastic model
    sst = {}
    for i in range(3):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    else:
        sst[3]=scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False)
    
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)
    
    # Make Plot
        # Plot Autocorrelation
    #labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
    xtk2 = np.arange(0,37,2)
    fig,ax = plt.subplots(1,1)
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
    ax.plot(lags,cesmauto,label="CESM SLAB",color='k')
    
    ax.plot(autocorr[ploti],label=labels[ploti])
    
    ax.legend(ncol=3,fontsize=10)
    ax3.set_ylabel("Forcing")
    ax3.yaxis.label.set_color('gray')
    plt.suptitle("MLD Max SST Autocorrelation, Forcing Max = %s"% (mons3[mmax[m]-1]))
    plt.tight_layout()
    plt.savefig("%sSeasonal_Forcing_mmax%i_model%i.png"%(outpath,mmax[m],ploti),dpi=200)
    
    
# **********************************
#%% Experiment Set 2 Varying Forcing (Magnitude)
# **********************************


F = np.ones(mld.shape) * np.sin(-1*np.pi*xtks/6+10)[None,None,:]
Fpt = []
Ffac = [10**x for x in np.linspace(-6,6,12)]
acall = []
for m in tqdm(range(12)):
    # Make Forcing
    Fpt = Ffac[m]*np.sin(-1*np.pi*xtks/6+10)
    Fs.append(Fpt)
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    
    # Run the stochastic model
    sst = {}
    for i in range(3):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    else:
        sst[3]=scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False)
    
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)
    acall.append(autocorr)



    
# Make Plot
# Plot Autocorrelation
#labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
ax.plot(lags,cesmauto,label="CESM SLAB",color='k')
for i in range(12):
    ax.plot(acall[i][1],label=Ffac[i])
ax.legend(ncol=1,fontsize=10)

ax3.set_ylabel("Forcing")
ax3.yaxis.label.set_color('gray')
plt.suptitle("MLD Max SST Autocorrelation, Forcing Max = %s"% (mons3[mmax[m]-1]))
plt.tight_layout()
plt.savefig("%sSeasonal_Forcing_mmax%i_model%i.png"%(outpath,mmax[m],ploti),dpi=200)


#
# Quantifying sensitivity to forcing...
#


#%%

for mm in tqdm(range(12)):
    Fpt = np.sin(-1*np.pi*xtks/6+mm)*0.3*20*-1
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    sst = {}
    for i in range(2):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)

    xtk2 = np.arange(0,37,2)
    fig,ax = plt.subplots(1,figsize=(6,4))
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
    ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
    #ax.plot(lags,autocorr[1],label=labels[i])
    ax.plot(lags,autocorr[1],label="Sinusoidal Forcing",color='b',ls='solid')
    #ax.plot(lags,autocorr[4],label="No Seasonal Cycle",color='k',ls='dotted')
    ax3.set_ylabel("Forcing (W/m2)")
    ax3.yaxis.label.set_color('gray')
    ax3.set_ylim([-12,12])
    ax3.axhline(y=0,color='gray',linestyle="dotted",lw=0.75)
    plt.suptitle("mm=%i"%mm)
    ax.legend()
    plt.savefig(outpath+"Sinusoida_Forcing_Slab_mm%i.png"%(mm),dpi=200)