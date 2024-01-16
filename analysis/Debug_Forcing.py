#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug Forcing Leading to Variance Differences


Involved Scripts: 
    - viz_synth_stochmod_combine
    - constant_v_variable
    

Created on Mon Jan 15 14:23:11 2024

@author: gliu
"""
# ---------------------

#%% Constant_v_variable

# ---------------------
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
#from scipy.ndimage.filters import uniform_filter1d

# Set Paths
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20240115/'
proc.makedir(outpath)

darkmode = False

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','orange','magenta','red')
hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab
if darkmode:
    dfcol      = 'w'
    dfalph     = 0.30
    dfalph_col = 0.40
else:
    dfcol      = 'k'
    dfalph     = 0.1
    dfalph_col = 0.25 

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
config['runid']       = "syn009"  # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy" #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1
config['query']       = [-30,50]
config['applyfac']    = 2 # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = outpath
config['smooth_forcing'] = False
config['method'] = 3
config['favg'] = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)


#%% Clean Run

ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt = params

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

# Calculate constant forcing value (Using CESM-SLAB values)
cp0=3996
rho=1026
dt = 3600*24*30
frcstd_slab = np.std(frc[1]) * (rho*cp0*hblt) / dt  # (Constant Forcing)
#frcstd_slab_raw = Fpt#frc[1].copy()

#%%

foriginal = np.copy(config['fname'])

if len(Fptdef.shape)>2:
    #original = config['fname']
    config['fname'] = 'FLXSTD' # Dummy Move for now to prevent forcing_flag
    
expids   = [] # String Indicating the Variable Type
acs      = []
ssts     = []
damps    = []
mlds     = []
forces   = []
explongs = []

for vmld in [False,True]:
    
    if vmld:
        config['mldpt'] = mlddef
    else:
        config['mldpt'] = np.ones(12)*mlddef.mean()
    
    for vdamp in tqdm([False,True]):
        
        if vdamp:
            config['damppt'] = dampdef
        else:
            config['damppt'] = np.ones(12)*dampdef.mean()
        
        for vforce in [False,True]:
                
                if vforce:
                    
                    print(foriginal)
                    config['fname'] = np.copy(foriginal).item()
                    config.pop('Fpt',None)
                    config['favg'] = False
                    
                        #config['Fpt'] = Fptdef.mean(0)
                else: # Constant Forcing
                    
                    config['fname'] = 'FLXSTD'
                    config['favg'] = True
                    # Take stdev over EOFs, then take the mean
                    
                    Fptin = Fptdef.copy()
                    Fptin[Fptin==0] = np.nan                    
                    
                    config['Fpt'] = frcstd_slab * np.ones(12)
                    # This yields a value that is greater than Fptin
                    #config['Fpt'] = np.ones(12)*np.nanmean(np.nansum(Fptin,0))
                    #np.std(Fptdef,0).mean()
                
                # Set experiment name
                expid = "vdamp%i_vforce%i_vmld%i" % (vdamp,vforce,vmld)
                explong = "Vary Damping (%s) Forcing (%s) MLD (%s)" % (vdamp,vforce,vmld)
                
                
                # Run Model
                ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
                [o,a],damppt,mldpt,kprev,Fpt = params
                
                # Save variables
                acs.append(ac)
                ssts.append(sst)
                expids.append(expid)
                damps.append(damppt)
                mlds.append(mldpt)
                forces.append(Fpt)
                explongs.append(explong)
                print("Completed %s"%expid)
                #print(mldpt)
                
                # Clean Config, End Forcing Loop
                config.pop('Fpt',None)
                #print(Fpt)
                
        # Clean Config, End Damping Loop
        config.pop('damppt',None)
        #print(damppt)
    # Clean Config, End MLD Loop
    config.pop('mldpt',None)

#%% Lets Check the Output


# Make Some Plots
proc.get_monstr(nletters=3,)
fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(mons3,Fptdef,label=r"$\sigma (F')$ Forcing") # Default Forcing
ax.grid(True,ls='dashed')
ax.legend()
ax.set_ylabel(r"$W/m^2$")
savename = "%sConst_v_Vary_Forcing_Fstd.png" % outpath
plt.savefig(savename,dpi=150)


fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(mons3,dampdef,label=r"$\lambda ^a$") # Default Damping
ax.grid(True,ls='dashed')
ax.legend()
ax.set_ylabel(r"$\degree C$ per $W/m^2$")
savename = "%sConst_v_Vary_Damping.png" % outpath
plt.savefig(savename,dpi=150)

#%%Save Output for Analysis/Comparison/Debugging
outdir   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%sParameters_constant_v_vary.npz" % outdir
np.savez(savename,**{
    'forcing'  : Fptdef,
    'damping'  : dampdef,
    'mld'      : mlddef,
    'hblt'     : hblt,
    'mconfig'  : config['mconfig'],
    'fname'    : config['fname'],
    'ssts'     : ssts,
    'expnames' : explongs,
    'acs'      : acs,
    })

#%% Check Monthly variance



selids  = [0,1,2,3]
selcol  = ["blue","cyan","yellow","red"]

ssts_unpack = [ssts[ii][1] for ii in selids]
expnamein   = np.array(explongs)[selids]


monvars     = [proc.calc_monvar(sst) for sst in ssts_unpack]




fig,ax = plt.subplots(1,1,figsize=(8,6))
for i in range(len(monvars)):
    ax.plot(mons3,monvars[i],label=expnamein[i],lw=2.5,color=selcol[i])
    ax.legend(fontsize=8)
ax.grid(True,ls='dotted')

#%%
sst

#%%

#%%

#%%

