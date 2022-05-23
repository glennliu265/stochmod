#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check point autocorrelation for a continuous stochastic model output
    - Calculates AC for each 1000 year chunk
    - Takes the mean AC, also calculates AC for the whole simulation strung together
    - Compare with CESM PiC Data


Created on Mon May 23 13:14:09 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
from time import time

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220526/"
   
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

#%% Experimental Configurations

mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over
continuous = True # Set to True to Load a continuous run with the lines below


if continuous:
    # Do a continuous Run
    # -------------------
    fnames   = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
    frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
    exname   = "Fprime_amq0_method5_cont"
else:
    # Options to determine the experiment ID (Old Format)
    # --------------------------------------------------
    runid      = "011"
    
    # Indicate the experiment file name, display name (for plotting), and output save name
    fname       = 'forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method4_dmp0'
    frcnamelong = "EOF Forcing (90% Variance)"
    exname      ="90perc_variance"
    
    
#%% Other Settings

# Analysis Options
lags = np.arange(0,37,1)

# Set the Visualization mode
darkmode   = False
if darkmode:
    plt.style.use("dark_background")
    dfcol = "w"
else:
    plt.style.use("default")
    dfcol = "k"

# Calculate in place (or load output from sm_postprocess_output)
calc_inplace=False

#%% Labels and Plotting

# # Regional Analysis Setting (NEW, with STG Split)
# Regional Analysis Settings
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
bbox_NA_et  = [-80,0,20,60]
regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw","eNAT")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w,bbox_NA_et) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic",
               "North Atlantic","Subtropical (East)","Subtropical (West)",
               "Extratropical North Atlantic")
bbcol       = ["Blue","Red","Yellow","Black","Black"]
bbcol       = ["cornflowerblue","Red","Yellow","Black","Black","limegreen","indigo","Black"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted",'dashed']

# AMV Contours
cint        = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int      = np.arange(-0.45,0.50,0.05)

# SM Names and colors
modelnames  = ("Vary $F'$ and $\lambda_a$",
                    "Vary $F'$, $h$, and $\lambda_a$",
                    "Entraining")
mcolors     = ["red","magenta","orange"]
mlines      = ["solid","dotted","dashed"]


# CESM Names
cesmname    =  ["CESM-FULL","CESM-SLAB"]
cesmcolor   =  [dfcol,"gray"]
cesmline    =  ["dashed","dotted"]

# Autocorrelation Plot parameters
xtk2        = np.arange(0,37,2)
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
conf        = 0.95
tails       = 2
alw         = 3

# Power Spectra Options
speccolors = ["r","magenta","Orange","k","gray"]
specnames  = np.hstack([modelnames,cesmname])


# Linear-Power Spectra, < 2-yr
xlm   = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper  = np.array([100,50,20,10,5,2])
xtks  = 1/xper
xlm   = [xtks[0],xtks[-1]]


#%% Silly functions to repackage postprocessed output (need to stop using dicts)



def unpack_smdict(indict):
    """
    Takes a dict of [run][region][models][OTHERDIMS] and unpacks it into
    an array [unpackaged]

    """
    # Get "Outer Shell" dimensions
    nrun    = len(indict)
    nregion = len(indict[0])
    nmodels = len(indict[0][0])
    
    # For Autocorrelation
    otherdims = indict[0][0][0].shape
    print("Found... Runs (%i) Regions (%i) ; Models (%i) ; Otherdims (%s)" % (nrun,nregion,nmodels,str(otherdims)))
    
    # Preallocate
    newshape = np.concatenate([[nrun,nregion,nmodels],otherdims])
    unpacked = np.zeros(newshape) * np.nan
    
    # Loop thru dict
    for run in range(nrun):
        for reg in range(nregion):
            for mod in range(nmodels):
                unpacked[run,reg,mod,:] = indict[run][reg][mod]
    return unpacked

def repack_smdict(inarr,nregion,nmodels):
    """
    Repackages a numpy array of inarr[region x model x otherdims] to 
    outdict{region}{model}
    """
    outdict = {}
    for reg in range(nregion):
        indict = {}
        for mod in range(nmodels):
            indict[mod] = inarr[reg,mod,:]
        outdict[reg] = indict.copy()
    return outdict

#%% Load lat/lon regional

# Get one of the outputs
if continuous:
    fname = fnames[0] # Take first one to load lat/lon
ldname = "%sstoch_output_%s.npz" % (datpath,fname)
if exname == "numEOFs":
    ldname = ldname.replace("50","2")
ld     = np.load(ldname,allow_pickle=True)
lon    = ld['lon']
lat    = ld['lat']


# Load global lat/lon
lon180g,latg  = scm.load_latlon(rawpath)

#%% Get indices for selected point

lonf = -30
latf = 50
klon,klat = proc.find_latlon(lonf,latf,lon,lat)

locfn,locstr = proc.make_locstring(lonf,latf)


#%%

# Load in data and take annual average
ssts_all = np.zeros(())
for f, fname in tqdm(enumerate(fnames)):
    ld = np.load(datpath+"stoch_output_%s.npz" % fname, allow_pickle=True)
    ssts = ld['sst']
    if f == 0:
        
        nmod,nlon,nlat,ntime = ssts.shape
        ssts_all = np.zeros([nmod,ntime,len(fnames)])*np.nan
        
    ssts_all[:,:,f] = ssts[:,klon,klat,:].copy() # [model x time]
#%% Calculate Autocorrelation


# Settings
imodel = 2
lags = np.arange(0,61,1)
basemonth = 1

# Unpack to list
sst_in = [ssts_all[imodel,:,i] for i in range(len(fnames))]
sst_in.append(ssts_all[imodel,:,:].T.reshape(ntime*len(fnames)))

# Calculate
acs         = scm.calc_autocorr(sst_in,lags,basemonth)

#%% Load and calculate for cesm

cesm_ssts   = scm.load_cesm_pt(datpath+"../",grabpoint=[lonf,latf])
cesm_acs    = scm.calc_autocorr(cesm_ssts,lags,basemonth)

#%%


xtks    = np.arange(0,63,3)
xtk_lbls=viz.prep_monlag_labels(basemonth-1,xtks,2)

fig,ax  = plt.subplots(1,1,constrained_layout=True)

acs_mean = np.zeros(len(lags))
for k in range(12):
    
    if k < 10:
        label = "run%02i" % (k+1)
        ax.plot(lags,acs[k],label=label,lw=3)
        acs_mean += acs[k]
        
        
    elif k == 10:
        label = "All Together"
        ax.plot(lags,acs[k],label=label,lw=3,color='k',ls='solid')
        
        
    else:
        label = "10-member Mean"
        acs_mean /= len(fnames)
        ax.plot(lags,acs_mean,label=label,lw=3,color='gray',ls='dotted')

#Add CESM plots
if imodel > 0:
    ax.plot(lags,cesm_acs[0],label="CESM-FULL",lw=3,color='cyan',ls='solid')
else:
    ax.plot(lags,cesm_acs[1],label="CESM-SLAB",lw=3,color='cyan',ls='solid')


ax.axhline([0],color='k',ls='dashed',lw=0.75)
ax.legend()
ax.set_title("%s SST Autocorrelation at %s \n %s Stochastic Model" % (mons3[basemonth-1],locstr,modelnames[imodel]))

ax.set_xticks(xtks,label=xtk_lbls)
ax.grid(True,ls='dotted')
ax.set_xlim([xtks[0],xtks[-1]])



#%% Visualize a particular Run
sel_run =0

xtks = np.arange(0,63,3)
xtk_lbls=viz.prep_monlag_labels(basemonth-1,xtks,2)

fig,ax = plt.subplots(1,1,constrained_layout=True)



acs_mean = np.zeros(len(lags))
for k in range(12):
    
    
    if k < 10:
        
        if k == sel_run:
            
            label = "run%02i" % (k+1)
            ax.plot(lags,acs[k],label=label,lw=3)
            acs_mean += acs[k]
        else:
            continue
        
        
    elif k == 10:
        label = "All Together"
        ax.plot(lags,acs[k],label=label,lw=3,color='k',ls='solid')
        
        
    else:
        label = "10-member Mean"
        acs_mean /= len(fnames)
        ax.plot(lags,acs_mean,label=label,lw=3,color='gray',ls='dotted')
    

#Add CESM plots

if imodel > 0:
    ax.plot(lags,cesm_acs[0],label="CESM-FULL",lw=0.75,color='cyan',ls='solid')
else:
    ax.plot(lags,cesm_acs[1],label="CESM-SLAB",lw=0.75,color='cyan',ls='solid')


ax.axhline([0],color='k',ls='dashed',lw=0.75)
ax.legend()
ax.set_title("%s SST Autocorrelation at %s \n %s Stochastic Model" % (mons3[basemonth-1],locstr,modelnames[imodel]))

ax.set_xticks(xtks,label=xtk_lbls)
ax.grid(True,ls='dotted')
ax.set_xlim([xtks[0],xtks[-1]])