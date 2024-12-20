#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Model Paper Stylesheet

Stylesheet to track variable names and plotting params for SM manuscript
Copied from plot_temporal_region.py on Dec 18 2021

Created on Sat Dec 18 13:27:15 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
# # Regional Analysis Settings (NEW, with STG Split)
# Regional Analysis Settings
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic","Subtropical (East)","Subtropical (West)",)
bbcol       = ["Blue","Red","Yellow","Black","Black"]
bbcol       = ["Blue","Red","Yellow","Black","Black","magenta","red"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted"]

# AMV Pattern Contours
cint        = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int      = np.arange(-0.45,0.50,0.05)
cmax        = 0.5
cstep       = 0.025
lstep       = 0.05
cint,cl_int = viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]
bboxplot    = [-80,0,9,65]#[-80,0,5,60] # Updated 8/8/2022 for AMV

# SM Names and colors
modelnames  = ("Constant h","Vary h","Entraining")
mcolors     = ["red","magenta","orange"] 


# SM Lower Hierarchy (05/25/2021)
ecol_lower       = ["blue",'cyan','gold','red']
els_lower        = ["dotted","dashdot","dashed","solid"]
# labels_lower     = ["All Constant",
#                      r"Vary $F'$",
#                      r"Vary $\lambda_a$",
#                      "Vary $F'$ and $\lambda_a$"] 
labels_lower     = ["All Constant (Level 1)",
                     r"Vary $F'$ (Level 2b)",
                     r"Vary $\lambda_a$ (Level 2a)",
                     "Vary $F'$ and $\lambda_a$ (Level 3)"]  # Added Level Labels

# SM Upper Hierarchy (05/25/2021)



# SM Upper Hierarchy (05/25/2021)
# labels_upper = ["h=50m",
#                  "Vary $F'$ and $\lambda_a$",
#                  "Vary $F'$, $h$, and $\lambda_a$",
#                  "Entraining"]
labels_upper = ["h=50m",
                 "Vary $F'$ and $\lambda_a$ (Level 3)",
                 "Vary $F'$, $h$, and $\lambda_a$ (Level 4)",
                 "Entraining (Level 5)"] # Added Level Labels
ecol_upper = ('mediumorchid','red','magenta','orange')
els_upper = ["dashdot","solid","dotted","dashed"]



# CESM Names
cesmname    =  ["CESM-FULL","CESM-SLAB"]
cesmcolor   =  ["k","gray"]
cesmline    =  ["dashed","dotted"]

# Autocorrelation Plot parameters
xtk2        = np.arange(0,37,2)
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
conf        = 0.95
tails       = 2 # Tails for Significance Calculation
alw         = 3 # Autocorrelation Line Width

# Combined plots
innames  = modelnames + tuple(cesmname)
incolors = tuple(mcolors) + tuple(cesmcolor)

#%% Power Spectra Options
speccolors = ["r","magenta","Orange","k","gray"]
specnames  = np.hstack([modelnames,cesmname])

# Linear-Power Spectra, < 2-yr (Current SM Draft Choice)
xlm = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0,3.0]

# Shorter Timescales (Patrizio et al)
xper = np.array([25,10,5,2.5,1,0.5,0.2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]

# Clement et al. 2015 range
xper = np.array([50,25,10,5,2.5,1.0])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
#%% Darkmode, Lightmode

darkmode = True
if darkmode:
    plt.style.use("dark_background")
    dfcol = "w"
else:
    plt.style.use("default")
    dfcol = "k"

#%% Flux Dictionary
namedict = {
    "LHFLX": "Latent Heat Flux",
    "SHFLX": "Sensible Heat Flux",
    "FSNS" : "Shortwave Flux",
    "FLNS" : "Longwave Flux",
    "qnet" : "Net Heat Flux",
    "NHFLX": "Net Heat Flux",
    "RHFLX": "Radiative Heat Flux",
    "THFLX": "Turbulent Heat Flux"
    }

