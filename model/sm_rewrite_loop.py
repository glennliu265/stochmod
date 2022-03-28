#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loopable version of sm_rewrite
Created on Tue Sep 21 10:42:25 2021

@author: gliu
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tqdm
import time

#%% Set the location

stormtrack = 0

if stormtrack == 0:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    
    input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"

elif stormtrack == 1:
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/" 

from amv import proc,viz
import scm

# Run Mode
# pointmode = 0 # Set to 1 to output data for the point speficied below
# points=[-30,50] # Lon, Lat for pointmode

# # Forcing Type
# # 0 = completely random in space time
# # 1 = spatially unform forcing, temporally varying
# # 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# # 3 = NAO-like NHFLX Forcing, with NAO (DJFM) and NHFLX (Monthly)
# # 4 = NAO-like NHFLX Forcing, with NAO (Monthly) and NHFLX (Monthly)
# funiform = 0     # Forcing Mode (see options above)
# fscale   = 1     # Value to scale forcing by

# # ID of the run (determines random number sequence if loading or generating)
# runid = "002"

# # White Noise Options. Set to 1 to load data
# genrand   = 1  # Set to 1 to regenerate white noise time series, with runid above
    
# # Integration Options
# nyr      = 1000        # Number of years to integrate over
# fstd     = 0.3         # Standard deviation of the forcing
# bboxsim  = [-100,20,-20,90] # Simulation Box

# # Running Location
# stormtrack = 1 # Set to 1 if running in stormtrack

# applyfac options
# 0) Forcing is just the White Noise For ing
# 1) Forcing is white noise (numerator) and includes MLD
# 2) Forcing includes both MLD seasonal cycle AND integration factor
# 3) Forcing just includes integration factor

# Types of forcing
# "allrandom" : Completely random in space or time
# "uniform"   : Uniform in space, random in time

#%%


# Directories

# Landice Mask Name
limaskname = "limask180_FULL-HTR.npy" 

# Model Params
mconfig    = "SLAB_PIC"

#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_q-ek_090pct_SLAB-PIC_eofcorr1" #"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_qek_50eofs_SLAB-PIC" #"uniform" "flxeof_5eofs_SLAB-PIC"
#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_5eofs_SLAB-PIC"
#"flxeof_080pct_SLAB-PIC"
#flxeof_qek_50eofs_SLAB-PIC

# Running Parameters
runids      = ["2%02d"%i for i in range(10)]#"011"
pointmode   = 0 
points      = [-30,50]
bboxsim     = [-80,0,0,65] # Simulation Box
hconfigs    = [2,] # Which MLD configuration to use

# Toggles
useslab     = False # Set to True to use SLAB_CESM parameters for all...
savesep     = False # Set to True to save the outputs differently

# Continuous Run Options
continuous  = True  # Set to True to continue run between each runid. Set to False to start new run from T0
# Select startfile for when run == 0. Default is to start from zero
startfile   = None  
#"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/stoch_output_forcingforcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method5_dmp0_1000yr_run011_ampq3_method2_dmp0.npz"

# ADD EKMAN FORCING (or other custom parameters)
custom_params         = {}
custom_params['q_ek'] = "Qek_eof_090pct_FULL_PIC_eofcorr0_Fprime_rolln0.npy"

# Save for Budget Analyses
budget = True # Set to True to only run entrain, and save budget analyses separately

# Additional Constants
# --------------------
t_end      = 12000 # Sim Length
dt         = 3600*24*30 # Timestep
T0         = 0 # Init Temp

# Forcing Correction Method (q-corr)
ampq       = 0 #0 = none 1 = old method, 2 = method 1, 3 = method 2

# Damping Significance Test Method
method     = 5 # 1 = No Testing; 2 = SST autocorr; 3 = SST-FLX crosscorr, 4 = Both 
lagstr     = "lag1"

# Point information
lonf       = -30
latf       = 50
debug      = False

# Indicate Forcing Files to Loop Thru

# Testing different # of EOFs
frcnames = (
            "flxeof_50eofs_SLAB-PIC_eofcorr2",
            "flxeof_25eofs_SLAB-PIC_eofcorr2",
            "flxeof_10eofs_SLAB-PIC_eofcorr2",
            "flxeof_5eofs_SLAB-PIC_eofcorr2",
            "flxeof_3eofs_SLAB-PIC_eofcorr2",
            "flxeof_2eofs_SLAB-PIC_eofcorr2",
            "flxeof_1eofs_SLAB-PIC_eofcorr2"
            )

# Testing seasonal EOFs
frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',
            'flxeof_090pct_SLAB-PIC_eofcorr2_DJF',
            'flxeof_090pct_SLAB-PIC_eofcorr2_MAM',
            'flxeof_090pct_SLAB-PIC_eofcorr2_JJA',
            'flxeof_090pct_SLAB-PIC_eofcorr2_SON')

frcnames = ('flxeof_EOF1_SLAB-PIC_eofcorr1',
            'flxeof_EOF2_SLAB-PIC_eofcorr1')

# Testing different # of EOFs, simpler correction
frcnames = (
            "flxeof_50eofs_SLAB-PIC_eofcorr1",
            "flxeof_25eofs_SLAB-PIC_eofcorr1",
            "flxeof_10eofs_SLAB-PIC_eofcorr1",
            "flxeof_5eofs_SLAB-PIC_eofcorr1",
            "flxeof_3eofs_SLAB-PIC_eofcorr1",
            "flxeof_2eofs_SLAB-PIC_eofcorr1",
            "flxeof_1eofs_SLAB-PIC_eofcorr1"
            )

frcnames = ('flxeof_EOF1_SLAB-PIC_eofcorr1',
            'flxeof_EOF2_SLAB-PIC_eofcorr1')

frcnames = (
            "flxeof_50eofs_SLAB-PIC_eofcorr0",
            "flxeof_25eofs_SLAB-PIC_eofcorr0",
            "flxeof_10eofs_SLAB-PIC_eofcorr0",
            "flxeof_5eofs_SLAB-PIC_eofcorr0",
            "flxeof_3eofs_SLAB-PIC_eofcorr0",
            "flxeof_2eofs_SLAB-PIC_eofcorr0",
            "flxeof_1eofs_SLAB-PIC_eofcorr0"
            )

#frcnames = ("flxeof_2eofs_SLAB-PIC_eofcorr0",)

#frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',)

# Test the effect of increasing the number of EOFs
frcnames = (
            "flxeof_50eofs_SLAB-PIC_eofcorr0",
            "flxeof_25eofs_SLAB-PIC_eofcorr0",
            "flxeof_10eofs_SLAB-PIC_eofcorr0",
            "flxeof_5eofs_SLAB-PIC_eofcorr0",
            "flxeof_3eofs_SLAB-PIC_eofcorr0",
            "flxeof_2eofs_SLAB-PIC_eofcorr0",
            "flxeof_1eofs_SLAB-PIC_eofcorr0"
            )

#frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0',)
frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0',)


# Updated EOF TSest
frcnames = ('flxeof_EOF1_SLAB-PIC_eofcorr0_Fprime_rolln0',
            'flxeof_EOF2_SLAB-PIC_eofcorr0_Fprime_rolln0',
            'flxeof_2eofs_SLAB-PIC_eofcorr0_Fprime_rolln0')


# Updated Seasonal EOFs
frcnames = ('flxeof_090pct_FULL-PIC_eofcorr2_DJF_Fprime_rolln0',
            'flxeof_090pct_FULL-PIC_eofcorr2_MAM_Fprime_rolln0',
            'flxeof_090pct_FULL-PIC_eofcorr2_JJA_Fprime_rolln0',
            'flxeof_090pct_FULL-PIC_eofcorr2_SON_Fprime_rolln0')


frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0',)


# Single run test (90% variance forcing)
#frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0',)

# Rerun, 
#frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',)

# # Testing NAO, EAP Forcing
# frcnames = ('flxeof_EOF1_SLAB-PIC_eofcorr0',
#             'flxeof_EOF2_SLAB-PIC_eofcorr0')

# # Test the effect of increasing the number of EOFs
# frcnames = (
#             "flxeof_50eofs_SLAB-PIC_eofcorr0",
#             "flxeof_25eofs_SLAB-PIC_eofcorr0",
#             "flxeof_10eofs_SLAB-PIC_eofcorr0",
#             "flxeof_5eofs_SLAB-PIC_eofcorr0",
#             "flxeof_3eofs_SLAB-PIC_eofcorr0",
#             "flxeof_2eofs_SLAB-PIC_eofcorr0",
#             "flxeof_1eofs_SLAB-PIC_eofcorr0"
#             )

# # Testing seasonal EOFs
# frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2_DJF',
#             'flxeof_090pct_SLAB-PIC_eofcorr2_MAM',
#             'flxeof_090pct_SLAB-PIC_eofcorr2_JJA',
#             'flxeof_090pct_SLAB-PIC_eofcorr2_SON')

print("Running the following forcings: \n")
print(*frcnames, sep='\n')

#%%
st = time.time()

for f in range(len(frcnames)):
    frcname    = frcnames[f]
    expnames = []
    for r,runid in enumerate(runids):
        
        
        expname    = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i_method%i_dmp0.npz" % (output_path,frcname,int(t_end/12),runid,ampq,method,)
        if budget:
            expname = proc.addstrtoext(expname,"_budget")
        # dmp0 indicates that points with insignificant lbd_a were set to zero.
        # previously, they were set to np.nan, or the whole damping term was set to zero
        if 'q_ek' in custom_params.keys():
            print("Using Ekman Forcing!")
            correction_str = '_Qek'
            expname = proc.addstrtoext(expname,correction_str)
        
        
        if continuous:
            if r == 0: # First Run
                if startfile is False: # Initialize from zero
                    continue_run = False 
                else: # Initialize from startfile
                    print("Initializing run 0 from %s" % startfile)
                    continue_run = startfile
            else:
                continue_run=expnames[r-1] # Use Previous file
        
        # Check if results exist
        query = glob.glob(expname)
        if len(query) > 0:
            overwrite = input("Found existing file(s) \n %s. \n Overwite? (y/n)" % (str(query)))
        else:
            overwrite = 'y'
        # Skip forcing
        if overwrite == 'n':
            continue
        
        scm.run_sm_rewrite(expname,mconfig,input_path,limaskname,
                           runid,t_end,frcname,ampq,
                           bboxsim,pointmode,points=[lonf,latf],
                           dt=3600*24*30,
                           debug=False,check=False,useslab=useslab,savesep=savesep
                           ,method=method,lagstr=lagstr,hconfigs=hconfigs,
                           custom_params=custom_params,budget=budget)
        expnames.append(expname)
    print("Completed in %.2fs" % (time.time()-st))

