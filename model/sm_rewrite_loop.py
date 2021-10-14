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

stormtrack = 1

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
ampq       = True # Set to true to multiply stochastic forcing by a set value
mconfig    = "SLAB_PIC"
#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_q-ek_090pct_SLAB-PIC_eofcorr1" #"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_qek_50eofs_SLAB-PIC" #"uniform" "flxeof_5eofs_SLAB-PIC"
#"flxeof_090pct_SLAB-PIC_eofcorr1"
#"flxeof_5eofs_SLAB-PIC"
#"flxeof_080pct_SLAB-PIC"
#flxeof_qek_50eofs_SLAB-PIC

# Running Parameters
runid      = "010"
pointmode  = 0 
points     = [-30,50]
bboxsim    = [-80,0,0,65] # Simulation Box

useslab    = False # Set to True to use SLAB_CESM parameters for all...
savesep    = False # Set to True to save the outputs differently

# Additional Constants
t_end      = 12000 # Sim Length
dt         = 3600*24*30 # Timestep
T0         = 0 # Init Temp

# Forcing Correction Method
ampq = 3 #0 = none 1 = old method, 2 = method 1, 3 = method 2

lonf = -30
latf = 50
debug = False

# Loop by forcing

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


## Testing NAO, EAP Forcing
# frcnames = ('flxeof_EOF1_SLAB-PIC_eofcorr0',
#             'flxeof_EOF2_SLAB-PIC_eofcorr0')

#frcnames = ("flxeof_2eofs_SLAB-PIC_eofcorr0",)

#frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',)

# Testing seasonal EOFs
frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',
            'flxeof_090pct_SLAB-PIC_eofcorr2_DJF',
            'flxeof_090pct_SLAB-PIC_eofcorr2_MAM',
            'flxeof_090pct_SLAB-PIC_eofcorr2_JJA',
            'flxeof_090pct_SLAB-PIC_eofcorr2_SON')

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
print("Running the following forcings: \n %s"%(str(frcnames)))
#%%
st = time.time()
for f in range(len(frcnames)):
    frcname    = frcnames[f]
    expname    = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i.npz" % (output_path,frcname,int(t_end/12),runid,ampq) 
    
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
                       debug=False,check=False,useslab=useslab,savesep=savesep)

print("Completed in %.2fs" % (time.time()-st))

