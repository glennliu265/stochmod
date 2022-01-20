#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

sm_rewrite Custom

copied from sm_rewrite (1/20/2022)
but customized to support input of custom parameters (h,F', or lambda')

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

stormtrack =0

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

# Custom Parameters

"""
Fixed MLD Experiment
"""
cexpname = "hfix50_slab"
h_cust = np.ones((288,192,12)) * 50 # 50 m slab
custom_params = {}
custom_params['h'] = h_cust
#'forcing' and 'lambda' are two other options
hconfigs      = [0] # Just run the slab simulation

# Landice Mask Name
limaskname = "limask180_FULL-HTR.npy" 

# Model Params
ampq       = True # Set to true to multiply stochastic forcing by a set value
mconfig    = "SLAB_PIC"

# Running Parameters
runid      = "011"
pointmode  = 0 
points     = [-30,50]
bboxsim    = [-80,0,0,65] # Simulation Box

useslab    = False # Set to True to use SLAB_CESM parameters for all...
savesep    = False # Set to True to save the outputs differently

# Additional Constants
t_end      = 12000 # Sim Length
dt         = 3600*24*30 # Timestep
T0         = 0 # Init Temp

# Forcing Correction Method (q-corr)
ampq   = 3 #0 = none 1 = old method, 2 = method 1, 3 = method 2

# Damping Significance Test Method
method = 4 # 1 = No Testing; 2 = SST autocorr; 3 = SST-FLX crosscorr, 4 = Both 

# Point information
lonf = -30
latf = 50
debug = False

# Indicate Forcing Files to Loop Thru
# Single run test (90% variance forcing)
frcnames = ('flxeof_090pct_SLAB-PIC_eofcorr2',)

print("Running the following forcings: \n")
print(*frcnames, sep='\n')

#%%
st = time.time()
for f in range(len(frcnames)):
    frcname    = frcnames[f]
    expname    = "%sstoch_output_forcing%s_%iyr_run%s_ampq%i_method%i_dmp0_%s.npz" % (output_path,
                                                                                      frcname,int(t_end/12),runid,
                                                                                      ampq,method,
                                                                                      cexpname) 
    # dmp0 indicates that points with insignificant lbd_a were set to zero.
    # previously, they were set to np.nan, or the whole damping term was set to zero
    
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
                       debug=False,check=False,useslab=useslab,
                       savesep=savesep,method=method,
                       custom_params=custom_params,
                       hconfigs=hconfigs)
print("Completed in %.2fs" % (time.time()-st))


