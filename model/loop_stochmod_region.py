#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

loop_stochmod_region
Created on Sun Aug 23 17:10:35 2020

@author: gliu
"""

import sys
import time
import glob
#%% Determine System
startall = time.time()
stormtrack = 1
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
#    scriptpath  = projpath + '03_Scripts/stochmod/'
    datpath     = projpath + '01_Data/'
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")


elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

#%%

import stochmod_region as sr

# # of runs
pointmode = 0 # Set to 1 to output data for the point speficied below
points    = [-30,50] # Lon, aLat for pointmode
naoscale  = 10 # Number to scale NAO and other forcings by
parallel  = False

# Integration Options
nyr      = 1000        # Number of years to integrate over
fstd     = 0.3         # Standard deviation of the forcing
bboxsim  = [-100,20,-20,90] # Simulation Box

# Do a stormtrackloop
runids    = ["202"]
funiforms = [5.5,7]
applyfac  = 2
mconfig   = "SLAB_PIC"#"FULL_HTR"

fscale = naoscale

for runid in runids:
    
    for funiform in funiforms:
        
        if (len(glob.glob(datpath+'model_output/' + "stoch_output_%iyr_run%s_randts.npy"%(nyr,runid)))==0):
            genrand=1
        else:
            
            genrand=0

        sr.stochmod_region(pointmode,funiform,fscale,runid,genrand,nyr,fstd,bboxsim,stormtrack,
                           mconfig=mconfig,applyfac=applyfac,parallel=parallel)
        print("Completed run %s funiform %s (Runtime Total: %.2fs)" % (runid,funiform,time.time()-startall))