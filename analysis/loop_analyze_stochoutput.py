#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 05:31:36 2020

@author: gliu
"""
import numpy as np
import sys


#%%
stormtrack = 1
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
   
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

#%% User Edits

# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
naoscale  = 10 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over

# Do a stormtrackloop
runids = ("001","002","003","004","005")
funiforms = (0,1,2,5,6)
#funiforms=[0,1,2,5,6]
#runids=['006']

for runid in runids:
    for funiform in funiforms:
        
        if funiform < 2:
            fscale = 1
        else:
            fscale = naoscale
            
        # Set experiment ID
        expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
        
        # Generate Output
        scm.postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags)
        print("Completed Post-processing for Experiment: %s" % expid)

