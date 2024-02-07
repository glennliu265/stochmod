#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Stochastic Model Parameter Files

Created on Thu Nov 16 17:34:14 2023
@author: gliu

"""

#%% Experiments Dictionary

"""
Notes on runid

runid		Notes/Info
-----		------------
003		Runs where the latitude weights were applied (can toss, placed in old_runs folder)
004		Runs including the correction to q based on Claude's calculations
005		Runs not including the q correction (except q_ek, 50-eofs run includes q-corr...)
006              90% Variance-based threshold (68 EOFs) without q-correction
007              90% Variance-based threshold (68 EOFs), with q-correction (ann-avg), or also 

011              90% Variance based threshold + Fprime, Qek, etc
012              90% Variance based threshold, Qnet re-run (2022.02.23)

200-209          90% Variance threshold, Fprime, CONTINUOUS RUN (2022.02.24)
200-209          (useslab2) - Same as above, but using SLAB damping (2022.07.19)
200-209          (useslab4) - Same as above, but using FULL damping (2022.07.19)


e00              90% Variance threshold, Fprime, SLAB PARAMS for Damping/Forcing, Lag 1 Mode 4

*****************
Other Terminology
*****************

------
 ampq
------
Corrections applied to Qnet. Not applicable for Fprime runs.
0 : No q-correction was applied
1 : Old q-correction method was applied (2a^2/(1-a))
2 : Method 1 q-corr applied (instantaneous Q)
3 : Method 2 q-corr applied (averaged Q)

--------
 method
--------
Damping Significance Testing/Treatment. Insig. HFF values set to zero except for method 5.
*based on "mode" nomenclature from calc_HF_func.py
1 : No Testing Applied
2 : SST Autocorrelation Testing
3 : SST-FLX Crosscorrelation Testing
4 : Both 2 and 3
5 : Same as 4, but replace insig. CESM-FULL values with those from SLAB

-----
 dmp
-----
*from documentation in sm_rewrite_loop.py
dmp0 indicates that points with insignificant lbd_a were set to zero.
previously, they were set to np.nan, or the whole damping term was set to zero
--> This is a residual name, may remove from chain.

"""

# 0. Default Run
expname0 = "default"
list0    = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run2%02d_ampq3_method5_dmp0" %i for i in range(10)]

# 1. Default Run, but add damping to bottom temperature Td
expname1 = "Tddamp"
list1    = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_useslab0_ensorem0_Tddamp"%i for i in range(10)]

# 2. Default Run, but add Stochastic Ekman Forcing
expname2 = "Qek"
list2    = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0_Qek"%i for i in range(10)]

# 3. Seasonally Fixed Forcing
expname3 = "seasonal"
list3    = [
    "forcingflxeof_090pct_FULL-PIC_eofcorr2_DJF_Fprime_rolln0_1000yr_run011_ampq0_method5_dmp0",
    "forcingflxeof_090pct_FULL-PIC_eofcorr2_MAM_Fprime_rolln0_1000yr_run011_ampq0_method5_dmp0",
    "forcingflxeof_090pct_FULL-PIC_eofcorr2_JJA_Fprime_rolln0_1000yr_run011_ampq0_method5_dmp0",
    "forcingflxeof_090pct_FULL-PIC_eofcorr2_SON_Fprime_rolln0_1000yr_run011_ampq0_method5_dmp0"
    ]

# Make Dictionary (copied form predict_amv_params)
runlists  = (list0,list1,list2,list3)
runnames  = (expname0,expname1,expname2,expname3)
rundicts  = dict(zip(runnames,runlists))

#%% General Names for Plotting, Display, ETC

# Bounding Boxes [LonW, LonE, LatS, latN]
amvbbox       = [-80,0,0,65]   # AMV Calculation box
bbox_crop     = [-90,20,0,90]  # Preprocessing box


#%% Regional Analysis

# Copied below from sm_paper_stylesheet, 2024.02.05 ---------------------------
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
# -----------------------------------------------------------------------------



#%% Stochastic Model Hierarchy Names

# Taken some sections from analysis/sm_paper_stylesheet/
modelnames  = ("Constant h (Level 3)","Vary h (Level 4)","Entraining (Level 5)")
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


#%%

