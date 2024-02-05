#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basinwide expeirment parameters for run_SSS_basinwide
Created on Sun Feb  4 19:06:41 2024

@author: gliu
"""


"""
Run of half-shifted damping/forcing/mld over the Atlantic Basin


"""
expname     = "Test_Td0.1_SPG_allroll1_halfmode"

expparams   = {
    'varname'       : "SSS"
    'bbox_sim'      : [-65,0,45,65],
    'nyrs'          : 1000,
    'runids'        : ["test%02i" % i for i in np.arange(1,6,1)],
    'runid_path'    : None, # If true, load a runid from another directory
    'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'         : 0.10,
    'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'          : None, # If None, just compute entrainment damping
    'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : True,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'         : 1,
    'mroll'         : 1,
    'droll'         : 1,
    'halfmode'      : True,
    }

"""


"""




"""
NAT Extratropics run with Expfit Lbda
"""


expname     = "SST_expfit_damping_20to65"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,6,1)],
    'runid_path'        : "SST_covariance_damping_20to65", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_Expfitlbda123_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : False,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    }

"""
Same as above but with SST ACF fit (full rather than just lbd_a estimate)
"""

expname     = "SST_expfit_SST_damping_20to65"

expparams   = {
    'varname'           : "SST",
    'bbox_sim'          : [-80,0,20,65],
    'nyrs'              : 1000,
    'runids'            : ["run%02i" % i for i in np.arange(1,6,1)],
    'runid_path'        : "SST_covariance_damping_20to65", # If true, load a runid from another directory
    'Fprime'            : "CESM1_HTR_FULL_Fprime_ExpfitSST123_nroll0_NAtl_EnsAvg.nc",       
    'PRECTOT'           : None,
    'LHFLX'             : None,
    'h'                 : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'             : None,
    'Sbar'              : None,
    'beta'              : None, # If None, just compute entrainment damping
    'kprev'             : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'             : "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123_EnsAvg.nc", # NEEDS TO BE CONVERTED TO 1/Mon !!!
    'convert_Fprime'    : True,
    'convert_lbd_a'     : False,
    'convert_PRECTOT'   : True,
    'convert_LHFLX'     : True,
    'froll'             : 0,
    'mroll'             : 0,
    'droll'             : 0,
    'halfmode'          : False,
    }