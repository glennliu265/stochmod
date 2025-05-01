#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:41:54 2025


scm scripts organization

-------------------------------------------------------------------------------
~~ Stochastic Model Code
    Key scripts for running the stochastic model
    
no_entrain      : Run non-entraining stochastic model at a point
no_entrain_2d   : 2D implementation of no_entrain
entrain         : Run entraining stochastic model at a point

-------------------------------------------------------------------------------
~~ Stochastic Model Utilities
    Components to run the stochastic model
    
calc_Td                 : Calculate entraining temperature (Td) using linear interpolation
calc_Td_decay_factor    : Compute exponential decay given the lbdd (TdCorr=True)

find_kprev              : Find month of detrainment given seasonal cycle of MLD
calc_kprev_dmon         : Calculate # of months between detrainment/entrainment

load_pathdict           : <In development> Load .csv containing paths to all files
    
-------------------------------------------------------------------------------
~~ Parameterization and Preprocessing
    Scripts for Parameterizing/calculating/preprocessing stochastic model inputs

get_detrain_depth       : Given detrianment time and MLD, get approximate detrainment depth
calc_tau_detrain        : Retrieve timescale tau at depth and time of detrainment

-------------------------------------------------------------------------------
~~ Analysis
    Scripts for analysis of stochastic model output (and others)
    
calc_autocorr           : Calculate autocorrelation for output of stochastic models
calc_autocorr_mon       : Given 1-D Array, compute monhtly lag correlation

quick_spectrum          : Compute spectra for an array of timeseries [ts1,ts2,...]
get_freqdim             : Get the frequency dimension from spectra calculation
point_spectra           : Compute power spectra at a point, copied from viz_atmospheric_persistence

-------------------------------------------------------------------------------
~~ stochmod Legacy Scripts 
    Legacy scripts for the old stochmod simulations... (first paper and older?)


    <NAO Processing> ---
cut_NAOregion           : Prepare SLP DataArray for NAO calculation (crop region and month average)
make_naoforcing         : Make forcing timeseries, given NAO forcing pattern (old stochmod scripts)
convert_NAO             : Conver NAO Forcing from W/m2 to degC/S given seasonal MLD

    <Wrapper Scripts> ---
set_stochprams          : Given MLD and Heat Flux Feedback, calculate parameters
get_data                : Wrapper to return data based on pointmode
load_data               : <from Synthetic Stochastic Model> Load Inputs for stochastic model
synth_stochmod          : <from Synthetic Stochastic Model> Wrapper to run stochastic model from config script
postprocess_stochoutput : Post-process stochastic model output (region-average analysis)

    <In Development> ---
calc_kprev_lin          : <In development> Find detrainment time given timeseries of mld
entrain_parallel        : <In development> Rewrote [entrain] to take all inputs as 1

-------------------------------------------------------------------------------
~~ Data Loading
    Convenience Scripts to load datasets/variables/etc
    
load_hadisst : Load Obs (HadISST)
load_ersst : Load Obs (ERSST)
load_cesm_pt : Load CESM output at a point
load_latlon : Load CESM Lat and Lon
load_dmasks : Load Damping Masks
load_cesm_le : Load CESM LE output at a point
load_limopt_sst : Load SST data detrended by LIM-opt (Frankignoul et al. 2017)
load_limopt_amv : Load AMV from (Frankignoul et al. 2017)


-------------------------------------------------------------------------------
~~ Heat Flux Feedback (HFF) Calculations
    Functions to estimate the heat flux feedback (inclues preprocessing)

indexwindow     : Index months/years for an odd sliding window for lag analysis, reducing year at year crossings 
calc_HF         : Compute HFF using lag covariance approach
prep_HF         : Apply significance testing to estimated HFF
postprocess_HF  : Apply land/ice mask to HFF and average across selected lags
check_ENSO_sign : Check sign of ENSO from EOF analysis output
calc_ENSO       : Compute ENSO using EOF-based approach
remove_ENSO     : Remove ENSO-related component (determined through regression) fro a variable
compute_qnet    : Given flux components, compute the net heat flux


-------------------------------------------------------------------------------

@author: gliu
"""


