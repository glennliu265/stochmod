#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:41:54 2025


scm scripts organization

2025.05.01
    Attempted to reorganized scripts into logical sections.
    There appears to be several versions of the stochastic model floating around :(...
    Future project is to organize this all into their own scripts (or write a single polished version)
    
    Below is a description of some of the main categories/groupings (not everything is tagged)
    
    [SSS_paper] : SSS and SST stochastic salinity model for Liu et al. 2025
        - Main script: reemergence/stochastic_model/run_SSS_basinwide.py
        - Runs based on parameter dictionary 'expdict'
        
    [SST_paper] : Original stochastic model hierarchy paper for SST (Liu et al. 2023)
    
    <In development>
        - Scripts that are not associated with a finished project, work in progress.
        
    

-------------------------------------------------------------------------------
~~ Stochastic Model Code
    Key scripts for running the stochastic model
    
no_entrain              : Run non-entraining stochastic model at a point
no_entrain_2d           : 2D implementation of no_entrain
entrain                 : Run entraining stochastic model at a point

integrate_noentrain     : 2-D implementation of entraining stochastic model
integr_noentrain        : copied version of integrate_noentrain with different indexing?
integrate_entrain       : Point-by-point loop of entraining stochastic model

-------------------------------------------------------------------------------
~~ Stochastic Model Utilities
    Components to run the stochastic model
    
calc_Td                 : Calculate entraining temperature (Td) using linear interpolation
calc_Td_decay_factor    : Compute exponential decay given the lbdd (TdCorr=True)

find_kprev              : Find month of detrainment given seasonal cycle of MLD
calc_kprev_dmon         : Calculate # of months between detrainment/entrainment

calc_FAC                : Calculate the integration factor (FAC) given total damping (lbd)
calc_beta               : Calculate the discretized entrainment velocity (beta)

repair_expparams        : (SSS_paper) Repair parameter dictionary
patch_expparams         : (SSS_paper) Check parameter dictionary for new inputs
gen_expdir              : (SSS_paper) Generate experiment directory
load_params             : (SSS_paper) Load input parameters
convert_inputs          : (SSS_paper) Convert inputs based on parameter dictionary toggles

load_pathdict           : <In development> Load .csv containing paths to all files
unpack_smdict           : (SST_paper) Reprocesses sm analysis output from dict to array

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

compute_sm_metrics      : Compute basic metrics for the stochastic model (ACF, Spectra, MonVar)

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
integrate_Q             : Integrate Net Heat flux to compare with CESM (copied from SST_Paper Wrappers, formerly SCM Rewritten)
method1                 : Discretization/Test with Lambda (from Claude's derivations)
method2                 : Discretization/Test with Lambda (from Claude's derivations)

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
~~ SST_paper Wrappers
    Scripts for use in the Liu et al. 2023 stochastic model hierarchy paper...

convert_Wm2     : (SST_paper) Convert forcing and damping to degC/sec
load_inputs     : (SST_paper) Load stochastic model inputs given <mconfig>
make_forcing    : (SST_paper) Make scaled white noise timeseries 
tile_forcing    : (SST_paper) Repeat monthly forcing to full simulation length
make_forcing_pt : (SST_paper) 1-D version of make_forcing
cut_regions     : (SST_paper) Crop stochastic model output to selected regions

run_sm_rewrite  : (SST_paper) Run stochastic model given <mconfig>




-------------------------------------------------------------------------------

@author: gliu
"""


