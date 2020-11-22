# -*- coding: utf-8 -*-
"""
stochmod_sst, Python Version


This is a temporary script file.
"""

from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#import seaborn as sns
import xarray as xr
import time
import sys

from dask.distributed import Client,progress
import dask

#%% User Edits

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
#%% DEBUG CHOICES

# Integration/Model Options
nyr        = 1000        # Number of years to integrate over
pointmode  = 0
bboxsim    = [-100,20,-20,90] # Simulation Box
mconfig    = "FULL_HTR"
points     = [-30,50]

# Forcing Options
runid      = "100"
genrand    = 1 
fstd       = 0.3         # Standard deviation of the forcing
fscale     = 1
funiform   = 6
applyfac   = 2

# Running Location
stormtrack = 0 # Set to 1 if running in stormtrack


#%%




"""
Inputs

1) pointmode [int]
    (0) Full Run (1) Single Point (2) Regional Average
2) funiform [int]
    Forcing Type:
        (0) Spatially Random
        (1) Spatially Uniform
        (2) NAO pattern (Fixed DJFM)
        (3) NAO pattern (Varying DJFM)
        (4) NAO pattern (Monthly)
        (5) EAP pattern (Fixed DJFM)
        (6) NAO+EAP pattern (Fixed DJFM)
        (7) NAO+EAP pattern (Monthly)
3) applyfac [int]
    (0) Forcing is just the White Noise
    (1) Forcing is white noise (numerator) and includes MLD
    (2) Forcing includes both MLD seasonal cycle AND integration factor
    
    Apply integration factor and mixed layer depth pattern to forcing...

"""


def stochmod_region(pointmode,funiform,fscale,runid,genrand,nyr,fstd,bboxsim,stormtrack,points=[-30,50],mconfig='FULL_HTR',applyfac=1):
                    
    # --------------
    # %% Set Parameters--------------------------------------------------------
    # --------------
    # Unpack Points if in pointmode
    lonf,latf = points
    
    # Other intengration Options (not set by user)
    t_end    = 12*nyr      # Calculates Integration Period
    dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
    T0       = 0           # Initial temperature [degC]
    hfix     = 50          # Fixed MLD value (meters)
    
    # Set Constants
    cp0      = 3850 # Specific Heat [J/(kg*C)]
    rho      = 1025 # Density of Seawater [kg/m3]
    
    # Set Integration Region
    lonW,lonE,latS,latN = bboxsim
    
    # Save Option
    saveforcing = 0 # Save Forcing for each point (after scaling, etc)
    
    #Set Paths (stormtrack and local)
    if stormtrack == 0:
        projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
        datpath     = projpath + '01_Data/'
        sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
        sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    elif stormtrack == 1:
        datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
        sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
        sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    import scm
    from amv import proc
    input_path  = datpath + 'model_input/'
    output_path = datpath + 'model_output/'   
    
    ## ------------ Script Start -------------------------------------------------
    
    print("Now Running stochmod_region with the following settings: \n")
    print("mconfig   = " + mconfig)
    print("funiform  = " + str(funiform))
    print("genrand   = " + str(genrand))
    print("fstd      = " + str(fstd))
    print("runid     = " + runid)
    print("pointmode = " + str(pointmode))
    print("fscale    = " + str(fscale))
    print("nyr       = " + str(nyr))
    print("bbox      = " + str(bboxsim))
    print("Data will be saved to %s" % datpath)
    allstart = time.time()
    
    # Set experiment ID
    
    #expid = "%iyr_funiform%i_run%s_fscale%03d" %(nyr,funiform,runid,fscale)
    expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyr,funiform,runid,fscale,applyfac)
    
    # --------------
    # %% Load Variables ------------------------------------------------------
    # --------------
    
    # Load Latitude and Longitude
    dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp    = loadmat(input_path+dampmat)
    LON         = np.squeeze(loaddamp['LON1'])
    LAT         = np.squeeze(loaddamp['LAT'])
    
    # Load Atmospheric Heat Flux Feedback/Damping
    if mconfig == "FULL_HTR":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")
    
    elif mconfig == "SLAB_PIC":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
    
    # Load Mixed layer variables (preprocessed in prep_mld.py)
    mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
    kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month
    
    # ------------------
    # %% Restrict to region --------------------------------------------------
    # ------------------
    
    # Note: what is the second dimension for?
    dampingr,lonr,latr = proc.sel_region(damping,LON,LAT,bboxsim)
    hclim,_,_ = proc.sel_region(mld,LON,LAT,bboxsim)
    kprev,_,_ = proc.sel_region(kprevall,LON,LAT,bboxsim)
    
    # Get lat and long sizes
    lonsize = lonr.shape[0]
    latsize = latr.shape[0]
    np.save(datpath+"lat.npy",latr)
    np.save(datpath+"lon.npy",lonr)
    
    # ------------------
    # %% Prep NAO Forcing ----------------------------------------------------
    # ------------------
    
    # Consider moving this section to another script?
    if funiform > 1: # For NAO-like forcings (and EAP forcings, load in data and setup)
    
        # Load Longitude for processing
        lon360 =  np.load(datpath+"CESM_lon360.npy")
    
        if funiform == 2: # Load (NAO-NHFLX)_DJFM Forcing
            
            # [lon x lat x pc]
            naoforcing = np.load(input_path+mconfig+"_NAO_EAP_NHFLX_Forcing_DJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC1 # [lon x lat x 1]
            NAO1 = naoforcing[:,:,[0]]
                
        elif funiform == 3: # NAO (DJFM) regressed to monthly NHFLX
            
        
            # NOTE: THESE HAVE NOT BEEN RECALCULATED. NEED TO REDO FOR PIC SLAB ----
            # Load NAO Forcing and take ensemble average
            naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
            NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
            
            
        elif funiform == 4: # Monthly NAO and NHFLX
            
            # NOTE: THESE HAVE NOT BEEN RECALCULATED. NEED TO REDO FOR PIC SLAB ----
            # # Load Forcing and take ensemble average
            # naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
            # NAO1 = np.nanmean(naoforcing,0)
            
            # Load Forcing and take ensemble average
            naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
            
            # Select PC1 Take ensemble average
            NAO1 = naoforcing[:,:,:,:,0].mean(0)
        
        elif funiform == 5: # EAP (DJFM) ONLY
            
            # [lon x lat x pc]
            naoforcing = np.load(input_path+mconfig+"_NAO_EAP_NHFLX_Forcing_DJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC1 # [lon x lat x 1]
            NAO1 = naoforcing[:,:,[1]]
            
            
        elif funiform == 6: # DJFM NAO and EAP
            
            # [lon x lat x pc]
            naoforcing = np.load(input_path+mconfig+"_NAO_EAP_NHFLX_Forcing_DJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC 1 and 2 # [lon x lat x 2]
            NAO1 = naoforcing[:,:,[1,2]]
        
        
        # Restrict to region 
        NAO1,_,_ = proc.sel_region(NAO1,LON,LAT,bboxsim)
        
    else: # For funiform= uniform or random forcing, just make array of ones
        
        NAO1 = np.ones(hclim.shape)
    
    
    
    # Convert NAO from W/m2 to degC/sec. Returns dict with keys 0-2
    NAOF  = {}
    NAOF1 = {}
    if applyfac == 0: # Don't Apply MLD Cycle
        
        if funiform > 1:
            NAO1 = NAO1 * dt / rho / cp0 # Do conversions (minus MLD)
        
        for i in range(3):
            if funiform > 5: 
                NAOF[i]  = NAO1[:,:,0].copy() # NAO Forcing
                NAOF1[i] = NAO1[:,:,1].copy() # EAP Forcing
            else:
                NAOF[i] = NAO1.copy()
            
    else: # Apply seasonal MLD cycle and convert
            
            
        if funiform > 5: # Separately convert NAO and EAP forcing
            NAOF  = scm.convert_NAO(hclim,NAO1[:,:,0],dt,rho=rho,cp0=cp0,hfix=hfix) # NAO Forcing
            NAOF1 = scm.convert_NAO(hclim,NAO1[:,:,1],dt,rho=rho,cp0=cp0,hfix=hfix) # EAP Forcing
    
        else:
            NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)

    # Out: Dict. (keys 0-2) with [lon x lat x mon]
    """     
    # Outformat: Dict. (keys 0-2, representing MLD type) with [lon x lat x mon]
    # We have prepared NAO forcing patterns for the 3 different MLD treatments (if
    # applyfac is set. All it requires now is scaling by both the chosen factor and
    # white nosie timeseries)
    """
    
    # ----------------------------
    # %% Set-up damping parameters
    # ----------------------------
    
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)
    
    """
    Out Format:
        lbd -> Dict (keys 0-3) representing each mode, damping parameter
        lbd_entr -> array of entrainment damping
        FAC -> Dict (keys 0-3) representing each model, integration factor
        beta ->array [Lon x Lat x Mon]
    """
    
    # ----------------------------
    # %% Set Up Forcing           ------------------------------------------------
    # ----------------------------
    
    startf = time.time()
    
    

    startf = time.time()
        
    # Prepare or load random time series
    if genrand == 1: # Generate new time series
    
        print("Generating New Time Series")
        if funiform == 0: # Create entire forcing array [lon x lat x time] and apply scaling factor
            F = np.random.normal(0,fstd,size=(lonsize,latsize,t_end)) * fscale # Removed Divide by 4 to scale between -1 and 1
            # Save Forcing
            np.save(output_path+"stoch_output_%s_Forcing.npy"%(expid),F)
            
        else: # Just generate the time series
            randts = np.random.normal(0,fstd,size=t_end) # Removed Divide by 4 to scale between -1 and 1
            np.save(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)
    
    else: # Load old data
    
        print("Loading Old Data")
        if funiform == 0:# Directly load full forcing
            F = np.load(output_path+"stoch_output_%s_Forcing.npy"%(expid))
        else: # Load random time series
            randts = np.load(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))
    
    
    # Generate extra time series for EAP forcing
    if funiform in [5,6,7]:
        numforce = 1 # In the future, incoporate forcing for other EOFs
        if genrand == 1:
            print("Generating Additional New Time Series for EAP")
            randts1 = np.random.normal(0,fstd,size=t_end) # Removed Divide by 4 to scale between -1 and 1
            np.save(output_path+"stoch_output_%iyr_run%s_randts_%03d.npy"%(nyr,runid,numforce),randts)
        else:
            print("Loading Additional New Time Series for EAP")
            randts1 = np.load(output_path+"stoch_output_%iyr_run%s_randts_%03d.npy"%(nyr,runid,numforce))
            
        if funiform == 5: # Assign EAP Forcing white noise time series
            randts = randts1
        
    
    # Use random time series to scale the forcing pattern
    if funiform != 0:
        
            
        if funiform in [5,6,7]: # NAO + EAP Forcing
            F,Fseas   = scm.make_naoforcing(NAOF,randts,fscale,nyr) # Scale NAO Focing
            F1,Fseas1 = scm.make_naoforcing(NAOF1,randts1,fscale,nyr) # Scale EAP forcing
            
            
            # Add the two forcings together
            for hi in range(3):
                F[hi]     += F1[hi]
                Fseas[hi] += Fseas1[hi]
                
        else: # NAO Like Forcing of funiform with mld/lbd factors, apply scaling and randts
        
            F,Fseas = scm.make_naoforcing(NAOF,randts,fscale,nyr)
        
    
        # Save Forcing if option is set
        if saveforcing == 1:
            np.save(output_path+"stoch_output_%s_Forcing.npy"%(runid),F)
            
    print("Forcing Setup in %.2fs" % (time.time() - startf))
    
    """
    Output:
        F - dict (keys = 0-2, representing each MLD treatment) [ lon x lat x time (simulation length)]
        Fseas - dict (keys = 0-2, representing each MLD treatment) [ lon x lat x  month]
    """
    
    # ----------------------------
    # %% Additional setup based on pointmode  ------------------------------------------------
    # ----------------------------    
    
    if pointmode == 1: # Find indices for pointmode
    
        # Get point indices
        ko,ka = proc.find_latlon(lonf,latf,lonr,latr)
        locstring = "lon%02d_lat%02d" % (lonf,latf)
        
        # Select variable at point
        hclima   = hclim[ko,ka,:]
        dampinga = dampingr[ko,ka,:]
        kpreva   = kprev[ko,ka,:]
        lbd_entr = lbd_entr[ko,ka,:]
        beta     = beta[ko,ka,:]
        naoa     = NAO1[ko,ka,...]
        
        # Select forcing at point
        Fa    = {} # Forcing
        Fseasa = {} # Seasonal Forcing pattern
        for hi in range(3):
            Fa[hi] = F[hi][ko,ka,:]
            Fseasa = Fseas[hi][ko,ka,:]
        F = Fa.copy()
        Fseas=Fseasa.copy()
        
        # Do the same but for each model type (hfdamping and FAC)
        lbda = {}
        FACa = {}
        for model in range(4):
            FACa[model] = FAC[model][ko,ka,:]
            lbda[model] = lbd[model][ko,ka,:]
        lbd = lbda.copy()
        FAC = FACa.copy()

    if pointmode == 2: # Take regionally averaged parameters (need to recalculate some things)
    
        # Make string for plotting
        locstring = "lon%02d_%02d_lat%02d_%02d" % (lonW,lonE,latS,latN)
        
        # Current setup: Average raw variables, assuming
        # that bboxsim is the region you want to average over
        hclima    = np.nanmean(hclim,(0,1))    # Take lon,lat mean, ignoring nans
        kpreva    = scm.find_kprev(hclima)[0]  # Recalculate entrainment month
        dampinga  = np.nanmean(dampingr,(0,1)) # Repeat for damping
        naoa      = np.nanmean(NAO1,(0,1))     # Repeat for nao forcing
        
        # Get regionally averaged forcing based on mld config
        rNAOF = {}
        rF    = {}
        for hi in range(3):
            rNAOF[hi] = proc.sel_region(NAOF[hi],lonr,latr,bboxsim,reg_avg=1)
            rF[hi] = randts * np.tile(rNAOF[hi],nyr)
            
        # Add in EAP Forcing [consider making separate file to save?]
        if funiform in [6,7]: # NAO + EAP Forcing
            for hi in range(3):
                rNAOF1 = proc.sel_region(NAOF1[hi],lonr,latr,bboxsim,reg_avg=1)
                rF1 = randts1 * np.tile(rNAOF1,nyr)
                
                # Add to forcing
                rNAOF[hi] += rNAOF1
                rF[hi] += rF1
        # Copy over forcing
        F = rF.copy()
        Fseas = rNAOF.copy()
        

        
        # Convert units
        lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclima,dampinga,dt,ND=0,rho=rho,cp0=cp0,hfix=hfix)
    
    
    """
    Output:
        
    Dict with keys 0-2 for MLD configuation
        - F (Forcing, full timeseries)
        - Fseas (Forcing, seasonal pattern)
    
    Dict with keys 0-3 for Model Type
        - lbd (damping parameter)
        - FAC (integration factor)
    
    Just Arrays...
        - beta (entrainment velocity)
        - dampinga (atmospheric damping)
        - hclima (mixed layer depth)
        - kpreva (entraining month)
        - naoa (NAO forcing pattern)
        
    """
    
    # ----------
    # %%RUN MODELS -----------------------------------------------------------------
    # ----------
    
    # Set mulFAC condition based on applyfac
    if applyfac == 2:
        multFAC = 1 # Don't apply integrationreduction factor if applyfac is set to 0 or 1
    else:
        multFAC = 0
    
    # Run Model Without Entrainment
    sst = {}
    # Loop for each Mixed Layer Depth Treatment
    for hi in range(3):
        start = time.time()
        
        # Select damping and FAC based on MLD
        FACh = FAC[hi]
        lbdh = lbd[hi]
        
        # Select Forcing
        Fh  = F[hi]
        
    
        # # Match Forcing and FAC shape
        # if (len(Fh.shape)>2) & (Fh.shape[2] != FACh.shape[2]):
        #     FACh = np.tile(FACh,int(t_end/12))
    
        
        if pointmode == 0: #simulate all points
            if Fh.shape[2] < 12: # Adjust for cases where Fh is not seasonal
                Fh = np.tile(Fh,12)
            if funiform == 0:
                randts = np.copy(Fh)
            
            sst[hi],_ =  scm.noentrain_2d(randts,lbdh,T0,Fh,FACh,multFAC=multFAC)
            print("Simulation for No Entrain Model, hvarmode %s completed in %s" % (hi,time.time() - start))
        
    elif pointmode == 1: # simulate for 1 point
        start = time.time()
    
        # Run Point Model
        sst[hi],_,_=scm.noentrain(t_end,lbdh[ko,ka,:],T0,Fh[ko,ka,:],FACh[ko,ka,:],multFAC=multFAC)
    
        elapsed = time.time() - start
        tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
        print(tprint)
    
    elif pointmode == 2: # simulate using regionally averaged params
    
         # Run Point Model
        start = time.time()
        sst[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)
        
        elapsed = time.time() - start
        tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
        print(tprint)    
        
    
    #%%
    # Scrap test
    
    # import matplotlib.pyplot as plt
    
    # diff_h0 = new_h0-old_h0
    # fig,ax = plt.subplots(1,1)
    # ax.plot(old_h0,color='k',linewidth=0.5,label='old-pt')
    # ax.plot(new_h0,color='b',linewidth=0.5,label='new-region')
    # ax.plot(diff_h0,label='diff')
    # plt.legend()
    # np.nanmax(np.abs(diff_h0))
    # plt.xlim(12,600)
    
    # #%% Set up dask client
    # client = Client(threads_per_worker=4,n_workers=1)
    
    #
    #%%
    
    
    # Run Model With Entrainment
    start = time.time()
    
    icount = 0
    if funiform > 1:
    Fh = np.copy(F[2])
    else:
    Fh = np.copy(F)
    FACh = FAC[3]
    
    if pointmode == 1:
    
    sst[3]= scm.entrain(t_end,lbd_entr[ko,ka,:],T0,Fh[ko,ka,:],beta[ko,ka,:],hclim[ko,ka,:],kprev[ko,ka,:],FACh[ko,ka,:],multFAC=multFAC)
    
    elif pointmode == 2:
    
    sst[3]= scm.entrain(t_end,lbd_entr,T0,Fh,beta,hclima,kpreva,FACh,multFAC=multFAC)
    
    else:
    
    T_entr1 = np.ones((lonsize,latsize,t_end))*np.nan
    for o in range(0,lonsize):
        # Get Longitude Value
        lonf = lonr[o]
        
        # Convert to degrees East
        if lonf < 0:
            lonf = lonf + 360
        
        for a in range(0,latsize):
    
            # Get latitude indices
            latf = latr[a]
            
            
            # Skip if the point is land
            if np.isnan(np.mean(dampingr[o,a,:])):
                #msg = "Land Point @ lon %f lat %f" % (lonf,latf)
                icount += 1
                continue
                #print(msg)
    
            else:
                T_entr1[o,a,:] = scm.entrain(t_end,lbd_entr[o,a,:],T0,Fh[o,a,:],beta[o,a,:],hclim[o,a,:],kprev[o,a,:],FACh[o,a,:],multFAC=multFAC)
                
                # lbdin   = np.copy(lbd_entr[o,a,:])
                # Fin     = np.copy(Fh[o,a,:])
                # betain  = np.copy(beta[o,a,:])
                # hclimin = np.copy(hclim[o,a,:])
                # kprevin = np.copy(kprev[o,a,:])
                # FACin   = np.copy(FAC[o,a,:])
                # delayedtask = dask.delayed(scm.entrain)(t_end,lbdin,T0,Fin,betain,hclimin,kprevin,FACin)
                # T_entr1[o,a,:] = delayedtask
            icount += 1
            msg = '\rCompleted Entrain Run for %i of %i points' % (icount,lonsize*latsize)
            print(msg,end="\r",flush=True)
        #End Latitude Loop
    #End Longitude Loop
    
    # Copy over to sst dictionary
    sst[3] = T_entr1.copy()
    
    #T_entr1 = dask.compute(*T_entr1)
    elapsed = time.time() - start
    tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
    print(tprint)    
        
    #%% Combine dimensions
    
    
    # def combine_spatial(var):
    #     """
    #     Assuming spatial dimensions are axis = 0,1, combine them.
    #     Also assumes that the variable is 3d
    #     """
    
    #     spsize = var.shape[0] * var.shape[1]
    #     varrs = np.reshape(var,(spsize,var.shape[2]))
    #     return varrs
    
    
    # lbd2d  =  combine_spatial(lbd_entr)
    # if funiform > 1:
    #     F2d    =  combine_spatial(F[2])
    # else:
    #     F2d    =  combine_spatial(F)
    # beta2d =  combine_spatial(beta)
    # h2d    =  combine_spatial(hclim)
    # kprev2d = combine_spatial(kprev)
    # FAC2d   = combine_spatial(FAC)
    # spsize = lbd2d.shape[0]
    
    
    # T_entr1 = dask.delayed([])
    # for i in range(spsize):
    #     lbdin   = np.copy(lbd2d[i,:])
    #     Fin     = np.copy(F2d[i,:])
    #     betain  = np.copy(beta2d[i,:])
    #     hclimin = np.copy(h2d[i,:])
    #     kprevin = np.copy(kprev2d[i,:])
    #     FACin   = np.copy(FAC2d[i,:])
    #     delayedtask = dask.delayed(scm.entrain)(t_end,lbdin,T0,Fin,betain,hclimin,kprevin,FACin)
    #     T_entr1.append(delayedtask)
    
    
    # start = time.time()
    # dask.compute(*T_entr1)
    # elapsed = time.time() - start
    # tprint = "\nEntrain Model ran in %.2fs" % (elapsed)
    
    # %% save output
    
    if pointmode > 0:
    
    if pointmode == 1:
        np.savez(output_path+"stoch_output_point%s_%s.npz"%(locstring,expid),sst=sst,hpt=hclim[ko,ka,:])
        
    elif pointmode == 2:
        np.savez(output_path+"stoch_output_point%s_%s.npz"%(locstring,expid),
                 sst=sst,
                 hclim=hclima,
                 kprev=kpreva,
                 dampping=dampinga,
                 F=F,
                 lbd=lbd,
                 lbd_entr=lbd_entr,
                 beta=beta,
                 FAC=FAC,
                 NAO1=NAO1,
                 NAOF=NAOF
                 )
    
    else:
    
    # SAVE ALL in 1
    np.save(output_path+"stoch_output_%s.npy"%(expid),sst)
    
    #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain0_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr0_all)
    #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain1_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr1)
    
    print("stochmod_region.py ran in %.2fs"% (time.time()-allstart))
    print("Output saved as %s" + output_path + "stoch_output_%s.npy"%(expid))
