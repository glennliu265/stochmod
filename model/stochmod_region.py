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

def stochmod_region(pointmode,funiform,fscale,runid,genrand,nyr,fstd,bboxsim,stormtrack,points=[-30,50]):
                    

    # Unpack Points if in pointmode
    lonf,latf = points
        
    # Other intengration Options
    t_end    = 12*nyr      # Calculates Integration Period
    dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
    T0       = 0           # Initial temperature [degC]
    hfix     = 50          # Fixed MLD value (meters)

    # Set Constants
    cp0      = 3850 # Specific Heat [J/(kg*C)]
    rho      = 1025 # Density of Seawater [kg/m3]

    # Set Integration Region
    lonW,lonE,latS,latN = bboxsim

    
    
    #Set Paths
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
    
    import scm
    from amv import proc
    input_path  = datpath + 'model_input/'
    output_path = datpath + 'model_output/'   
      

    # Set up some strings for labeling
    #mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    #monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')
    
    ## ------------ Script Start -------------------------------------------------
    print("Now Running stochmod_region with the following settings: \n")
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
    expid = "%iyr_funiform%i_run%s_fscale%03d" %(nyr,funiform,runid,fscale)
    # --------------
    # %% Load Variables -------------------------------------------------------------
    # --------------
    
    # Load damping variables (calculated in hfdamping matlab scripts...)
    dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp    = loadmat(input_path+dampmat)
    LON         = np.squeeze(loaddamp['LON1'])
    LAT         = np.squeeze(loaddamp['LAT'])
    damping     = loaddamp['ensavg']
    
    # Load Mixed layer variables (preprocessed in prep_mld.py)
    mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
    kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month
    
    # Save Options are here
    saveforcing = 0 # Save Forcing for each point
    
    # ------------------
    # %% Restrict to region ---------------------------------------------------------
    # ------------------
    
    # Note: what is the second dimension for?
    klat = np.where((LAT >= latS) & (LAT <= latN))[0]
    if lonW < 0:
        klon = np.where((LON >= lonW) & (LON <= lonE))[0]
    else:
            
        klon = np.where((LON <= lonW) & (LON >= lonE))[0]
              
    # Restrict Damping Region
    dampingr = damping[klon[:,None],klat[None,:],:]
    lonr = np.squeeze(LON[klon])
    latr = np.squeeze(LAT[klat])
    
    # Restrict MLD variables to region
    hclim = mld[klon[:,None],klat[None,:],:]
    kprev = kprevall[klon[:,None],klat[None,:],:]
    
    # Get lat and long sizes
    lonsize = lonr.shape[0]
    latsize = latr.shape[0]
    np.save(datpath+"lat.npy",latr)
    np.save(datpath+"lon.npy",lonr)
    
    # %% Load and Prep NAO Forcing... <Move to separate script?>
    
    
    if funiform > 1: # For NAO-like forcings (and EAP forcings, load in data and setup)
        # Load Longitude for processing
        lon360 =  np.load(datpath+"CESM_lon360.npy")
        
        # Load (NAO-NHFLX)_DJFM Forcing
        if funiform == 2:
            
            # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
            naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC1 and take ensemble average
            NAO1 = np.mean(naoforcing[0,:,:,:],0) # [Lat x Lon]
            NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
            
        elif funiform == 3:
            
            # Load NAO Forcing and take ensemble average
            naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
            NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
            
            
        elif funiform == 4:
            
            # # Load Forcing and take ensemble average
            # naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
            # NAO1 = np.nanmean(naoforcing,0)
        
              # Load Forcing and take ensemble average
            naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
            
            # Take ensemble average, then sum EOF 1 and EOF2
            NAO1 = naoforcing[:,:,:,:,0].mean(0)
        
        elif funiform == 5: # Apply EAP only
            # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
            naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC2 and take ensemble average
            NAO1 = naoforcing[1,:,:,:].mean(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
            NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
            
        elif funiform == 6:
            # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
            naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
            
            # Select PC1-2 and take ensemble average + sum
            NAO1 = naoforcing[0:2,:,:,:].mean(1).sum(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
            NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
            
        elif funiform == 7:
            
            # Load Forcing and take ensemble average
            naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
            
            # Take ensemble average, then sum EOF 1 and EOF2
            NAO1 = naoforcing[:,:,:,:,:2].mean(0).sum(3)
        
        # Transpose to [Lon x Lat x Time]
        NAO1 = np.transpose(NAO1,(2,1,0))
        
        # Convert Longitude to Degrees East
        lon180,NAO1 = proc.lon360to180(lon360,NAO1)
        
        # Test Plot
        #plt.pcolormesh(NAO1[:,:,0].T)
        
        NAO1 = NAO1[klon[:,None],klat[None,:],:]
        
        # Convert from W/m2 to C/S for the three different mld options
        NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)
        
    else:
        NAOF = 1
        
        
    # ----------------------------
    # %% Set-up damping parameters
    # ----------------------------
    
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)
    
    # ----------------------------
    # %% Set Up Forcing           ------------------------------------------------
    # ----------------------------
    
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
    
    
    # Use random time series to scale the forcing pattern
    if funiform != 0: 
        if funiform == 1:# Spatially Uniform Forcing, replicate to domain and apply scaling factor
            F      = np.ones((lonsize,latsize,t_end)) * fscale
            F      = np.multiply(F,randts[None,None,:])
            Fseas  = np.ones((lonsize,latsize,12)) # Dummy variable 
        
        else: # NAO Like Forcing..., apply scaling and randts
        
            F,Fseas = scm.make_naoforcing(NAOF,randts,fscale,nyr)

        # Save Forcing if option is set
        if saveforcing == 1:
            np.save(output_path+"stoch_output_%s_Forcing.npy"%(runid),F)
    
    print("Forcing Setup in %s" % (time.time() - startf))
    

    # ----------------------------
    # %% Additional setup based on pointmode  ------------------------------------------------
    # ----------------------------    
    if pointmode == 1: # Find indices for pointmode
        ko,ka = proc.find_latlon(lonf,latf,lonr,latr)
        locstring = "lon%02d_lat%02d" % (lonf,latf)
    
    if pointmode == 2: # Take regionally averaged parameters (need to recalculate some things)
        
        locstring = "lon%02d_%02d_lat%02d_%02d" % (lonW,lonE,latS,latN)
        
        # For this current setup, raw variables are averaged
        # Note: This assumes that bboxsim is the region you want to average over
        hclima    = np.nanmean(hclim,(0,1)) # Take lon,lat mean, ignoring nans
        kpreva    = scm.find_kprev(hclima)
        dampinga  = np.nanmean(dampingr,(0,1)) # Repeat for damping
        
        
        # Set up forcing
        if funiform > 1:
            Favg = {}
            NAOavg = {}
            NAOFavg = {}
            for hi in range(3): # Take regional average for each forcing
                Favg[hi] = proc.sel_region(F[hi],lonr,latr,bboxsim,reg_avg=1) # Average over region
                NAOavg[hi] = proc.sel_region(NAO1[hi],lonr,latr,bboxsim,reg_avg=1)
                NAOFavg[hi] = proc.sel_region(NAOF[hi],lonr,latr,bboxsim,reg_avg=1)
            F    = Favg.copy()    
            NAO1 = NAOavg.copy() 
            NAOF = NAOFavg.copy()
        else:
            F = proc.sel_region(F,lonr,latr,bboxsim,reg_avg=1)
        
        
        
        # Convert units
        lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclima,dampinga,dt,ND=0,rho=rho,cp0=cp0,hfix=hfix)
        
        # Set up forcing
        if funiform < 2: # Forcing is just the random time series
            F = randts.copy()
        else:
            F = {}
            for hi in range(3):
                F[hi] = proc.sel_region(Fseas[hi],lonr,latr,bboxsim,reg_avg=1) # Take regional average
        
    

    # ----------
    # %%RUN MODELS -----------------------------------------------------------------
    # ----------
    
    
    # Set mulFAC condition based on forcing
    if funiform < 2:
        multFAC = 0 # Don't apply reduction factor to forcing if no pattern is applied
    else:
        multFAC = 1
    
    # Run Model Without Entrainment
    T_entr0_all = {}
    # Loop for each Mixed Layer Depth Treatment
    for hi in range(3):
        start = time.time()
        
        # Select damping and FAC based on MLD
        FACh = FAC[hi]
        lbdh = lbd[hi]
        
        # Select Forcing
        if funiform > 1:
            Fh = F[hi]
        else:
            Fh = F
        
        if len(Fh.shape) > 2:
            FACh = np.tile(FACh[:,:],int(t_end/12))
            
        if pointmode == 0: #simulate all points
            if Fh.shape[2] < 12: # Adjust for cases where Fh is not seasonal
                Fh = np.tile(Fh,12)
            if funiform == 0:
                randts = np.copy(Fh)
                
            T_entr0_all[hi],_ =  scm.noentrain_2d(randts,lbdh,T0,Fh,FACh,multFAC=multFAC)
            print("Simulation for No Entrain Model, hvarmode %s completed in %s" % (hi,time.time() - start))
            
        elif pointmode == 1: # simulate for 1 point
            start = time.time()
    
            # Run Point Model
            T_entr0_all[hi],_,_=scm.noentrain(t_end,lbdh[ko,ka,:],T0,Fh[ko,ka,:],FACh,multFAC=multFAC)

            elapsed = time.time() - start
            tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
            print(tprint)
        
        elif pointmode == 2: # simulate using regionally averaged params
        
             # Run Point Model
            start = time.time()
            T_entr0_all[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)
            
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
    
    
    #output = dask.delayed({})#= dask.delayed(np.ones((lonsize,latsize,t_end))*np.nan)
    
    
    icount = 0
    if funiform > 1:
        Fh = np.copy(F[2])
    else:
        Fh = np.copy(F)
    FACh = FAC[3]
    
    
    if pointmode == 1:
        
        T_entr1= scm.entrain(t_end,lbd_entr[ko,ka,:],T0,Fh[ko,ka,:],beta[ko,ka,:],hclim[ko,ka,:],kprev[ko,ka,:],FACh[ko,ka,:],multFAC=multFAC)
        
    elif pointmode == 2:
        
        T_entr1= scm.entrain(t_end,lbd_entr,T0,Fh,beta,hclima,kpreva,FACh,multFAC=multFAC)
        
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
        
        # Combine entraining and nonentraining models into 1 dictionary
        sst = T_entr0_all.copy()
        sst[3] = T_entr1
        
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
        
        # Combine entraining and nonentraining models into 1 dictionary
        sst = T_entr0_all.copy()
        sst[3] = T_entr1
        
        
        # SAVE ALL in 1
        np.save(output_path+"stoch_output_%s.npy"%(expid),sst)
        
        #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain0_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr0_all)
        #np.save(output_path+"stoch_output_%iyr_funiform%i_entrain1_run%s_fscale%03d.npy"%(nyr,funiform,runid,fscale),T_entr1)
    
    print("stochmod_region.py ran in %.2fs"% (time.time()-allstart))
    print("Output saved as %s" + output_path + "stoch_output_%s.npy"%(expid))
