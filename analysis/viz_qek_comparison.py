#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize/Compare How Addition of Qek impacts things (using )


Created on Tue Mar 15 15:02:43 2022

@author: gliu
"""

import numpy as np


import sys

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import proc,viz
import scm
import tbx

from matplotlib import gridspec

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#%% Load in the spectra, and compute

datpath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/'
fnames  = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0" %i for i in range(10)]
figpath = datpath + "../../02_Figures/20220422/"
proc.makedir(figpath)

#%% From SM Stylesheet

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
bbcol       = ["cornflowerblue","Red","Yellow","Black","Black","limegreen","indigo"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted"]


cstep = 0.025
lstep = 0.05
cmax  = 0.5
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]
bboxplot    = [-80,0,5,60]

# Linear-Power Spectra, < 2-yr (Current SM Draft Choice)
xlm = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0,3.0]

#%% Load Lat/Lon
ld      = np.load("%sstoch_output_%s.npz" % (datpath,fnames[0]),allow_pickle=True)
lonr    = ld['lon']
latr    = ld['lat']



# Load Masks
dmsks = scm.load_dmasks(bbox=[lonr[0],lonr[-1],latr[0],latr[-1]])
dmsks.append(dmsks[-1])

#%% Load in the regional SSTs

# Unpack both Qek and Regular forcing
reg_ssts    = np.zeros((2,10,8,12000)) * np.nan # [Forcing, Run, Region, Time] 
amvpats_all = np.zeros((2,10,8,65,69)) * np.nan # [Forcing, Run, Region, LON X LAT] 
amvids_all  = np.zeros((2,10,8,1000)) * np.nan
for q in range(2):
    
    
    # Unpack and load Regional SSTs
    sstdict = []
    for f in range(len(fnames)):
        if q == 0:
            rsst_fn = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,fnames[f])
            mid  = 2
        else:
            rsst_fn = "%s/proc/SST_RegionAvg_%s_Qek.npy" % (datpath,fnames[f])
            mid  = 0
        
        sstdict.append(np.load(rsst_fn,allow_pickle=True).item())
    
    # Unpack Dictionary to numpy array
    sstup = scm.unpack_smdict(sstdict) # [run, region, model, time]
    reg_ssts[q,:,:,:] = sstup[:,:,mid,:]
    
    
    # Unpack and load AMV Patterns
    amvpats = []
    amvids  = []
    for f in range(len(fnames)):
        if q == 0:
            expid = fnames[f]
            mid  = 2
        else:
            expid = fnames[f] + "_Qek"
            mid  = 0
        
        
        rsst_fn = "%sproc/AMV_Region_%s.npz" % (datpath,expid)
        print("Loading %s" % rsst_fn)
        ld = np.load(rsst_fn,allow_pickle=True)#.item()
        
        amvidx = ld['amvidx_region'].item()
        amvpat = ld['amvpat_region'].item()
        
        amvpats.append(amvpat)
        amvids.append(amvidx)
    
    # Unpack dicts into array [nrun,nreg,nmod,nlon,nlat]
    patup = scm.unpack_smdict(amvpats) # 
    idxup = scm.unpack_smdict(amvids)
    
    amvpats_all[q,:,:,:,:] = patup[:,:,mid,:,:]
    amvids_all[q,:,:,:] = idxup[:,:,mid,:]
    
#%% Compute the spectra (stochastic model)

"""
reg_ssts DIMS  :: (2, 10, 8, 12000)
"""

ssmooth    = 30        # Stochastic Model Smoothing
pct        = 0.10
alpha      = 0.05      # For CI Calculatuions

nforce,nruns,nreg,ntime = reg_ssts.shape

smspecs = np.zeros((nforce,nruns,nreg,int(ntime/2)))

for q in range(nforce):
    for rid in range(nruns):
        for reg in range(nreg):
            # Get SST
            inssts = [reg_ssts[q,rid,reg,:],]
            # Compute
            specs,freqs,_,_,_ = scm.quick_spectrum(inssts,ssmooth,pct)
            smspecs[q,rid,reg,:] = specs[0].copy()


#%% Repeat for CESM

#% Load corresponding CESM Data ------------------
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_RegionAvg_%s_ensorem0.npy" % (datpath,expid)
sstcesm    = np.load(rsst_fn,allow_pickle=True).item() # [Region x Model x Time]
cesmname   =  ["CESM-FULL","CESM-SLAB"]


csmooth    = 100        # Stochastic Model Smoothing

cspecs = [] # [region][model]
cfreqs = []
for reg in range(nreg-1):
    inssts = [ sstcesm[reg][0], sstcesm[reg][1], ]
    specs,cfreqs0,_,_,_ = scm.quick_spectrum(inssts,ssmooth,pct)
    cspecs.append(specs)
    cfreqs.append(cfreqs0)
    
#%% Compute the Spectral Ratios using interpolation

freqslab = cfreqs[0][1]
freqfull = cfreqs[0][0]
freqsm   = freqs[0]

#%% Make the Plot

dtplot     = 3600*24*365
rid_sel    =[0,5,6,]
specnames_ratio = ("log( With $Q_{ek}$ / Without $Q_{ek}$)","log( With $Q_{ek}$ / CESM-FULL)")
notitle    = False
useC       = True



# Take Mean values for plotting
plotvar     = np.var(amvids_all[1,:,7,:])
amvplot    = np.nanmean(amvpats_all[1,:,7,:,:],0)
numerspec  = smspecs[1,:,:,:].mean(0) # [region x freq]
denomspec  = smspecs[0,:,:,:].mean(0)
cid        = 0 # 0 = FULL, 1 = SLAB


# Initialize figure
fig = plt.figure(constrained_layout=True, facecolor='w',figsize=(12,8))

gs = fig.add_gridspec(nrows=6, ncols=6, left=.02, right=1,
                      hspace=.075, wspace=0.25)

ax1 = plt.subplot(gs[:, :3],projection=ccrs.PlateCarree())
ax2 = plt.subplot(gs[1:3, 3:])
ax3 = plt.subplot(gs[3:5, 3:])

# ------------
# Plot the AMV
# ------------
ax = ax1
f   = 0
rid = 4
mid = 0
spid = 0


#if notitle is False:
    #ax.set_title("%s ($\sigma^2_{AMV}$ = %.4f$K^2$)"%(expnames[mid],plotvar))
#else:
if useC:
    ptitle = "Entraining with $Q_{ek}$ ($\sigma^2_{AMV}$ = %.4f$\degree C^2$)"%(plotvar)
else:
    ptitle = "Entraining with $Q_{ek}$ ($\sigma^2_{AMV}$ = %.4f$K^2$)"%(plotvar)
    
ax.set_title(ptitle)

# Make the Plot
ax = viz.add_coast_grid(ax,bboxplot,line_color='k',
                        fill_color='gray')
pcm = ax.contourf(lonr,latr,amvplot.T,levels=cint,cmap='cmo.balance')
cl = ax.contour(lonr,latr,amvplot.T,levels=cint,colors="k",linewidths=0.5)
ax.clabel(cl,levels=cint[::2],fontsize=10,fmt="%.02f")
viz.plot_mask(lonr,latr,dmsks[1],ax=ax,markersize=0.5)


ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
spid += 1


#
# For each axis, do a ratio
#

def calc_specratio(targspec,refspec,targfreq,reffreq,return_interp=False):
    """
    Calculate ratio of 2 spectra targspec/refspec.
    Interpolates to the coarser of targspec or refspec

    Parameters
    ----------
    targspec : ARRAY
        Spectra in numerator
    refspec : ARRAY
        Spectra in demonimator
    targfreq : ARRAY
        Frequencies for spectra in numerator
    reffreq : ARRAY
        Frequencies for spectra in denominator
    return_interp : BOOL
        Set to True to return interpolated spectra

    Returns
    -------
    specratio
        Ratio of targspec/refspec

    """
    # Interpolate to the minimum 
    inspecs  = [targspec,refspec]
    infreqs  = [targfreq,reffreq]
    
    # Interpolate to coarser frequency (smaller)
    if len(targspec) != len(refspec):
        # Find index of min/max
        kmin = np.argmin([len(targspec),len(refspec)])
        kmax = np.argmax([len(targspec),len(refspec)])
        # Perform Interpolation
        spec_interp   = np.interp(infreqs[kmin],infreqs[kmax],inspecs[kmax])
        inspecs[kmax] = spec_interp # Replace with interpolated values
        
    specratio = inspecs[0]/inspecs[1]
    if return_interp:
        return specratio,spec_interp,
    return specratio
    
    

for a,ax in enumerate([ax2,ax3]):
    
    for r,rid in enumerate(rid_sel):
        
        targspec = numerspec[rid,:]
        targfreq = freqsm
        if a == 0:
            refspec = denomspec[rid,:]
            reffreq = freqsm
        else:
            refspec = cspecs[rid][cid]
            reffreq = freqfull
        
        specratio = calc_specratio(targspec,refspec,targfreq,reffreq)

        
        
        
        ax.semilogx(targfreq*dtplot,np.log(specratio),
                   lw=4,color=bbcol[rid],label=regions[rid])
    
    
    
    ax.set_xticks(xtks)
    if a == 1:
        ax.set_xticklabels(xper)
        ax.set_xlabel("Period (Years)")
    elif a ==0:
        ax.legend(ncol=3,loc='lower center')
        ax.set_xticklabels([])
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.axhline(0,ls='dashed',color="k")
    ax.grid(True,ls='dotted')
    
    ax.set_ylim([-1.5,1.5])
    
    
    #ax.set_ylabel("log(%s/%s)" % (specnames[targid],specnames[refid]))
    ax.set_ylabel(specnames_ratio[a])
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
    spid += 1



plt.savefig("%sEkmanForcing_Summary_Figure.png"%figpath,dpi=200,bbox_inches='tight')

#%%


# #%% Load in data (Full SSTs, might be too big...)

# # Ekman Forcing
# for q in range(2):
#     for f,fname in enumerate(fnames):
        
#         if q == 0:
#             ld = np.load("%sstoch_output_%s.npz" % (datpath,fname),allow_pickle=True)
#             sst = ld['sst'][-1,...] # Take entraining model [model (3) x lon x lat x time]
#         else:
#             ld = np.load("%sstoch_output_%s_Qek.npz" % (datpath,fname),allow_pickle=True)
#             sst = ld['sst'][0,...] # Take entraining model [model (1) x lon x lat x time]
        
#         if (q == 0) and (f == 0): # Read out lat/lon as well
#             lonr = ld['lon']
#             latr = ld['lat']
#             nlon,nlat,ntime = sst.shape
        
#             sstall = np.zeros((2,len(fnames),nlon,nlat,ntime)) * np.nan
#         sstall[q,f,...] = sst.copy()


#%%
# Restrict to Region

# Compute Spectra

# Interpolate to each other

# Compute Spectral Ratios

# Load + Plot the AMV Pattern





# Quick Comparison of SST from 2 experiments
# Debug (Load 1 data and check quickly)
fname1   = 'forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run204_ampq0_method5_dmp0'
ld1      = np.load("%sstoch_output_%s.npz" % (datpath,fname1),allow_pickle=True)
ld2      = np.load("%sstoch_output_%s_Qek.npz" % (datpath,fname1),allow_pickle=True)

# Get the Ekman related pumping
sst_noek   = ld1['sst'][-1,...]
sst_ek     = ld2['sst'][0,...]

# Compute the variance Difference
diff = np.var(sst_ek,-1) - np.var(sst_noek,-1)
