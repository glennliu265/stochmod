#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug Forcing Leading to Variance Differences


Involved Scripts: 
    - viz_synth_stochmod_combine
    - constant_v_variable
    

Created on Mon Jan 15 14:23:11 2024

@author: gliu
"""
# ---------------------

#%% Constant_v_variable

# ---------------------
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs
import xarray as xr

from scipy import signal
#from scipy.ndimage.filters import uniform_filter1d

# Set Paths
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20240115/'
proc.makedir(outpath)

darkmode = False

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','orange','magenta','red')
hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab
if darkmode:
    dfcol      = 'w'
    dfalph     = 0.30
    dfalph_col = 0.40
else:
    dfcol      = 'k'
    dfalph     = 0.1
    dfalph_col = 0.25 

# UPDATED Colors and names for generals (5/25/2021)
ecol = ["blue",'cyan','gold','red']
els  = ["dotted","dashdot","dashed","solid"]
ename = ["All Constant",
         r"Vary $\alpha$",
         r"Vary $\lambda_a$",
         "All Varying"]


config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0
config['fstd']        = 1
config['t_end']       = 120000    # Number of months in simulation
config['runid']       = "syn009"  # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy" #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1
config['query']       = [-30,50]
config['applyfac']    = 2 # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = outpath
config['smooth_forcing'] = False
config['method'] = 3
config['favg'] = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)


#%% Clean Run

ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt = params

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

# Calculate constant forcing value (Using CESM-SLAB values)
cp0=3996
rho=1026
dt = 3600*24*30
frcstd_slab = np.std(frc[1]) * (rho*cp0*hblt) / dt  # (Constant Forcing)
#frcstd_slab_raw = Fpt#frc[1].copy()

#%%

foriginal = np.copy(config['fname'])

if len(Fptdef.shape)>2:
    #original = config['fname']
    config['fname'] = 'FLXSTD' # Dummy Move for now to prevent forcing_flag
    
expids   = [] # String Indicating the Variable Type
acs      = []
ssts     = []
damps    = []
mlds     = []
forces   = []
explongs = []

for vmld in [False,True]:
    
    if vmld:
        config['mldpt'] = mlddef
    else:
        config['mldpt'] = np.ones(12)*mlddef.mean()
    
    for vdamp in tqdm([False,True]):
        
        if vdamp:
            config['damppt'] = dampdef
        else:
            config['damppt'] = np.ones(12)*dampdef.mean()
        
        for vforce in [False,True]:
                
                if vforce:
                    
                    print(foriginal)
                    config['fname'] = np.copy(foriginal).item()
                    config.pop('Fpt',None)
                    config['favg'] = False
                    
                        #config['Fpt'] = Fptdef.mean(0)
                else: # Constant Forcing
                    
                    config['fname'] = 'FLXSTD'
                    config['favg'] = True
                    # Take stdev over EOFs, then take the mean
                    
                    Fptin = Fptdef.copy()
                    Fptin[Fptin==0] = np.nan                    
                    
                    config['Fpt'] = frcstd_slab * np.ones(12)
                    # This yields a value that is greater than Fptin
                    #config['Fpt'] = np.ones(12)*np.nanmean(np.nansum(Fptin,0))
                    #np.std(Fptdef,0).mean()
                
                # Set experiment name
                expid = "vdamp%i_vforce%i_vmld%i" % (vdamp,vforce,vmld)
                explong = "Vary Damping (%s) Forcing (%s) MLD (%s)" % (vdamp,vforce,vmld)
                
                
                # Run Model
                ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
                [o,a],damppt,mldpt,kprev,Fpt = params
                
                # Save variables
                acs.append(ac)
                ssts.append(sst)
                expids.append(expid)
                damps.append(damppt)
                mlds.append(mldpt)
                forces.append(Fpt)
                explongs.append(explong)
                print("Completed %s"%expid)
                #print(mldpt)
                
                # Clean Config, End Forcing Loop
                config.pop('Fpt',None)
                #print(Fpt)
                
        # Clean Config, End Damping Loop
        config.pop('damppt',None)
        #print(damppt)
    # Clean Config, End MLD Loop
    config.pop('mldpt',None)

#%% Lets Check the Output


# Make Some Plots
proc.get_monstr(nletters=3,)
fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(mons3,Fptdef,label=r"$\sigma (F')$ Forcing") # Default Forcing
ax.grid(True,ls='dashed')
ax.legend()
ax.set_ylabel(r"$W/m^2$")
savename = "%sConst_v_Vary_Forcing_Fstd.png" % outpath
plt.savefig(savename,dpi=150)


fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
ax.plot(mons3,dampdef,label=r"$\lambda ^a$") # Default Damping
ax.grid(True,ls='dashed')
ax.legend()
ax.set_ylabel(r"$\degree C$ per $W/m^2$")
savename = "%sConst_v_Vary_Damping.png" % outpath
plt.savefig(savename,dpi=150)

#%%Save Output for Analysis/Comparison/Debugging
outdir   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%sParameters_constant_v_vary.npz" % outdir
np.savez(savename,**{
    'forcing'  : Fptdef,
    'damping'  : dampdef,
    'mld'      : mlddef,
    'hblt'     : hblt,
    'mconfig'  : config['mconfig'],
    'fname'    : config['fname'],
    'ssts'     : ssts,
    'expnames' : explongs,
    'acs'      : acs,
    })

#%% Check Monthly variance



selids  = [0,1,2,3]
selcol  = ["blue","cyan","yellow","red"]

ssts_unpack = [ssts[ii][1] for ii in selids]
expnamein   = np.array(explongs)[selids]


monvars     = [proc.calc_monvar(sst) for sst in ssts_unpack]




fig,ax = plt.subplots(1,1,figsize=(8,6))
for i in range(len(monvars)):
    ax.plot(mons3,monvars[i],label=expnamein[i],lw=2.5,color=selcol[i])
    ax.legend(fontsize=8)
ax.grid(True,ls='dotted')

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#%% Load the Different Forcings

input_path = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/'

# Insert File Names
fnames = [
    
    "SLAB_PIC_NAO_EAP_NHFLX_Forcing_DJFM-MON.npy",
    "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy",
    "flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0.npy",
    
    ]

fnames_long = ["NAO-EAP DJFM (pt)","FLXSTD EOF (pt,SLAB)","FLXSTD EOF (pt,FULL)"]

# Load and print dimensions
fload   = [np.load(input_path+fn) for fn in fnames]
[print(fl.shape)for fl in fload]

# Square and sum the EOFs
floadsq = [np.sqrt(np.nansum((fl**2),2)) for fl in fload] 

# Load Lat/Lon for plotting
lon,lat = scm.load_latlon()
lonf,latf=-30,50
klon,klat = proc.find_latlon(lonf,latf,lon,lat)

# Compare EOF 1 Month 1 of both forcings
imode = 1
sq    = True
imon  = 0
vmax  = 80
fig,axs = viz.geosubplots(1,2,figsize=(12,8))
for a in range(2):
    ax = axs[a]
    ax = viz.add_coast_grid(ax,bbox=[-80,0,0,65],fill_color="k")
    if sq:
        plotvar = floadsq[a][:,:,imon].T
        eofstr  = "SUM"
    else:
        fload[a][:,:,imode,imon].T
        eofstr  = "%s" % (imode+1)
    
    pcm=ax.pcolormesh(lon,lat,plotvar,vmin=-vmax,vmax=vmax,cmap="RdBu_r")
    ax.set_title(fnames_long[a],fontsize=18)
plt.suptitle("EOF %s, Month %02i" % (eofstr,imon+1),fontsize=18)
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)

#%% Add in Fprime Std (takes awhile to load)

# Load Fprime 
fname    = "Fprime_PIC_SLAB_rolln0.nc"
dsf      = xr.open_dataset(input_path+"../"+fname).Fprime.load()
dsf      = proc.format_ds(dsf)

# Flip Longitude
dsf = proc.format_ds(dsf)

# Compute Monthly variance
dsmonvar = dsf.groupby('time.month').var('time')
fprimestd = dsmonvar.values.transpose(2,1,0)

# Append
#dsf     = proc.format(

#Fprime  = dsf.sel(lon=lonf+360,lat=latf,method='nearest').Fprime.values # Time
#ntime   = len(Fprime)
#Fpt_new     = Fprime.reshape(int(ntime/12),12).std(0) 


#%% Examine what is going on at the point

# Restrict to Point
fpt   = [fl[klon,klat,:,:] for fl in fload]
fptsq = [np.sqrt(np.nansum(f**2,0)) for f in fpt] 

fprimept = np.sqrt(fprimestd[klon,klat,:])

# Load plotting variables
mons3 = proc.get_monstr(nletters=3)
fig,ax = viz.init_monplot(1,1)

for ff in range(len(fptsq)):
    plotvar = fptsq[ff]
    ax.plot(mons3,plotvar,label=fnames_long[ff],marker="d")

ax.plot(mons3,fprimept,color='gray',ls='dashed',marker="x",label="std(Fprime) (SLAB)")
ax.legend()


#plt.pcolormesh(fload[1][:,:,0,0].T),plt.colorbar() # Debug Plot

#%% Ok, now try loading the damping
"""

"New" descriptes the settings in sm_rewrite_loop where:
    mode    = 5
    ensostr = ""
    lag     = lag1
    
"""
# Assuming new is default lagstr1, ensolag is removed
# Method 5 (not sure what this is again?)

dampfn = [
    
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy",
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode5_lag1.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode4.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode5_lag1.npy",
    
    ]

dampnames = [
    
    "SLAB PIC (old)",
    "SLAB PIC (new)",
    "FULL PIC (old)",
    "FULL PIC (new)",
    
    ]

# Load and print shape
dampload = [np.load(input_path+fn) for fn in dampfn]
[print(fl.shape) for fl in dampload]

# Restrict to Point
dpt   = [fl[klon,klat,:] for fl in dampload]


#%% Plot Damping to Compare


fig,ax = viz.init_monplot(1,1)

for ff in range(len(dpt)):
    plotvar = dpt[ff]
    ax.plot(mons3,plotvar,label=dampnames[ff],marker="d")

#ax.plot(mons3,np.sqrt(fprimestd[klon,klat,:]),color='gray',ls='dashed',marker="x",label="std(Fprime) (SLAB)")
ax.legend()

#%% Run a silly simulation with this set

# Set up forcing and other parameters
dt       = 30*3600*24
nyrs     = 10000
eta      = np.random.normal(0,1,nyrs*12)

hblt     = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

#%% Set Forcings and dampings
forcings = [fptsq[1],fptsq[2],fprimept]
dampings = [dpt[0],dpt[1],dpt[1]]
expnames = ["Old","New","Fstd"]
nexps    = len(forcings)

#%% Load output from synth_stochmod before for comparison

outdir   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%ssynth_stochmod_combine_output.npz" % outdir

ld  = np.load(savename,allow_pickle=True)
mvs = ld['monvars']
lbs = ld['labels'] 



# Experiment with different rolls
forcingroll = 0
dampingroll = 0

rollstr = "damproll%i_forceroll%i" % (dampingroll,forcingroll)

#%%

debug = True

outputs = []
for ex in range(nexps):
    
    f_in = np.roll(forcings[ex].copy(),forcingroll)
    d_in = np.roll(dampings[ex].copy(),dampingroll)
    
    smconfig = {}
    smconfig['forcing'] = f_in.copy()[None,None,:] # [x x 12]
    smconfig['damping'] = d_in.copy()[None,None,:]
    smconfig['h']       = np.ones((1,1,12)) * hblt
    
    if debug:
        fig,ax = viz.init_monplot(1,1,)
        ax.plot(mons3,smconfig['forcing'].squeeze(),label='forcing')
        ax2 = ax.twinx()
        ax2.plot(mons3,smconfig['damping'].squeeze(),label='damping',color='red')
        ax.legend()
        ax.set_title("Experiment %i" % (ex+1))
    
    
    
    # Convert units
    smconfig['damping']=scm.convert_Wm2(smconfig['damping'],smconfig['h'],dt)[None,None,:]
    smconfig['forcing']=scm.convert_Wm2(smconfig['forcing'],smconfig['h'],dt)
    
    
    # Make Forcing
    smconfig['Fprime']= np.tile(smconfig['forcing'],nyrs) * eta[None,None,:]
    
    
    
    output = scm.integrate_noentrain(smconfig['damping'],smconfig['Fprime'],debug=True)
    outputs.append(output)
    
#% Calculate some diagnostics
ssts    = [o[0].squeeze() for o in outputs]
monvars = [proc.calc_monvar(s) for s in ssts]



#%% Plot forcing and damping

ytks  = np.arange(10,80,10)
ytks2 = np.arange(0,36,5)

fig,axs = viz.init_monplot(3,1,figsize=(6,5.5))

for ff in range(nexps):
    
    ax = axs[ff]
    
    f_in = np.roll(forcings[ff].copy(),forcingroll)
    d_in = np.roll(dampings[ff].copy(),dampingroll)
    
    ax.plot(mons3,f_in,color="cornflowerblue",lw=2.5,label="forcing",marker="o")
    ax.tick_params(axis='y', colors='cornflowerblue')
    
    ax2 = ax.twinx()
    ax2.plot(mons3,d_in,color="red",lw=2.5,label="damping",marker='d',ls='dashed')
    ax2.tick_params(axis='y', colors='red')
    
    
    ax.set_ylim([10,70])
    ax.set_yticks(ytks)
    
    ax2.set_ylim([0,35])
    ax2.set_yticks(ytks2)
    
    viz.label_sp(expnames[ff],x=0.45,ax=ax,labelstyle="%s",usenumber=True)
plt.suptitle("Damping Shift (%i) | Forcing Shift (%i)" % (dampingroll, forcingroll))
savename = "%sDebug_Forcing_v_Damping_%s.png" % (outpath,rollstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')




#%% PLot monvar

fig,ax = viz.init_monplot(1,1)

for ff in range(nexps):
    plotvar = monvars[ff]
    ax.plot(mons3,plotvar,label=expnames[ff],marker="o")


ax.plot(mons3,mvs[-1],label="SLAB",color="gray",ls="dashed")
ax.legend()
ax.set_ylim([0.5,1.5])

plt.suptitle("Damping Shift (%i) | Forcing Shift (%i)" % (dampingroll, forcingroll))
savename = "%sDebug_Monthly_Variance_%s.png" % (outpath,rollstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%


#

#%% Inform

# Include these sections

#%%

#%%

#%%

