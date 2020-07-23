#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:04:03 2020

@author: gliu
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend


from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% Functions
def find_latlon(lonf,latf,lon,lat):
    """
    Find lat and lon indices
    """
    if((np.any(np.where(lon>180)) & (lonf < 0)) or (np.any(np.where(lon<0)) & (lonf > 180))):
        print("Potential mis-match detected between lonf and longitude coordinates")
    
    klon = np.abs(lon - lonf).argmin()
    klat = np.abs(lat - latf).argmin()
    
    msg1 = "Closest lon to %.2f was %.2f" % (lonf,lon[klon])
    msg2 = "Closest lat to %.2f was %.2f" % (latf,lat[klat])
    print(msg1)
    print(msg2)
    
    return klon,klat

def detrendlin_nd(var_in):
    
    
    # Reshape to combine all other dimensions
    alldims = var_in.shape[1:]
    combinedims = 1
    for ele in alldims:
        combinedims *= ele
    var_rs     = np.reshape(var_in,(var_in.shape[0],combinedims))
    var_dt = np.zeros(var_rs.shape)
    
    
    # Loop over each dimension
    for i in range(0,combinedims):
        
        # Select timeseries for that point
        vloop = np.copy(var_rs[:,i])
        
        # Skip if all values are nan
        if np.all(np.isnan(vloop)):
            
            # Assign values to nan
            var_dt[:,i] = np.ones(vloop.shape)*np.nan
            
        else:
            
            # Detrend using 1d function
            var_dt[:,i] = detrendlin(vloop)
            
    
    var_dt = np.reshape(var_dt,var_in.shape)
    
    return var_dt

def detrendlin(var_in):
    debug = 0
    if debug == 1:
        var_in = sssr[:,32]
    
    x = np.arange(0,len(var_in))
    
    # Limit to non-nan values
    inotnan = ~np.isnan(var_in)
    
    # Perform Regression
    m,b,r_val,p_val,std_err=stats.linregress(x[inotnan],var_in[inotnan])
    
    # Detrend
    var_detrend = var_in - (m * x +b)
    
    return var_detrend


def area_avg(data,bbox,lon,lat,wgt):
    """
    Function to find the area average of data [lon x lat x otherdim]
    
    Given a bounding box [lonW, lonE, latS, latN]
    
    and corresponding lon/lat variables

    """
        
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
        
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if wgt == 1:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        wgtm = np.cos(np.radians(lat))
        wgta = np.cos(np.radians(yy)).T
        
        # Remove nanpts from weight
        nansearch = np.sum(sel_data,2) # Sum along time
        wgta[np.isnan(nansearch)] = 0
        
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt ==1:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa


def calc_AMV(lon,lat,sst,bbox,order,cutofftime,awgt):
    """
    Calculate AMV Index for detrended/anomalized SST data [LON x LAT x Time]
    given bounding box [bbox]. Applies cosine area weighing
    

    Parameters
    ----------
    lon : ARRAY [LON]
        Longitude values
    lat : ARRAY [LAT]
        Latitude Values
    sst : ARRAY [LON x LAT x TIME]
        Sea Surface Temperature
    bbox : ARRAY [LonW,LonE,LonS,LonN]
        Bounding Box for Area Average
    order : INT
        Butterworth Filter Order
    cutofftime : INT
        Filter Cutoff, expressed in same timesteps as input data
        
    Returns
    -------
    amv: ARRAY [TIME]
        AMV Index (Not Standardized)
    
    aa_sst: ARRAY [TIME]
        Area Averaged SST

    """
    
    """
    
    # Dependencies
    functions: area_avg, detrendlin
    
    numpy as np
    from scipy.signal import butter,filtfilt
    """
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,awgt)


    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Compute AMV Index
    amv = filtfilt(b,a,aa_sst)
    
    return amv,aa_sst
    
def regress2ts(var,ts,normalizeall,method):
    
    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    londim = var.shape[0]
    latdim = var.shape[1]
    
    
    



    
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 

        var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        
        # Find Nan Points
        # sumvar = np.sum(var,1)
        
        # # Find indices of nan pts and non-nan (ok) pts
        # nanpts = np.isnan(sumvar)
        # okpts  = np.invert(nanpts)
    
        # # Drop nan pts and reshape again to separate space and time dimensions
        # var_ok = var[okpts,:]
        #var[np.isnan(var)] = 0
        
        
        # Perform regression
        #var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        var_reg,_ = regress_2d(ts,var)
        
        
        # Reshape to match lon x lat dim
        var_reg = np.reshape(var_reg,(londim,latdim))
    
    
    
    
    # 2nd method is looping point by poin  t
    elif method == 2:
        
        
        # Preallocate       
        var_reg = np.zeros((londim,latdim))
        
        # Loop lat and long
        for o in range(londim):
            for a in range(latdim):
                
                # Get time series for that period
                vartime = np.squeeze(var[o,a,:])
                
                # Skip nan points
                if any(np.isnan(vartime)):
                    var_reg[o,a]=np.nan
                    continue
                
                # Perform regression 
                r = np.polyfit(ts,vartime,1)
                #r=stats.linregress(vartime,ts)
                var_reg[o,a] = r[0]
                #var_reg[o,a]=stats.pearsonr(vartime,ts)[0]
    
    return var_reg


def regress_2d(A,B):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b


def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to search along. 0 or 1.
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        

    """
    
    # Sum along select dimension
    datasum = np.sum(data,axis=dim)
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if dim == 0:
        okdata = data[:,okpts]
    elif dim == 1:    
        okdata = data[okpts,:]
    
    return okdata,knan,okpts
    
    
def year2mon(ts):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    ts = ts.T
    return ts
    
def ann_avg(ts,dim):
    """
    # Take Annual Average of a monthly time series
    where time is axis "dim"
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    annavg = np.reshape(ts,newshape)
    annavg = np.nanmean(annavg,axis=dim+1)
    return annavg


def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None):

    
    if ax is None:
        ax = plt.gca()
        
        
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    

    # Set up projections and extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    if len(cint) == 1:
        # Draw contours
        cs = ax.contourf(lon1,lat,var,cmap=cmap)
    
    else:
        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap)
    
    
    
        # Negative contours
        cln = ax.contour(lon1,lat,var,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
        # Positive Contours
        clp = ax.contour(lon1,lat,var,
                    cint[cint>=0],
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())    
                      
        #ax.clabel(cln,colors=None)
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if len(clab) == 1:
        bc = plt.colorbar(cs)
    else:
        bc = plt.colorbar(cs,ticks=clab)
    
    
    return ax

    
def plot_AMV(amv,ax=None):
    
    """
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax

def array_nan_equal(a, b):
    """
    Check if two arrays are equal ignoring nans
    source: https://stackoverflow.com/questions/49245963/how-to-compare-numpy-arrays-ignoring-nans
    """
    m = np.isfinite(a) & np.isfinite(b)
    return np.array_equal(a[m], b[m])


def detrend_dim(invar,dim):
    
    """
    Detrends n-dimensional variable [invar] at each point along axis [dim].
    Performs appropriate reshaping and NaN removal, and returns
    variable in the same shape+order. Assumes equal spacing along [dim] for 
    detrending
    
    Also outputs linear model and coefficients.
    
    Dependencies: 
        numpy as np
        find_nan (function)
        regress_2d (function)
    
    Inputs:
        1) invar: variable to detrend
        2) dim: dimension of axis to detrend along
        
    Outputs:
        1) dtvar: detrended variable
        2) linmod: computed trend at each point
        3) beta: regression coefficient (slope) at each point
        4) interept: y intercept at each point
    
    
    """
    
    # Reshape variable
    varshape = invar.shape
    
    # Reshape to move time to first dim
    newshape = np.hstack([dim,np.arange(0,dim,1),np.arange(dim+1,len(varshape),1)])
    newvar = np.transpose(invar,newshape)
    
    # Combine all other dims and reshape to [time x otherdims]
    tdim = newvar.shape[0]
    otherdims = newvar.shape[1::]
    proddims = np.prod(otherdims)
    newvar = np.reshape(newvar,(tdim,proddims))
    
    # Find non nan points
    varok,knan,okpts = find_nan(newvar,0)
    
    # Ordinary Least Squares Regression
    tper = np.arange(0,tdim)
    m,b = regress_2d(tper,varok)
    
    # Detrend
    ymod = (m[:,None]*tper + b[:,None]).T
    dtvarok = varok - ymod
    
    # Replace into variable of original size
    dtvar  = np.zeros(newvar.shape) * np.nan
    linmod = np.copy(dtvar)
    beta   = np.zeros(okpts.shape) * np.nan
    intercept = np.copy(beta)
    
    dtvar[:,okpts] = dtvarok
    linmod[:,okpts] = ymod
    beta[okpts] = m
    intercept[okpts] = b
    
    # Reshape to original size
    dtvar  = np.reshape(dtvar,((tdim,)+otherdims))
    linmod = np.reshape(linmod,((tdim,)+otherdims))
    beta = np.reshape(beta,(otherdims))
    intercept = np.reshape(beta,(otherdims))
    
    # Tranpose to original order
    oldshape = [dtvar.shape.index(x) for x in varshape]
    dtvar = np.transpose(dtvar,oldshape)
    linmod = np.transpose(linmod,oldshape)
    
    return dtvar,linmod,beta,intercept
    
def init_map(bbox,ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    return ax
#%% Load in HadISST Data
# Path to SST data from obsv
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20200723/'
datpath = projpath + '01_Data/'
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")
hlat = np.squeeze(obvhad['LAT'])
hlon = np.squeeze(obvhad['LON'])
hyr  = obvhad['YR']
hsst = obvhad['SST']



# Change hsst to lon x lat x time
hsst = np.transpose(hsst,(2,1,0))

# Take the set time period
startyr = 1920
monstart = (1920+1-hyr[0,0])*12
hsst = hsst[:,:,monstart::]


#%% Fix Latitude Dimensions for HSST
# currently it is arranged 90:-90, need to flip to south first

# Find north and south latitude points
hsouth = np.where(hlat <= 0)
hnorth = np.where(hlat > 0)

# Find corresponding points in data
hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]

# Stitch things together, reversing the order 
hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)





#%% Perform Linear Detrend on SST and annually averaged sst (Detrend First)

usedtfunction = 1  # Set to 1 to use the new detrending function

if usedtfunction == 1:
    
    start= time.time()
    dt_hsst,ymodall,_,_ = detrend_dim(hsstnew,2)
    print("Detrended in %.2fs" % (time.time()-start))
else:
    
    # Reshape to [Time x Space] and remove NaN Points
    start= time.time()
    hsstnew = np.reshape(hsstnew,(360*180,1176)).T
    hsstok,knan,okpts = find_nan(hsstnew,0)
    
    
    
    tper = np.arange(0,hsstok.shape[0])
    beta,b = regress_2d(tper,hsstok) # Perform regression
    
    # Detrend
    dt_hsst = hsstnew[:,okpts] - (beta[:,None] * tper + b[:,None]).T
    
    # Replace NaN vaues back into the system
    hsstall = np.zeros(hsstnew.shape) * np.nan
    hsstall[:,okpts] = dt_hsst
    
    # Also save the linear model
    ymodall = np.zeros(hsstnew.shape) * np.nan
    ymodall[:,okpts] = (beta[:,None] * tper + b[:,None]).T
    
    # Reshape again
    dt_hsst = np.reshape(hsstall.T,(360,180,1176))
    hsstnew = np.reshape(hsstnew.T,(360,180,1176))
    ymodall = np.reshape(ymodall.T,(360,180,1176))
    print("Detrended in %.2fs" % (time.time()-start))




#%% Visualize detrending for point [lonf,latf] (Visualize Detrending effect)

# Find Point
lonf = -30
latf = 64
klon,klat = find_latlon(lonf,latf,hlon,hlatnew)

tempts = hsstnew[klon,klat,:]
dtts = dt_hsst[klon,klat,:]
ymodts = ymodall[klon,klat,:]

olddt = detrendlin(tempts)
from scipy import signal
scidt = signal.detrend(tempts)

#% Plot Detrended and undetrended lines
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,tempts,color='k',label="raw")
ax.plot(tper,dtts,color='b',label="detrended")
#ax.plot(tper,olddt,color='r',label="olddetrending")
ax.plot(tper,scidt,color='g',label="scipy")
plt.legend()

#% Plot Scatter and fitted model...
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.scatter(tper,tempts,color='k',label="raw")
ax.plot(tper,ymodts,color='b',label="linear model")
plt.legend()

#%% Remove Seasonal Cycle (After Detrending)

# Deseasonalize [lon x lat x yr x mon]
ahsst = np.reshape(dt_hsst,(360,180,int(dt_hsst.shape[2]/12),12))
ahsst = ahsst - np.mean(ahsst,axis=2)[:,:,None,:]
ahsst = np.reshape(ahsst,(360,180,dt_hsst.shape[2]))


#%% Visualize Deseasonalization...

ats = ahsst[klon,klat,:]

fig,ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(tper,tempts,color='k',label="Raw")
ax.plot(tper,dtts,color='b',label="Detrended")
plt.plot(tper,ats,color='r',label="Detrended,Deseasonalized")
plt.legend()


#%% Save Data (Detrended First, then deseasonalized)

timecft = xr.cftime_range(start="1920-01-01",end="2017-12-01",freq="MS") 

da = xr.DataArray(ahsst,
                  dims=["lon","lat","time"],
                  coords={"lat":hlatnew,"lon":hlon,"time":timecft}
                 )
da.name = 'SST'
da.to_netcdf("%sHadISST_Detrended_Deanomalized_1920_2018.nc" % (datpath2))

#aa = xr.open_dataset("%sHadISST_Detrended_Deanomalized_1920_2018.nc" % (datpath2))


#%% Try removing seasonal cycle first, then detrending

# Remove Seasonal Cycle first and plot
dsfirst = np.reshape(hsstnew,(360,180,int(dt_hsst.shape[2]/12),12))
dsfirst = dsfirst - np.mean(dsfirst,axis=2)[:,:,None,:]
dsfirst = np.reshape(dsfirst,(360,180,hsstnew.shape[2]))

# Detrend
start= time.time()
dtdsfirst,dsymodall,_,_ = detrend_dim(dsfirst,2)
print("Detrended in %.2fs" % (time.time()-start))

# Plot Seasonal Cycle Removal and Detrended
lonf = -30
latf = 64
tper = np.arange(0,hsstnew.shape[2])
klon,klat = find_latlon(lonf,latf,hlon,hlatnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdsfirst[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

#%% Sample AMV Calculations.... <Detrended First, then Deseasonalized>

bbox  = [-80,0,0,60]
order = 5
cutofftime = 120 
cmap = cmocean.cm.balance
cint = np.arange(-0.8,0.9,0.1)
clab = cint
# Detrended, Deseasonalized
dtamv,dtaa = calc_AMV(hlon,hlatnew,ahsst,bbox,order,cutofftime,1)

dtr = regress2ts(ahsst,dtamv/np.std(dtamv),0,1)

plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap,cint=cint,clab=clab)
plt.title("Detrended, Deanomalized, Monthly SST")

#%%  AMV, <Detrended Only>
dtamv,dtaa = calc_AMV(hlon,hlatnew,dt_hsst,bbox,order,cutofftime,1)

dtr = regress2ts(dt_hsst,dtamv,0,1)

plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap)
plt.title("Detrended Monthly SST")

#%% AMV <Raw>

dtamv,dtaa = calc_AMV(hlon,hlatnew,hsstnew,bbox,order,cutofftime,1)

dtr = regress2ts(hsstnew,dtamv,0,1)

plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap)
plt.title("Raw Monthly SST")

#%% Trying annually averaged data

aaraw = ann_avg(hsstnew,2)
#dtraw = ann_avg(dt_hsst,2)
#adtraw = ann_avg(ahsst,2)

# Detrend annual data
invar = ahsst
cutofftime = 120
dtamv,dtaa = calc_AMV(hlon,hlatnew,invar,bbox,order,cutofftime,1)
dtr = regress2ts(invar,dtamv,0,1)
xtk = np.arange(1,101,10)
xlbs = np.arange(1920,2020,10)

#plt.subplots(211)
plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap,cint=cint,clab=clab)
#plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap)
#plt.title("Detrended, Deanomalized, Monthly SST")

plot_AMV(dtamv)
plt.xticks(xtk,xlbs)

#%% Perform detrend on annually averaged sst and try calculation from there



selvar = dtdsfirst
annsst = ann_avg(selvar,2)

start= time.time()
dtann,ymodann,_,_ = detrend_dim(annsst,2)
print("Detrended in %.2fs" % (time.time()-start))


invar = dtann
cutofftime = 10
dtamv,dtaa = calc_AMV(hlon,hlatnew,invar,bbox,order,cutofftime,1)
dtr = regress2ts(invar,dtamv/np.std(dtamv),0,1)
xtk = np.arange(1,101,10)
xlbs = np.arange(1920,2020,10)


fig,ax = plt.subplots(1,1,figsize=(6,4))
cint = np.arange(-0.8,0.9,0.1)
#plt.subplots(211)
plot_AMV_spatial(dtr.T,hlon,hlatnew,[-80,0,0,80],cmap,cint=cint,clab=cint)
#plot_AMV_spatial(dtr.T,hlon,hlatnew,bbox,cmap)
plt.title("AMV Spatial Pattern ($^{\circ}C/\sigma_{AMV Indx})$ \n HadISST (1920-2018)")
plt.savefig(outpath+"HADISST_AMV_Annual_LinDetrend_Deseas_First.png",dpi=200)

fig,ax = plt.subplots(1,1,figsize=(6,2))
plot_AMV(dtamv)
plt.xticks(xtk,xlbs)
plt.title("AMV Index HadISST (1920-2018)")
plt.savefig(outpath+"HADISST_AMVIndex_Annual_LinDetrend.png",dpi=200)

#%% Explore Differences between if you Detrend first or Deseasonalize first...

diff = ahsst - dtdsfirst
diffmax = np.nanmax(diff,axis=2)

fig.ax = plt.subplots(1,1,figsize=(12,8))
cint = np.arange(0,0.11,0.01)
#pcm = plt.contourf(hlon,hlatnew,np.squeeze(diffmax).T,cmap=cmocean.cm.dense,vmax=0.10,vmin=0)
pcm = plt.contourf(hlon,hlatnew,np.squeeze(diffmax).T,cint,cmap=cmocean.cm.dense)
plt.colorbar(pcm)
plt.title("Detrend first - Deseasonalized First (Max Differencem HadISST)")
plt.savefig(outpath+"HADISST_Differences_dtfirst.png",dpi=200)

#%% Explore Differences between if you use monthly versus annual data

annsst = ann_avg(dtdsfirst,2)
annamv,annaa = calc_AMV(hlon,hlatnew,annsst,bbox,order,cutofftime,1)
annregr = regress2ts(annsst,annamv/np.nanstd(annamv),0,1)

monamv,monaa = calc_AMV(hlon,hlatnew,dtdsfirst,bbox,order,cutofftime,1)
monregr = regress2ts(dtdsfirst,monamv/np.nanstd(monamv),0,1)

diff = annregr - monregr

fig,axs = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})
axs = init_map(bbox,ax=axs)
cint=np.arange(-0.16,0.20,0.04)
pcm=axs.contourf(hlon,hlatnew,diff.T,cint,cmap=cmap)
axs.set_title("Annual - Monthly, Max Difference (AMV Pattern)")
plt.colorbar(pcm)
plt.savefig(outpath+"HADISST_AMV_Ann_vs_Mon_dsfirst.png",dpi=200)



#%% Plot Maximum Anomaly


maxanom = np.max(np.abs(dtdsfirst),axis=2)

fig,axs = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})
axs = init_map(bbox,ax=axs)
cint=np.arange(0,11,1)
pcm=axs.contourf(hlon,hlatnew,maxanom.T,cint,cmap=cmocean.cm.dense)
axs.set_title("HadISST (1920-2018), Max SST Anomaly",fontsize=20)

plt.colorbar(pcm)
plt.savefig(outpath+"HADISST_AMV_MaxAnom_dsfirst.png",dpi=200)

#%% Plot time series at a single point

lonf = -50
latf = 12

klon,klat = find_latlon(lonf,latf,hlon,hlatnew)

#sstpt = dtdsfirst[klon,klat,:]
sstpt = dtdsfirst[klon,klat,:]

fig,ax = plt.subplots(1,1,figsize=(6,3))
plt.style.use('ggplot')
plt.plot(tper,sstpt)
plt.ylabel("SST Anomaly ($^{\circ}C$)")
plt.xlabel("Time(Months)")
plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.tight_layout()

plt.savefig(outpath+"HadISST_dsdt_SST_lon%02d_lat%02d.png"%(lonf,latf),dpi=200)



#%% Try with entrain data
# Load data and plot
funiform = 2
entraint = np.load(datpath+"stoch_output_1000yr_funiform2_entrain1_hvar2.npy")
Fn =np.load(datpath+"stoch_output_1000yr_run000_funiform2_Forcing.npy",allow_pickle=True)
#forcing = np.load(datpath+"stoch_output_1000yr_funiform%i_Forcing.npy"%(funiform))
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

maxanom = np.max(np.abs(entraint),axis=2)

fig,axs = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})
axs = init_map(bbox,ax=axs)
#cint=np.arange(0,11,1)
pcm=axs.contourf(lonr,latr,maxanom.T,cmap=cmocean.cm.dense)
axs.set_title("Entrain, Max SST Anomaly",fontsize=20)

plt.colorbar(pcm)
plt.savefig(outpath+"Entrain_MaxAnom_dsfirst.png",dpi=200)

#%% Plot Point for entrain data...


lonf = -30
latf = 50

klon,klat = find_latlon(lonf,latf,lonr,latr)


sstpt = entraint[klon,klat,:]
F=Fn.item().get(0)[klon,klat,:]


tper = np.arange(0,entraint.shape[2])
fig,ax = plt.subplots(2,1,figsize=(6,4))
plt.style.use('ggplot')


plt.subplot(2,1,1)
plt.plot(tper,F)
plt.ylabel("Forcing ($^{\circ}C/s$)",fontsize=10)
plt.title("Forcing at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(F),np.nanstd(F),np.nanmax(np.abs(F))))



plt.subplot(2,1,2)
plt.plot(tper,sstpt)
plt.ylabel("SST ($^{\circ}C$)",fontsize=10)
plt.xlabel("Time(Months)",fontsize=10)
#plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.title("Detrended, Deseasonalized SST \n Mean: %.2f || Std: %.2f || Max: %.2f" % (np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))


plt.tight_layout()

plt.savefig(outpath+"HadISST_dsdt_SST_lon%02d_lat%02d_hvarmode%i_fscale%i.png"%(lonf,latf,hvarmode,fscale),dpi=200)









