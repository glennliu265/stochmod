function [corr_ts] = calc_laglead(var_in,lag_rng,basemonth)
% [corr_ts] = calc_laglead(var_in,lag_rng,basemonth)
% Lead Lag Correlation Analysis: Compute the lead-lag 
% correlation of input variable (var_in) for lags (lag_rng) 
% in months, centered on a specified month (basemonth). Note
% that the script detrends the data (linear dt) before calculating
% correlation.
%
% Inputs:
%     1) var_in   : Input Variable of dimensions [month x year]
%     2) lag_rng  : Range of lead (- values) and lag (+ values) in months
%     3) basemonth: Base month to perform lag correlation on (lag 0)
% Output:
%     1) corr_ts  : Correlation time series with indices matching to lag_rng
%
% Status:      Finalized, Needs Testing
% Last Edited: Glenn Liu, 12/08/2019
%
% %Testing
% var_in = nanmean(varnew_rm,[3,4]);
% lag_rng = [-24:24]
% basemonth = 2
%
%% Prepare to calculate lag
 
% Get size of lag
lagdim = length(lag_rng);
 
% Get total timeseries length (assumed 2nd dimension)
totyr  = size(var_in,2);
 
% Get lead and lag sizes
leadsize = ceil(length(find(lag_rng<0))/12);
lagsize = ceil(length(find(lag_rng>0))/12);
 
% Preallocate variable to store correlation
corr_ts = NaN(1,lagdim);
 
% Get base timeseries to perform lag on.
% Select period according to lag and lead size
base_ts = (1+leadsize):(totyr-lagsize);
var_base = var_in(basemonth,base_ts);
 
% Detrend to remove spurious correlations
var_base = detrend(var_base,1);
 
% Perform Lag Analysis
nxtyr = 0; % Add year on next step
addyr = 0; % Flag to add to nxtyr
lagid = 1; 
 
for lag = lag_rng(1):lag_rng(end)
 
    % Determine month to pull data for
    lagm = mod(basemonth + lag,12);
    if lagm == 0
        lagm = 12;
        addyr = 1;         % Flag to add to nxtyr
        modswitch = lag+1; % Add year on next step
    end
 
    % Add to year if month switched from Dec -> Jan
    if addyr==1 && lag==modswitch
%         fprintf(['Now adding a year starting on lag ',...
%             num2str(lag),' which is mon', num2str(lagm),'\n'])
        addyr = 0;
        nxtyr = nxtyr + 1;
    end
 
    % Take data, shifting by year if necessary
    lag_ts = (1+nxtyr):(length(var_base))+nxtyr;
    var_lag = squeeze(var_in(lagm,lag_ts));
    
    % Detrend data
    var_lag = detrend(var_lag);
 
    % Compute correlation
    cc = corrcoef(var_base,var_lag);
    corr_ts(lagid) = cc(2);
    
    % Add to counter
    lagid = lagid + 1;
end
end
