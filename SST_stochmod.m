%% User Edits -------------------------------------------------------------

% Constants
cp0    = 3850 ;% Specific Heat       [J/(kg*C)]
rho    = 1025 ;% Density of Seawater [kg/m^3]

% Initial Conditions/Other Presets
h      =  150 ;% Typical Height Scale [m]
T0     =  0;% Temperature Anomaly at time 0 [C]

% Integration Options
t_end   = 120000;% Timestep to integrate up to

% Filtering Options (expressed in terms of timesteps
filter_cutoff = 240 % Filter out freqs below this # of timesteps
butterord     =   5 % Order of the butterworth filter

% Path to data
datpath  = '/Users/gyl/Downloads/02_Research/01_Data/AMV_hfdamping/';
outpath  = '/Users/gyl/Downloads/02_Research/02_Figures/20200221/';

% Select Point to model at
lonf = 330; % [0 360]
latf = 50; % [-90 90]
mon  = 11;

% White Noise Generator Options
rescaler  = 10^0; % Multiplier to rescale whitenoise
genrand   = 1    ;   % Option to generate white noise (1 to gen, 0 to load)
noisemat  = 'whitenoise.mat'; % matfile for stored white noise
savenoise = 1    ;   % Option to save generated white noise

%% Script Start / Setup

% Load in Damping Coefficients if not already in workspace
if exist('dnhflx','var') ~= 1
    addpath(datpath)
    
    % Load in corrected damping coefficients
    load('Damping_Coeffs_Net_Corrected.mat')
    
    % Load in climatological MLD from levitus 1994 wod
    load('wod_1994_mldclim.mat')
end

% Load in Lat/Lon Variables
load('CESM1_LATLON.mat')

% Load in MLD Climatology and get point Lat/Lon
[mld_lon,mld_lat] = findcoords(lonf,latf,2,{lonlev,latlev});

% Get Ensemble and Lag Average
vrb = permute(dnhflx,[1,2,4,3,5]);% Move month element to the 3rd position
vrb = nanmean(vrb(:,:,:,:),4);

% Find the point
[oid,aid] = findcoords(lonf,latf,2,{LON,LAT});

% Generate Random White Noise Time Series based on nondimensionalized
% parameters
if genrand == 1
    a = rand(1,t_end);
    F = (a-nanmean(a)) .* rescaler;
    if savenoise == 1
        save([datpath,noisemat],'F')
    end
else
    load([datpath,noisemat]);
end



% Preallocate
temp_ts = NaN(1,t_end) ;% Temperature Time Series
noise_ts = NaN(size(temp_ts));
damp_ts = NaN(size(temp_ts));

%% Run Model Forward
for t = 1:t_end

    m = mod(mon + t,12);
    if m == 0
        m = 12;
    end
    
    % Use damping coefficient to compute e-folding time
    l = vrb(oid,aid,m)/(rho*cp0*h);
    
    % Get mixed layer depth
    h = mld(mld_lon,mld_lat,m);
    
    if t == 1
        T = T0;
    else
        T = temp_ts(1,t-1);
    end
    
    % Calculate Noise term (include seasonal correction)
    %noise_term = F(1,t) / (rho * cp0 * h) * ((1 - exp(-l)) / l);
    noise_term = F(1,t);
    
    % Calculate temp at next timestep
    %T1 = T*-1*l + noise_term;
    T1 = exp(-l*t)*T + noise_term;
    
    % Record values
    damp_ts(:,t) = exp(-l*t)*T;
    noise_ts(:,t) = noise_term;
    temp_ts(:,t) = T1;
end


%% Filter data
FiltFreq = length(temp_ts)/filter_cutoff;
NyFreq   = length(temp_ts)/2;

% Use a 10-yr lowpass on time series (normal and detrended)
[b,a]          = butter(butterord,FiltFreq/NyFreq);
SSTfilt        = filtfilt(b,a,double(temp_ts));


%% Plotting Options
%% Plot of the statistical model results -----------------------------------
figure(1)
clf
hold on

yrs = [1:12:t_end/12]

yyaxis right
l1 = plot(1:t_end,temp_ts,'Color',[.75 .75 .75],'LineWidth',1.5)
ylabel('SST Anomaly')
l3 = plot(1:t_end,SSTfilt,'-k','LineWidth',2.0)

% yyaxis left
% l2 = plot(1:t_end,F,'b','LineWidth',0.25)
% ylabel('Forcing')

%legend([l1,l2,l3],{'SST','F','10-yr LP-Filt'})
legend([l1,l3],{'SST','20-yr LP-Filt'})
xlabel('Months')
title({'Statistical Model Results',['LON: ',num2str(lonlev(mld_lon)),...
    ' LAT: ',num2str(latlev(mld_lat))]},'FontSize',18)

figname = [outpath,'statmod_result_LON',num2str(lonlev(mld_lon)),...
    '_LAT',num2str(latlev(mld_lat)),'.png'];
saveas(gcf,figname)
%% Seasonal Cycle of MLD and Damping coefficient at Point -----------------
figure(2)
clf

% Plot MLD on left axis
yyaxis left
mld_scycle = squeeze(mld(mld_lon,mld_lat,:));
l1 = plot(1:12,mld_scycle,'b','LineWidth',2.5)
set(gca,'Ydir','reverse')
ylabel(['Mixed Layer Depth'],'FontSize',14)

% Plot Damping Coeff on right axis
yyaxis right
damp_scycle = squeeze(vrb(oid,aid,:));
l2 = plot(1:12,damp_scycle,'r','LineWidth',1.5)
ylabel(['NHFLX Damping Coefficient'],'FontSize',14)

legend([l1,l2],{'MLD','\lambda'},'Location','northwest','FontSize',14)
xlim([1 12])
title({'Seasonal MLD and \lambda^a Cycle',['LON: ',num2str(lonlev(mld_lon)),...
    ' LAT: ',num2str(latlev(mld_lat))]},'FontSize',18)
xlabel(['Months'],'FontSize',14)
figname = [outpath,'mld_scycle_LON',num2str(lonlev(mld_lon)),...
    '_LAT',num2str(latlev(mld_lat)),'.png'];
saveas(gcf,figname)



