%% User Edits -------------------------------------------------------------

% Constants
cp0    = 3850 ;% Specific Heat       [J/(kg*C)]
rho    = 1025 ;% Density of Seawater [kg/m^3]

% Initial Conditions/Other Presets
h      =  150 ;% Typical Height Scale [m]
T0     =  0;% Temperature Anomaly at time 0 [C]

% Integration Options
t_end   = 12*1000;% Timestep to integrate up to
dt      = 60*60*24*30; % Timestepping (Will be used to multiply lambda
usetau  = 1

% Correlation Options
detrendopt = 0;

% Filtering Options (expressed in terms of timesteps
% filter_cutoff = 240 % Filter out freqs below this # of timesteps
% butterord     =   5 % Order of the butterworth filter

% Path to data
projpath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/';
scriptpath  = [projpath,'/03_Scripts/stochmod/'];
datpath  = [projpath,'/01_Data/'];
outpath  = [projpath,'/02_Figures/20200602/'];

% Add paths
addpath(scriptpath)
addpath(datpath)
addpath('/Users/gliu/')
startup

% Select Point to model at
lonf = 330; % [0 360]
latf = 50; % [-90 90]
mon  = 3;


% White Noise Generator Options
rescaler  = 10^0; % Multiplier to rescale whitenoise
genrand   = 1    ;   % Option to generate white noise (1 to gen, 0 to load)
noisemat  = 'whitenoise.mat'; % matfile for stored white noise
savenoise = 1    ;   % Option to save generated white noise

%% Script Start / Setup

% Load damping variable (ensemble and lag averaged)
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/';
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat';
load([damppath,dampmat],'ensavg','LON1','LAT') ;
load([datpath,'stochrun.mat'])

% Find the point
[oid,aid] = findcoords(lonf,latf,2,{LON1,LAT});
  
% % Load in climatological MLD from levitus 1994 wod
% load('wod_1994_mldclim.mat')
% % Load in MLD Climatology and get point Lat/Lon
% [mld_lon,mld_lat] = findcoords(lonf,latf,2,{lonlev,latlev});


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



% 
lambda_new = exp(-1* squeeze(ensavg(oid,aid,:))/(rho*cp0*h) * monscale);
lambda_old = exp(-1* 1 ./ mean(TAUall,2));

% explam = exp(-lambda);
% lambda = squeeze(ensavg(oid,aid,:))/(rho*cp0*h)


% Preallocate
temp_ts = NaN(1,t_end) ;% Temperature Time Series
noise_ts = NaN(size(temp_ts));
damp_ts = NaN(size(temp_ts));

%% Run Model Forward
for t = 1:t_end

    m = mod(t,12);
    if m == 0
        m = 12;
    end
    
    lambda = ensavg(oid,aid,m);
    
    % Use damping coefficient to compute e-folding time
    if usetau == 1
        l = 1 ./ nanmean(TAUall(m,:),2)
    else
        l = lambda / (rho*cp0*h);
    end

    
    % Get mixed layer depth
    %h = mld(mld_lon,mld_lat,m);
    
    % Get SST from the last timestep
    if t == 1
        T = T0;
    else
        T = temp_ts(1,t-1);
    end
    
    % Calculate Noise term (include seasonal correction)
    %noise_term = F(1,t) %/ (rho * cp0 * h) %* ((1 - exp(-l)) / l);
    noise_term = eta(1,t);
    
    % Calculate temp at next timestep
    %T1 = T*-1*l + noise_term;
    if usetau == 1
        T1 = exp(-l)*T + noise_term;
    else
        T1 = exp(-l*dt)*T + noise_term;
    end
    
    % Record values
    damp_ts(:,t) = exp(-1*l*dt)*T;
    noise_ts(:,t) = noise_term;
    temp_ts(:,t) = T1;
end

% Reshape to separate month and year dimensions
tempr = reshape(temp_ts,12,length(temp_ts)/12);


R_march2a = []
R_march2a(1) = 1

 for i=1:60
          lag_yr=fix((i+kmonth-1)/12);
          R_march2a(1+i)=yo_cor(temp_ts(kmonth:12:end-lag_yr*12)',temp_ts(kmonth+i:12:end)',-1,-1);
          %R_march1a(1+i)=yo_cor(SST1(24+kmonth:12:end-lag_yr*12),SST1(24+kmonth+i:12:end),-1,-1);
 end

% Calculate autocorrelation
tot_lag = 60; % Total Lag in Months
kmonth  = 3 ; % Lag Base Month (Lag 0)
lag_rng = 0:60;
[corr_ts] = calc_lagcovar(tempr,tempr,lag_rng,kmonth,detrendopt)

if detrendopt == 1
    
    R0 = NaN(60,40);
    R1 = NaN(60,40);
    R2 = NaN(60,40);
       
    for en = 1:40
        
        T0 = SSTall(:,en);
        
        T1 = SSTall(:,en);
        T2 =
        
        
        
    end
    
    
end


%%
figure(1)
clf
hold on

lagplot = [0:60];
plot(lagplot,AVG0,'LineWidth',2.5,'color','b')
plot(lagplot,AVG1,'LineWidth',2.5,'color','c')
plot(lagplot,AVG2,'LineWidth',2.5,'color','g')
plot(lagplot,corr_ts,'LineWidth',2.5,'color','r')
legend('CESM1','No-Entrain (YO)','Entrain (YO)','No-Entrain (GL)')

title(sprintf('SST Autocorrelation for month %i',kmonth),'FontSize',20)
grid on
ylabel('Lags (Months)')
xlabel('Correlation')

figname = sprintf('%sacplot_mon%i_usetau%i_detrend%i.png',outpath,kmonth,usetau,detrendopt);
saveas(gcf,figname,'png')





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



