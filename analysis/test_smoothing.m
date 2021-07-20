

%%
fnmat = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/Stochastic_Model_Test_lon-30_lat50.mat"

ld = load(fnmat)
ssts = ld.ssts
cesmslab = ld.cesmslab
cesmfull = ld.cesmfull


addpath '/Users/gliu/Documents/MatlabScripts/Toolboxes/yo_box'


%% Perform Spectral Analysis
close all

x   = ssts(4,:)
opt     = 1
pct     = 0.0
nsmooth = 200
tunit   = char("Months")
dt      = 3600*24*30
axopt   = 3
cls     = [0.95,0.99]
clopt   = 1



[P,freq,dof,r1] = yo_spec(x,opt,nsmooth,pct)

argsin = [tunit,dt,cls,axopt,clopt]
yo_specplot(freq,P,dof,r1,tunit,dt,cls,axopt,clopt)


%% Observe the window


clf
plot(win)
ylim([-0.006,0.0060])
xlim([0,105])



clf
plot(ifft(win))



clf
plot(ifft(P0refl),'k')
hold on
%plot(x,'r')


%% Do another test

x = 

