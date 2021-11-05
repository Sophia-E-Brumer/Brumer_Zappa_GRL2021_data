# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:06:13 2021

@author: sbrumer
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:47:07 2021

@author: sbrumer
"""

import scipy.io as sio
from scipy.odr import *
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot  as plt

import matplotlib
import scipy.stats as sis #.pearsonr(x, y)[source]¶
import seaborn as sns
sns.set_style("ticks",{'axes.grid' : True})
#import seabornfig2grid as sfg
import SeabornFig2Grid as sfg

import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rcParams['savefig.dpi'] = 100
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
import xarray as xr

#%% Loading data archive
Brumer_Zappa_GRL_2021 = xr.open_dataset('Brumer_Zappa_GRL_2021.nc')
#%% Calculations of dissipation and volume flux
B = 0.1
Ub = 0.1 # m/s
g = 9.81
Brumer_Zappa_GRL_2021['eps'] = Brumer_Zappa_GRL_2021.beff/g* (Brumer_Zappa_GRL_2021.Lambda.fillna(0)*Brumer_Zappa_GRL_2021.c**5).integrate("c")
Brumer_Zappa_GRL_2021['eps'].attrs['units'] = 'm2/s3'
Brumer_Zappa_GRL_2021['eps'].attrs['long_name'] = 'Wave Breaking Induced Dissipation Rate'
    
Brumer_Zappa_GRL_2021['Fa_eq19'] = 4*Brumer_Zappa_GRL_2021.beff*B/Ub/g**2*(Brumer_Zappa_GRL_2021.Lambda.fillna(0)*Brumer_Zappa_GRL_2021.c**4).integrate("c")*3600*100
Brumer_Zappa_GRL_2021['Fa_eq19'].attrs['units'] = 'cm/hr'
Brumer_Zappa_GRL_2021['Fa_eq19'].attrs['long_name'] = 'Wave Breaking Entrained Air Volume Flux'
nu = Brumer_Zappa_GRL_2021.nu_w.values
HW_wave_age = Brumer_Zappa_GRL_2021.cp.values/Brumer_Zappa_GRL_2021.usr.values
#%% Functions
def smooth(a,WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def woolf_1993_1(Sc,alpha,Qb):
    """    
    INPUT:   alpha  - Ostwald solubility (inverse of H the Henry coeff)
             Sc     - Schmidt number
             Qb     - volume flux of bubbles in (cm/h)/m^3 (Woolf uses 24.4)
    
    OUTPUT:  kb - bubble mediated gas transfer according to Woolf 1993
    """
    #-- bubble mediated gas transfer velocities
    # Qb = 24.5; % volume flux of bubbles 24.5 (cm/h)/m^3
    f = 1.2; # constant related to the breath of the plume distribution
    Chi = Sc**0.5/(14*alpha);
    # eq (8)
    kbb = (Qb/alpha)*(1+Chi**(1/f))**(-f);
    return kbb 

def model_1(x,a,b,c):
    """
    x[0] - K_eps
    x[1] - Fa/alpha
    """
    Sc = 660
    n = 0.5
    f = 1.2
    y = a*x[0]+b*x[1]*(1+(c*Sc**(-n)*x[1])**(1/f))**(-f)
    return y

def model_ODR(B,x):
    """
    x[0] - K_eps
    x[1] - Fa/alpha
    """
    a = B[0];b=B[1];c=B[2]
    Sc = 660
    n = 0.5
    f = 1.2
    y = a*x[0]+b*x[1]*(1+(c*Sc**(-n)*x[1])**(1/f))**(-f)
    return y
#%% Figure 1
df = pd.DataFrame({'SST [$^\circ$C]': Brumer_Zappa_GRL_2021.SST, 
                   '$T_a$ [$^\circ$C]': Brumer_Zappa_GRL_2021.Ta,
                   '$H_{s}$ [m]': Brumer_Zappa_GRL_2021.Hs,
                   '$c_{p}/u_{*}$':Brumer_Zappa_GRL_2021.cp/Brumer_Zappa_GRL_2021.usr,
                   '$U_{10N}$ [m s$^{-1}$]':Brumer_Zappa_GRL_2021.U10N
                  })
g2 = sns.jointplot(data = df, x = 'SST [$^\circ$C]',y = '$T_a$ [$^\circ$C]', s=100, alpha=.5, marginal_ticks=True, marginal_kws=dict(bins=40, fill=True))
g1 = sns.jointplot(df['$H_{s}$ [m]'],df['$c_{p}/u_{*}$'], s=100, alpha=.5, marginal_ticks=True, marginal_kws=dict(bins=40, fill=True))
g0 = sns.jointplot(df['$H_{s}$ [m]'],df['$U_{10N}$ [m s$^{-1}$]'], s=100, alpha=.5,marginal_ticks=True, marginal_kws=dict(bins=40, fill=True))
g0.ax_joint.annotate('(a)', xy=(-0.25, 1.2), xycoords="axes fraction")
g1.ax_joint.annotate('(b)', xy=(-0.25, 1.2), xycoords="axes fraction")
g2.ax_joint.annotate('(c)', xy=(-0.25, 1.2), xycoords="axes fraction")

fig = plt.figure(figsize=(17,6))
gs = gridspec.GridSpec(1,3)
mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
mg2 = sfg.SeabornFig2Grid(g2, fig, gs[2])
gs.tight_layout(fig)
fig.savefig('joint_hist.png',dpi = 72)
#
print(df.min())
print(df.max())
#%% Figure 2
cp_star_bins = np.arange(0,40+2.5,2.5)

bins = Brumer_Zappa_GRL_2021.c.values
upper_limit = 20;
ind_keep = np.argwhere((bins>=1)& (bins<upper_limit)).ravel()

s = []
l = []
wave_age=[]
cc = 0
for i in Brumer_Zappa_GRL_2021.time:
    ref = Brumer_Zappa_GRL_2021.Lambda[i,:].values
    
    if (HW_wave_age[i] > 0)  & (np.nanmax(ref)>10**(-5)):
        wave_age.append(HW_wave_age[i])
        l.append(ref[ind_keep])
        s.append(bins[ind_keep]+0.5*np.diff(bins[np.argwhere((bins>=1)& (bins<=upper_limit)).ravel()]))
        cc = cc + 1
Y = np.digitize(wave_age,cp_star_bins)
l = np.vstack(l)
s = np.vstack(s)
ll = np.empty((len(cp_star_bins),np.size(s,1)))
ss = np.empty_like(ll)
aaa = np.empty(len(cp_star_bins))
for i,wa in enumerate(np.unique(Y)):
    ll[i,:] = 10**smooth(np.nanmean(np.log10(l[Y==wa,:]),0));
    ss[i,:] = np.nanmean((s[(Y==wa),:]),0);
    aaa[i] = np.size(s[(Y==wa),:],0);


n = len(cp_star_bins)
cmap = plt.get_cmap("GnBu", n+3)
cmap = ListedColormap(cmap(np.linspace(0.2, 1, n)))

fig,ax = plt.subplots(1,2,figsize=(13,9))

s_wave_age = Brumer_Zappa_GRL_2021.wa_SM.values
c = 0;
for i in range(len(s_wave_age)):
    cind = np.argwhere(np.abs(cp_star_bins - s_wave_age[c]) == np.min(np.abs(cp_star_bins - s_wave_age[c]) )).ravel()[0];  
    ax[0].plot(Brumer_Zappa_GRL_2021.c_SM[:,i],Brumer_Zappa_GRL_2021['SM_Lambda'][:,i],color=cmap(cind),label = ('' if c==0 else '_') + 'Southerland and Melville [2015]')
    ax[1].plot(Brumer_Zappa_GRL_2021.c_SM[:,i],Brumer_Zappa_GRL_2021['SM_Lambda'][:,i]*Brumer_Zappa_GRL_2021.c_SM[:,i]**5,color=cmap(cind))
    c = c + 1
 

k_wave_age = Brumer_Zappa_GRL_2021.wa_KM.values
c = 0;
for ij in range(len(k_wave_age)):
    cind = np.argwhere(np.abs(cp_star_bins - k_wave_age[c]) == np.min(np.abs(cp_star_bins - k_wave_age[c]) )).ravel()[0];
    k = ax[0].plot(Brumer_Zappa_GRL_2021.c_KM[:,i],Brumer_Zappa_GRL_2021['KM_Lambda'][:,i],linestyle = '-.',linewidth=1,color=cmap(cind),label = ('' if ij==0 else '_') + 'Kleiss and Meville [2010]');
    ax[1].plot(Brumer_Zappa_GRL_2021.c_KM[:,i],Brumer_Zappa_GRL_2021['KM_Lambda'][:,i]*Brumer_Zappa_GRL_2021.c_KM[:,i]**5,linestyle ='-.',linewidth=1,color=cmap(cind))#,'Color',Colors(cind,:),'linewidth',2);
    c = c+1;

for ij in range(0,len(Brumer_Zappa_GRL_2021.wa_STG.values)):
    cind = np.argwhere(np.abs(cp_star_bins - 10) == np.min(np.abs(cp_star_bins -10) )).ravel()[0];
    sch = ax[0].plot(Brumer_Zappa_GRL_2021.c_STG[:,i],Brumer_Zappa_GRL_2021['STG_Lambda'][:,i],linestyle = ':',linewidth=2,color=cmap(cind),label=('' if ij==0 else '_') + 'Schwendeman et al. [2014]')#,'Color',Colors(cind,:),'linewidth',2);
    ax[1].plot(Brumer_Zappa_GRL_2021.c_STG[:,i],Brumer_Zappa_GRL_2021['STG_Lambda'][:,i]*Brumer_Zappa_GRL_2021.c_STG[:,i]**5,linestyle = ':',linewidth=2, color=cmap(cind))

b_wave_age = Brumer_Zappa_GRL_2021.wa_B.values;
c = 0;
for i in range(len(b_wave_age)):
    cind = np.argwhere(np.abs(cp_star_bins - b_wave_age[c]) == np.min(np.abs(cp_star_bins - b_wave_age[c]) )).ravel()[0];
    b = ax[0].plot(Brumer_Zappa_GRL_2021.c_B[:,i],Brumer_Zappa_GRL_2021['B_Lambda'][:,i],color=cmap(cind),label=('' if c==0 else '_') + 'Banner et al. [2014]',marker = 'o',linewidth = 1)
    b = ax[1].plot(Brumer_Zappa_GRL_2021.c_B[:,i],Brumer_Zappa_GRL_2021['B_Lambda'][:,i]*Brumer_Zappa_GRL_2021.c_B[:,i]**5,'-o','linewidth',1,color=cmap(cind))
    c = c + 1;

for i,wa in enumerate(np.unique(Y)):
    h = ax[0].plot(ss[i,:],ll[i,:],color=cmap(wa),label=('' if i==0 else '_') + 'HiWinGS',linewidth = 2);
    h = ax[1].plot(ss[i,:],ll[i,:]*ss[i,:]**5,color=cmap(wa),linewidth = 2);
    
    
ccc =np.arange(15.,31.)
l6 = ax[0].plot(ccc,10000*ccc**(-6),color = 'gray',linewidth=4,label='$c^{-6}$')    
    
norm= matplotlib.colors.BoundaryNorm(cp_star_bins+np.diff(cp_star_bins)[0]/2, n)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for axs in ax:
    axs.set_xscale('log')
    axs.set_yscale('log')
 #   axs.grid(True,which="both")
    axs.grid(which='both', axis='both', linestyle='--')
    axs.minorticks_on()
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    axs.yaxis.set_minor_locator(locmin)
    axs.set_xlabel('$c$ [m s$^{-1}$]')
    axs.set_ylim([10**-8,10])
    axs.set_xlim([0.1,50])

ax[0].set_ylabel('$\Lambda(c)$ [m$^{-2}$ s]')
ax[1].set_ylabel('$c^5 \Lambda(c)$ [m$^{3}$ s$^{-4}$]')
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
clb = fig.colorbar(sm, ticks=cp_star_bins[::2], cax=cbar_ax)
clb.ax.set_title('$c_p/u_*$')
leg = ax[0].legend(loc = 1,fontsize = 'small',borderpad = 0.1,handlelength=1.2,handletextpad=0.3)
for handels in leg.legendHandles[:-1]:
    handels.set_color('k')
    
ax[0].annotate('(a)', xy=(-0.18, 1.01), xycoords="axes fraction")
ax[1].annotate('(b)', xy=(-0.18, 1.01), xycoords="axes fraction")
plt.savefig('lambda_fig_vpython.png')
#%% First estimates of Klambda coefficients
Sc = 660
Ke = np.real(Sc**(-1/2)*(nu*Brumer_Zappa_GRL_2021.eps.values)**(1/4)*100*3600)
kb_CO2 = woolf_1993_1(Sc,Brumer_Zappa_GRL_2021.alphaCO2,Brumer_Zappa_GRL_2021.Fa_eq19)
kb_DMS = woolf_1993_1(Sc,Brumer_Zappa_GRL_2021.alphaDMS,Brumer_Zappa_GRL_2021.Fa_eq19)

kb_CO2[kb_CO2==0]=np.nan
kb_DMS[kb_DMS==0]=np.nan

KE = np.concatenate([Ke,Ke])
KB = np.concatenate([kb_CO2,kb_DMS])
ydata = np.concatenate([Brumer_Zappa_GRL_2021.k660CO2,Brumer_Zappa_GRL_2021.k660DMS])
COAREG_all = np.concatenate([Brumer_Zappa_GRL_2021.COAREG_k660CO2,Brumer_Zappa_GRL_2021.COAREG_k660DMS])

KE = np.concatenate([Ke,Ke])
FA = np.concatenate([Brumer_Zappa_GRL_2021.Fa_eq19.values,Brumer_Zappa_GRL_2021.Fa_eq19.values])
ALPHA =  np.concatenate([Brumer_Zappa_GRL_2021.alphaCO2,Brumer_Zappa_GRL_2021.alphaDMS])
index_1 = ~(np.isnan(KE)| np.isnan(FA) | np.isnan(ALPHA) | np.isnan(ydata))

xdata_1 = [KE[index_1],FA[index_1]/ALPHA[index_1]]

popt_1, pcov_1 = curve_fit(model_1, xdata_1, ydata[index_1])
print(popt_1, pcov_1)
stdevs = np.sqrt(np.diag(pcov_1))
print(stdevs)
#%% Estimates of Klambda coefficients with beff error propagation
KE = np.concatenate([Ke,Ke])
FA = np.concatenate([Brumer_Zappa_GRL_2021.Fa_eq19.values,Brumer_Zappa_GRL_2021.Fa_eq19.values])
ALPHA =  np.concatenate([Brumer_Zappa_GRL_2021.alphaCO2,Brumer_Zappa_GRL_2021.alphaDMS])
ydata = np.concatenate([Brumer_Zappa_GRL_2021.k660CO2,Brumer_Zappa_GRL_2021.k660DMS])
sx = np.sqrt((6.481*10**(-4))**2+HW_wave_age*(1.935*10**(-5))**2)
sx = np.concatenate([sx,sx])
index_1 = ~(np.isnan(KE)| np.isnan(FA) | np.isnan(ALPHA) | np.isnan(ydata) | np.isnan(sx))
sx = np.vstack((sx[index_1],sx[index_1]))
gas_transfer = Model(model_ODR)


xdata_1 = [KE[index_1],FA[index_1]/ALPHA[index_1]]
mydata = RealData(xdata_1,ydata[index_1],sx = sx)

myodr = ODR(mydata, gas_transfer, beta0=[popt_1[0],popt_1[1],popt_1[2]])
myoutput = myodr.run()
myoutput.pprint()
CO2_model_ODR = model_1([Ke,Brumer_Zappa_GRL_2021.Fa_eq19/Brumer_Zappa_GRL_2021.alphaCO2],myoutput.beta[0],myoutput.beta[1],myoutput.beta[2])
DMS_model_ODR = model_1([Ke,Brumer_Zappa_GRL_2021.Fa_eq19/Brumer_Zappa_GRL_2021.alphaDMS],myoutput.beta[0],myoutput.beta[1],myoutput.beta[2])
all_model_ODR = np.concatenate([CO2_model_ODR,DMS_model_ODR])
print('Model coeffs')
print(myoutput.beta)
#%% Calculation of fit statistics
# dealing with NaNs
index_CO2 = ~(np.isnan(Ke)| np.isnan(Brumer_Zappa_GRL_2021.Fa_eq19) | np.isnan(Brumer_Zappa_GRL_2021.alphaCO2) | np.isnan(Brumer_Zappa_GRL_2021.k660CO2))
index_DMS = ~(np.isnan(Ke)| np.isnan(Brumer_Zappa_GRL_2021.Fa_eq19) | np.isnan(Brumer_Zappa_GRL_2021.alphaDMS) | np.isnan(Brumer_Zappa_GRL_2021.k660DMS))

#COAREG performance stats
COAREG_all = np.concatenate([Brumer_Zappa_GRL_2021.COAREG_k660CO2.values,Brumer_Zappa_GRL_2021.COAREG_k660DMS.values])

print('COAREG')
COAREG_r2_all = sis.pearsonr(COAREG_all[index_1], ydata[index_1])[0]**2
COAREG_rmse_all = np.sqrt(np.mean((COAREG_all[index_1]- ydata[index_1])**2))
print(r'all: r^2=',COAREG_r2_all,'rmse =', COAREG_rmse_all)

COAREG_r2_CO2 = sis.pearsonr(Brumer_Zappa_GRL_2021.COAREG_k660CO2[index_CO2].values, Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)[0]**2
COAREG_rmse_CO2 = np.sqrt(np.mean((Brumer_Zappa_GRL_2021.COAREG_k660CO2[index_CO2].values- Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)**2))
print(r'CO2: r^2=',COAREG_r2_CO2,'rmse =', COAREG_rmse_CO2)

COAREG_r2_DMS = sis.pearsonr(Brumer_Zappa_GRL_2021.COAREG_k660DMS[index_DMS].values, Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)[0]**2
COAREG_rmse_DMS = np.sqrt(np.mean((Brumer_Zappa_GRL_2021.COAREG_k660DMS[index_DMS].values- Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)**2))
print(r'DMS: r^2=',COAREG_r2_DMS,'rmse =', COAREG_rmse_DMS)

#model_ODR performance
print('ODR')

mODR_r2_all = sis.pearsonr(all_model_ODR[index_1], ydata[index_1])[0]**2
mODR_rmse_all = np.sqrt(np.mean((all_model_ODR[index_1]- ydata[index_1])**2))
print(r'all: r^2=',mODR_r2_all,'rmse =', mODR_rmse_all)

mODR_r2_CO2 = sis.pearsonr(CO2_model_ODR[index_CO2].values, Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)[0]**2
mODR_rmse_CO2 = np.sqrt(np.mean((CO2_model_ODR[index_CO2].values- Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)**2))
print(r'CO2: r^2=',mODR_r2_CO2,'rmse =', mODR_rmse_CO2)

mODR_r2_DMS = sis.pearsonr(DMS_model_ODR[index_DMS].values, Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)[0]**2
mODR_rmse_DMS = np.sqrt(np.mean((DMS_model_ODR[index_DMS].values- Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)**2))
print(r'DMS: r^2=',mODR_r2_DMS,'rmse =', mODR_rmse_DMS)
#%% Figure 3
fig,ax = plt.subplots(1,2,figsize=(17,8),sharex='all',sharey='all')
for axs in ax:
    axs.plot([0, 350],[0, 350],'k',label='1:1 line')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.grid(True,which="both")
    axs.set_xticks([10,50,100,200,300])
    axs.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs.set_xlim([8.5,350])
    axs.set_yticks([10,50,100,200,300])
    axs.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs.set_ylim([8.5,350])
ax[0].set_xlabel('measured k$_{660}$ [cm hr$^{-1}$]')
    
ax[0].scatter(Brumer_Zappa_GRL_2021.k660CO2[index_CO2],CO2_model_ODR[index_CO2],color = 'indigo',label = 'CO$_2$: r$^2$ = ' + '{0:.2f}'.format(mODR_r2_CO2)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_CO2))
ax[0].scatter(Brumer_Zappa_GRL_2021.k660DMS[index_DMS],DMS_model_ODR[index_DMS],color = 'crimson',marker='s',label = 'DMS: r$^2$ = '+ '{0:.2f}'.format(mODR_r2_DMS)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_DMS))
ax[0].set_ylabel('k$_{\Lambda 660}$ [cm hr$^{-1}$]')#' = '+'{0:.2f}'.format(popt_1[0])+' $K_\epsilon$ + '
                # + '{0:.2f}'.format(popt_1[1])+r'$F_a \alpha^{-1}$(1+('
                # + '{0:.2e}'.format(popt_1[2]) 
                # + r'$Sc^{-1/2}F_a \alpha^{-1})^{1/1.2})^{-1.2}$ ')


ax[1].scatter(Brumer_Zappa_GRL_2021.k660CO2[index_CO2],Brumer_Zappa_GRL_2021.COAREG_k660CO2[index_CO2],color = 'indigo',label = 'CO$_2$: r$^2$ = ' + '{0:.2f}'.format(COAREG_r2_CO2)+'; rmse = ' + '{0:.2f}'.format(COAREG_rmse_CO2))
ax[1].scatter(Brumer_Zappa_GRL_2021.k660DMS[index_DMS],Brumer_Zappa_GRL_2021.COAREG_k660DMS[index_DMS],color = 'crimson',marker='s',label = 'DMS: r$^2$ = '+ '{0:.2f}'.format(COAREG_r2_DMS)+'; rmse = ' + '{0:.2f}'.format(COAREG_rmse_DMS))
ax[1].set_xlabel('measured k$_{660}$ [cm hr$^{-1}$]')
ax[1].set_ylabel('COAREG k$_{660}$ [cm hr$^{-1}$]')

ax[1].legend(title='HiWINGS all: r$^2$ = ' + '{0:.2f}'.format(COAREG_r2_all)+'; rmse = ' + '{0:.2f}'.format(COAREG_rmse_all))

ax[0].legend(title='HiWINGS all: r$^2$ = ' + '{0:.2f}'.format(mODR_r2_all)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_all))

ax[0].annotate('(a)', xy=(-0.12, 1.05), xycoords="axes fraction")
ax[1].annotate('(b)', xy=(-0.12, 1.05), xycoords="axes fraction")
plt.savefig('/HiWinGS/framework_fig_vR1.png')
#%% fixing dissipation part with DMS first then bubble with CO2, i.e. considering non bubble effects on dms
# for reviewer
def model_ODR_Keps(B,x):
    """
    x[0] - K_eps
    """
    a = B
    Sc = 660
    n = 0.5
    f = 1.2
    y = a*x
    return y

KE = Ke
ydata = Brumer_Zappa_GRL_2021.k660DMS.values
sx = np.sqrt((6.481*10**(-4))**2+HW_wave_age*(1.935*10**(-5))**2)
index_eps = ~(np.isnan(KE)|np.isnan(ydata) | np.isnan(sx))

gas_transfer_eps = Model(model_ODR_Keps)

mydata = RealData(KE[index_eps],ydata[index_eps],sx = sx[index_eps])

myodr = ODR(mydata, gas_transfer_eps, beta0=[popt_1[0]])
myoutput_eps = myodr.run()
myoutput_eps.pprint()


def model_ODR_co2(B,x):
    """
    x[0] - K_eps
    x[1] - Fa/alpha
    """
    b = B[0];c=B[1]
    Sc = 660
    n = 0.5
    f = 1.2
    y =b*x*(1+(c*Sc**(-n)*x)**(1/f))**(-f)
    return y

FA = Brumer_Zappa_GRL_2021.Fa_eq19.values
ALPHA = Brumer_Zappa_GRL_2021.alphaCO2.values
ydata = Brumer_Zappa_GRL_2021.k660CO2.values - myoutput_eps.beta[0]*KE
index_2 = ~(np.isnan(KE)| np.isnan(FA) | np.isnan(ALPHA) | np.isnan(ydata) | np.isnan(sx))
gas_transfer_co2 = Model(model_ODR_co2)


xdata_1 = FA[index_2]/ALPHA[index_2]
mydata = RealData(xdata_1,ydata[index_2],sx = sx[index_2])

myodr_c02 = ODR(mydata, gas_transfer_co2, beta0=[popt_1[1],popt_1[2]])
myoutput_c02 = myodr_c02.run()
myoutput_c02.pprint()


ydata = np.concatenate([Brumer_Zappa_GRL_2021.k660CO2.values,Brumer_Zappa_GRL_2021.k660DMS.values])
CO2_model_ODR2 = model_1([Ke,Brumer_Zappa_GRL_2021["Fa_eq19"]/Brumer_Zappa_GRL_2021.alphaCO2],myoutput_eps.beta[0],myoutput_c02.beta[0],myoutput_c02.beta[1])
DMS_model_ODR2 = model_1([Ke,Brumer_Zappa_GRL_2021["Fa_eq19"]/Brumer_Zappa_GRL_2021.alphaDMS],myoutput_eps.beta[0],0,0)
all_model_ODR2 = np.concatenate([CO2_model_ODR2,DMS_model_ODR2])
#model_ODR performance
print('ODR2')
mODR2_r2_all = sis.pearsonr(all_model_ODR2[index_1], ydata[index_1])[0]**2
mODR2_rmse_all = np.sqrt(np.mean((all_model_ODR2[index_1]- ydata[index_1])**2))
print(r'all: r^2=',mODR_r2_all,'rmse =', mODR_rmse_all)

mODR2_r2_CO2 = sis.pearsonr(CO2_model_ODR2[index_CO2].values, Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)[0]**2
mODR2_rmse_CO2 = np.sqrt(np.mean((CO2_model_ODR2[index_CO2].values- Brumer_Zappa_GRL_2021.k660CO2[index_CO2].values)**2))
print(r'CO2: r^2=',mODR_r2_CO2,'rmse =', mODR_rmse_CO2)

mODR2_r2_DMS = sis.pearsonr(DMS_model_ODR2[index_DMS].values, Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)[0]**2
mODR2_rmse_DMS = np.sqrt(np.mean((DMS_model_ODR2[index_DMS].values- Brumer_Zappa_GRL_2021.k660DMS[index_DMS].values)**2))
print(r'DMS: r^2=',mODR_r2_DMS,'rmse =', mODR_rmse_DMS)


fig,ax = plt.subplots(1,2,figsize=(17,8),sharex='all',sharey='all')
for axs in ax:
    axs.plot([0, 350],[0, 350],'k',label='1:1 line')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.grid(True,which="both")
    axs.set_xticks([10,50,100,200,300])
    axs.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs.set_xlim([8.5,350])
    axs.set_yticks([10,50,100,200,300])
    axs.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs.set_ylim([8.5,350])
ax[0].set_xlabel('measured k$_{660}$ [cm hr$^{-1}$]')
    
ax[0].scatter(Brumer_Zappa_GRL_2021.k660CO2[index_CO2],CO2_model_ODR2[index_CO2],color = 'indigo',label = 'CO$_2$: r$^2$ = ' + '{0:.2f}'.format(mODR2_r2_CO2)+'; rmse = ' + '{0:.2f}'.format(mODR2_rmse_CO2))
ax[0].scatter(Brumer_Zappa_GRL_2021.k660DMS[index_DMS],DMS_model_ODR2[index_DMS],color = 'crimson',marker='s',label = 'DMS: r$^2$ = '+ '{0:.2f}'.format(mODR2_r2_DMS)+'; rmse = ' + '{0:.2f}'.format(mODR2_rmse_DMS))
ax[0].set_ylabel('k$_{\Lambda 660}$ v fix $\mathcal{A}$ DMS[cm hr$^{-1}$]')

ax[1].scatter(Brumer_Zappa_GRL_2021.k660CO2[index_CO2],CO2_model_ODR[index_CO2],color = 'indigo',label = 'CO$_2$: r$^2$ = ' + '{0:.2f}'.format(mODR_r2_CO2)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_CO2))
ax[1].scatter(Brumer_Zappa_GRL_2021.k660DMS[index_DMS],DMS_model_ODR[index_DMS],color = 'crimson',marker='s',label = 'DMS: r$^2$ = '+ '{0:.2f}'.format(mODR_r2_DMS)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_DMS))
ax[1].set_ylabel('k$_{\Lambda 660}$ v manuscript [cm hr$^{-1}$] ')
ax[1].set_xlabel('measured k$_{660}$ [cm hr$^{-1}$]')


ax[1].legend(title='HiWINGS all: r$^2$ = ' + '{0:.2f}'.format(mODR_r2_all)+'; rmse = ' + '{0:.2f}'.format(mODR_rmse_all))

ax[0].legend(title='HiWINGS all: r$^2$ = ' + '{0:.2f}'.format(mODR2_r2_all)+'; rmse = ' + '{0:.2f}'.format(mODR2_rmse_all))
ax[0].annotate('(a)', xy=(-0.12, 1.05), xycoords="axes fraction")
ax[1].annotate('(b)', xy=(-0.12, 1.05), xycoords="axes fraction")
plt.savefig('/HiWinGS/framework_fig_vresponse.png')

#%% SI Figure S1 - Deike scaling 
g = 9.81

c = np.tile(np.arange(0,19,0.2),(len(Brumer_Zappa_GRL_2021.Hs.values),1))
c2 = 0.85*np.sqrt(g*Brumer_Zappa_GRL_2021.Hs.values)
c[[c[i,:]<c2[i]for i in range(len(c2))]] = np.nan
lambda_deike = np.empty_like(c)
for i in range(len(c2)):
    lambda_deike[i,:] = np.sqrt(g*Brumer_Zappa_GRL_2021.Hs.values[i])**(-3)*g*0.25*((c[i,:]/np.sqrt(g*Brumer_Zappa_GRL_2021.Hs.values[i]))**(-6))*(Brumer_Zappa_GRL_2021.usr.values[i]/np.sqrt(g*Brumer_Zappa_GRL_2021.Hs.values[i]))**(5/3)
    
Y_all = np.digitize(HW_wave_age,cp_star_bins)
fig,ax = plt.subplots(1,2,figsize=(13,9))

ll_deike = np.empty((len(cp_star_bins),np.size(c,1)))
for i,wa in enumerate(np.unique(Y_all)):
    ll_deike[i,:] = np.nanmean(lambda_deike[Y_all==wa,:],0);
    ax[0].plot(c[0,:],ll_deike[i,:],color=cmap(wa),
               label=('' if i==0 else '_') + 'Deike Scaling for HiWinGS',
               linewidth = 1,linestyle = '--')
    ax[1].plot(c[0,:],ll_deike[i,:]*c[0,:]**5,color=cmap(wa),
               label=('' if i==0 else '_') + 'Deike Scaling for HiWinGS',
               linewidth = 1,linestyle = '--')


for i,wa in enumerate(np.unique(Y)):
    h = ax[0].plot(ss[i,:],ll[i,:],color=cmap(wa),
                   label=('' if i==0 else '_') + 'HiWinGS',linewidth = 2);
    h = ax[1].plot(ss[i,:],ll[i,:]*ss[i,:]**5,color=cmap(wa),linewidth = 2);

norm= matplotlib.colors.BoundaryNorm(cp_star_bins+np.diff(cp_star_bins)[0]/2, n)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
for axs in ax:
    axs.set_xscale('log')
    axs.set_yscale('log')
 #   axs.grid(True,which="both")
    axs.grid(which='both', axis='both', linestyle='--')
    axs.minorticks_on()
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    axs.yaxis.set_minor_locator(locmin)
    axs.set_xlabel('$c$ [m s$^{-1}$]')
    axs.set_ylim([10**-8,10])
    axs.set_xlim([0.1,50])

ax[0].set_ylabel('$\Lambda(c)$ [m$^{-2}$ s]')
ax[1].set_ylabel('$c^5 \Lambda(c)$ [m$^{3}$ s$^{-4}$]')
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
clb = fig.colorbar(sm, ticks=cp_star_bins[::2], cax=cbar_ax)
clb.ax.set_title('$c_p/u_*$')
leg = ax[0].legend(loc = 1,fontsize = 'small',borderpad = 0.1,handlelength=1.2,handletextpad=0.3)
for handels in leg.legendHandles[:-1]:
    handels.set_color('k')
    
ax[0].annotate('(a)', xy=(-0.18, 1.01), xycoords="axes fraction")
ax[1].annotate('(b)', xy=(-0.18, 1.01), xycoords="axes fraction")
plt.savefig('lambda_deike_scaling.png')
#%% COAREG Fa
RH_w = Brumer_Zappa_GRL_2021.usr*Brumer_Zappa_GRL_2021.Hs/Brumer_Zappa_GRL_2021.nu_w;
f = 4.48*10**(-6)*RH_w**0.90/100
Fa_COAREG = 2450*f
Fa_COAREG / Brumer_Zappa_GRL_2021["Fa_eq19"]