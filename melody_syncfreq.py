"""
Plots the synchrotron frequency distribution and its derivative at LLD threshold calculated by MELODY.
Obtains MELODY results from '../LHC/results_threshold_bbr_eff_imp/n_m97'
Also plots and compares to case with no intensity effects. This data is produced by running 
'synchrotronfreq_stationary.py'.
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
plt.rcParams.update({'font.size':15})
c = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=c)
from effective_impedance_params import *

f_r2_factor_array = np.array([0.005,0.2,0.32,0.5,0.7])
all_freqs = False # If False, only use f_r2/f_r in f_r2_factor_array
Q_2 = 200

# Get data for no intensity effects
noint_npzfile = np.load('fs_E_no_int_eff.npz')
fs_fs0_noint = noint_npzfile['fs_noint_arr']
fs_fs0_der_noint = noint_npzfile['fs_noint_der_arr']
E_noint = noint_npzfile['e_noint_arr']
# Interpolate in order to compare at E given by MELODY
interp_der_noint = interpolate.interp1d(E_noint,fs_fs0_der_noint)

# Get data for bbr only
bbr_npzfile = np.load('../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_kmax_20.0_nrh_10.0_int5.65e+10_phimax1.3_exact_mu2.0_VRF6.0MV_Q_hom10_nr_hom178200_f_hom0/fs_e_arr.npz')
E_bbr = bbr_npzfile['e_arr']
fs_fs0_bbr = bbr_npzfile['fs_arr']
fs_fs0_der_bbr = np.gradient(fs_fs0_bbr,E_bbr)
# Interpolate in order to compare at E given by MELODY
interp_der_bbr = interpolate.interp1d(E_bbr,fs_fs0_der_bbr)

files = np.array(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_ImZn0.07_kmax20.0_nrh10.0_phimax1.30_exact_mu2.0_VRF6.0MV_Q_hom{Q_2}_nr_hom*_f_hom1.0_all_data.txt'))
nr_hom_array = np.array([int(file.split('nr_hom',1)[1].split('_f_hom',1)[0]) for file in files])
nr_hom_array.sort()

inset = False
fig, ax = plt.subplots(1,3,sharex=True,figsize=(15,5))
axins = ax[0].inset_axes([0.03,0.03,0.4,0.4])
x1, x2, y1, y2 = -0.005, 0.05, 0.977, 0.99
axins.set_xticklabels('')
axins.set_yticklabels('')

for nr_hom in nr_hom_array:
    f_r2_factor = nr_hom*f_0/f_r
    if all_freqs:
        folder = max(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_kmax_20.0_nrh_10.0_*_phimax1.3_exact_mu2.0_VRF6.0MV_Q_hom{Q_2}_nr_hom{nr_hom}_f_hom1.0'),key=os.path.getctime)
        npzfile = np.load(folder+'/fs_e_arr.npz')
        E = npzfile['e_arr']
        fs_fs0 = npzfile['fs_arr']
        fs_fs0_der = np.gradient(fs_fs0,E)
        fs_fs0_der_error = (fs_fs0_der+1/8)/(fs_fs0_der)
        fs_fs0_der_error_noint = (interp_der_noint(E)-fs_fs0_der)/interp_der_noint(E) # Error wrt no intensity effects
        fs_fs0_der_error_bbr = (interp_der_bbr(E)-fs_fs0_der)/interp_der_bbr(E) # Error wrt BBR
    
        ax[0].plot(E,fs_fs0,label=f'$f_{{r,2}}={f_r2_factor:1.3f}f_r$')        
        axins.plot(E,fs_fs0)
        ax[1].plot(E,fs_fs0_der)
        # ax[2].plot(E,fs_fs0_der_error)
        # ax[2].plot(E,fs_fs0_der_error_noint)
        ax[2].plot(E,fs_fs0_der_error_bbr)
    else:
        if np.any(f_r2_factor_array==np.round(f_r2_factor,3)):
            folder = max(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_kmax_20.0_nrh_10.0_*_phimax1.3_exact_mu2.0_VRF6.0MV_Q_hom{Q_2}_nr_hom{nr_hom}_f_hom1.0'),key=os.path.getctime)
            npzfile = np.load(folder+'/fs_e_arr.npz')
            E = npzfile['e_arr']
            fs_fs0 = npzfile['fs_arr']
            fs_fs0_der = np.gradient(fs_fs0,E)
            fs_fs0_der_error = (fs_fs0_der+1/8)/(fs_fs0_der)   
            fs_fs0_der_error_noint = (interp_der_noint(E)-fs_fs0_der)/interp_der_noint(E) # Error wrt no intensity effects  
            fs_fs0_der_error_bbr = (interp_der_bbr(E[:-1])-fs_fs0_der[:-1])/interp_der_bbr(E[:-1]) # Error wrt BBR    

            ax[0].plot(E,fs_fs0,label=f'$f_{{r,2}}={f_r2_factor:1.3f}f_r$')
            axins.plot(E,fs_fs0)
            ax[1].plot(E,fs_fs0_der)
            # ax[2].plot(E,fs_fs0_der_error)
            # ax[2].plot(E,fs_fs0_der_error_noint)
            ax[2].plot(E[:-1],fs_fs0_der_error_bbr)


ax[0].plot(E_noint,fs_fs0_noint,'--',color='grey')
ax[1].plot(E_noint,fs_fs0_der_noint,'--',color='grey',label='Without intensity effects')

ax[0].plot(E_bbr,fs_fs0_bbr,'--',color='lime')
ax[1].plot(E_bbr,fs_fs0_der_bbr,'--',color='lime',label='BBR only')

ax[0].set_xlim(right = np.max(E))
ax[0].set_xlabel('$\mathcal{E}$')
ax[0].set_ylabel('$f_s/f_{s0}$')
ax[0].legend(handletextpad=0.2,frameon=True,labelspacing=0.1)

ax[1].axhline(-1/8,linestyle='--',color='black',label='-1/8')
ax[1].set_xlabel('$\mathcal{E}$')
ax[1].set_ylabel('$d(f_s/f_{s0})/d\mathcal{E}$')
ax[1].legend()

ax[2].set_xlabel('$\mathcal{E}$')
# ax[2].set_ylabel('Relative error of $-1/8$ from $d(f_s/f_{s0})/d\mathcal{E}$')
# ax[2].set_ylabel('Relative error wrt. no intensity effects')
ax[2].set_ylabel('Relative error wrt. BBR')
ax[2].grid()

axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)

if inset:
    ax.indicate_inset_zoom(axins, edgecolor='black')
else:
    axins.remove()

ax[1].set_title(f'$Q_2={Q_2}$')
plt.tight_layout()
plt.show()