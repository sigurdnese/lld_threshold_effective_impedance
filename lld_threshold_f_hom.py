"""
Plots LLD threshold from MELODY as a function of f_hom (hom contribution fraction)
and compares to effective impedance calculation.
MELODY results obtained from '../LHC/results_threshold_bbr_eff_imp/n_m97'
"""
import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15})
import mpmath
import time
from effective_impedance_funcs import *
from effective_impedance_params import *

Q_2 = 200
f_r2 = 0.2*f_r 

# Obtain MELODY results
files = np.array(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_ImZn0.07_kmax20.0_nrh10.0_phimax1.30_exact_mu2.0_VRF6.0MV_Q_hom{Q_2}_nr_hom{int(f_r2/f_0)}_f_hom*_threshold.txt'))

melody_f_hom = np.array([])
melody_xi_down = np.array([])
melody_xi_up = np.array([])

for f in files:
    melody_f_hom = np.append(melody_f_hom,float(f.split('f_hom',1)[1].split('_threshold',1)[0]))
    table = np.loadtxt(f)
    melody_xi_down = np.append(melody_xi_down,table[1,0])
    melody_xi_up = np.append(melody_xi_up,table[0,0])

melody_xi_down = melody_xi_down[melody_f_hom.argsort()]
melody_xi_up = melody_xi_up[melody_f_hom.argsort()]
melody_f_hom.sort()

melody_xi = np.mean([melody_xi_down,melody_xi_up],axis=0)

plt.scatter(melody_f_hom,melody_xi,label='Melody')

xi_threshold_fixed = []
xi_threshold_adaptive = []
for f_hom in melody_f_hom:
    print(f'f_hom={f_hom}')
    for adaptive_k_eff in [True,False]:
        k_eff_bbr = int(f_r/f_0)
        if adaptive_k_eff:
            k_top = int(1.1*f_r/f_0) # Slightly wider range allowing finding of zero crossing at f_r
            pcolor = 'black'
            plabel = 'Adaptive $k_{eff}$'
        else:
            k_top = k_eff_bbr
            pcolor = 'red'
            plabel = 'Fixed $k_{eff}$'
        
        k_full = np.arange(-k_top,k_top+1,1)
        k_full = np.delete(k_full,np.argwhere(k_full==0))
        
        k_half_sampled = np.arange(1,k_top+1,10)

        # Choose k list
        k = k_half_sampled

        # Set up Z_k
        broadband_resonator_impedance = resonator_impedance(k,1,f_r)
        xi_threshold_array = []
        effective_impedance_array = []
        impedance_model = np.imag(broadband_resonator_impedance+f_hom*resonator_impedance(k,Q_2,f_r2))
        if adaptive_k_eff:
            # -- k_eff at last zero crossing of impedance model --
            zero_crossing_index = np.where(np.diff(np.sign(impedance_model/k)))[0][-1]
            k_bool = np.array(np.abs(k)<k[zero_crossing_index])
            k_max = k[zero_crossing_index]
        else:
            k_bool = np.array([True])
            k_max = k_full[-1]
        effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,k_bool))
        print(f'(ImZ/k)_eff = {effective_impedance}')

        y_max = k_max*phi_max/h
        xi_threshold = get_xi_threshold(mu,y_max,effective_impedance,phi_max)
        # plt.axhline(xi_threshold,label=plabel,color=pcolor)
        if adaptive_k_eff:
            xi_threshold_adaptive.append(xi_threshold)
        else:
            xi_threshold_fixed.append(xi_threshold)

plt.scatter(melody_f_hom,xi_threshold_adaptive,color='black',label='Adaptive $k_{eff}$')
plt.scatter(melody_f_hom,xi_threshold_fixed,color='red',label='Fixed $k_{eff}$')

plt.xlabel('f_hom')
plt.ylabel('$\zeta_\\mathrm{th}$')
plt.title(f'$Q_2=${Q_2}, $f_{{r,2}}/f_r=${f_r2/f_r}')
plt.legend()
plt.tight_layout()
plt.show()