"""
Plots impedance model and cumulative sum corresponding to numerator of effective impedance
"""
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13})
import mpmath
import time
from effective_impedance_funcs import *
from effective_impedance_params import *
T = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M_%S', T)
start_time = time.time()

k_eff = int(f_r/f_0)

k_full = np.arange(-k_eff,k_eff+1,1)
k_full = np.delete(k_full,np.argwhere(k_full==0))

k_sampled = np.arange(-4*k_eff,4*k_eff+1,100)
k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

k_half = np.arange(1,1*k_eff+1,1)

# Accurate in effective impedance to 5 decimals, at Q_2 = 1 and 1000
k_half_sampled = np.arange(1,1.5*k_eff+1,10)

# Choose k list
k = k_half_sampled
y = k*phi_max/h

# Set up impedance model
Q = 1
hom = 1
Q_2_list = [200]
f_r2_list = f_r*np.array([0.15,0.2,0.3])

broadband_resonator_impedance = resonator_impedance(k,Q,f_r)

fig, ax = plt.subplots(1,2,figsize=(10,4),sharex=False)
cmap = plt.get_cmap("tab10")
for j in range(len(f_r2_list)):
    for i in range(len(Q_2_list)):
        impedance_model = np.imag(broadband_resonator_impedance+hom*resonator_impedance(k,Q_2_list[i],f_r2_list[j]))

        impedance_model_k = impedance_model/k
        ax[0].plot(k,impedance_model_k,label=f'$f_{{r,2}}/f_r={f_r2_list[j]/f_r}$',color=cmap(j))
        # ax[0].axvline(-k_eff,linestyle='--',color='red')
        ax[0].axvline(k_eff,linestyle='--',color='red')
        ax[0].set_ylabel("$Im(Z_k/k)\;[\Omega]$")
        ax[0].set_xlabel("k")
        ax[0].set_title(f'$Q_2={Q_2_list[i]}$')
        ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        n_cumsum = np.imag(numerator_cumsum(impedance_model,mu,phi_max,k))
        zero_crossing_index = np.where(np.diff(np.sign(impedance_model_k)))[0][-1]
        k_max = k[zero_crossing_index]
        ax[1].plot(k,n_cumsum,label=f'$f_{{r,2}}/f_r={f_r2_list[j]/f_r}$',color=cmap(j))
        ax[1].axvline(k_eff,linestyle='--',color='red')
        ax[1].plot(k_max,n_cumsum[zero_crossing_index],'o',color='black')
        ax[1].set_xlabel('$k$')
        ax[1].set_ylabel('$\sum_{{k\'=0}}^k Im(Z_{{k\'}}/k\') G_{k\'k\'}$')
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # G_kk_array = [G_kk(mu,yy,phi_max) for yy in y]
        # ax[2].plot(k,np.imag(G_kk_array))
        # ax[2].axvline(-k_eff,linestyle='--',color='red')
        # ax[2].axvline(k_eff,linestyle='--',color='red')
        # ax[2].set_xlabel('$k$')
        # ax[2].set_ylabel('$G_{kk}$')
        # ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))


# fig.suptitle(f'$Q={Q_2_list},\; f_{{r,2}}/f_r={f_r2_list/f_r},\; \mu={mu},\; hom={hom}$')
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(f'Figures/Z_eff_multiplot_{timestamp}.png',dpi=300)
plt.show()