"""
Creates contourplot of the effective impedance as a function of Q_2 and f_r,2
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15})
import mpmath
import time
from effective_impedance_funcs import *
from effective_impedance_params import *
T = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M_%S', T)
start_time = time.time()
print("Timer starts now")

k_eff = int(f_r/f_0)

# ---- Calculating effective impedance ----

k_full = np.arange(-k_eff,k_eff+1,1)
k_full = np.delete(k_full,np.argwhere(k_full==0))

k_sampled = np.arange(-k_eff,k_eff+1,10)
k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

k_half = np.arange(1,k_eff+1,1)

# Accurate in effective impedance to 5 decimals, at Q_2 = 1 and 1000
k_half_sampled = np.arange(1,k_eff+1,10)

# Choose k list
k = k_half_sampled
y = k*phi_max/h

# Set up Z_k
broadband_resonator_impedance = resonator_impedance(k,1,f_r)
Q_2 = np.linspace(1,1000,50,endpoint=True)
f_r2 = np.linspace(0,1*f_r,50,endpoint=True)
Q_2_grid, f_r2_grid = np.meshgrid(Q_2,f_r2)

effective_impedance_grid = np.zeros_like(Q_2_grid)
impedance_model_grid = np.zeros((len(Q_2),len(f_r2_grid),len(k)))

for i in range(len(Q_2)):
    for j in range(len(f_r2)):
        impedance_model_grid[i,j] = np.imag(broadband_resonator_impedance+resonator_impedance(k,Q_2[i],f_r2[j]))
        print(f'Creating impedance grid: {i} of {len(Q_2)}   ',end='\r')
impedance_model_grid = impedance_model_grid.swapaxes(0,1)
print('Finished creating impedance model grid.          ')

effective_impedance_grid = np.real_if_close(get_effective_impedance(impedance_model_grid,mu,phi_max,k))
print("Process finished --- %s seconds ---" % (time.time() - start_time))

cnt = plt.contourf(Q_2,f_r2/f_r,effective_impedance_grid,1000)
for c in cnt.collections:
    c.set_edgecolor('face')
plt.xlabel('$Q_2$')
plt.ylabel('$f_{r,2}/f_r$')
# plt.title(f'{len(Q_2)}x{len(f_r2)} grid, $\mu=${mu}')
plt.tight_layout()
plt.colorbar(label='$(ImZ/k)_{eff}$',format=matplotlib.ticker.FormatStrFormatter('%.2f'))
plt.savefig(f'Figures/Zeff_contourplots/effective_impedance_{timestamp}.pdf')
# plt.close()
# plt.show()

# f_r2_slice = 25
# plt.plot(Q_2,effective_impedance_grid[f_r2_slice])
# plt.xlabel('$Q_2$')
# plt.ylabel(f'$(ImZ/k)_{{eff}}(f_{{r,2}}/f_r={f_r2[f_r2_slice]/f_r:.2f})$')
# plt.savefig(f'Figures/Zeff_contourplots/Zeff_f_r2slice{timestamp}.png',dpi=300)
# plt.close()

# Q_2_slice = 30
# plt.plot(f_r2/f_r,effective_impedance_grid[:,Q_2_slice])
# # plt.axvline(0.066,linestyle='--',color='orange')
# plt.xlabel('$f_{r,2}/f_r$')
# plt.ylabel(f'$(ImZ/k)_{{eff}}(Q_2={Q_2[Q_2_slice]:.2f})$')
# plt.title(f'$\mu={mu}$')
# plt.savefig(f'Figures/Zeff_contourplots/Zeff_Q_2slice{timestamp}.png',dpi=300)
# plt.close()

# plt.plot(f_r2/f_r,np.sum(impedance_model_grid[:,Q_2_slice]/k,axis=1))
# plt.xlabel('$f_{r,2}/f_r$')
# plt.ylabel(f'$\sum_k Im(Z_k/k) (Q_2={Q_2[Q_2_slice]:.2f})$')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.savefig(f'Figures/Zsum_Q_2slice_{timestamp}.png',dpi=300)









