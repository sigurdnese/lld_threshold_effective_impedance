"""
Functions for calculating the LLD threshold using effective impedance
"""
import numpy as np
import matplotlib.pyplot as plt
import mpmath
from scipy.special import jv
from effective_impedance_params import *

def resonator_impedance(K,quality,resonance_freq):
    shunt = ImZ_over_k*quality*resonance_freq/f_0
    if resonance_freq != 0:
        return shunt/(1+1j*(quality*((K*f_0)/(resonance_freq)-(resonance_freq)/(K*f_0))))
    else:
        # Handle apparent singularity
        return np.zeros_like(K)

def constant_inductive(K):
    return 1j*K*ImZ_over_k

def hg1f2(Mu,Y):
    """
    Generalized hypergeometric function.
    Takes single number Y.
    """
    return float(mpmath.hyp1f2(0.5,2,Mu,-Y**2))

def hg2f3(Mu,Y):
    """
    Generalized hypergeometric function.
    Takes single number Y.
    """
    return float(mpmath.hyp2f3(0.5,0.5,1.5,2,Mu,-Y**2))

def G_kk(Mu,Y,Phi_max):
    """
    Calculates the matrix element G_kk in the cases where Bessel functions cannot be used.
    """
    return np.complex64(1j*((16*Mu*(Mu+1))/(np.pi*Phi_max**4)*(1-hg1f2(Mu,Y))))

def make_G_diag(Mu,Y,Phi_max):
    """
    Gives the diagnoal matrix elements G_kk as an array.
    Uses Bessel functions for special values of Mu.
    """
    if Mu == 0.5:
        print('mu=0.5, using Bessel functions')
        G_diag = np.complex64(1j*((16*Mu*(Mu+1))/(np.pi*Phi_max**4)*(1-jv(1,2*Y)/Y)))
    elif Mu == 2:
        print('mu=2, using Bessel functions')
        G_diag = np.complex64(1j*((16*Mu*(Mu+1))/(np.pi*Phi_max**4)*(0.5-(jv(0,Y))**2-(jv(1,Y))**2+jv(0,Y)*jv(1,Y)/Y)))
    else:
        G_diag = []
        step = 0
        total = len(Y)
        for yy in Y:
            step += 1
            if step%1000 == 0:
                print(f'{step} of {total}',end='\r')
            G_diag.append(G_kk(Mu,yy,Phi_max))
        print()
        G_diag = np.array(G_diag)
    return G_diag

def get_effective_impedance(imag_Z_k,Mu,Phi_max,K,K_bool=np.array([True]),G_diag=[]):
    """
    Calculates the effective impedance given an impedance model imag_Z_k = Im(Z_k)
    and a corresponding range K. K_bool is an array of bools determining which values are included in the sum.
    Also works for a grid of K arrays/impedances.
    """
    Y = K*Phi_max/h
    # G_kk is pure imaginary. i will cancel, so take imag to save memory
    if len(G_diag)==0:
        G_diag = np.imag(make_G_diag(Mu,Y,Phi_max))
    # Sum over last axis, works for impedance grid and single list
    numerator = np.sum(G_diag*imag_Z_k/K,axis=-1,where=K_bool)
    if len(imag_Z_k.shape)>1:
        G_diag_grid = np.array([[G_diag for i in range(imag_Z_k.shape[0])] for j in range(imag_Z_k.shape[1])])
        denominator = np.sum(G_diag_grid,axis=-1,where=K_bool)
    else:
        denominator = np.sum(G_diag,axis=-1,where=K_bool)
    return numerator/denominator

def numerator_cumsum(imag_Z_k,Mu,Phi_max,K,G_diag=[]):
    """
    Calculates the cumulative sum corresponding to the numerator in the effective impedance.
    """
    # Creates G_kk array, takes impedance model array
    Y = K*Phi_max/h
    if len(G_diag)==0:
        G_diag = make_G_diag(Mu,Y,Phi_max)
    return np.cumsum(G_diag*imag_Z_k/K)

def chi(Mu, Y):
    """
    The function chi(mu,y) appearing in the LLD intensity threshold formula.
    """
    return Y*(1-hg2f3(Mu,Y))

def get_xi_threshold(Mu,Y_max,Effective_Impedance,Phi_max):
    """
    Calculates the intensity parameter at the LLD threshold using effective impedance.
    """
    return (np.pi*Phi_max**5*ImZ_over_k)/(32*Mu*(Mu+1)*chi(Mu,Y_max)*Effective_Impedance)

