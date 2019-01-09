import numpy as np
import scipy.constants as sc
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,AutoLocator,LogLocator

def vdf_M(n,T,v,species="proton"):

    v = v.astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    vdf = n*np.exp(-m_s*(v**2)/(2*sc.k*T))*(m_s/(2*np.pi*sc.k*T))**0.5

    return vdf

def vdf_k(n,T,v,k=0.5,species="proton"):

    #NOT FINAL

    v = v.astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    theta = np.sqrt(2*sc.k*T/m_s)

    vdf = 2*n*np.pi**(-0.5)*theta**(-1)*(k)**(-0.5)*(scipy.special.gamma(k+1.5)/scipy.special.gamma(k))*(1+v**2/(theta**2*(k)))**(-k-1.5)

    return vdf