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

    v = v.astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    theta = np.sqrt(2*sc.k*T/m_s)

    vdf = n*np.pi**(-0.5)*theta**(-1)*(k)**(-0.5)*(scipy.special.gamma(k+1.5)/scipy.special.gamma(k+1))*(1+v**2/(theta**2*(k)))**(-k-1.5)

    return vdf

def vdf_dM(n1,n2,T1,T2,v,species="proton"):

    v = v.astype(float)

    vdf = vdf_M(n1,T1,v,species)+vdf_M(n2,T2,v,species)

    return vdf

def vdf_brM(n1,n2,T1,T2,v0,v,species="proton"):

    v = v.astype(float)

    res1 = vdf_M(n2,T2,v[v<=-v0],species)
    res2 = vdf_M(n1,T1,v[(v>-v0)&(v<v0)],species)
    res3 = vdf_M(n2,T2,v[v>=v0],species)

    res2 /= (res1[-1]/res2[0])

    return np.concatenate((res1,res2,res3))

def generate_vdf(n,T,species="proton",type="maxwell",k=0.5,n2=0,T2=1):

    v = np.arange(-1000000,1000001,10).astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    vdf = {"maxwell":vdf_M(n,T,v,species),"kappa":vdf_k(n,T,v,k,species),"double_maxwell":vdf_dM(n,n2,T,T2,v,species)}[type]

    return [v,vdf]

def dm_fitter(xdata,a1,a2,a3,a4):

    return vdf_dM(a1,a2,a3,a4,xdata,species="proton")

def brm_fitter(xdata,a1,a2,a3,a4,a5):

    return vdf_brM(a1,a2,a3,a4,a5,xdata,species="proton")    

def fit_dmk(n,T,k=0.5,species="proton"):

    v = np.arange(-1000000,1000001,10).astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    kappa_vdf = vdf_k(n,T,v,k,species)

    popt,pcov = scipy.optimize.curve_fit(brm_fitter,v,kappa_vdf,p0=[n,n,T,T,50000])

    dm_vdf = dm_fitter(v,popt[0],popt[1],popt[2],popt[3])

    return [k,popt,v,kappa_vdf,dm_vdf]