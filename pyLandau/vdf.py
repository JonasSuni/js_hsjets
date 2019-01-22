import numpy as np
import scipy.constants as sc
import scipy.integrate as integrate
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,AutoLocator,LogLocator

def vdf_M(n,T,v,species="proton"):

    #v = v.astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    vdf = n*np.exp(-m_s*(v**2)/(2*sc.k*T))*(m_s/(2*np.pi*sc.k*T))**0.5

    return vdf

def vdf_k(n,T,v,k=2,species="proton"):

    #v = v.astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    #theta = np.sqrt(2*sc.k*T*(k-1.5)/(m_s*k))
    theta = np.sqrt(2*sc.k*T/m_s)

    vdf = n*(np.pi*k)**(-0.5)*(theta)**(-1)*(scipy.special.gamma(k)/scipy.special.gamma(k-0.5))*(1+v**2/(k*theta**2))**(-k)

    return vdf

def vdf_dM(n1,n2,T1,T2,v,species="proton"):

    v = v.astype(float)

    vdf = vdf_M(abs(n1),T1,v,species)+vdf_M(abs(n1),T2,v,species)
    vdf *= (float(abs(n1))/(abs(n1)+abs(n1)))

    return vdf

def vdf_dkM(n1,n2,T1,T2,k,v0,v,species="proton"):

    v = v.astype(float)

    res1 = vdf_k(n2,T2,v[v<=-v0],k,species)
    res2 = vdf_M(n1,T1,v[(v>-v0)&(v<v0)],species)
    res3 = vdf_k(n2,T2,v[v>=v0],k,species)

    vdf = np.concatenate((res1,res2,res3))

    return vdf

def vdf_brM(n1,n2,T1,T2,v0,v,species="proton"):

    v = v.astype(float)

    res1 = vdf_M(n2,T2,v[v<=-v0],species)
    res2 = vdf_M(n1,T1,v[(v>-v0)&(v<v0)],species)
    res3 = vdf_M(n2,T2,v[v>=v0],species)

    rest = np.concatenate((res1,res2,res3))

    return rest

def vdf_2d_m(Tpar,Tperp,vpar,vperp):

    m_s = sc.m_p

    theta_par = np.sqrt(2*sc.k*Tpar/m_s)
    theta_perp = np.sqrt(2*sc.k*Tperp/m_s)

    vdf = np.exp(-(vpar**2)/(theta_par**2)-(vperp**2)/(theta_perp**2))*(np.pi**(-1.5))*(theta_par*(theta_perp**2))**(-1)

    return vdf

def vdf_2d_k(Tpar,Tperp,vpar,vperp,k):

    m_s = sc.m_p

    #theta_par = np.sqrt(2*sc.k*Tpar*(k-1.5)/(m_s*k))
    #theta_perp = np.sqrt(2*sc.k*Tperp*(k-1.5)/(m_s*k))
    theta_par = np.sqrt(2*sc.k*Tpar/m_s)
    theta_perp = np.sqrt(2*sc.k*Tperp/m_s)

    vdf = (np.pi*k)**(-1.5)*(theta_par*theta_perp**2)**(-1)*(scipy.special.gamma(k+1)/scipy.special.gamma(k-0.5))*(1+vpar**2/(k*theta_par**2)+vperp**2/(k*theta_perp**2))**(-k-1)

    return vdf

def generate_2d_vdf(Tpar,Tperp,vmax,vstep,k):

    v = np.arange(-vmax,vmax+1,vstep).astype(float)
    v = v*np.abs(v)

    vpar,vperp = scipy.meshgrid(v,v)

    vdfm = vdf_2d_m(Tpar,Tperp,vpar,vperp)
    vdfk = vdf_2d_k(Tpar,Tperp,vpar,vperp,k)

    return [vpar,vperp,vdfm,vdfk,vpar[0]]

def calc_2d_moms(Tpar,Tperp,vmax,vstep,k):

    vpar,vperp,vdfm,vdfk,v_vec = generate_2d_vdf(Tpar,Tperp,vmax,vstep,k)

    mm_0 = np.trapz(np.pi*np.abs(v_vec)*np.trapz(vdfm,axis=1,x=v_vec),x=v_vec)
    mk_0 = np.trapz(np.pi*np.abs(v_vec)*np.trapz(vdfk,axis=1,x=v_vec),x=v_vec)

    mm_2_par = np.trapz(np.pi*np.abs(v_vec)*np.trapz(v_vec*v_vec*vdfm,axis=1,x=v_vec),x=v_vec)*sc.m_p/sc.k
    mk_2_par = np.trapz(np.pi*np.abs(v_vec)*np.trapz(v_vec*v_vec*vdfk,axis=1,x=v_vec),x=v_vec)*sc.m_p/sc.k

    mm_2_perp = np.trapz(np.pi*v_vec*v_vec*np.abs(v_vec)*np.trapz(vdfm,axis=1,x=v_vec),x=v_vec)*sc.m_p/sc.k/2
    mk_2_perp = np.trapz(np.pi*v_vec*v_vec*np.abs(v_vec)*np.trapz(vdfk,axis=1,x=v_vec),x=v_vec)*sc.m_p/sc.k/2

    return np.array([[mm_0,mk_0],[mm_2_par,mk_2_par],[mm_2_perp,mk_2_perp]])


def generate_vdf(n,T,species="proton",k=0.5):

    v = np.arange(-200000+50,200000,100).astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    vdfm = vdf_M(n,T,v,species)
    vdfk = vdf_k(n,T,v,k,species)
    #{"maxwell":vdf_M(n,T,v,species),"kappa":vdf_k(n,T,v,k,species),"double_maxwell":vdf_dM(n,n2,T,T2,v,species)}[type]

    return [v,vdfm,vdfk]

def dm_fitter(xdata,a1,a2,a3,a4):

    return vdf_dM(a1,a2,a3,a4,xdata,species="proton")

def brm_fitter(xdata,a1,a2,a3,a4,a5):

    return vdf_brM(a1,a2,a3,a4,a5,xdata,species="proton")

def dkm_fitter(xdata,a1,a2,a3,a4,a5,a6):

    return vdf_dkM(a1,a2,a3,a4,a5,a6,xdata,species="proton")    

def fit_dmk(n,T,k=0.5,species="proton"):

    v = np.arange(-1000000+50,1000001,100).astype(float)

    m_s = {"proton":sc.m_p,"electron":sc.m_e}[species]

    kappa_vdf = vdf_k(n,T,v,k,species)

    popt,pcov = scipy.optimize.curve_fit(dm_fitter,v,kappa_vdf,p0=[n,n,T,T])

    dm_vdf = dm_fitter(v,popt[0],popt[1],popt[2],popt[3])
    #dm_vdf = brm_fitter(v,popt[0],popt[1],popt[2],popt[3],popt[4])

    return [k,popt,v,kappa_vdf,dm_vdf]