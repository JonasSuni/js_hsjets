import numpy as np
import scipy.constants as sc
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,AutoLocator,LogLocator

def vdf_M(n,T,v,species="proton"):

    if species == "proton":
        m_s = sc.m_p
    else:
        m_s = sc.m_e

    vdf = n*(m_s/(2*np.pi*sc.k*T))**(3/2)*exp(-m_s*(v**2)/(2*sc.k*T))