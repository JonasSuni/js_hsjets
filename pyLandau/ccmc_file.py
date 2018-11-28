import numpy as np
import scipy.constants as sc
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class ExoReader:

    def __init__(self,filename):

        self.filename = filename

        self.ccmcobj = pd.read_csv(filename,sep=" ",header=None,skiprows=6,skipinitialspace=True)
        self.ccmc_arr = self.ccmcobj.as_matrix()
        with open(filename) as f:
            r = f.read()
            varnames = r.split("\n")[4]
            varunits = r.split("\n")[5]
        self.varnames = varnames.split(" ")[1::]
        self.varunits = varunits.replace("[]","").split(" ")[1::]

    def read(self,varname):

        if varname in self.varnames:
            ind = self.varnames.index(varname)
        else:
            print("Variable name nonexistent")
            return None

        output = ExoVariable(varname=varname,varunit=self.varunits[ind],r_unit=self.varunits[0],r_data=self.ccmc_arr[:,0],var_data=self.ccmc_arr[:,ind])

        return output

class ExoVariable:

    def __init__(self,varname,varunit,r_unit,r_data,var_data):

        self.label_dict = {"R":"R [$R_s$]","G_p":"Total normalized proton potential","PHI":"$\\phi$ [$V$]","N_e":"n$_e$ [$cm^{-3}$]","N_p":"n$_p$ [$cm^{-3}$]","F_e":"j$_e$ [$10^8 cm^{-2} s^{-1}$]","F_p":"j$_p$ [$10^8 cm^{-2} s^{-1}$]","V_e":"V$_e$ [$km s^{-1}$]","V_p":"V$_p$ [$km s^{-1}$]","T_e_par":"T$_{e\\parallel}$ [$K$]","T_e_perp":"T$_{e\\perp}$ [$K$]","T_p_par":"T$_{p\\parallel}$ [$K$]","T_p_perp":"T$_{p\\perp}$ [$K$]","T_e":"T$_e$ [$K$]","T_p":"T$_p$ [$K$]","T_e_ani":"T$_{e\\parallel}$ / T$_{e\\perp}$","T_p_ani":"T$_{p\\parallel}$ / T$_{p\\perp}$","Q_e":"q$_e$ [$J m^{-2} s^{-1}$]","Q_p":"q$_p$ [$J m^{-2} s^{-1}$]"}

        self.name = varname
        self.unit = varunit
        self.r_unit = r_unit
        self.r = r_data
        self.data = var_data
        self.label = self.label_dict[varname]
        self.r_label = self.label_dict["R"]

def plot_variable(filename,varname,rmin=0,log=None):

    var = ExoReader(filename).read(varname)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(var.r_label,labelpad=10,fontsize=20)
    ax.set_ylabel(var.label,labelpad=10,fontsize=20)
    ax.tick_params(labelsize=20)

    r_data = var.r[var.r>=rmin]
    var_data = var.data[var.r>=rmin]

    ax.set_xlim(rmin,max(r_data))

    ax.plot(r_data,var_data)

    if not not log:
        plt.yscale("log")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if not log:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune="lower"))

    plt.tight_layout()
    plt.grid()
    plt.show()

    return None