import numpy as np
import scipy.constants as sc
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt

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
        self.varunits = varunits.replace("[]","").split(" ")

    def read(varname):

        if varname in varnames:
            ind = self.varnames.index(varname)
        else:
            print("Variable name nonexistent")
            return None

        output = ExoVariable(varname=varname,varunit=varunits[ind],r_unit=varunits[0],r_data=ccmc_arr[:,0],var_data=ccmc_arr[:,ind])

class ExoVariable:

    def __init__(self,varname,varunit,r_unit,r_data,var_data):

        self.name = varname
        self.unit = varunit
        self.r_unit = r_unit
        self.r = r_data
        self.data = var_data