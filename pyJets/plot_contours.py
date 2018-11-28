import pytools as pt
import numpy as np
import jetfile_make as jfm
import jet_contours as jc
import pandas as pd
import matplotlib.pyplot as plt
import os

m_p = 1.672621898e-27
r_e = 6.371e+6

###EXTERNAL FUNCTIONS###



###EXPRESSIONS###

def expr_cone_angle(exprmaps):

  B = exprmaps[0]

  Bx = B[:,:,0]
  Bmag = np.linalg.norm(B,axis=-1)

  cone_ang = np.rad2deg(np.arccos(np.divide(Bx,Bmag)))

  return cone_ang

def expr_pdyn_gen(exprmaps):
  # for use with cust_contour
  # exprmaps is ["rho","v"]
  # returns dynamic pressure in nanopascals

  # find variables for the time step to be plotted
  timewidth = len(exprmaps)
  curr_step = (timewidth-1)/2
  curr_maps = exprmaps[curr_step]

  rho = curr_maps[0]
  v = curr_maps[1]

  # calculate dynamic pressure
  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  # pdyn in nanopascals
  pdyn /= 1.0e-9

  return pdyn

def expr_pdyn(exprmaps):
  # exprmaps is ["rho","v"]
  # returns dynamic pressure in nanopascals

  rho = exprmaps[0]
  v = exprmaps[1]

  # calculate dynamic pressure
  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  # pdyn in nanopascals
  pdyn /= 1.0e-9

  return pdyn

def expr_srho(exprmaps):
  # exprmaps is ["rho","CellID"]
  # returns number density in cm^-3

  rho = exprmaps[0][:,:]
  cellids = exprmaps[1][:,:]

  # rho in cm^-3
  srho = rho/1.0e+6

  return srho
    
###PLOTTERS###
