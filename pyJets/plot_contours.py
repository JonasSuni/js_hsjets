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

  B = exprmaps["B"]

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

  rho = curr_maps["rho"]
  v = curr_maps["v"]

  # calculate dynamic pressure
  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  # pdyn in nanopascals
  pdyn /= 1.0e-9

  return pdyn

def expr_coreheating(exprmaps):

    T_sw = 0.5e+6

    r_rhonbs = exprmaps["RhoNonBackstream"]
    pr_PTDNBS = exprmaps["PTensorNonBackstreamDiagonal"]

    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    return pr_TNBS/T_sw

def expr_pdyn(exprmaps):
  # exprmaps is ["rho","v"]
  # returns dynamic pressure in nanopascals

  rho = exprmaps["rho"]
  v = exprmaps["v"]

  # calculate dynamic pressure
  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  # pdyn in nanopascals
  pdyn /= 1.0e-9

  return pdyn

def expr_srho(exprmaps):
  # exprmaps is ["rho","CellID"]
  # returns number density in cm^-3

  rho = exprmaps["rho"]
  cellids = exprmaps["CellID"]

  # rho in cm^-3
  srho = rho/1.0e+6

  return srho

###PLOTTERS###
