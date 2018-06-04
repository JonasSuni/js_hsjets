import pytools as pt
import numpy as np
import jetfile_make as jfm
import jet_contours as jc
import jet_analyser as ja
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

m_p = 1.672621898e-27
r_e = 6.371e+6

###EXTERNAL FUNCTIONS###

def cust_contour(ax,XmeshXY,YmeshXY,extmaps,ext_pars):
  # extmaps is ["rho","v","X","Y"]

  if type(extmaps[0]) is not list:
    print("Requires pass_times to be larger than 0. Exiting.")
    quit()

  timewidth = len(extmaps)
  curr_step = (timewidth-1)/2
  curr_maps = extmaps[curr_step]

  rho = np.ma.masked_less_equal(curr_maps[0][:,:], 0)
  v = curr_maps[1][:,:,:]
  vx = curr_maps[1][:,:,0]
  pdyn = rho*(np.linalg.norm(v,axis=-1)**2)
  pdynx = rho*(vx**2)

  X = curr_maps[2][:,:]
  Y = curr_maps[3][:,:]

  sw_mask = np.ma.masked_greater(X,16*r_e)
  sw_mask.mask[X < 14*r_e] = True
  sw_mask.mask[Y < -4*r_e] = True
  sw_mask.mask[Y > 4*r_e] = True

  rho_sw = np.mean(np.ma.array(rho,mask=sw_mask.mask).compressed())
  pdyn_sw = np.mean(np.ma.array(pdyn,mask=sw_mask.mask).compressed())

  avgpdyn = np.zeros(np.array(pdyn.shape))

  for i in range(timewidth):
    if i == curr_step:
        continue
    tmaps = extmaps[i]
    tpdyn = tmaps[0]*(np.linalg.norm(tmaps[1],axis=-1)**2)
    avgpdyn = np.add(avgpdyn,tpdyn)

  avgpdyn = np.divide(np.ma.masked_less_equal(avgpdyn,0),np.array([timewidth-1]))
  avgpdyn = np.ma.masked_less_equal(avgpdyn,0)

  Plaschke = pdynx/pdyn_sw
  SWCrit = rho/rho_sw
  ArcherHorbury = np.divide(pdyn,avgpdyn)

  jet = np.ma.masked_greater(Plaschke,0.25)
  jet.mask[SWCrit < 3.5] = False
  jet.mask[ArcherHorbury > 2] = True
  jet.fill_value = 0
  jet[jet.mask == False] = 1

  contour_jet = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=1.0,colors="black")

def expr_pdyn_gen(exprmaps):
  # for use with cust_contour
  # exprmaps is ["rho","v"]
  # returns dynamic pressure in nanopascals

  timewidth = len(exprmaps)
  curr_step = (timewidth-1)/2
  curr_maps = exprmaps[curr_step]

  rho = curr_maps[0]
  v = curr_maps[1]

  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  pdyn /= 1.0e-9

  return pdyn

def expr_pdyn(exprmaps):
  # exprmaps is ["rho","v"]
  # returns dynamic pressure in nanopascals

  rho = exprmaps[0]
  v = exprmaps[1]

  pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

  pdyn /= 1.0e-9

  return pdyn

def expr_srho(exprmaps):
  # exprmaps is ["rho","CellID"]
  # returns number density in cm^-3

  rho = exprmaps[0][:,:]
  cellids = exprmaps[1][:,:]

  srho = rho/1.0e+6

  return srho
    
###PLOTTERS###

def plot_new(runid,filenumber,vmax=1.5):

  # get colormap
  parula = make_parula()

  # create outputdir if it doesn't already exist
  outputdir = "Contours/"+runid+"/"
  if not os.path.exists(outputdir):
        os.makedirs(outputdir)

  bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"
  bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"
  print(bulkname)

  pt.plot.plot_colormap(filename=bulkpath+bulkname,run=runid,step=filenumber,outputdir=outputdir,colormap=parula,lin=1,usesci=0,title="",cbtitle="nPa",vmin=0,vmax=vmax,boxre=[6,16,-6,6],expression=expr_pdyn_gen,external=cust_contour,pass_vars=["rho","v","X","Y"],pass_times=180)

def plot_plaschke(filenumber,run,newfile=True,cmap="viridis",draw_pic=None):

    # create temporary vlsv file for plotting
    if newfile:
        jfm.pfmake(filenumber,run)

    # create the plot with contours
    pt.plot.plot_colormap(filename="VLSV/temp_plaschke.vlsv",var="npdynx",colormap=cmap,outputdir="Contours/"+run+"_P"+str(filenumber)+"_",usesci=0,lin=1,draw=draw_pic,boxre=[6,16,-6,6],vmax=1,cbtitle="",title="",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

def plot_archerhorbury(filenumber,run,halftimewidth,draw_pic=None):

    # create temporary vlsv file for plotting
    jfm.ahfmake(filenumber,run,halftimewidth)

    # create the plot with contours
    pt.plot.plot_colormap(filename="VLSV/temp_archerhorbury.vlsv",var="tapdyn",colormap="viridis",outputdir="Contours/"+run+"_AH"+str(filenumber)+"_"+str(halftimewidth)+"_",usesci=0,lin=1,boxre=[8,16,-6,6],vmax=4,cbtitle="",title="",draw=draw_pic,external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

def plot_karlsson(filenumber,run,halftimewidth):

    # create temporary vlsv file for plotting
    jfm.kfmake(filenumber,run,halftimewidth)

    # create the plot with contours
    pt.plot.plot_colormap(filename="VLSV/temp_karlsson.vlsv",var="tarho",colormap="viridis",outputdir="Contours/"+run+"_K"+str(filenumber)+"_"+str(halftimewidth)+"_",usesci=0,lin=1,boxre=[8,16,-6,6],vmax=2,cbtitle="",title="",external=jc.jc_karlsson,pass_vars=["tarho"])

def plot_all(filenumber,run,halftimewidth,newfile=True,cmap="viridis",draw_pic=None,draw_var="srho",v_min=0.0,v_max=5,box_re=[8,16,-6,6]):

    # create temporary vlsv file for plotting
    if newfile:
        jfm.pahkmake(filenumber,run,halftimewidth)

    # create the plot with contours
    pt.plot.plot_colormap(filename="VLSV/temp_all.vlsv",var=draw_var,colormap=cmap,outputdir="Contours/"+run+"_ALL"+str(filenumber)+"_"+str(halftimewidth)+"_",boxre=box_re,cbtitle="",title="",usesci=0,lin=1,draw=draw_pic,vmin=v_min,vmax=v_max,external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])

def plot_all_cust(filenumber,run,halftimewidth,box_re=[8,16,-6,6],draw_pic=None):

    # create temporary vlsv file for plotting
    jfm.pahkmake(filenumber,run,halftimewidth,sw_params=[1.0e+6,750.0e+3])

    # create the plot with contours
    pt.plot.plot_colormap(filename="VLSV/temp_all.vlsv",var="srho",colormap="viridis",outputdir="Contours/"+run+"_ALL_CUST"+str(filenumber)+"_"+str(halftimewidth)+"_",boxre=box_re,vmin=0.8,vmax=5.0,cbtitle="$n_\mathrm{p}$ [cm$^{-3}$]",usesci=0,lin=1,draw=draw_pic,external=jc.jc_all_cust,pass_vars=["npdynx","nrho","tapdyn","tarho","identifiers"])

def make_parula():

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    test_cm = parula_map

    return test_cm