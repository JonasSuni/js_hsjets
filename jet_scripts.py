import plot_contours as pc
import pytools as pt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc

def prop_file_maker(run,start,stop,halftimewidth):
    # create properties files, with custom jet criteria for bulk files in 
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script_cust(n,run,halftimewidth,boxre=[8,16,-6,6],min_size=100,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def prop_file_maker_AH(run,start,stop,halftimewidth):
    # create properties files, with AH jet criteria, for bulk files in
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script(n,run,halftimewidth,criterion="AH",boxre=[8,16,-6,6],min_size=100,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def hist_xy(runid,var1,var2,figname):
    # create 2D histogram of the specified variables

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    x = np.array([])
    y = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_dict = dict(zip(xrange(21),["$n_{avg} [cm^{-3}]$","$n_{med} [cm^{-3}]$","$n_{max} [cm^{-3}]$","$v_{avg} [km/s]$","$v_{med} [km/s]$","$v_{max} [km/s]$","$B_{avg} [nT]$","$B_{med} [nT]$","$B_{max} [nT]$","$T_{avg} [MK]$","$T_{med} [MK]$","$T_{max} [MK]$","$X_{vmax} [R_e]$","$Y_{vmax} [R_e]$","$Z_{vmax} [R_e]$","$A [km^2]$","$Nr\_cells$","$phi [deg]$","$mag\_p\_bool$","$x\_size [R_e]$","$y\_size [R_e]$"]))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        x = np.append(x,props[:,var1])
        y = np.append(y,props[:,var2])
        nr_cells = np.append(nr_cells,props[:,16])

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw histogram
    xy_hist = ax.hist2d(x,y,bins=15,normed=True,weights=nr_cells)
    plt.colorbar(xy_hist[3], ax=ax)

    # save figure
    plt.savefig("Figures/"+figname+".png")

def plot_xy(runid,var1,var2,figname):
    # plot the two specified variables against each other

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    x = np.array([])
    y = np.array([])

    # create dictionary for axis labels
    label_dict = dict(zip(xrange(21),["$n_{avg} [cm^{-3}]$","$n_{med} [cm^{-3}]$","$n_{max} [cm^{-3}]$","$v_{avg} [km/s]$","$v_{med} [km/s]$","$v_{max} [km/s]$","$B_{avg} [nT]$","$B_{med} [nT]$","$B_{max} [nT]$","$T_{avg} [MK]$","$T_{med} [MK]$","$T_{max} [MK]$","$X_{vmax} [R_e]$","$Y_{vmax} [R_e]$","$Z_{vmax} [R_e]$","$A [km^2]$","$Nr\_cells$","$phi [deg]$","$mag\_p\_bool$","$x\_size [R_e]$","$y\_size [R_e]$"]))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        x = np.append(x,props[:,var1])
        y = np.append(y,props[:,var2])

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw plot
    xy_plot = ax.plot(x,y,"x",color="black")

    # save figure
    plt.savefig("Figures/"+figname+".png")

def var_hist_mult(runid,var1,figname):
    # create histogram of specified variable

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    hist_var = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_dict = dict(zip(xrange(21),["$n_{avg} [cm^{-3}]$","$n_{med} [cm^{-3}]$","$n_{max} [cm^{-3}]$","$v_{avg} [km/s]$","$v_{med} [km/s]$","$v_{max} [km/s]$","$B_{avg} [nT]$","$B_{med} [nT]$","$B_{max} [nT]$","$T_{avg} [MK]$","$T_{med} [MK]$","$T_{max} [MK]$","$X_{vmax} [R_e]$","$Y_{vmax} [R_e]$","$Z_{vmax} [R_e]$","$A [km^2]$","$Nr\_cells$","$phi [deg]$","$mag\_p\_bool$","$x\_size [R_e]$","$y\_size [R_e]$"]))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        hist_var = np.append(hist_var,props[:,var1])
        nr_cells = np.append(nr_cells,props[:,16])

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel("Probability density")

    # draw histogram
    var_h = ax.hist(hist_var,bins=15,weights=nr_cells,normed=True)

    # save figure
    plt.savefig("Figures/"+figname+".png")

def y_hist_mult(runid,figname):

  filenames = os.listdir("Props/"+runid)

  y = np.array([])
  nr_cells = np.array([])

  for filename in filenames:

    props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

    y = np.append(y,props[:,13])
    nr_cells = np.append(nr_cells,props[:,16])

  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlabel("$Y_{vmax}$ $[R_e]$")
  ax.set_ylabel("Probability density")

  y_h = ax.hist(y,bins=list(xrange(-6,7)),weights=nr_cells,normed=True)

  fig.show()

  plt.savefig("Figures/"+figname+".png")

def phi_hist_mult(runid,figname):

  filenames = os.listdir("Props/"+runid)

  phi = np.array([])
  nr_cells = np.array([])

  for filename in filenames:

    props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

    phi = np.append(phi,props[:,17])
    nr_cells = np.append(nr_cells,props[:,16])

  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlabel("$Angle [deg]$")
  ax.set_ylabel("Probability density")

  phi_h = ax.hist(phi,bins=list(xrange(-40,41,5)),weights=nr_cells,normed=True)

  fig.show()

  plt.savefig("Figures/"+figname+".png")

def contour_gen(run,start,stop,contour_type,halftimewidth):

  if contour_type == "Plaschke":
    for n in xrange(start,stop+1):
      pc.plot_plaschke(n,run)


  elif contour_type == "ArcherHorbury":
    for n in xrange(start,stop+1):
      pc.plot_archerhorbury(n,run,halftimewidth)

  elif contour_type == "Karlsson":
    for n in xrange(start,stop+1):
      pc.plot_karlsson(n,run,halftimewidth)

  elif contour_type == "All":
    for n in xrange(start,stop+1):
        pc.plot_all(n,run,halftimewidth)

  else:
    pass

  return None

def make_figs(outputfolder,box_re=[8,16,-6,6],plaschkemax=1,rhomax=6,rhomax5=6,rhomin=0,pdynmax=1.5):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    file_name = "VLSV/temp_all.vlsv"
    parula = pc.make_parula()

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap="viridis",outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="$\\rho v_x^2/\\rho_{sw} v_{sw}^2$",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

    pt.plot.plot_colormap(filename=file_name,var="tapdyn",colormap=parula,outputdir=outputfolder+"/Fig3a_",usesci=0,lin=1,boxre=box_re,vmax=4,vmin=0,cbtitle="",title="$\\rho v^2/<\\rho v^2>_{3min}$",external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

    pt.plot.plot_colormap(filename=file_name,var="spdyn",colormap=parula,outputdir=outputfolder+"/Fig3b_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="nPa",title="$\\rho v^2$")

    pt.plot.plot_colormap(filename=file_name,var="tpdynavg",colormap=parula,outputdir=outputfolder+"/Fig3c_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="nPa",title="$<\\rho v^2>_{3min}$")

    pt.plot.plot_colormap(filename=file_name,var="tarho",colormap=parula,outputdir=outputfolder+"/Fig4a_",usesci=0,lin=1,boxre=box_re,vmax=2,vmin=0,cbtitle="",title="$\\rho/<\\rho>_{3min}$",external=jc.jc_karlsson,pass_vars=["tarho"])

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig4b_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=rhomin,cbtitle="cm$^{-3}$",title="$\\rho$")

    pt.plot.plot_colormap(filename=file_name,var="trhoavg",colormap=parula,outputdir=outputfolder+"/Fig4c_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=rhomin,cbtitle="cm$^{-3}$",title="$<\\rho>_{3min}$")

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig5_",usesci=0,lin=1,boxre=box_re,vmax=rhomax5,vmin=rhomin,cbtitle="cm$^{-3}$",title="$\\rho$",external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])

def minna_figs(outputfolder,box_re=[8,16,-6,6],plaschkemax=1,rhomax=6,rhomax5=5,rhomin=0.8,pdynmax=1.5):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    file_name = "VLSV/temp_all.vlsv"
    parula = pc.make_parula()

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap="viridis",outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

    pt.plot.plot_colormap(filename=file_name,var="tapdyn",colormap=parula,outputdir=outputfolder+"/Fig3a_",usesci=0,lin=1,boxre=box_re,vmax=4,vmin=0,cbtitle="",title="",external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

    pt.plot.plot_colormap(filename=file_name,var="spdyn",colormap=parula,outputdir=outputfolder+"/Fig3b_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tpdynavg",colormap=parula,outputdir=outputfolder+"/Fig3c_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tarho",colormap=parula,outputdir=outputfolder+"/Fig4a_",usesci=0,lin=1,boxre=box_re,vmax=2,vmin=0,cbtitle="",title="",external=jc.jc_karlsson,pass_vars=["tarho"])

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig4b_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="trhoavg",colormap=parula,outputdir=outputfolder+"/Fig4c_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig5_",usesci=0,lin=1,boxre=box_re,vmax=rhomax5,vmin=rhomin,cbtitle="",title="",external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])