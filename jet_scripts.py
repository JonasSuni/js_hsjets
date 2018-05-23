import plot_contours as pc
import pytools as pt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc

m_p = 1.672621898e-27
r_e = 6.371e+6

def prop_file_maker(run,start,stop,halftimewidth):
    # create properties files, with custom jet criteria for bulk files in 
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script_cust(n,run,halftimewidth,boxre=[8,16,-6,6],min_size=50,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def prop_file_maker_AH(run,start,stop,halftimewidth):
    # create properties files, with AH jet criteria, for bulk files in
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script(n,run,halftimewidth,criterion="AH",boxre=[8,16,-6,6],min_size=100,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def hist_xy(runid,var1,var2,figname,normed_b=True,weight_b=True):
    # create 2D histogram of the specified variables

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    x = np.array([])
    y = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        x = np.append(x,props[:,var1])
        y = np.append(y,props[:,var2])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$"+label_dict[var1]+"$")
    ax.set_ylabel("$"+label_dict[var2]+"$")

    # draw histogram
    xy_hist = ax.hist2d(x,y,bins=15,normed=normed_b,weights=nr_cells)
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
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

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
    ax.set_xlabel("$"+label_dict[var1]+"$")
    ax.set_ylabel("$"+label_dict[var2]+"$")

    # draw plot
    xy_plot = ax.plot(x,y,"x",color="black")

    # save figure
    plt.savefig("Figures/"+figname+".png")

def var_hist_mult(runid,var1,figname,normed_b=True,weight_b=True):
    # create histogram of specified variable

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    hist_var = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        hist_var = np.append(hist_var,props[:,var1])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$"+label_dict[var1]+"$")
    ax.set_ylabel("Probability density")

    # draw histogram
    var_h = ax.hist(hist_var,bins=15,weights=nr_cells,normed=normed_b)

    # save figure
    plt.savefig("Figures/"+figname+".png")

def y_hist_mult(runid,figname,normed_b=True,weight_b=True):

    filenames = os.listdir("Props/"+runid)

    y = np.array([])
    nr_cells = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        y = np.append(y,props[:,19])
        nr_cells = np.append(nr_cells,props[:,22])

  
    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$Y_{vmax}$ $[R_e]$")
    ax.set_ylabel("Probability density")

    y_h = ax.hist(y,bins=list(xrange(-6,7)),weights=nr_cells,normed=normed_b)

    fig.show()

    plt.savefig("Figures/"+figname+".png")

def phi_hist_mult(runid,figname,normed_b=True,weight_b=True):

    filenames = os.listdir("Props/"+runid)

    phi = np.array([])
    nr_cells = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        phi = np.append(phi,props[:,23])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$Angle [deg]$")
    ax.set_ylabel("Probability density")

    phi_h = ax.hist(phi,bins=list(xrange(-40,41,5)),weights=nr_cells,normed=normed_b)

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

def make_wave_figs(outputfolder,start,stop):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    f_path = "/proj/vlasov/2D/ABA/bulk/"

    for n in xrange(start,stop+1):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        pt.plot.plot_colormap(filename=f_path+f_name,outputdir=outputfolder+"/"+str(n)+"_",colormap="viridis",vmin=0,vmax=1.5,usesci=0,lin=1,cbtitle="nPa",boxre=[6,18,-10,2],expression=pc.expr_pdyn,external=sc_pos_marker,pass_vars=["rho","v"])

    return None

def sc_pos_marker(ax,XmeshXY,YmeshXY,extmaps):

    pos_mark = ax.plot(12,-4.4,marker="o",color="black",markersize=2)

def get_pos_index(posre,runid,file_number):

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/"+runid+"/bulk/bulk."+str(file_number).zfill(7)+".vlsv")

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x_i = np.where(abs(X-posre[0]*r_e)<120000)[0]
    y_i = np.where(abs(Y-posre[1]*r_e)<120000)[0]

    pos_index = np.intersect1d(x_i,y_i)

    pos_id = cellids[pos_index[0]]

    return pos_id

def wave_spacecraft(start,stop,step,pos,font_size):

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv")

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x_i = np.where(abs(X-pos[0]*r_e)<100000)[0]
    y_i = np.where(abs(Y-pos[1]*r_e)<100000)[0]

    pos_index = np.intersect1d(x_i,y_i)

    pos_id = cellids[pos_index[0]]
    f_path = "/proj/vlasov/2D/ABA/bulk/"

    B_arr = np.array([])
    v_arr = np.array([])
    rho_arr = np.array([])
    pdyn_arr = np.array([])

    for n in xrange(start,stop+1,step):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        f = pt.vlsvfile.VlsvReader(f_path+f_name)

        B = f.read_variable("B",cellids=pos_id,operator="magnitude")
        v = f.read_variable("v",cellids=pos_id,operator="magnitude")
        rho = f.read_variable("rho",cellids=pos_id)

        pdyn = m_p*rho*(v**2)

        B_arr = np.append(B_arr,B)
        v_arr = np.append(v_arr,v)
        rho_arr = np.append(rho_arr,rho)
        pdyn_arr = np.append(pdyn_arr,pdyn)

    time_arr = np.array(xrange(start,stop+1,step)).astype(float)/2

    B_arr /= 1.0e-9
    v_arr /= 1.0e+3
    rho_arr /= 1.0e+6
    pdyn_arr /= 1.0e-9

    plt.ion()
    fig = plt.figure()

    B_ax = fig.add_subplot(411)
    v_ax = fig.add_subplot(412)
    rho_ax = fig.add_subplot(413)
    pdyn_ax = fig.add_subplot(414)

    B_ax.set_xlim(250,305)
    v_ax.set_xlim(250,305)
    rho_ax.set_xlim(250,305)
    pdyn_ax.set_xlim(250,305)

    B_ax.set_ylim(2,10)
    v_ax.set_ylim(600,800)
    rho_ax.set_ylim(0.6,2.0)
    pdyn_ax.set_ylim(0.6,1.8)

    B_ax.set_ylabel("B [nT]",fontsize=font_size)
    v_ax.set_ylabel("v [km/s]",fontsize=font_size)
    rho_ax.set_ylabel("$\\rho$ [cm$^{-3}$]",fontsize=font_size)
    pdyn_ax.set_ylabel("$P_{dyn}$ [nPa]",fontsize=font_size)
    pdyn_ax.set_xlabel("Time [s]",fontsize=font_size)

    B_ax.set_yticks([4,6,8,10])
    v_ax.set_yticks([650,700,750,800])
    rho_ax.set_yticks([0.8,1.2,1.6,2.0])
    pdyn_ax.set_yticks([0.8,1.0,1.2,1.4,1.6,1.8])

    plt.tight_layout()

    B_ax.plot(time_arr,B_arr)
    v_ax.plot(time_arr,v_arr)
    rho_ax.plot(time_arr,rho_arr)
    pdyn_ax.plot(time_arr,pdyn_arr)

    B_ax.axvline(280,linestyle="dashed")
    v_ax.axvline(280,linestyle="dashed")
    rho_ax.axvline(280,linestyle="dashed")
    pdyn_ax.axvline(280,linestyle="dashed")

    fig.show()

    plt.savefig("Figures/"+str(start)+"_"+str(stop)+".png")

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