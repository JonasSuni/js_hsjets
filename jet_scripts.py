import plot_contours as pc
import pytools as pt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc
import jetfile_make as jfm

from matplotlib import rc

parula = pc.make_parula()

m_p = 1.672621898e-27
r_e = 6.371e+6

###PROP MAKER FILES HERE###

def prop_file_maker(run,start,stop,halftimewidth):
    # create properties files, with custom jet criteria for bulk files in 
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script_cust(n,run,halftimewidth,boxre=[6,16,-6,6],min_size=50,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def linsize_maker(run,start,stop):
    # create linsize properties files with custom jet criteria for bulk files in range

    timerange = xrange(start,stop+1)

    for n in timerange:

        jet_script_dim(n,run)

    return None

def jet_script_dim(filenumber,runid):
    # script for creating linsize properties files

    # find correct file based on runid and filenumber
    file_nr = str(filenumber).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    # open file
    vlsvobj=pt.vlsvfile.VlsvReader(file_path)

    # make mask
    msk = ja.make_cust_mask(filenumber,runid,180,[8,16,-6,6])
    
    # sort jets
    jets = ja.sort_jets(vlsvobj,msk,100,3000,[1,1])
    
    # create linsize properties file
    calc_linsize(vlsvobj,jets,runid,filenumber)

    return None

def calc_linsize(vlsvobj,jets,runid,file_number):

    # Area of one cell
    dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]

    # erase contents of outputfile if it already exists
    open("lin_sizes/"+runid+"/linsizes_"+runid+"_"+str(file_number)+".csv","w").close()

    # open outputfile in append mode
    outputfile = open("lin_sizes/"+runid+"/linsizes_"+runid+"_"+str(file_number)+".csv","a")

    # write header to outputfile
    outputfile.write("x_mean [R_e],y_mean [R_e],A [R_e^2],Nr_cells,phi [deg],r_d [R_e],size_rad [R_e],size_tan [R_e]")

    # read variables
    v,X,Y,cellids = ja.read_mult_vars(vlsvobj,["v","X","Y","CellID"])

    # calculate magnitude
    vmag = np.linalg.norm(v,axis=-1)

    for event in jets:

        outputfile.write("\n")

        # restrict variables to cellids corresponding to jets
        jvmag,jX,jY = ja.ci2vars_nofile([vmag,X,Y],cellids,event)

        # calculate geometric center of jet
        x_mean = np.mean([max(jX),min(jX)])/r_e
        y_mean = np.mean([max(jY),min(jY)])/r_e

        # calculate jet size
        A = dA*event.size/(r_e**2)
        Nr_cells = event.size

        # geometric center of jet in polar coordinates
        phi = np.rad2deg(np.arctan(y_mean/x_mean))
        r_d = np.linalg.norm([x_mean,y_mean])

        # r-coordinates corresponding to all (x,y)-points in jet
        r = np.linalg.norm(np.array([jX,jY]),axis=0)/r_e

        # calculate linear sizes of jet
        size_rad = max(r)-min(r)
        size_tan = A/size_rad

        # properties array
        temp_arr = [x_mean,y_mean,A,Nr_cells,phi,r_d,size_rad,size_tan]

        # write array to file
        outputfile.write(",".join(map(str,temp_arr)))

    outputfile.close()

    print("lin_sizes/"+runid+"/linsizes_"+runid+"_"+str(file_number)+".csv")

    return None

###FIGURE MAKERS HERE###

def linsize_fig(figsize=(10,10)):
    # script for creating time series of jet linear sizes and area

    linsizes = pd.read_csv("jet_linsize.csv").as_matrix()

    time_arr = linsizes[:,0]
    area_arr = linsizes[:,3]
    rad_size_arr = linsizes[:,7]
    tan_size_arr = linsizes[:,8]

    plt.ion()
    fig = plt.figure(figsize=figsize)

    area_ax = fig.add_subplot(311)
    rad_size_ax = fig.add_subplot(312)
    tan_size_ax = fig.add_subplot(313)

    area_ax.grid()
    rad_size_ax.grid()
    tan_size_ax.grid()

    area_ax.set_xlim(290,320)
    rad_size_ax.set_xlim(290,320)
    tan_size_ax.set_xlim(290,320)

    area_ax.set_ylim(0.4,2.6)
    rad_size_ax.set_ylim(0.6,2.8)
    tan_size_ax.set_ylim(0.4,1.4)

    area_ax.set_yticks([0.5,1,1.5,2,2.5])
    rad_size_ax.set_yticks([0.8,1.2,1.6,2,2.4,2.8])
    tan_size_ax.set_yticks([0.6,1,1.4])

    area_ax.set_xticks([295,300,305,310,315,320])
    rad_size_ax.set_xticks([295,300,305,310,315,320])
    tan_size_ax.set_xticks([295,300,305,310,315,320])

    area_ax.set_xticklabels([])
    rad_size_ax.set_xticklabels([])

    area_ax.set_ylabel("Area [R$_{e}^{2}$]",fontsize=20)
    rad_size_ax.set_ylabel("Radial size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_ylabel("Tangential size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_xlabel("Time [s]",fontsize=20)

    area_ax.tick_params(labelsize=16)
    rad_size_ax.tick_params(labelsize=16)
    tan_size_ax.tick_params(labelsize=16)

    area_ax.plot(time_arr,area_arr,color="black",linewidth=2)
    rad_size_ax.plot(time_arr,rad_size_arr,color="black",linewidth=2)
    tan_size_ax.plot(time_arr,tan_size_arr,color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    plt.savefig("lin_sizes/sizefig.png")
    print("lin_sizes/sizefig.png")

    return None


def magp_ratio(runid):

    filenames = os.listdir("Props/"+runid)

    mag_p_bool = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        mag_p_bool = np.append(mag_p_bool,props[:,25])

    magp_ratio = float(mag_p_bool[mag_p_bool>0].size)/float(mag_p_bool.size)

    return magp_ratio

def hist_xy(runid,var1,var2,figname,normed_b=True,weight_b=True,bins=15):
    # create 2D histogram of the specified variables

    rc('text', usetex=False)

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
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw histogram
    xy_hist = ax.hist2d(x,y,bins=bins,normed=normed_b,weights=nr_cells)
    plt.colorbar(xy_hist[3], ax=ax)

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)

def plot_xy(runid,var1,var2,figname):
    # plot the two specified variables against each other

    rc('text', usetex=False)

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
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw plot
    xy_plot = ax.plot(x,y,"x",color="black")

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)

def var_hist_mult(runid,var1,figname,normed_b=True,weight_b=True):
    # create histogram of specified variable

    rc('text', usetex=False)

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
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel("Probability density")

    # draw histogram
    var_h = ax.hist(hist_var,bins=15,weights=nr_cells,normed=normed_b)

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)



###PLOT MAKER HERE###

def plotmake(runid,start,stop,vmax=1.5):

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    outputdir = "Plots/"+runid+"/"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for n in xrange(start,stop+1):

        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        pv_1 = "rho"
        pv_2 = "v"

        if type(pt.vlsvfile.VlsvReader(bulkpath+bulkname).read_variable("rho")) is not np.ndarray:
            pv_1 = "proton/rho"
            pv_2 = "proton/V"

        pt.plot.plot_colormap(filename=bulkpath+bulkname,run=runid,step=n,outputdir=outputdir,colormap=parula,lin=1,usesci=0,cbtitle="nPa",vmin=0,vmax=vmax,expression=pc.expr_pdyn,pass_vars=[pv_1,pv_2])

    return None

###CONTOUR MAKER HERE###

def contour_gen(runid,start,stop,vmax=1.5):

    for n in xrange(start,stop+1):

        pc.plot_new(runid,n,vmax)

    return None

def contour_gen_ff(runid,start,stop,vmax=1.5,boxre=[6,16,-6,6]):

    outputfilename = "new_"+runid+"_"+str(start)+"_"+str(stop)+".vlsv"
    outputdir = "/wrk/sunijona/DONOTREMOVE/Contours/"+runid+"/"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for n in xrange(start,stop+1):

        jfm.custmake(runid,n,outputfilename)

        pt.plot.plot_colormap(filename="/wrk/sunijona/VLSV/"+outputfilename,var="spdyn",run=runid,step=n,outputdir=outputdir,colormap=parula,lin=1,usesci=0,cbtitle="nPa",vmin=0,vmax=vmax,boxre=boxre,external=jc.jc_cust_scr,pass_vars=["npdynx","nrho","tapdyn"])

    return None

###VIRTUAL SPACECRAFT MAKER HERE###


# MOVED TO vspacecraft.py




###MULTI FILE SCRIPTS HERE###

def presentation_script(run_id,fig_name):

    '''props are 
    0: n_avg [cm^-3],   1: n_med [cm^-3],   2: n_max [cm^-3],
    3: v_avg [km/s],    4: v_med [km/s],    5: v_max [km/s],
    6: B_avg [nT],      7: B_med [nT],      8: B_max [nT],
    9: T_avg [MK],      10: T_med [MK],     11: T_max [MK],
    12: Tpar_avg [MK],  13: Tpar_med [MK],  14: Tpar_max [MK],
    15: Tperp_avg [MK], 16: Tperp_med [MK], 17: Tperp_max [MK],
    18: X_vmax [R_e],   19: Y_vmax [R_e],   20: Z_vmax [R_e],
    21: A [km^2],       22: Nr_cells,       23: phi [deg],
    24: r_d [R_e],      25: mag_p_bool,     26: size_x [R_e],
    27: size_y [R_e],   28: MMS,            29: MA'''

    hist_xy(run_id,18,19,fig_name+run_id+"_x_y",normed_b=False,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(-4,4,17)])

    hist_xy(run_id,18,5,fig_name+run_id+"_x_vmax",normed_b=True,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(100,900,17)])
    #hist_xy(run_id,18,12,fig_name+run_id+"_x_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,18,15,fig_name+run_id+"_x_Tperp_avg",normed_b=True,weight_b=True)

    hist_xy(run_id,19,5,fig_name+run_id+"_y_vmax",normed_b=True,weight_b=True,bins=[np.linspace(-4,4,17),np.linspace(100,900,17)])
    #hist_xy(run_id,19,12,fig_name+run_id+"_y_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,19,15,fig_name+run_id+"_y_Tperp_avg",normed_b=True,weight_b=True)

    hist_xy(run_id,18,22,fig_name+run_id+"_x_nrcells",normed_b=True,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(50,1950,17)])
    hist_xy(run_id,19,22,fig_name+run_id+"_y_nrcells",normed_b=True,weight_b=True,bins=[np.linspace(-4,4,17),np.linspace(50,1950,17)])
    #hist_xy(run_id,22,12,fig_name+run_id+"_nrcells_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,22,15,fig_name+run_id+"_nrcells_Tperp_avg",normed_b=True,weight_b=True)
    hist_xy(run_id,22,5,fig_name+run_id+"_nrcells_vmax",normed_b=True,weight_b=True,bins=[np.linspace(50,1950,17),np.linspace(100,900,17)])

    var_hist_mult(run_id,18,fig_name+run_id+"_x_hist",normed_b=True,weight_b=True)
    var_hist_mult(run_id,19,fig_name+run_id+"_y_hist",normed_b=True,weight_b=True)
    var_hist_mult(run_id,5,fig_name+run_id+"_vmax_hist",normed_b=True,weight_b=True)
    #var_hist_mult(run_id,12,fig_name+run_id+"_Tpar_avg_hist",normed_b=True,weight_b=True)
    #var_hist_mult(run_id,15,fig_name+run_id+"_Tperp_avg_hist",normed_b=True,weight_b=True)

    print("Magp_ratio is "+str(magp_ratio(run_id)))

    plt.close("all")

def fromfile_cont_movie(outputfolder,runid,start,stop):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    if runid == "ABA":

        v_max = 5.0

    elif runid == "ABC":

        v_max = 15.0

    parula = pc.make_parula()

    for n in xrange(start,stop+1):

        pt.plot.plot_colormap(filename="/proj/vlasov/2D/"+runid+"/bulk/bulk."+str(n).zfill(7)+".vlsv",outputdir="Contours/"+outputfolder+"/"+runid+"_"+str(n)+"_",usesci=0,lin=1,vmin=0.8,vmax=v_max,colormap=parula,boxre=[4,16,-6,6],cbtitle="",expression=pc.expr_srho,external=jc.jc_fromfile,pass_vars=["rho","CellID"],ext_pars=[runid,n,180])

def make_figs(outputfolder,box_re=[8,16,-6,6],plaschkemax=1,rhomax=6,rhomax5=6,rhomin=0,pdynmax=1.5):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    file_name = "VLSV/temp_all.vlsv"

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap=parula,outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="$\\rho v_x^2/\\rho_{sw} v_{sw}^2$",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

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

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap=parula,outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

    pt.plot.plot_colormap(filename=file_name,var="tapdyn",colormap=parula,outputdir=outputfolder+"/Fig3a_",usesci=0,lin=1,boxre=box_re,vmax=4,vmin=0,cbtitle="",title="",external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

    pt.plot.plot_colormap(filename=file_name,var="spdyn",colormap=parula,outputdir=outputfolder+"/Fig3b_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tpdynavg",colormap=parula,outputdir=outputfolder+"/Fig3c_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tarho",colormap=parula,outputdir=outputfolder+"/Fig4a_",usesci=0,lin=1,boxre=box_re,vmax=2,vmin=0,cbtitle="",title="",external=jc.jc_karlsson,pass_vars=["tarho"])

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig4b_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="trhoavg",colormap=parula,outputdir=outputfolder+"/Fig4c_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig5_",usesci=0,lin=1,boxre=box_re,vmax=rhomax5,vmin=rhomin,cbtitle="",title="",external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])