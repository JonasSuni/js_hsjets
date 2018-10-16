import plot_contours as pc
import pytools as pt
import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc
import jetfile_make as jfm
import jet_io as jio
from matplotlib.ticker import MaxNLocator

from matplotlib import rcParams

m_p = 1.672621898e-27
r_e = 6.371e+6

###TEMPORARY SCRIPTS HERE###

def expr_smooth(exprmaps):

    rho = exprmaps[0]/1.0e+6

    rho = scipy.ndimage.uniform_filter(rho,size=9,mode="nearest")

    return rho

###PROP MAKER FILES HERE###



###FIGURE MAKERS HERE###

def jet_pos_graph(runid):

    filenames = os.listdir("jets/"+runid)

    propfiles = [filename for filename in filenames if ".props" in filename]

    r_list = []
    phi_list = []
    size_list = []

    for fname in propfiles:
        props = pd.read_csv("jets/"+runid+"/"+fname).as_matrix()
        r = props[:,6]
        phi = props[r==max(r)][0][8]
        r_list.append(max(r))
        phi_list.append(phi)
        size_list.append(r.size)

    r_list = np.asarray(r_list)
    phi_list = np.asarray(phi_list)
    size_list = np.asarray(size_list)

    r_list = r_list[size_list > 20]
    phi_list = phi_list[size_list > 20]

    plt.ion()
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_xlabel("r$_{mean,max}$ [R$_{e}$]",fontsize=20)
    ax.set_ylabel("$\\phi _{mean}$ [deg]",fontsize=20)
    ax.set_xlim(0,18)
    ax2.set_xlabel("r$_{mean,max}$ [R$_{e}$]",fontsize=20)
    ax2.set_ylabel("Number of jets",fontsize=20)
    ax2.set_xlim(0,18)
    plt.title(runid+"\nN = "+str(r_list.size),fontsize=20)

    rphi_graph = ax.plot(r_list,phi_list,"x",color="black")
    r_hist = ax2.hist(r_list,bins=list(xrange(0,19)))

    plt.tight_layout()

    if not os.path.exists("Figures/jets/debugging/"):
        try:
            os.makedirs("Figures/jets/debugging/")
        except OSError:
            pass

    fig.savefig("Figures/jets/debugging/"+runid+"_"+"rmax.png")

    plt.close(fig)

    return None

def jet_paper_pos():

    runids = ["ABA","ABC","AEA","AEC"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC"],[10,8,10,8]))
    run_marker_dict = dict(zip(["ABA","ABC","AEA","AEC"],["x","o","^","d"]))
    run_color_dict = dict(zip(["ABA","ABC","AEA","AEC"],["black","red","blue","green"]))

    x_list_list = [[],[],[],[]]
    y_list_list = [[],[],[],[]]

    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > 10 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    x_list_list[n].append(props.read_at_amax("x_mean"))
                    y_list_list[n].append(props.read_at_amax("y_mean"))

    plt.ioff()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X [R$_{e}$]",fontsize=20)
    ax.set_ylabel("Y [R$_{e}$]",fontsize=20)
    ax.set_xlim(6,18)
    ax.set_ylim(-9,7)

    lines = []
    labs = []

    for n in xrange(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=10,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=20)
    plt.legend(lines,labs,numpoints=1)
    plt.tight_layout()

    if not os.path.exists("Figures/paper/misc/"+"_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/misc/"+"_".join(runids)+"/")
        except OSError:
            pass

    fig.savefig("Figures/paper/misc/"+"_".join(runids)+"/"+"pos.png")
    print("Figures/paper/misc/"+"_".join(runids)+"/"+"pos.png")

    plt.close(fig)

    return None

def jet_2d_hist(runids,var1,var2,time_thresh=10):
    # Create 2D histogram of var1 and var2

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(xrange(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise input variable list and variable list
    inp_var_list = [var1,var2]
    var_list = [[],[]]

    # Append variable values to var lists
    for ind in xrange(len(inp_var_list)):
        for n in xrange(len(runids)):
            for fname in file_list_list[n]:
                props = jio.PropReader("",runids[n],fname=fname)
                if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    if inp_var_list[ind] == "duration":
                        var_list[ind].append(props.read("time")[-1]-props.read("time")[0])
                    elif inp_var_list[ind] == "size_ratio":
                        var_list[ind].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                    elif inp_var_list[ind] in ["n_max","n_avg","n_med","rho_vmax"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[0])
                    elif inp_var_list[ind] in ["v_max","v_avg","v_med"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[1])
                    elif inp_var_list[ind] in ["B_max","B_avg","B_med"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[2])
                    elif inp_var_list[ind] in ["beta_max","beta_avg","beta_med","b_vmax"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[4])
                    elif inp_var_list[ind] in ["pdyn_vmax"]:
                        var_list[ind].append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                    elif inp_var_list[ind] in ["pd_avg","pd_med","pd_max"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[3])
                    elif inp_var_list[ind] == "death_distance":
                        var_list[ind].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]]))
                    else:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind]))

    # Labels for figure
    label_list = ["Duration [s]",
    "Radial size [R$_{e}$]","Tangential size [R$_{e}$]","Radial size/Tangential size",
    "P$_{dyn,vmax}$ [P$_{dyn,sw}$]","P$_{dyn,avg}$ [P$_{dyn,sw}$]","P$_{dyn,med}$ [P$_{dyn,sw}$]","P$_{dyn,max}$ [P$_{dyn,sw}$]",
    "n$_{max}$ [n$_{sw}$]","n$_{avg}$ [n$_{sw}$]","n$_{med}$ [n$_{sw}$]","n$_{v,max}$ [n$_{sw}$]",
    "v$_{max}$ [v$_{sw}$]","v$_{avg}$ [v$_{sw}$]","v$_{med}$ [v$_{sw}$]",
    "B$_{max}$ [B$_{IMF}$]","B$_{avg}$ [B$_{IMF}$]","B$_{med}$ [B$_{IMF}$]",
    "$\\beta _{max}$ [$\\beta _{sw}$]","$\\beta _{avg}$ [$\\beta _{sw}$]","$\\beta _{med}$ [$\\beta _{sw}$]","$\\beta _{v,max}$ [$\\beta _{sw}$]",
    "T$_{avg}$ [MK]","T$_{med}$ [MK]","T$_{max}$ [MK]",
    "T$_{Parallel,avg}$ [MK]","T$_{Parallel,med}$ [MK]","T$_{Parallel,max}$ [MK]",
    "T$_{Perpendicular,avg}$ [MK]","T$_{Perpendicular,med}$ [MK]","T$_{Perpendicular,max}$ [MK]",
    "Area [R$_{e}^{2}$]",
    "r$_{v,max}$ at time of death [R$_{e}$]"]

    # X limits and bin widths for figure
    xmax_list=[120,
    3.5,3.5,7,
    5,5,5,5,
    10,10,10,10,
    1.5,1.5,1.5,
    8,8,8,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    4,
    18]

    step_list = [5,
    0.25,0.25,0.2,
    0.2,0.2,0.2,0.2,
    0.5,0.5,0.5,0.5,
    0.1,0.1,0.1,
    0.5,0.5,0.5,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.2,
    0.5]

    # Create figure
    plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_list[var_dict[var1]],fontsize=24)
    ax.set_ylabel(label_list[var_dict[var2]],fontsize=24)
    ax.tick_params(labelsize=20)
    #weights = [[1/float(len(var_list[n]))]*len(var_list[n]) for n in xrange(len(var_list))]
    weights = [1/float(len(var_list[0]))]*len(var_list[0]) # Normalise by total number of jets
    bins = [np.linspace(0,xmax_list[var_dict[var]],21).tolist() for var in inp_var_list]

    hist = ax.hist2d(var_list[0],var_list[1],bins=bins,weights=weights)

    if xmax_list[var_dict[inp_var_list[0]]] == xmax_list[var_dict[inp_var_list[1]]]:
        ax.plot([0,xmax_list[var_dict[inp_var_list[0]]]],[0,xmax_list[var_dict[inp_var_list[0]]]],"r--")

    if var not in ["death_distance"]:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower'))

    plt.title(",".join(runids),fontsize=20)
    plt.colorbar(hist[3], ax=ax)
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("Figures/paper/histograms/"+"_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/histograms/"+"_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("Figures/paper/histograms/"+"_".join(runids)+"/"+var1+"_"+var2+"_"+str(time_thresh)+"_2d.png")
    print("Figures/paper/histograms/"+"_".join(runids)+"/"+var1+"_"+var2+"_"+str(time_thresh)+"_2d.png")

    plt.close(fig)

    return None

def jet_paper_vs_hist(runids,var,time_thresh=10):

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Different colors for different runs
    run_colors_dict = dict(zip([runids[0],runids[1]],["red","blue"]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(xrange(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = [[],[]]

    val_dict = dict(zip(runids,var_list))

    # Append variable values to var lists
    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    val_dict[runids[n]].append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    val_dict[runids[n]].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[4])
                elif var in ["pdyn_vmax"]:
                    val_dict[runids[n]].append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    val_dict[runids[n]].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-ja.bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    val_dict[runids[n]].append(props.read_at_amax(var))

    # Labels for figure
    label_list = ["Duration [s]",
    "Radial size [R$_{e}$]","Tangential size [R$_{e}$]","Radial size/Tangential size",
    "P$_{dyn,vmax}$ [P$_{dyn,sw}$]","P$_{dyn,avg}$ [P$_{dyn,sw}$]","P$_{dyn,med}$ [P$_{dyn,sw}$]","P$_{dyn,max}$ [P$_{dyn,sw}$]",
    "n$_{max}$ [n$_{sw}$]","n$_{avg}$ [n$_{sw}$]","n$_{med}$ [n$_{sw}$]","n$_{v,max}$ [n$_{sw}$]",
    "v$_{max}$ [v$_{sw}$]","v$_{avg}$ [v$_{sw}$]","v$_{med}$ [v$_{sw}$]",
    "B$_{max}$ [B$_{IMF}$]","B$_{avg}$ [B$_{IMF}$]","B$_{med}$ [B$_{IMF}$]",
    "$\\beta _{max}$ [$\\beta _{sw}$]","$\\beta _{avg}$ [$\\beta _{sw}$]","$\\beta _{med}$ [$\\beta _{sw}$]","$\\beta _{v,max}$ [$\\beta _{sw}$]",
    "T$_{avg}$ [MK]","T$_{med}$ [MK]","T$_{max}$ [MK]",
    "T$_{Parallel,avg}$ [MK]","T$_{Parallel,med}$ [MK]","T$_{Parallel,max}$ [MK]",
    "T$_{Perpendicular,avg}$ [MK]","T$_{Perpendicular,med}$ [MK]","T$_{Perpendicular,max}$ [MK]",
    "Area [R$_{e}^{2}$]",
    "$r_{v,max}-r_{BS}$ at time of death [R$_{e}$]"]

    # X limits and bin widths for figure
    xmax_list=[120,
    3.5,3.5,7,
    5,5,5,5,
    10,10,10,10,
    1.5,1.5,1.5,
    8,8,8,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    5,
    5]

    step_list = [5,
    0.25,0.25,0.2,
    0.2,0.2,0.2,0.2,
    0.5,0.5,0.5,0.5,
    0.1,0.1,0.1,
    0.5,0.5,0.5,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.2,
    0.5]

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_list[var_dict[var]],fontsize=24)
    ax.set_ylabel("Fraction of jets",fontsize=24)
    ax.set_xlim(0,xmax_list[var_dict[var]])
    ax.set_ylim(0,0.75)
    ax.tick_params(labelsize=20)
    weights = [[1/float(len(val_dict[runids[n]]))]*len(val_dict[runids[n]]) for n in xrange(len(runids))] # Normalise by total number of jets

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax_list[var_dict[var]])
        
        #for n in xrange(len(runids)):
        #    hist = ax.hist(var_list[n],weights=weights[n],bins=bins,color=run_colors_dict[runids[n]],alpha=0.5,label=runids[n])
        
        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],weights=weights,bins=bins,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

    else:
        bins = np.arange(0,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        if var == "death_distance":
            ax.set_xlim(-5,xmax_list[var_dict[var]])
            bins = np.arange(-5,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        #for n in xrange(len(runids)):
        #    hist = ax.hist(var_list[n],bins=bins,weights=weights[n],color=run_colors_dict[runids[n]],alpha=0.5,label=runids[n])

        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],bins=bins,weights=weights,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

    #for n in xrange(len(runids)):
    #    ax.axvline(np.median(val_dict[runids[n]]), linestyle="dashed", linewidth=2, color=run_colors_dict[runids[n]])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=7,prune='lower'))

    plt.title(",".join(runids),fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("Figures/paper/histograms/"+"_vs_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/histograms/"+"_vs_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("Figures/paper/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("Figures/paper/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

def jet_paper_all_hist(runids,var,time_thresh=10):
    # Creates histogram specified var

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff values for elimination of false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(xrange(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = []

    # Append variable values to var list
    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    var_list.append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    var_list.append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[4])
                elif var in ["pdyn_vmax"]:
                    var_list.append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    var_list.append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-ja.bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    var_list.append(props.read_at_amax(var))

    var_list = np.asarray(var_list)

    # Labels for figure
    label_list = ["Duration [s]",
    "Radial size [R$_{e}$]","Tangential size [R$_{e}$]","Radial size/Tangential size",
    "P$_{dyn,vmax}$ [P$_{dyn,sw}$]","P$_{dyn,avg}$ [P$_{dyn,sw}$]","P$_{dyn,med}$ [P$_{dyn,sw}$]","P$_{dyn,max}$ [P$_{dyn,sw}$]",
    "n$_{max}$ [n$_{sw}$]","n$_{avg}$ [n$_{sw}$]","n$_{med}$ [n$_{sw}$]","n$_{v,max}$ [n$_{sw}$]",
    "v$_{max}$ [v$_{sw}$]","v$_{avg}$ [v$_{sw}$]","v$_{med}$ [v$_{sw}$]",
    "B$_{max}$ [B$_{IMF}$]","B$_{avg}$ [B$_{IMF}$]","B$_{med}$ [B$_{IMF}$]",
    "$\\beta _{max}$ [$\\beta _{sw}$]","$\\beta _{avg}$ [$\\beta _{sw}$]","$\\beta _{med}$ [$\\beta _{sw}$]","$\\beta _{v,max}$ [$\\beta _{sw}$]",
    "T$_{avg}$ [MK]","T$_{med}$ [MK]","T$_{max}$ [MK]",
    "T$_{Parallel,avg}$ [MK]","T$_{Parallel,med}$ [MK]","T$_{Parallel,max}$ [MK]",
    "T$_{Perpendicular,avg}$ [MK]","T$_{Perpendicular,med}$ [MK]","T$_{Perpendicular,max}$ [MK]",
    "Area [R$_{e}^{2}$]",
    "$r_{v,max}-r_{BS}$ at time of death [R$_{e}$]"]

    # X-limits and bin widths for figure
    xmax_list=[120,
    3.5,3.5,7,
    5,5,5,5,
    10,10,10,10,
    1.5,1.5,1.5,
    8,8,8,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    5,
    5]

    step_list = [5,
    0.25,0.25,0.2,
    0.2,0.2,0.2,0.2,
    0.5,0.5,0.5,0.5,
    0.1,0.1,0.1,
    0.5,0.5,0.5,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.2,
    0.5]

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_list[var_dict[var]],fontsize=24)
    ax.set_ylabel("Fraction of jets",fontsize=24)
    ax.set_xlim(0,xmax_list[var_dict[var]])
    ax.set_ylim(0,0.6)
    ax.tick_params(labelsize=20)
    weights = np.ones(var_list.shape)/float(var_list.size) # Normalise by total number of jets

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax_list[var_dict[var]])
        hist = ax.hist(var_list,weights=weights,bins=bins)
    else:
        bins = np.arange(0,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        if var == "death_distance":
            ax.set_xlim(-5,xmax_list[var_dict[var]])
            bins = np.arange(-5,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        hist = ax.hist(var_list,bins=bins,weights=weights)

    #ax.axvline(np.median(var_list), linestyle="dashed", color="black", linewidth=2)
    ax.annotate("med: %.1f\nstd: %.1f"%(np.median(var_list),np.std(var_list,ddof=1)), xy=(0.8,0.85), xycoords='axes fraction', fontsize=20)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6,prune='lower'))

    plt.title(",".join(runids),fontsize=20)
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("Figures/paper/histograms/"+"_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/histograms/"+"_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("Figures/paper/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("Figures/paper/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

###PLOT MAKER HERE###



###CONTOUR MAKER HERE###



###VIRTUAL SPACECRAFT MAKER HERE###


# MOVED TO vspacecraft.py




###MULTI FILE SCRIPTS HERE###



### MISC SCRIPTS ###

def find_missing(inputfolder,start,stop):

    filenames = os.listdir(inputfolder)

    filenums = []

    for filename in filenames:

        filenums.append(int(filename[14:-4]))

    rangelist = list(xrange(start,stop+1))

    for n in rangelist:

        if n not in filenums:

            print(n)

    return None

def find_missing_bulk(inputfolder):

    filenames = os.listdir(inputfolder)

    file_list = []

    for filename in filenames:
        if "bulk." in filename and ".vlsv" in filename:
            file_list.append(filename)

    file_nums = []

    for filename in file_list:
        file_num = int("".join(i for i in filename if i.isdigit()))
        file_nums.append(file_num)

    file_nums.sort()

    comp_range = xrange(min(file_nums),max(file_nums)+1)

    for n in comp_range:
        if n not in file_nums:
            print(str(n)+" is missing!")

    print(str(len(comp_range)-len(file_nums))+" files missing!")

    return None

def find_missing_jetsizes(runid):

    jetfile_names = os.listdir("jets/"+runid)

    propfile_list = [int(s[4:-6]) for s in jetfile_names if ".props" in s]

    jetsize_names = os.listdir("jet_sizes/"+runid)

    jetsize_list = [int(s[:-4]) for s in jetsize_names]

    for jetid in propfile_list:
        if jetid not in jetsize_list:
            print("Jet with ID "+str(jetid)+" has no time series!")

    return None

def jethist_paper_script(runtype="ecl"):

    if runtype == "ecl":
        runids = ["ABA","ABC","AEA","AEC"]
    elif runtype == "pol":
        runids = ["BFD"]
    else:
        print("Runtype must be ecl or pol. Exiting.")
        return 1

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        jet_paper_all_hist(runids,var,time_thresh=10)

    return None

def jethist_paper_script_vs(runids):

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        jet_paper_vs_hist(runids,var,time_thresh=10)

    return None

def jethist_paper_script_2d(runtype="ecl"):

    if runtype == "ecl":
        runids_list = [["ABA"],["ABC"],["AEA"],["AEC"],["ABA","ABC","AEA","AEC"]]
    elif runtype == "pol":
        runids_list = [["BFD"]]

    var_list = [["pd_max","pdyn_vmax"],["pd_max","n_max"]]

    for runids in runids_list:
        for var_pair in var_list:
            jet_2d_hist(runids,var_pair[0],var_pair[1])

    return None