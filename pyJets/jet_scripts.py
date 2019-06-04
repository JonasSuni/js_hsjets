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

from matplotlib import rc
# font = {'family' : 'monospace',
#         'monospace' : 'Computer Modern Typewriter',
#         'weight' : 'bold'}

#rc('font', **font)
rc('mathtext', fontset='custom')
rc('mathtext', default='regular')

m_p = 1.672621898e-27
r_e = 6.371e+6

###TEMPORARY SCRIPTS HERE###

def bs_plotter(runid,file_nr,thresh,rho_par):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(file_nr).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(file_nr)+" not found, continuing")
        return 1

    global rho_thresh
    global rho_sw
    global vlsvobj

    rho_thresh = thresh
    rho_sw = rho_par
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    pt.plot.plot_colormap(filename=bulkpath+bulkname,draw=1,usesci=0,lin=1,var="rho",colormap="parula",external=ext_bsp,pass_vars=["rho","CellID"])

def ext_bsp(ax,XmeshXY,YmeshXY,pass_maps):

    rho = pass_maps["rho"]
    cellids = pass_maps["CellID"].flatten()

    X,Y,Z = ja.xyz_reconstruct(vlsvobj,cellids)

    if vlsvobj.get_spatial_mesh_size()[2]==1:
        r_angle = np.rad2deg(np.arctan(np.abs(Y)/X))
    else:
        r_angle = np.rad2deg(np.arctan(np.abs(Z)/X))

    r_angle = np.reshape(r_angle,rho.shape)
    X = np.reshape(X,rho.shape)

    mask1 = (rho>=rho_thresh*rho_sw)
    mask2 = (r_angle<=45)
    mask3 = (X>=0)

    mask = np.logical_and(np.logical_and(mask1,mask2),mask3).astype(int)

    contour = ax.contour(XmeshXY,YmeshXY,mask,[0.5],linewidths=0.8, colors="black")

def jet_plotter(start,stop,runid,vmax=1.5,boxre=[6,18,-8,6]):

    outputdir = "/wrk/sunijona/DONOTREMOVE/contours/JETS/{}/".format(runid)
    
    global jet_cells
    global full_cells
    global xmean_list
    global ymean_list
    global xvmax_list
    global yvmax_list

    for n in range(start,stop+1):

        xmean_list = []
        ymean_list = []
        xvmax_list = []
        yvmax_list = []

        for itr in range(700):
            try:
                props = jio.PropReader(str(itr).zfill(5),runid,580,transient="jet")
                xmean_list.append(props.read_at_time("x_mean",float(n)/2))
                ymean_list.append(props.read_at_time("y_mean",float(n)/2))
                xvmax_list.append(props.read_at_time("x_vmax",float(n)/2))
                yvmax_list.append(props.read_at_time("y_vmax",float(n)/2))
            except IOError:
                pass
        try:
            fileobj = open("events/{}/{}.events".format(runid,n),"r")
            contents = fileobj.read()
            jet_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            jet_cells = [] 

        try:
            full_cells = np.loadtxt("Masks/{}/{}.mask".format(runid,n)).astype(int)
        except IOError:
            full_cells = []


        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/{}/".format(runid)
        elif runid == "AEA":
            bulkpath = "/proj/vlasov/2D/{}/round_3_boundary_sw/".format(runid)
        else:
            bulkpath = "/proj/vlasov/2D/{}/bulk/".format(runid)

        bulkname = "bulk.{}.vlsv".format(str(n).zfill(7))

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file {} not found, continuing".format(str(n)))
            continue

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir=outputdir,usesci=0,lin=1,boxre=boxre,var="Pdyn",vmax=vmax*1.0e-9,colormap="parula",cbtitle="nPa",external=ext_jet,pass_vars=["CellID"])

def ext_jet(ax,XmeshXY,YmeshXY,pass_maps):

    cellids = pass_maps["CellID"]

    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    full_mask = np.in1d(cellids,full_cells).astype(int)
    full_mask = np.reshape(full_mask,cellids.shape)

    full_cont = ax.contour(XmeshXY,YmeshXY,full_mask,[0.5],linewidths=0.8,colors="magenta")
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="black")

    line1, = ax.plot(xmean_list,ymean_list,"o",color="red",markersize=4)
    line2, = ax.plot(xvmax_list,yvmax_list,"o",color="white",markersize=4)

def slamjet_plotter(start,stop,runid,vmax=1.5,boxre=[6,18,-8,6]):

    outputdir = "/wrk/sunijona/DONOTREMOVE/contours/SLAMSJETS/{}/".format(runid)
    
    global jet_cells
    global slams_cells
    global xmean_list
    global ymean_list

    for n in xrange(start,stop+1):

        xmean_list = []
        ymean_list = []

        for itr in range(500):
            try:
                props = jio.PropReader(str(itr).zfill(5),runid,580,transient="slamsjet")
                xmean_list.append(props.read_at_time("x_mean",float(n)/2))
                ymean_list.append(props.read_at_time("y_mean",float(n)/2))
            except IOError:
                pass
        try:
            fileobj = open("events/{}/{}.events".format(runid,n),"r")
            contents = fileobj.read()
            jet_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            jet_cells = [] 

        try:
            fileobj = open("SLAMS/events/{}/{}.events".format(runid,n),"r")
            contents = fileobj.read()
            slams_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            slams_cells = []


        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/{}/".format(runid)
        elif runid == "AEA":
            bulkpath = "/proj/vlasov/2D/{}/round_3_boundary_sw/".format(runid)
        else:
            bulkpath = "/proj/vlasov/2D/{}/bulk/".format(runid)

        bulkname = "bulk.{}.vlsv".format(str(n).zfill(7))

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file {} not found, continuing".format(str(n)))
            continue

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir=outputdir,usesci=0,lin=1,boxre=boxre,var="Pdyn",vmax=vmax*1.0e-9,colormap="parula",cbtitle="nPa",external=ext_slamjet,pass_vars=["CellID"])

def ext_slamjet(ax,XmeshXY,YmeshXY,pass_maps):

    cellids = pass_maps["CellID"]

    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    slams_mask = np.in1d(cellids,slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask,cellids.shape)

    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="magenta")
    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors="black")

    line1, = ax.plot(xmean_list,ymean_list,"o",color="red",markersize=4)

def draw_all_cont():

    pt.plot.plot_colormap(filename="/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv",outputdir="Contours/ALLCONT_",usesci=0,lin=1,boxre=[6,18,-8,6],colormap="parula",cbtitle="nPa",scale=1,expression=pc.expr_pdyn,external=ext_crit,var="rho",vmin=0,vmax=1.5,wmark=1,pass_vars=["rho","v","CellID"])

def ext_crit(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho = extmaps["rho"].flatten()
    vx = extmaps["v"][:,:,0].flatten()
    vy = extmaps["v"][:,:,1].flatten()
    vz = extmaps["v"][:,:,2].flatten()
    vmag = np.linalg.norm([vx,vy,vz],axis=0)
    cellids = extmaps["CellID"].flatten()
    XmeshXY = XmeshXY.flatten()
    YmeshXY = YmeshXY.flatten()
    shp = extmaps["rho"].shape

    pdyn = m_p*rho*(vmag**2)
    pdyn_x = m_p*rho*(vx**2)

    pdyn = pdyn[cellids.argsort()]
    pdyn_x = pdyn_x[cellids.argsort()]
    rho = rho[cellids.argsort()]
    XmeshXY = XmeshXY[cellids.argsort()]
    YmeshXY = YmeshXY[cellids.argsort()]

    fullcells = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv").read_variable("CellID")
    fullcells.sort()

    trho = np.loadtxt("/wrk/sunijona/DONOTREMOVE/tavg/ABA/611_rho.tavg")[np.in1d(fullcells,cellids)]
    tpdyn = np.loadtxt("/wrk/sunijona/DONOTREMOVE/tavg/ABA/611_pdyn.tavg")[np.in1d(fullcells,cellids)]

    rho_sw = 1000000
    v_sw = 750000
    pdyn_sw = m_p*rho_sw*(v_sw**2)

    pdyn = scipy.ndimage.zoom(np.reshape(pdyn,shp),3)
    pdyn_x = scipy.ndimage.zoom(np.reshape(pdyn_x,shp),3)
    rho = scipy.ndimage.zoom(np.reshape(rho,shp),3)
    XmeshXY = scipy.ndimage.zoom(np.reshape(XmeshXY,shp),3)
    YmeshXY = scipy.ndimage.zoom(np.reshape(YmeshXY,shp),3)
    trho = scipy.ndimage.zoom(np.reshape(trho,shp),3)
    tpdyn = scipy.ndimage.zoom(np.reshape(tpdyn,shp),3)

    jetp = np.ma.masked_greater(pdyn_x,0.25*pdyn_sw)
    #jetp.mask[nrho < level_sw] = False
    jetp.fill_value = 0
    jetp[jetp.mask == False] = 1

    jetah = np.ma.masked_greater(pdyn,2*tpdyn)
    jetah.fill_value = 0
    jetah[jetah.mask == False] = 1

    # make karlsson mask
    jetk = np.ma.masked_greater(rho,1.5*trho)
    jetk.fill_value = 0
    jetk[jetk.mask == False] = 1

    # draw contours
    contour_plaschke = ax.contour(XmeshXY,YmeshXY,jetp.filled(),[0.5],linewidths=0.8, colors="black",label="Plaschke")

    contour_archer = ax.contour(XmeshXY,YmeshXY,jetah.filled(),[0.5],linewidths=0.8, colors="yellow",label="ArcherHorbury")

    contour_karlsson = ax.contour(XmeshXY,YmeshXY,jetk.filled(),[0.5],linewidths=0.8, colors="magenta",label="Karlsson")

    return None

def lineout_plot(runid,filenumber,p1,p2,var):

    # find correct file based on file number and run id
    if runid in ["AEC"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    var_dict = {"rho":[1e+6,"$\\rho~[cm^{-3}]$",1],"v":[1e+3,"$v~[km/s]$",750]}

    bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    lin = pt.calculations.lineout(pt.vlsvfile.VlsvReader(bulkpath+bulkname),np.array(p1)*r_e,np.array(p2)*r_e,var,interpolation_order=1,points=100)

    var_arr = lin[2]
    if len(var_arr.shape) == 2:
        var_arr = np.linalg.norm(var_arr,axis=-1)
    r_arr = np.linalg.norm(lin[1],axis=-1)/r_e

    if var in var_dict:
        var_arr /= var_dict[var][0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(r_arr,var_arr)
    ax.set_xlabel("$R~[R_e]$",labelpad=10,fontsize=20)
    ax.tick_params(labelsize=20)
    if var in var_dict:
        ax.set_ylabel(var_dict[var][1],labelpad=10,fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,prune="lower"))

    plt.tight_layout()

    fig.show()

    fig.savefig("Contours/"+"lineout_"+runid+"_"+str(filenumber)+"_"+var+".png")

def find_broken_BCQ(start,stop):

    bulkpath = "/proj/vlasov/2D/BCQ/bulk/"
    time_list = np.array([])
    n_list = np.array(xrange(start,stop+1))

    for n in n_list:
        bulkname = "bulk."+str(n).zfill(7)+".vlsv"
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        time_list = np.append(time_list,vlsvobj.read_parameter("time"))

    tdiff = np.append(0.5,np.ediff1d(time_list))

    plt.plot(n_list,tdiff,"x")
    plt.show()

    return n_list[tdiff==0]

def plot_streamlines(boxre=[4,20,-10,10]):

    pt.plot.plot_colormap(filename="/proj/vlasov/2D/BFD/bulk/bulk.0000611.vlsv",outputdir="Contours/STREAMPLOT_",usesci=0,lin=1,boxre=boxre,colormap="parula",cbtitle="",expression=pc.expr_srho,external=B_streamline,vmin=0,vmax=10,wmark=1,pass_vars=["proton/rho","CellID","B"])

def B_streamline(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    B = extmaps["B"]

    Bx = B[:,:,0]
    By = B[:,:,2]

    dx = Bx*0+1
    dy = np.divide(By,Bx)

    dl = np.linalg.norm([dx,dy],axis=0)
    dx /= dl
    dy /= dl

    X = XmeshXY
    Y = YmeshXY

    ax.streamplot(X,Y,dx,dy,arrowstyle="-",linewidth=0.5,color="black",density=2)

def expr_smooth(exprmaps):

    rho = exprmaps["rho"]/1.0e+6

    rho = scipy.ndimage.uniform_filter(rho,size=9,mode="nearest")

    return rho

###PROP MAKER FILES HERE###



###HELPER FUNCTIONS HERE###

def sheath_pars_list(var):

    key_list = [
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med"]

    label_list = [
    "$P_{dyn,vmax}~[P_{dyn,sh}]$","$P_{dyn,avg}~[P_{dyn,sh}]$","$P_{dyn,med}~[P_{dyn,sh}]$","$P_{dyn,max}~[P_{dyn,sh}]$",
    "$n_{max}~[n_{sh}]$","$n_{avg}~[n_{sh}]$","$n_{med}~[n_{sh}]$","$n_{v,max}~[n_{sh}]$",
    "$v_{max}~[v_{sh}]$","$v_{avg}~[v_{sh}]$","$v_{med}~[v_{sh}]$"
    ]

    norm_list = [
    0.25,0.25,0.25,0.25,
    4,4,4,4,
    0.25,0.25,0.25]

    return [label_list[key_list.index(var)],norm_list[key_list.index(var)]]

def var_pars_list(var):

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

    label_list = ["$Duration~[s]$",
    "$Radial~size~[R_{e}]$","$Tangential~size~[R_{e}]$","$Radial~size/Tangential~size$",
    "$P_{dyn,vmax}~[P_{dyn,sw}]$","$P_{dyn,avg}~[P_{dyn,sw}]$","$P_{dyn,med}~[P_{dyn,sw}]$","$P_{dyn,max}~[P_{dyn,sw}]$",
    "$n_{max}~[n_{sw}]$","$n_{avg}~[n_{sw}]$","$n_{med}~[n_{sw}]$","$n_{v,max}~[n_{sw}]$",
    "$v_{max}~[v_{sw}]$","$v_{avg}~[v_{sw}]$","$v_{med}~[v_{sw}]$",
    "$B_{max}~[B_{IMF}]$","$B_{avg}~[B_{IMF}]$","$B_{med}~[B_{IMF}]$",
    "$\\beta _{max}~[\\beta _{sw}]$","$\\beta _{avg}~[\\beta _{sw}]$","$\\beta _{med}~[\\beta _{sw}]$","$\\beta _{v,max}~[\\beta _{sw}]$",
    "$T_{avg}~[T_{sw}]$","$T_{med}~[T_{sw}]$","$T_{max}~[T_{sw}]$",
    "$T_{Parallel,avg}~[T_{sw}]$","$T_{Parallel,med}~[T_{sw}]$","$T_{Parallel,max}~[T_{sw}]$",
    "$T_{Perpendicular,avg}~[T_{sw}]$","$T_{Perpendicular,med}~[T_{sw}]$","$T_{Perpendicular,max}~[T_{sw}]$",
    "$Area~[R_{e}^{2}]$",
    "$(r_{v,max}-r_{BS})~at~time~of~death~[R_{e}]$"]

    xmin_list=[0,
    0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,0,
    0,0,0,
    0,0,0,
    0,0,0,
    0,
    -5]

    xmax_list=[120,
    3.5,3.5,7,
    5,5,5,5,
    10,10,10,10,
    1.5,1.5,1.5,
    8,8,8,
    1000,1000,1000,1000,
    50,50,50,
    50,50,50,
    50,50,50,
    4,
    5]

    step_list = [5,
    0.25,0.25,0.2,
    0.2,0.2,0.2,0.2,
    0.5,0.5,0.5,0.5,
    0.1,0.1,0.1,
    0.5,0.5,0.5,
    100,100,100,100,
    2,2,2,
    2,2,2,
    2,2,2,
    0.2,
    0.5]

    tickstep_list = [20,
    0.5,0.5,1,
    1,1,1,1,
    2,2,2,2,
    0.2,0.2,0.2,
    1,1,1,
    100,100,100,100,
    5,5,5,
    5,5,5,
    5,5,5,
    1,
    2]

    return [label_list[key_list.index(var)],xmin_list[key_list.index(var)],xmax_list[key_list.index(var)],step_list[key_list.index(var)],tickstep_list[key_list.index(var)]]

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

def jet_paper_counter():

    runids = ["ABA","ABC","AEA","AEC","BFD"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    run_cutoff_dict = dict(zip(runids,[10,8,10,8,10]))

    count_list_list = [0,0,0,0,0]

    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > 10 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    count_list_list[n] += 1

    return count_list_list

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
    ax.set_xlabel("X [R$_{e}$]",fontsize=24,labelpad=10)
    ax.set_ylabel("Y [R$_{e}$]",fontsize=24,labelpad=10)
    ax.set_xlim(6,18)
    ax.set_ylim(-9,7)
    ax.tick_params(labelsize=20)
    lines = []
    labs = []

    for n in xrange(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=10,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    #plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=24)
    plt.legend(lines,labs,numpoints=1,prop={"size":20})
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

def jet_mult_time_series(runid,start,jetid,thresh = 0.0,transient="jet"):

    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = "SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    props = jio.PropReader(jetid,runid,start,transient=transient)

    var_list = ["time","A","n_max","v_max","pd_max","r_mean"]

    time_arr,area_arr,n_arr,v_arr,pd_arr,r_arr = [props.read(var)/ja.sw_normalisation(runid,var) for var in var_list]

    if np.max(area_arr) < thresh or time_arr.size < 10:
        print("Jet smaller than threshold, exiting!")
        return None

    plt.ioff()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel("Time [s]",fontsize=20)
    ax.set_ylabel("Fraction of maximum",fontsize=20)
    ax2.set_ylabel("Fraction of maximum",fontsize=20)
    ax.grid()
    ax2.grid()
    plt.title("Run: {}, ID: {}".format(runid,jetid))
    ax2.plot(time_arr,area_arr/np.max(area_arr),label="A\_max = {:.3g}".format(np.max(area_arr)))
    ax.plot(time_arr,n_arr/np.max(n_arr),label="n\_max = {:.3g}".format(np.max(n_arr)))
    ax.plot(time_arr,v_arr/np.max(v_arr),label="v\_max = {:.3g}".format(np.max(v_arr)))
    ax.plot(time_arr,pd_arr/np.max(pd_arr),label="pd\_max = {:.3g}".format(np.max(pd_arr)))
    ax2.plot(time_arr,r_arr/np.max(r_arr),label="r\_max = {:.3g}".format(np.max(r_arr)))
    ax2.plot(time_arr,ja.bow_shock_r(runid,time_arr)/np.max(r_arr),label="Bow shock")

    ax.legend(loc="lower right")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    fig.savefig("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))
    print("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))

    plt.close(fig)

    return None

def jet_time_series(runid,start,jetid,var,thresh = 0.0,transient="jet"):

    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = "SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    props = jio.PropReader(jetid,runid,start,transient=transient)

    time_arr = props.read("time")
    area_arr = props.read("A")
    var_arr = props.read(var)/ja.sw_normalisation(runid,var)

    if np.max(area_arr) < thresh:
        print("Jet smaller than threshold, exiting!")
        return None

    plt.ioff()
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()
    plt.title("Run: {}, ID: {}".format(runid,jetid))
    ax.plot(time_arr,var_arr,color="black")

    plt.tight_layout()

    fig.savefig("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))
    print("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))

    plt.close(fig)

    return None

def jts_make(runid,start,startid,stopid,thresh = 0.0,transient="jet"):

    for n in range(startid,stopid+1):
        try:
            jet_mult_time_series(runid,start,str(n).zfill(5),thresh=thresh,transient=transient)
        except IOError:
            print("Could not create time series!")

    return None

def SEA_make(runid,var,centering="pd_avg",thresh=5):

    #jetids = dict(zip(["ABA","ABC","AEA","AEC"],[[2,29,79,120,123,129],[6,12,45,55,60,97,111,141,146,156,162,179,196,213,223,235,259,271],[57,62,80,167,182,210,252,282,302,401,408,465,496],[2,3,8,72,78,109,117,127,130]]))[runid]

    jetids = np.arange(1,1000,1)

    epoch_arr = np.arange(-60.0,60.1,0.5)
    SEA_arr = np.zeros_like(epoch_arr)

    for n in jetids:
        try:
            props = jio.PropReader(str(n).zfill(5),runid,580)
        except:
            continue

        time_arr = props.read("time")
        cent_arr = props.read(centering)/ja.sw_normalisation(runid,centering)
        if time_arr.size < thresh:
            continue

        var_arr = props.read(var)/ja.sw_normalisation(runid,var)
        try:
            var_arr /= sheath_pars_list(var)[1]
            var_arr -= 1
        except:
            pass

        res_arr = np.interp(epoch_arr,time_arr-time_arr[np.argmax(cent_arr)],var_arr,left=0.0,right=0.0)
        SEA_arr = np.vstack((SEA_arr,res_arr))

    SEA_arr = SEA_arr[1:]

    SEA_arr_mean = np.mean(SEA_arr,axis=0)
    SEA_arr_std = np.std(SEA_arr,ddof=1,axis=0)

    plt.ioff()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epoch time [s]",fontsize=20)
    
    try:
        ax.set_ylabel("Fractional increase {}".format(sheath_pars_list(var)[0]),fontsize=20)
    except:
        ax.set_ylabel("Averaged {}".format(var_pars_list(var)[0]),fontsize=20)
    
    plt.grid()
    plt.title("Run: {}, Epoch centering: {}".format(runid,centering.replace("_",r"\_")))
    ax.plot(epoch_arr,SEA_arr_mean,color="black")
    ax.fill_between(epoch_arr,SEA_arr_mean-SEA_arr_std,SEA_arr_mean+SEA_arr_std,alpha=0.3)

    plt.tight_layout()

    if not os.path.exists("Figures/SEA/"+runid+"/"):
        try:
            os.makedirs("Figures/SEA/"+runid+"/")
        except OSError:
            pass

    fig.savefig("Figures/SEA/{}/SEA_{}.png".format(runid,var))
    print("Figures/SEA/{}/SEA_{}.png".format(runid,var))

    plt.close(fig)

    return None

def SEA_script(centering="pd_avg",thresh=5):

    runids = ["ABA","ABC","AEA","AEC"]
    var = ["n_max","v_max","pd_max","n_avg","n_med","v_avg","v_med","pd_avg","pd_med","pdyn_vmax"]

    for runid in runids:
        for v in var:
            SEA_make(runid,v,centering=centering,thresh=thresh)

    return None


def jet_lifetime_plots(var,amax=True):

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
                    x_list_list[n].append(props.read("time")[-1]-props.read("time")[0])
                    if amax:
                        y_list_list[n].append(props.read_at_amax(var)/ja.sw_normalisation(runids[n],var))
                    else:
                        y_list_list[n].append(np.max(props.read(var))/ja.sw_normalisation(runids[n],var))
    plt.ioff()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Lifetime [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()

    lines = []
    labs = []

    for n in xrange(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=5,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=20)
    plt.legend(lines,labs,numpoints=1)
    plt.tight_layout()

    x_list_full = []
    y_list_full = []

    for n in range(len(x_list_list)):
        x_list_full+=x_list_list[n]
        y_list_full+=y_list_list[n]

    p = np.polyfit(x_list_full,y_list_full,deg=1)
    x_arr = np.arange(np.min(x_list_full),np.max(x_list_full),1)
    y_arr = np.polyval(p,x_arr)

    ax.plot(x_arr,y_arr,linestyle="dashed")
    ax.annotate(str(p[0])+"\n"+str(p[1]),xy=(0.1,0.9),xycoords="axes fraction")

    if not os.path.exists("Figures/paper/misc/scatter/"+"_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/misc/scatter/"+"_".join(runids)+"/")
        except OSError:
            pass

    if amax:
        fig.savefig("Figures/paper/misc/scatter/{}/{}_{}_amax.png".format("_".join(runids),"lifetime",var))
        print("Figures/paper/misc/scatter/{}/{}_{}_amax.png".format("_".join(runids),"lifetime",var))
    else:
        fig.savefig("Figures/paper/misc/scatter/{}/{}_{}.png".format("_".join(runids),"lifetime",var))
        print("Figures/paper/misc/scatter/{}/{}_{}.png".format("_".join(runids),"lifetime",var))

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
                    elif inp_var_list[ind] == "death_distance":
                        var_list[ind].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]]))
                    else:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/ja.sw_normalisation(runids[n],inp_var_list[ind]))

    v1_label,v1_xmin,v1_xmax,v1_step,v1_tickstep = var_pars_list(var1)
    v2_label,v2_xmin,v2_xmax,v2_step,v2_tickstep = var_pars_list(var2)

    # Create figure
    plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+v1_label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{"+v2_label[1:-1]+"}$",fontsize=24)
    ax.tick_params(labelsize=20)
    weights = [1/float(len(var_list[0]))]*len(var_list[0]) # Normalise by total number of jets
    bins = [np.linspace(xlims[0],xlims[1],21).tolist() for xlims in [[v1_xmin,v1_xmax],[v2_xmin,v2_xmax]]]

    hist = ax.hist2d(var_list[0],var_list[1],bins=bins,weights=weights)

    if v1_xmax == v2_xmax:
        ax.plot([0,v1_xmax],[0,v1_xmax],"r--")

    ax.set_xticks(np.arange(v1_xmin+v1_tickstep,v1_xmax+v1_tickstep,v1_tickstep))
    ax.set_yticks(np.arange(v2_xmin+v2_tickstep,v2_xmax+v2_tickstep,v2_tickstep))

    ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(v1_xmin+v1_tickstep,v1_xmax+v1_tickstep,v1_tickstep).astype(str)])
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(v2_xmin+v2_tickstep,v2_xmax+v2_tickstep,v2_tickstep).astype(str)])

    plt.title(",".join(runids),fontsize=24)
    plt.colorbar(hist[3], ax=ax)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
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

def jet_paper_vs_hist_new(runids_list,var,time_thresh=10):

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Different colors for different runs
    run_colors_list = ["red","blue"]

    var_list = [[],[]]

    for n in range(len(runids_list)):
        for runid in runids_list[n]:
            for jetid_nr in range(1,700):
                try:
                    props = jio.PropReader(ID=str(jetid_nr).zfill(5),runid=runid,start=580,transient="jet")
                    if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runid]:
                        if var == "duration":
                            var_list[n].append(props.read("time")[-1]-props.read("time")[0])
                        elif var == "size_ratio":
                            var_list[n].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                        elif var == "death_distance":
                            var_list[n].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-ja.bow_shock_r(runid,props.read("time")[-1]))
                        else:
                            var_list[n].append(props.read_at_amax(var)/ja.sw_normalisation(runid,var))
                except IOError:
                    continue

    label,xmin,xmax,step,tickstep = var_pars_list(var)

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{Fraction~of~jets}$",fontsize=24)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,0.75)
    ax.tick_params(labelsize=20)

    weights = [[1/float(len(var_list[n]))]*len(var_list[n]) for n in range(len(runids_list))] # Normalise by total number of jets

    ax.set_yticks(np.arange(0.1,0.8,0.1))
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(0.1,0.8,0.1).astype(str)])

    var_med = map(np.median,var_list)

    std_f = lambda l: np.std(l,ddof=1)
    var_std = map(std_f,var_list)

    var_labels = ["{}\nmed: {:.2f}\nstd: {:.2f}".format(",".join(runids_list[n]),var_med[n],var_std[n]) for n in range(len(runids_list))]

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax)
        
        hist = ax.hist(var_list,weights=weights,bins=bins,color=run_colors_list,label=var_labels)

        ax.set_xticks(np.array([10**0,10**1,10**2,10**3]))
        ax.set_xticklabels(np.array(["$\\mathtt{10^0}$","$\\mathtt{10^1}$","$\\mathtt{10^2}$","$\\mathtt{10^3}$"]))

    else:
        bins = np.arange(xmin,xmax+step,step)

        hist = ax.hist(var_list,weights=weights,bins=bins,color=run_colors_list,label=var_labels)

        ax.set_xticks(np.arange(xmin,xmax+tickstep,tickstep))
        ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin,xmax+tickstep,tickstep).astype(str)])

    if xmin == -xmax and 0.5*(xmax-xmin)%tickstep != 0.0:
        ax.set_xticks(np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep))
        if tickstep%1 != 0:
            ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep).astype(str)])
        else:
            ax.set_xticklabels(["$\\mathtt{"+str(int(lab))+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep)])

    plt.title(" vs. ".join([",".join(runids_list[n]) for n in range(len(runids_list))]),fontsize=24)
    plt.legend(fontsize=20)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    plt.tight_layout()

    outputfolder = "Figures/paper/histograms/{}/".format("_vs_".join([",".join(runids_list[n]) for n in range(len(runids_list))]))

    outputfilename = "{}_{}.png".format(var,time_thresh)

    # Create output directory
    if not os.path.exists(outputfolder):
        try:
            os.makedirs(outputfolder)
        except OSError:
            pass

    # Save figure
    fig.savefig(outputfolder+outputfilename)
    print(outputfolder+outputfilename)

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
                elif var == "death_distance":
                    val_dict[runids[n]].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-ja.bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    val_dict[runids[n]].append(props.read_at_amax(var)/ja.sw_normalisation(runids[n],var))


    label,xmin,xmax,step,tickstep = var_pars_list(var)

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{Fraction~of~jets}$",fontsize=24)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,0.75)
    ax.tick_params(labelsize=20)
    weights = [[1/float(len(val_dict[runids[n]]))]*len(val_dict[runids[n]]) for n in xrange(len(runids))] # Normalise by total number of jets

    ax.set_yticks(np.arange(0.1,0.8,0.1))
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(0.1,0.8,0.1).astype(str)])

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax)
        
        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],weights=weights,bins=bins,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

        ax.set_xticks(np.array([10**0,10**1,10**2,10**3]))
        ax.set_xticklabels(np.array(["$\\mathtt{10^0}$","$\\mathtt{10^1}$","$\\mathtt{10^2}$","$\\mathtt{10^3}$"]))

    else:
        bins = np.arange(xmin,xmax+step,step)

        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],bins=bins,weights=weights,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.2f\nstd: %.2f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.2f\nstd: %.2f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

        ax.set_xticks(np.arange(xmin,xmax+tickstep,tickstep))
        ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin,xmax+tickstep,tickstep).astype(str)])

    if xmin == -xmax and 0.5*(xmax-xmin)%tickstep != 0.0:
        ax.set_xticks(np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep))
        if tickstep%1 != 0:
            ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep).astype(str)])
        else:
            ax.set_xticklabels(["$\\mathtt{"+str(int(lab))+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep)])

    plt.title(" vs. ".join(runids),fontsize=24)
    plt.legend(fontsize=20)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
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
                elif var == "death_distance":
                    var_list.append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-ja.bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    var_list.append(props.read_at_amax(var)/ja.sw_normalisation(runids[n],var))

    var_list = np.asarray(var_list)

    label,xmin,xmax,step,tickstep = var_pars_list(var)

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{Fraction~of~jets}$",fontsize=24)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,0.6)
    ax.tick_params(labelsize=20)
    weights = np.ones(var_list.shape)/float(var_list.size) # Normalise by total number of jets

    ax.set_yticks(np.arange(0.1,0.7,0.1))
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(0.1,0.7,0.1).astype(str)])

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax)
        hist = ax.hist(var_list,weights=weights,bins=bins)
        ax.set_xticks(np.array([10**0,10**1,10**2,10**3]))
        ax.set_xticklabels(np.array(["$\\mathtt{10^0}$","$\\mathtt{10^1}$","$\\mathtt{10^2}$","$\\mathtt{10^3}$"]))

    else:
        bins = np.arange(xmin,xmax+step,step)
        hist = ax.hist(var_list,bins=bins,weights=weights)
        ax.set_xticks(np.arange(xmin,xmax+tickstep,tickstep))
        ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin,xmax+tickstep,tickstep).astype(str)])

    if xmin == -xmax and 0.5*(xmax-xmin)%tickstep != 0.0:
        ax.set_xticks(np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep))
        if tickstep%1 != 0:
            ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep).astype(str)])
        else:
            ax.set_xticklabels(["$\\mathtt{"+str(int(lab))+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep)])

    #ax.axvline(np.median(var_list), linestyle="dashed", color="black", linewidth=2)
    ax.annotate("med: %.2f\nstd: %.2f"%(np.median(var_list),np.std(var_list,ddof=1)), xy=(0.75,0.85), xycoords='axes fraction', fontsize=20, fontname="Computer Modern Typewriter")

    plt.title(",".join(runids),fontsize=24)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
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
        jet_paper_vs_hist_new(runids,var,time_thresh=10)

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