# import matplotlib
# #matplotlib.use('ps')
# from matplotlib import rc

# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
import matplotlib as mpl
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['black', 'blue', 'red', 'green'])
mpl.rcParams['axes.color_cycle'] = ['black', 'blue', 'red', 'green']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plot_contours as pc
import pytools as pt
import os
import scipy
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc
import jetfile_make as jfm
import jet_io as jio
import jet_aux as jx

#from matplotlib import rc
# font = {'family' : 'monospace',
#         'monospace' : 'Computer Modern Typewriter',
#         'weight' : 'bold'}

#rc('font', **font)
#rc('mathtext', fontset='custom')
#rc('mathtext', default='regular')

m_p = 1.672621898e-27
q_e = 1.602176565e-19
r_e = 6.371e+6
k_b = 1.3806488e-23
eVtoK = q_e/k_b

EkinBinEdges = np.logspace(np.log10(10),np.log10(2e4),66)

# wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"
# homedir = "/homeappl/home/sunijona/"
wrkdir_DNR = os.environ["WRK"]+"/"
homedir = os.environ["HOME"]+"/"


###TEMPORARY SCRIPTS HERE###


def jet_plotter(start,stop,runid,vmax=1.5,boxre=[6,18,-8,6],transient="jet"):
    # Plot jet countours and positions

    #outputdir = wrkdir_DNR+"contours/JETS/{}/".format(runid)
    if transient == "jet":
        outputdir = wrkdir_DNR+"contours/JETS/{}/".format(runid)
        inputpath = wrkdir_DNR+"working/"
    elif transient == "slams":
        outputdir = wrkdir_DNR+"contours/SLAMS/{}/".format(runid)
        inputpath = wrkdir_DNR+"working/SLAMS/"

    # Initialise required global variables
    global jet_cells
    global full_cells
    global xmean_list
    global ymean_list
    global xvmax_list
    global yvmax_list

    for n in range(start,stop+1):

        # Initialise lists of coordinates
        xmean_list = []
        ymean_list = []
        xvmax_list = []
        yvmax_list = []

        for itr in range(3000):

            # Try reading properties
            try:
                props = jio.PropReader(str(itr).zfill(5),runid,580,transient=transient)
                xmean_list.append(props.read_at_time("x_mean",float(n)/2))
                ymean_list.append(props.read_at_time("y_mean",float(n)/2))
                xvmax_list.append(props.read_at_time("x_vmax",float(n)/2))
                yvmax_list.append(props.read_at_time("y_vmax",float(n)/2))
            except IOError:
                pass

        # event_props = np.array(jio.eventprop_read(runid,n))
        # xmean_list = event_props[:,1]
        # ymean_list = event_props[:,2]
        # xvmax_list = event_props[:,11]
        # yvmax_list = event_props[:,12]


        # Try reading events file
        try:
            fileobj = open(inputpath+"events/{}/{}.events".format(runid,n),"r")
            contents = fileobj.read()
            fileobj.close()
            jet_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            jet_cells = []

        # Try reading mask file
        try:
            full_cells = np.loadtxt(inputpath+"Masks/{}/{}.mask".format(runid,n)).astype(int)
        except IOError:
            full_cells = []

        # Find correct file path
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

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir=outputdir,usesci=0,lin=1,boxre=boxre,expression=pc.expr_pdyn,vmin=0,vmax=vmax,colormap="parula",cbtitle="nPa",external=ext_jet,pass_vars=["rho","v","CellID"])

def ext_jet(ax,XmeshXY,YmeshXY,pass_maps):
    # External function for jet_plotter

    cellids = pass_maps["CellID"]

    # Mask jets
    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    # Mask full mask
    full_mask = np.in1d(cellids,full_cells).astype(int)
    full_mask = np.reshape(full_mask,cellids.shape)

    #full_cont = ax.contour(XmeshXY,YmeshXY,full_mask,[0.5],linewidths=0.8,colors="magenta") # Contour of full mask
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="black") # Contour of jets

    line1, = ax.plot(xmean_list,ymean_list,"o",color="red",markersize=2) # Mean positions
    line2, = ax.plot(xvmax_list,yvmax_list,"o",color="white",markersize=2) # v_max positions

def slamjet_plotter(start,stop,runid,vmax=1.5,boxre=[6,18,-8,6]):
    # Plot slamjets contours and positions

    outputdir = wrkdir_DNR+"contours/SLAMSJETS/{}/".format(runid)
    rid_list = ["ABA","ABC","AEA","AEC"]
    maxrange = [787,3000,3000,3000]

    # Initialise required global variables
    global jet_cells
    global slams_cells
    global xmean_list
    global ymean_list

    for n in range(start,stop+1):

        #Initialise lists of coordinaates
        xmean_list = []
        ymean_list = []

        for itr in range(1,maxrange[rid_list.index(runid)]+1):

            # Try reading properties
            try:
                props = jio.PropReader(str(itr).zfill(5),runid,580,transient="slamsjet")
                xmean_list.append(props.read_at_time("x_mean",float(n)/2))
                ymean_list.append(props.read_at_time("y_mean",float(n)/2))
            except IOError:
                pass

        # Try reading events file
        try:
            fileobj = open(wrkdir_DNR+"working/SLAMSJETS/events/{}/{}.events".format(runid,n),"r")
            contents = fileobj.read()
            fileobj.close()
            jet_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            jet_cells = []

        # Try reading SLAMS events file
        # try:
        #     fileobj = open("SLAMS/events/{}/{}.events".format(runid,n),"r")
        #     contents = fileobj.read()
        #     fileobj.close()
        #     slams_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        # except IOError:
        #     slams_cells = []

        # Find correct file path
        bulkpath = ja.find_bulkpath(runid)

        bulkname = "bulk.{}.vlsv".format(str(n).zfill(7))

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file {} not found, continuing".format(str(n)))
            continue

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir=outputdir,usesci=0,lin=1,boxre=boxre,expression=pc.expr_pdyn,vmin=0,vmax=vmax,colormap="parula",cbtitle="nPa",external=ext_slamjet,pass_vars=["rho","v","CellID"])

def ext_slamjet(ax,XmeshXY,YmeshXY,pass_maps):
    # External function for slamjet_plotter

    cellids = pass_maps["CellID"]

    # Mask jet cells
    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    # Mask SLAMS cells
    # slams_mask = np.in1d(cellids,slams_cells).astype(int)
    # slams_mask = np.reshape(slams_mask,cellids.shape)

    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="black") # Contour of jets
    # slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors="black") # Contour of SLAMS

    line1, = ax.plot(xmean_list,ymean_list,"o",color="red",markersize=2) # SLAMSJET mean positions

def draw_all_cont():
    # Draw contours for all criteria
    # NOT FUNCTIONAL
    #raise NotImplementedError("DEPRECATED")

    pt.plot.plot_colormap(filename="/proj/vlasov/2D/ABA/bulk/bulk.0000595.vlsv",outputdir="Contours/ALLCONT_",usesci=0,draw=1,lin=1,boxre=[4,18,-12,12],colormap="parula",cbtitle="nPa",scale=1,expression=pc.expr_pdyn,external=ext_crit,var="rho",vmin=0,vmax=1.5,wmark=1,pass_vars=["rho","v","CellID"])

def ext_crit(ax,XmeshXY,YmeshXY,extmaps):
    # NOT FUNCTIONAL
    #raise NotImplementedError("DEPRECATED")

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

    trho = np.loadtxt(wrkdir_DNR+"tavg/ABA/611_rho.tavg")[np.in1d(fullcells,cellids)]
    tpdyn = np.loadtxt(wrkdir_DNR+"tavg/ABA/611_pdyn.tavg")[np.in1d(fullcells,cellids)]

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
    #contour_plaschke = ax.contour(XmeshXY,YmeshXY,jetp.filled(),[0.5],linewidths=0.8, colors="black",label="Plaschke")

    contour_archer = ax.contour(XmeshXY,YmeshXY,jetah.filled(),[0.5],linewidths=0.8, colors="black",label="ArcherHorbury")

    #contour_karlsson = ax.contour(XmeshXY,YmeshXY,jetk.filled(),[0.5],linewidths=0.8, colors="magenta",label="Karlsson")

    return None

def lineout_plot(runid,filenumber,p1,p2,var):
    # DEPRECATED, new version incoming at some point
    #raise NotImplementedError("DEPRECATED, new version incoming at some point")

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
    ax.plot(var_arr)
    #ax.plot(r_arr,var_arr)
    #ax.set_xlabel("$R~[R_e]$",labelpad=10,fontsize=20)
    ax.tick_params(labelsize=20)
    if var in var_dict:
        ax.set_ylabel(var_dict[var][1],labelpad=10,fontsize=20)
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True,prune="lower"))

    plt.tight_layout()

    fig.show()

    fig.savefig("Contours/"+"lineout_"+runid+"_"+str(filenumber)+"_"+var+".png")

###PROP MAKER FILES HERE###



###HELPER FUNCTIONS HERE###

class MMSJet:

    def __init__(self):

        filepath = wrkdir_DNR+"working/MMS_jet/"

        ebins_f = open(filepath+"EnergyBins.txt","r+")
        ebins_c = ebins_f.read()
        ebins_f.close()
        ebins_cl = ebins_c.split("\r\n")[2:]
        self.energy_bins = np.asarray(ebins_cl,dtype=float)

        mfdata_f = open(filepath+"MagneticFieldData.txt","r+")
        mfdata_c = mfdata_f.read()
        mfdata_f.close()
        mfdata_cl = mfdata_c.split("\r\n")[1:-1]
        mfdata_mat = [line.split(",") for line in mfdata_cl]

        self.mfdata = np.asarray(mfdata_mat,dtype=float)

        jetdata_f = open(filepath+"TimeseriesData_Jet.txt","r+")
        jetdata_c = jetdata_f.read()
        jetdata_f.close()
        jetdata_cl = jetdata_c.split("\r\n")[1:-1]
        jetdata_mat = [line.split(",") for line in jetdata_cl]

        self.jetdata = np.asarray(jetdata_mat,dtype=float)

        self.mfvar_list = ["Year","Month","Day","Hour","Minute","Second","Bx","By","Bz","B"]
        self.jetvar_list = ["Year","Month","Day","Hour","Minute","Second","Pdyn","pdr","vx","vy","vz","v","rho","TParallel","TPerpendicular"]

    def read_mftime(self):
        time = self.mfdata[:,self.mfvar_list.index("Hour")]+self.mfdata[:,self.mfvar_list.index("Minute")]/60.0+self.mfdata[:,self.mfvar_list.index("Second")]/3600.0
        return time

    def read(self,name,tmin=0,tmax=999):
        if name == "time":
            time = self.jetdata[:,self.jetvar_list.index("Hour")]+self.jetdata[:,self.jetvar_list.index("Minute")]/60.0+self.jetdata[:,self.jetvar_list.index("Second")]/3600.0
            b_arr = np.logical_and(time>=tmin,time<=tmax)
            return time[b_arr]
        elif name == "flux":
            time = self.read("time")
            b_arr = np.logical_and(time>=tmin,time<=tmax)
            flux = self.jetdata[:,15:]
            flux[flux==0] = 1e-31
            return flux[b_arr]
        elif name in self.jetvar_list:
            time = self.read("time")
            b_arr = np.logical_and(time>=tmin,time<=tmax)
            if name in ["TParallel","TPerpendicular"]:
                return self.jetdata[:,self.jetvar_list.index(name)][b_arr]*eVtoK*1e-6
            elif name == "Pdyn":
                return self.jetdata[:,self.jetvar_list.index(name)][b_arr]*1.0e9
            else:
                return self.jetdata[:,self.jetvar_list.index(name)][b_arr]
        elif name in self.mfvar_list:
            in_data = self.mfdata[:,self.mfvar_list.index(name)]
            intpol_data = np.interp(self.read("time",tmin=tmin,tmax=tmax),self.read_mftime(),in_data)
            return intpol_data

    def read_mult(self,name_list,tmin=0,tmax=999):
        outm = np.array([self.read(name,tmin=tmin,tmax=tmax) for name in name_list])
        return outm

class MMSReader:

    def __init__(self,filename):

        filepath = wrkdir_DNR+"working/MMS_data/"+filename

        f = open(filepath,"r+")
        contents = f.read()
        f.close()
        contents_list = contents.split("\r\n")[1:-1]
        contents_matrix = [line.split(",") for line in contents_list]

        self.data_arr = np.asarray(contents_matrix,dtype=float)

        var_list = ["B_max","B_avg","beta_avg","size_rad","n_avg","n_max","pd_avg","pd_max","TPar_avg","TPerp_avg","T_avg","v_avg","v_max","DT","DT_SW","Dn_SW","Dv_SW","Dpd_SW","DB_SW"]

        n_list = range(len(var_list))

        self.var_dict = dict(zip(var_list,n_list))

        label_list = ["$|B|_{max}~[|B|_{IMF}]$","$|B|_{avg}~[|B|_{IMF}]$","$\\beta_{avg}~[\\beta_{sw}]$","$Extent~[R_e]$","$n_{avg}~[n_{sw}]$","$n_{max}~[n_{sw}]$","$P_{dyn,avg}~[P_{dyn,sw}]$","$P_{dyn,max}~[P_{dyn,sw}]$","$T_{\\parallel,avg}~[T_{sw}]$","$T_{\\perp,avg}~[T_{sw}]$","$T_{avg}~[T_{sw}]$","$|V|_{avg}~[V_{sw}]$","$|V|_{avg}~[V_{sw}]$","$\\Delta T~[K]$","$\\Delta T~[T_{sw}]$","\\Delta n~[n_{sw}]","\\Delta v~[v_{sw}]","\\Delta P_{dyn}~[P_{dyn,sw}]","\\Delta B~[B_{IMF}]"]

        self.label_dict = dict(zip(var_list,label_list))

    def read(self,name):
        if name in self.var_dict:
            outp = self.data_arr[:,self.var_dict[name]]
            return outp[~np.isnan(outp)]

    def read_mult(self,name_list):

        outp_list = [self.read(name) for name in name_list]

        return np.array(outp_list,dtype=object)

def sheath_pars_list(var):
    # Returns scaling factors for variables based on the maximum compression ratio of the RH conditions

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
    # Returns a list of parameters useful for plotting specified variable

    # Keys of variable names
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

    # Labels for figures
    label_list = ["$Lifetime~[s]$",
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

    # Minimum variable value
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

    # Maximum variable value
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

    # Histogram bin widths
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

    # Axis tick distance
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
    # Draws the location of all jets in specified run on an r-phi plane and a histogram of jet r-values
    # For easy identification of magnetopause false positive jets

    filenames = os.listdir(wrkdir_DNR+"working/jets/"+runid)

    propfiles = [filename for filename in filenames if ".props" in filename]

    r_list = []
    phi_list = []
    size_list = []

    for fname in propfiles:
        props = pd.read_csv(wrkdir_DNR+"working/jets/"+runid+"/"+fname).as_matrix()
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
    r_hist = ax2.hist(r_list,bins=list(range(0,19)))

    plt.tight_layout()

    if not os.path.exists("Figures/jets/debugging/"):
        try:
            os.makedirs("Figures/jets/debugging/")
        except OSError:
            pass

    fig.savefig("Figures/jets/debugging/"+runid+"_"+"rmax.png")

    plt.close(fig)

    return None

def try_test():
    try:
        a="b"+1
        print("Hello")
    except:
        print("No")

    return None

def slams_jet_counter():

    runids = ["ABA","ABC","AEA","AEC"]

    time_per_run = np.array([839-580,1179-580,1339-580,879-580])/2.0

    slams_counter = np.array([0,0,0,0],dtype=int)
    jet_counter = np.array([0,0,0,0],dtype=int)
    slamsjet_counter = np.array([0,0,0,0],dtype=int)

    for n in range(0,3000):
        pass

def jet_paper_counter():
    # Counts the number of jets in each run, excluding false positives and short durations

    # List of runids
    runids = ["ABA","ABC","AEA","AEC"]
    #runids = ["ABA"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir(wrkdir_DNR+"working/jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff for false positives
    #run_cutoff_dict = dict(zip(runids,[10]))
    run_cutoff_dict = dict(zip(runids,[10,8,10,8]))

    # Initialise list of counts
    count_list_list = [0,0,0,0]
    #count_list_list = [0]
    time_per_run = np.array([839-580,1179-580,1339-580,879-580])/2.0

    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)

            # Conditions
            if props.read("time")[-1]-props.read("time")[0] + 0.5 > 5 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    count_list_list[n] += 1 # Iterate counter if conditions fulfilled


    print(count_list_list)
    return np.array(count_list_list)/time_per_run

def jet_paper_pos():
    # Draws locations of all jets in ecliptic runs on xy-plane at time of maximum area

    # List of runids
    runids = ["ABA","ABC","AEA","AEC"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir(wrkdir_DNR+"working/jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Dictionaries for false positive cutoff, marker shape and colour
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC"],[10,8,10,8]))
    run_marker_dict = dict(zip(["ABA","ABC","AEA","AEC"],["x","o","^","d"]))
    run_color_dict = dict(zip(["ABA","ABC","AEA","AEC"],["black","red","blue","green"]))

    # Initialise lists of coordinates
    x_list_list = [[],[],[],[]]
    y_list_list = [[],[],[],[]]

    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)

            # Conditions
            if props.read("time")[-1]-props.read("time")[0] + 0.5 > 10 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    x_list_list[n].append(props.read_at_amax("x_mean"))
                    y_list_list[n].append(props.read_at_amax("y_mean"))

    plt.ioff()

    # Draw figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X [R$_{e}$]",fontsize=24,labelpad=10)
    ax.set_ylabel("Y [R$_{e}$]",fontsize=24,labelpad=10)
    ax.set_xlim(6,18)
    ax.set_ylim(-9,7)
    ax.tick_params(labelsize=20)
    lines = []
    labs = []

    for n in range(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=10,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    #plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=24)
    plt.legend(lines,labs,numpoints=1,prop={"size":20})
    plt.tight_layout()

    # Save figure
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
    # Creates multivariable time series for specified jet

    # Check transient type
    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    # Open properties file, read variable data
    props = jio.PropReader(jetid,runid,start,transient=transient)
    var_list = ["time","A","n_max","v_max","pd_max","r_mean"]
    time_arr,area_arr,n_arr,v_arr,pd_arr,r_arr = [props.read(var)/ja.sw_normalisation(runid,var) for var in var_list]

    # Threshold condition
    if np.max(area_arr) < thresh or time_arr.size < 10:
        print("Jet smaller than threshold, exiting!")
        return None

    # Draw figure
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
    #ax2.plot(time_arr,ja.bow_shock_r(runid,time_arr)/np.max(r_arr),label="Bow shock")

    ax.legend(loc="lower right")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    # Save figure
    fig.savefig("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))
    print("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))

    plt.close(fig)

    return None

def jet_time_series(runid,start,jetid,var,thresh = 0.0,transient="jet"):
    # Creates timeseries of specified variable for specified jet

    # Check transient type
    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    # Open props file, read time, area and variable data
    props = jio.PropReader(jetid,runid,start,transient=transient)
    time_arr = props.read("time")
    area_arr = props.read("A")
    var_arr = props.read(var)/ja.sw_normalisation(runid,var)

    # Threshold condition
    if np.max(area_arr) < thresh:
        print("Jet smaller than threshold, exiting!")
        return None

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()
    plt.title("Run: {}, ID: {}".format(runid,jetid))
    ax.plot(time_arr,var_arr,color="black")

    plt.tight_layout()

    # Save figure
    fig.savefig("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))
    print("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))

    plt.close(fig)

    return None

def jts_make(runid,start,startid,stopid,thresh = 0.0,transient="jet"):
    # Script for creating time series for multiple jets

    for n in range(startid,stopid+1):
        try:
            jet_mult_time_series(runid,start,str(n).zfill(5),thresh=thresh,transient=transient)
        except IOError:
            print("Could not create time series!")

    return None

def SEA_make(runid,var,centering="pd_avg",thresh=5):
    # Creates Superposed Epoch Analysis of jets in specified run, centering specified var around maximum of
    # specified centering variable

    #jetids = dict(zip(["ABA","ABC","AEA","AEC"],[[2,29,79,120,123,129],[6,12,45,55,60,97,111,141,146,156,162,179,196,213,223,235,259,271],[57,62,80,167,182,210,252,282,302,401,408,465,496],[2,3,8,72,78,109,117,127,130]]))[runid]

    # Range of jetids to attempt
    jetids = np.arange(1,2500,1)

    # Define epoch time array, +- 1 minute from center
    epoch_arr = np.arange(-60.0,60.1,0.5)
    SEA_arr = np.zeros_like(epoch_arr) # Initialise superposed epoch array

    for n in jetids:

        # Try reading jet
        try:
            props = jio.PropReader(str(n).zfill(5),runid,580)
        except:
            continue

        # Read time and centering
        time_arr = props.read("time")
        cent_arr = props.read(centering)/ja.sw_normalisation(runid,centering)

        # Threshold condition
        if time_arr.size < thresh:
            continue

        # Read variable data
        var_arr = props.read(var)/ja.sw_normalisation(runid,var)

        # Try scaling to fractional increase
        try:
            var_arr /= sheath_pars_list(var)[1]
            var_arr -= 1
        except:
            pass

        # Interpolate variable data to fit epoch time, and stack it with SEA array
        res_arr = np.interp(epoch_arr,time_arr-time_arr[np.argmax(cent_arr)],var_arr,left=0.0,right=0.0)
        SEA_arr = np.vstack((SEA_arr,res_arr))

    # Remove the row of zeros from stack
    SEA_arr = SEA_arr[1:]

    # Calculate mean and STD of the stack
    SEA_arr_mean = np.mean(SEA_arr,axis=0)
    SEA_arr_std = np.std(SEA_arr,ddof=1,axis=0)

    # Draw figure
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

    # Save figure
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
    # Script for making several SEA graphs for different runs

    runids = ["ABA","ABC","AEA","AEC"]
    var = ["n_max","v_max","pd_max","n_avg","n_med","v_avg","v_med","pd_avg","pd_med","pdyn_vmax"]

    for runid in runids:
        for v in var:
            SEA_make(runid,v,centering=centering,thresh=thresh)

    return None


def jet_lifetime_plots(var,amax=True):
    # Creates scatter plot of jet lifetime versus variable value either at time of maximum area or global
    # maximum for all ecliptical runs.

    # List of runids
    runids = ["ABA","ABC","AEA","AEC"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir(wrkdir_DNR+"working/jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Dictionaries for false positive cutoff, marker shape and colour
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC"],[10,8,10,8]))
    run_marker_dict = dict(zip(["ABA","ABC","AEA","AEC"],["x","o","^","d"]))
    run_color_dict = dict(zip(["ABA","ABC","AEA","AEC"],["black","red","blue","green"]))

    # Initialise lists of coordinates
    x_list_list = [[],[],[],[]]
    y_list_list = [[],[],[],[]]

    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)

            # Condition
            if props.read("time")[-1]-props.read("time")[0] + 0.5 > 10 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    x_list_list[n].append(props.read("time")[-1]-props.read("time")[0])
                    if amax:
                        y_list_list[n].append(props.read_at_amax(var)/ja.sw_normalisation(runids[n],var))
                    else:
                        y_list_list[n].append(np.max(props.read(var))/ja.sw_normalisation(runids[n],var))

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Lifetime [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()

    lines = []
    labs = []

    for n in range(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=5,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=20)
    plt.legend(lines,labs,numpoints=1)
    plt.tight_layout()

    # Fit line to data and draw it
    x_list_full = []
    y_list_full = []

    for n in range(len(x_list_list)):
        x_list_full+=x_list_list[n]
        y_list_full+=y_list_list[n]

    p = np.polyfit(x_list_full,y_list_full,deg=1)
    x_arr = np.arange(np.min(x_list_full),np.max(x_list_full),1)
    y_arr = np.polyval(p,x_arr)

    ax.plot(x_arr,y_arr,linestyle="dashed")

    # TO DO: Make annotation look nice DONE
    ax.annotate("y = {:5.3f}x + {:5.3f}".format(p[0],p[1]),xy=(0.1,0.9),xycoords="axes fraction")

    # Save figure
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

### HACKATHON 2019 SCRIPTS HERE ###

def hack_2019_fig4(time_thresh=5):

    vlas_norm = np.array([1,1,1,1,1,1/0.5,1/0.5])
    mms_norm = np.array([1,1,1,1,1,1.0e+6,1.0e+6])
    bins_list = np.array([np.linspace(0,1.6,10+1),np.linspace(1.5,7.5,10+1),np.linspace(0,1,10+1),np.linspace(0,3,10+1),np.linspace(0,12,10+1),np.linspace(0,15,10+1)])

    var_list = ["size_rad","n_max","v_max","pd_max","B_max","TPerp_avg","TPar_avg"]
    ylabel_list = ["$\mathrm{Extent~[R_e]}$","$\mathrm{n_{max}~[n_{sw}]}$","$\mathrm{|v|_{max}~[v_{sw}]}$","$\mathrm{P_{dyn,max}~[P_{dyn,sw}]}$","$\mathrm{|B|_{max}~[B_{IMF}]}$","$\mathrm{T_{mean}~[MK]}$"]
    xlabel_list = ["VLMax","VLRand","MMS"]

    mms_reader = MMSReader("StableJets.txt")

    MMS_ext,MMS_n,MMS_v,MMS_pd,MMS_B,MMS_TPerp,MMS_TPar = mms_reader.read_mult(var_list)
    VLH_ext,VLH_n,VLH_v,VLH_pd,VLH_B,VLH_TPerp,VLH_TPar = read_mult_runs(var_list,time_thresh,amax=True)
    VLR_ext,VLR_n,VLR_v,VLR_pd,VLR_B,VLR_TPerp,VLR_TPar = read_mult_runs(var_list,time_thresh)

    darr_list = [[VLH_ext,VLR_ext,MMS_ext],[VLH_n,VLR_n,MMS_n],[VLH_v,VLR_v,MMS_v],[VLH_pd,VLR_pd,MMS_pd],[VLH_B,VLR_B,MMS_B],[VLH_TPerp,VLR_TPerp,MMS_TPerp],[VLH_TPar,VLR_TPar,MMS_TPar]]

    fig,ax_list = plt.subplots(6,3,figsize=(10,15),sharey=True)

    for row in range(6):
        for col in range(3):
            if col != 2:
                norm = vlas_norm[row]
            else:
                norm = mms_norm[row]
            ax = ax_list[row][col]
            data_arr = darr_list[row][col]/norm
            weights = np.ones(data_arr.shape,dtype=float)/data_arr.size

            if col == 0 and row == 5:
                lab = "TPerp\nmed:{:.2f}\nstd:{:.2f}".format(np.median(data_arr),np.std(data_arr,ddof=1))
            else:
                lab = "med:{:.2f}\nstd:{:.2f}".format(np.median(data_arr),np.std(data_arr,ddof=1))
            ax.hist(data_arr,weights=weights,label=lab,histtype="step",bins=bins_list[row])

            ax.yaxis.set_major_locator(MaxNLocator(nbins=7,prune="lower"))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_ylim(0,0.8)
            ax.legend(fontsize=10,frameon=False)

            if row == 5:
                data_arr_2 = darr_list[6][col]/norm
                weights_2 = np.ones(data_arr_2.shape,dtype=float)/data_arr_2.size

                if col == 0:
                    lab = "TPar\nmed:{:.2f}\nstd:{:.2f}".format(np.median(data_arr_2),np.std(data_arr_2,ddof=1))
                else:
                    lab = "med:{:.2f}\nstd:{:.2f}".format(np.median(data_arr_2),np.std(data_arr_2,ddof=1))
                ax.hist(data_arr_2,weights=weights,label=lab,histtype="step",bins=bins_list[row])

                ax.set_xlabel(xlabel_list[col],labelpad=10,fontsize=20)
                ax.legend(fontsize=10,frameon=False)
            if col == 0:
                ax.set_ylabel(ylabel_list[row],labelpad=10,fontsize=15)

    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/fig4.png")
    plt.close(fig)

def hack_2019_fig6(time_thresh=5):

    var_list = ["duration","size_tan","size_ratio"]
    label_list = ["$\mathrm{Lifetime~[s]}$","$\mathrm{Tangential~size~[R_e]}$","$\mathrm{Size~ratio}$"]
    bins_list = np.array([np.linspace(time_thresh,60,10+1),np.linspace(0,0.5,10+1),np.linspace(0,5,10+1)])

    data_list = read_mult_runs(var_list,time_thresh,runids=["ABA","ABC","AEA","AEC"],amax=False)

    fig,ax_list = plt.subplots(1,3,figsize=(10,5),sharey=True)

    for col in range(3):
        ax = ax_list[col]
        var = data_list[col]
        weights = np.ones(var.shape,dtype=float)/var.size
        lab = "med:{:.2f}\nstd:{:.2f}".format(np.median(var),np.std(var))

        ax.hist(var,weights=weights,label=lab,histtype="step",bins=bins_list[col])
        ax.legend(fontsize=10,frameon=False)
        ax.set_xlabel(label_list[col],fontsize=15)
        ax.set_ylim(0,1)

    ax_list[0].set_ylabel("Fraction of jets",fontsize=15,labelpad=10)

    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/fig6.png")
    plt.close(fig)

def hack_2019_fig6_alt(time_thresh=5):

    runids_list = ["ABA","ABC","AEA","AEC"]
    cutoff_list = [10,8,10,8]
    cutoff_dict = dict(zip(runids_list,cutoff_list))
    bins_list = np.array([np.linspace(time_thresh,60,20+1),np.linspace(0,0.5,25+1),np.linspace(0,5,25+1)])

    var_list = ["duration","size_tan","size_ratio"]
    label_list = ["$\mathrm{Lifetime~[s]}$","$\mathrm{Tangential~size~[R_e]}$","$\mathrm{Size~ratio}$"]

    ABA_vars = read_mult_runs(var_list,time_thresh,runids=["ABA"],amax=False)
    ABC_vars = read_mult_runs(var_list,time_thresh,runids=["ABC"],amax=False)
    AEA_vars = read_mult_runs(var_list,time_thresh,runids=["AEA"],amax=False)
    AEC_vars = read_mult_runs(var_list,time_thresh,runids=["AEC"],amax=False)

    fig,ax_list = plt.subplots(1,3,figsize=(10,5),sharey=True)

    for col in range(3):
        ax = ax_list[col]
        var_arr = [ABA_vars[col],ABC_vars[col],AEA_vars[col],AEC_vars[col]]
        weights_arr = [np.ones(var.shape,dtype=float)/var.size for var in var_arr]
        med_arr = [np.median(var) for var in var_arr]
        std_arr = [np.std(var,ddof=1) for var in var_arr]
        #labs_arr = ["{} med:{:.2f} std:{:.2f}".format(runids_list[itr],med_arr[itr],std_arr[itr]) for itr in range(len(var_arr))]
        labs_arr = ["{} med:{:.2f}".format(runids_list[itr],med_arr[itr]) for itr in range(len(var_arr))]
        color_arr = ["black","blue","red","green"]

        ax.hist(var_arr,weights=weights_arr,label=labs_arr,color=color_arr,histtype="step",bins=bins_list[col])
        ax.legend(fontsize=10,frameon=False)
        ax.set_xlabel(label_list[col],fontsize=15)
        ax.set_ylim(0,0.5)


    ax_list[0].set_ylabel("Fraction of jets",fontsize=15,labelpad=10)

    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/fig6_alt.png")
    plt.close(fig)

def jetcand_vdf(runid):

    outputdir = "/homeappl/home/sunijona/Figures/paper/vdfs/"
    title_list = ["{} t0-30".format(runid),"{} t0".format(runid),"{} t0+30".format(runid)]

    if runid == "AEA":
        bulkpath = "/proj/vlasov/2D/AEA/round_3_boundary_sw/"
        fn_list = [760,820,880]
        cellid = 1301051
    else:
        bulkpath = "/proj/vlasov/2D/AEC/"
        fn_list = [700,760,820]
        cellid = 1700451

    vlsvobj_list = [pt.vlsvfile.VlsvReader(bulkpath+"bulk.{}.vlsv".format(str(fn).zfill(7))) for fn in fn_list]

    for fn in fn_list:

        #v = vlsvobj_list[fn_list.index(fn)].read_variable("v",cellids=cellid)
        #perp1 = [1,0,0]
        #perp2 = np.cross(v,perp1)

        #pt.plot.plot_vdf(vlsvobj=vlsvobj_list[fn_list.index(fn)],outputdir=outputdir,cellids=[cellid],run=runid,step=fn,box=[-5e+6,5e+6,-5e+6,5e+6],fmin=1e-14,fmax=1e-9,normal=perp2,normalx=v,slicethick=0,title=title_list[fn_list.index(fn)])

        pt.plot.plot_vdf(vlsvobj=vlsvobj_list[fn_list.index(fn)],outputdir=outputdir,cellids=[cellid],run=runid,step=fn,box=[-5e+6,5e+6,-5e+6,5e+6],fmin=1e-14,fmax=1e-9,bpara=True,slicethick=0,title=title_list[fn_list.index(fn)])

        pt.plot.plot_vdf(vlsvobj=vlsvobj_list[fn_list.index(fn)],outputdir=outputdir,cellids=[cellid],run=runid,step=fn,box=[-5e+6,5e+6,-5e+6,5e+6],fmin=1e-14,fmax=1e-9,bpara1=True,slicethick=0,title=title_list[fn_list.index(fn)])

        pt.plot.plot_vdf(vlsvobj=vlsvobj_list[fn_list.index(fn)],outputdir=outputdir,cellids=[cellid],run=runid,step=fn,box=[-5e+6,5e+6,-5e+6,5e+6],fmin=1e-14,fmax=1e-9,bperp=True,slicethick=0,title=title_list[fn_list.index(fn)])

def read_mult_runs(var_list,time_thresh,runids=["ABA","ABC","AEA","AEC"],amax=False):

    if type(var_list) == str:
        var_list = [var_list]

    runids_list = ["ABA","ABC","AEA","AEC"]
    cutoff_list = [10,8,10,8]
    cutoff_dict = dict(zip(runids_list,cutoff_list))

    data_list = [[] for var in var_list]

    for n in range(1,2500):
        for runid in runids:
            try:
                props = jio.PropReader(str(n).zfill(5),runid,580)
            except:
                continue

            if props.read("duration")[0] < time_thresh or max(props.read("r_mean")) < cutoff_dict[runid]:
                continue

            for var in var_list:
                if amax:
                    data_list[var_list.index(var)].append(props.read_at_amax(var)/ja.sw_normalisation(runid,var))
                else:
                    data_list[var_list.index(var)].append(props.read_at_randt(var)/ja.sw_normalisation(runid,var))

    return np.array(data_list)

def DT_comparison(time_thresh=5):

    xlabel_list = ["VLMax","VLRand","MMS"]
    ylabel_list = ["$\mathrm{\\Delta T~[MK]}$","$\mathrm{\\Delta n~[n_{SW}]}$","$\mathrm{\\Delta |v|~[v_{SW}]}$","$\mathrm{\\Delta P_{dyn}~[P_{dyn,SW}]}$","$\mathrm{\\Delta |B|~[B_{IMF}]}$"]
    hist_bins = [np.linspace(-5,5,10+1),np.linspace(-2,4,10+1),np.linspace(-0.1,0.4,10+1),np.linspace(0,2,10+1),np.linspace(-2,2,10+1)]

    mms_props = MMSReader("StableJets.txt")

    DT_MMS = mms_props.read("DT")/1.0e+6
    Dn_MMS = mms_props.read("Dn_SW")
    Dv_MMS = mms_props.read("Dv_SW")
    Dpd_MMS = mms_props.read("Dpd_SW")
    DB_MMS = mms_props.read("DB_SW")

    DT_vlas_amax = read_mult_runs("DT",time_thresh,amax=True)[0]/2.0
    DT_vlas_randt = read_mult_runs("DT",time_thresh)[0]/2.0

    Dn_vlas_amax = read_mult_runs("Dn",time_thresh,amax=True)[0]
    Dn_vlas_randt = read_mult_runs("Dn",time_thresh)[0]

    Dv_vlas_amax = read_mult_runs("Dv",time_thresh,amax=True)[0]
    Dv_vlas_randt = read_mult_runs("Dv",time_thresh)[0]

    Dpd_vlas_amax = read_mult_runs("Dpd",time_thresh,amax=True)[0]
    Dpd_vlas_randt = read_mult_runs("Dpd",time_thresh)[0]

    DB_vlas_amax = read_mult_runs("DB",time_thresh,amax=True)[0]
    DB_vlas_randt = read_mult_runs("DB",time_thresh)[0]

    var_list = [[DT_vlas_amax,DT_vlas_randt,DT_MMS],[Dn_vlas_amax,Dn_vlas_randt,Dn_MMS],[Dv_vlas_amax,Dv_vlas_randt,Dv_MMS],[Dpd_vlas_amax,Dpd_vlas_randt,Dpd_MMS],[DB_vlas_amax,DB_vlas_randt,DB_MMS]]

    fig,ax_list = plt.subplots(5,3,figsize=(10,12),sharey=True)

    for row in range(5):
        for col in range(3):
            ax = ax_list[row][col]
            var = var_list[row][col]
            weights = np.ones(var.shape,dtype=float)/var.size

            ax.hist(var,weights=weights,label="med:{:.2f}\nstd:{:.2f}".format(np.median(var),np.std(var,ddof=1)),histtype="step",bins=hist_bins[row])
            ax.legend(handlelength=0,frameon=False)
            ax.set_ylim(0,1)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == 4:
                ax.set_xlabel(xlabel_list[col],labelpad=10,fontsize=20)
            if col == 0:
                ax.set_ylabel(ylabel_list[row],labelpad=10,fontsize=20)

    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/DT_comp.png")
    plt.close(fig)

def DT_mach_comparison(time_thresh=5):

    xlabel_list = ["VLMax","VLRand","MMS"]
    ylabel_list = ["$\mathrm{\\Delta T~[MK]}$","$\mathrm{\\Delta n~[n_{sw}]}$","$\mathrm{\\Delta |v|~[v_{sw}]}$","$\mathrm{\\Delta P_{dyn}~[P_{dyn,sw}]}$","$\mathrm{\\Delta |B|~[B_{IMF}]}$"]
    hist_bins = [np.linspace(-5,5,10+1),np.linspace(-2,4,10+1),np.linspace(-0.1,0.4,10+1),np.linspace(0,2,10+1),np.linspace(-2,2,10+1)]

    mms_var_list = ["DT","Dn_SW","Dv_SW","Dpd_SW","DB_SW"]
    mms_norm_list = [1.0e+6,1,1,1,1]

    vlas_var_list = ["DT","Dn","Dv","Dpd","DB"]
    vlas_norm_list = [2.0,1,1,1,1]

    mms_props_low = MMSReader("LowMachJets.txt")
    mms_props_high = MMSReader("HighMachJets.txt")

    MMS_low = [mms_props_low.read(var)/mms_norm_list[mms_var_list.index(var)] for var in mms_var_list]
    MMS_high = [mms_props_high.read(var)/mms_norm_list[mms_var_list.index(var)] for var in mms_var_list]

    VLMax_low = [read_mult_runs(var,time_thresh,amax=True,runids=["AEA","AEC"])[0]/vlas_norm_list[vlas_var_list.index(var)] for var in vlas_var_list]
    VLMax_high = [read_mult_runs(var,time_thresh,amax=True,runids=["ABA","ABC"])[0]/vlas_norm_list[vlas_var_list.index(var)] for var in vlas_var_list]

    VLRand_low = [read_mult_runs(var,time_thresh,runids=["AEA","AEC"])[0]/vlas_norm_list[vlas_var_list.index(var)] for var in vlas_var_list]
    VLRand_high = [read_mult_runs(var,time_thresh,runids=["ABA","ABC"])[0]/vlas_norm_list[vlas_var_list.index(var)] for var in vlas_var_list]

    var_list_low = [VLMax_low,VLRand_low,MMS_low]
    var_list_high = [VLMax_high,VLRand_high,MMS_high]

    fig,ax_list = plt.subplots(5,3,figsize=(10,12),sharey=True)

    for row in range(5):
        for col in range(3):
            ax = ax_list[row][col]
            var_low = var_list_low[col][row]
            var_high = var_list_high[col][row]
            weights_low = np.ones(var_low.shape,dtype=float)/var_low.size
            weights_high = np.ones(var_high.shape,dtype=float)/var_high.size

            lab = ["med:{:.2f}\nstd:{:.2f}".format(np.median(var_low),np.std(var_low,ddof=1)),"med:{:.2f}\nstd:{:.2f}".format(np.median(var_high),np.std(var_high,ddof=1))]

            ax.hist([var_low,var_high],weights=[weights_low,weights_high],label=lab,color=["blue","red"],histtype="step",bins=hist_bins[row])
            ax.legend(fontsize=10,frameon=False)
            ax.set_ylim(0,1)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == 4:
                ax.set_xlabel(xlabel_list[col],labelpad=10,fontsize=20)
            if col == 0:
                ax.set_ylabel(ylabel_list[row],labelpad=10,fontsize=20)

    ax_list[0][0].set_title("Low Mach Jets",color="blue")
    ax_list[0][1].set_title("High Mach Jets",color="red")

    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/DT_mach_comp.png")
    plt.close(fig)

def colorbar(mappable,ax_list):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    div_list = [make_axes_locatable(a) for a in ax_list if a!=ax]
    cax_list = [divider.append_axes("right",size="5%",pad=0.05) for divider in div_list]
    [cx.set_axis_off() for cx in cax_list]

    return fig.colorbar(mappable, cax=cax)

def get_timeseries(runid,start,stop,var_list,cellids):

    if type(var_list) == str:
        var_list = [var_list]

    norm_list = [1.0e-9,1.0e+3,1.0e-9,1.0e+6,1.0e+6,1.0e+6]
    all_vars = ["Pdyn","v","B","rho","TParallel","TPerpendicular"]
    v_vars = ["vx","vy","vz"]
    B_vars = ["Bx","By","Bz"]
    coord_vars = ["x","y","z"]

    if len(var_list) == 1:
        data_arr = np.zeros(stop-start+1)
    else:
        data_arr = np.zeros((len(var_list),stop-start+1),dtype=float)

    time_arr = np.zeros_like(data_arr)

    bulkpath = ja.find_bulkpath(runid)
    itr = 0

    for n in range(start,stop+1):
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+"bulk.{}.vlsv".format(str(n).zfill(7)))
        vlsvobj.optimize_open_file()
        for var in var_list:
            if var in v_vars:
                if len(var_list) == 1:
                    data_arr[itr] = vlsvobj.read_variable("v",operator=coord_vars[v_vars.index(var)],cellids=cellids)/norm_list[all_vars.index("v")]
                    time_arr[itr] = vlsvobj.read_parameter("t")
                else:
                    data_arr[var_list.index(var),itr] = vlsvobj.read_variable("v",operator=coord_vars[v_vars.index(var)],cellids=cellids)/norm_list[all_vars.index("v")]
                    time_arr[var_list.index(var),itr] = vlsvobj.read_parameter("t")
            elif var in B_vars:
                if len(var_list) == 1:
                    data_arr[itr] = vlsvobj.read_variable("B",operator=coord_vars[v_vars.index(var)],cellids=cellids)/norm_list[all_vars.index("B")]
                    time_arr[itr] = vlsvobj.read_parameter("t")
                else:
                    data_arr[var_list.index(var),itr] = vlsvobj.read_variable("B",operator=coord_vars[B_vars.index(var)],cellids=cellids)/norm_list[all_vars.index("B")]
                    time_arr[var_list.index(var),itr] = vlsvobj.read_parameter("t")
            else:
                if len(var_list) == 1:
                    data_arr[itr] = vlsvobj.read_variable(var,operator="magnitude",cellids=cellids)/norm_list[all_vars.index(var)]
                    time_arr[itr] = vlsvobj.read_parameter("t")
                else:
                    data_arr[var_list.index(var),itr] = vlsvobj.read_variable(var,operator="magnitude",cellids=cellids)/norm_list[all_vars.index(var)]
                    time_arr[var_list.index(var),itr] = vlsvobj.read_parameter("t")

        itr += 1
        vlsvobj.optimize_clear_fileindex_for_cellid()
        vlsvobj.optimize_close_file()

    return (time_arr,data_arr)

def MMS_pos(filename):

    filepath = wrkdir_DNR+"working/MMS_data/"+filename

    f = open(filepath,"r+")
    contents = f.read()
    f.close()
    contents_list = contents.split("\r\n")[0:-1]
    contents_matrix = [line.split(",") for line in contents_list]

    coords = np.array(contents_matrix)[:,:-1].T

    return coords

def hack_2019_fig35():

    var_list = ["x_mean","y_mean"]

    coords_high = read_mult_runs(var_list,5,["ABA","ABC"],amax=True)
    coords_low = read_mult_runs(var_list,5,["AEA","AEC"],amax=True)

    mms_high = MMS_pos("HighMachJetsPosition.txt")
    mms_low = MMS_pos("LowMachJetsPosition.txt")

    # bs_y = np.arange(-10,10,0.01)
    # mp_p,bs_p = ja.bs_mp_fit("AEA",1320,[5,20,-10,10])
    # bs_x = np.polyval(bs_p,bs_y)
    # mp_x = np.polyval(mp_p,bs_y)+1.5
    bs_x,bs_y=jx.BS_xy()
    mp_x,mp_y=jx.MP_xy()

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    ax.plot(mp_x,bs_y,color="black")
    ax.plot(bs_x,bs_y,color="black")
    ax.plot(mms_high[0],mms_high[1],"x",color=jx.violet,mec=jx.violet,markersize=4,label="MMS High Mach")
    ax.plot(mms_low[0],mms_low[1],"x",color=jx.medium_blue,mec=jx.medium_blue,markersize=4,label="MMS Low Mach")
    ax.plot(coords_low[0],coords_low[1],"o",color=jx.dark_blue,mec=jx.dark_blue,markersize=5,label="Vlas Low Mach")
    ax.plot(coords_high[0],coords_high[1],"o",color=jx.orange,mec=jx.orange,markersize=5,label="Vlas High Mach")


    ax.set_xlim(6,20)
    ax.set_ylim(-8,6)
    ax.legend(frameon=False,numpoints=1,markerscale=3)
    ax.set_xlabel("X [$\mathrm{R_e}$]",fontsize=20,labelpad=10)
    ax.set_ylabel("Y [$\mathrm{R_e}$]",fontsize=20,labelpad=10)
    plt.tight_layout()

    fig.savefig(homedir+"Figures/hackathon_paper/fig35.png")
    plt.close(fig)

def hack_2019_fig2(runid,htw = 60):

    runids = ["AEA","AEC"]
    r_id = runids.index(runid)
    color_list = ["black","blue","red","green"]
    mms_min = 5+34.2/60-240.0/3600
    mms_max = 5+34.2/60+240.0/3600

    filenr = [820,760][r_id]
    cellid = [1301051,1700451][r_id]
    outputfolder = homedir+"Figures/hackathon_paper/"
    outpfn = ["fig2_AEA.png","fig2_AEC.png"][r_id]

    var_list_list = ["Pdyn",["vx","vy","vz","v"],["Bx","By","Bz","B"],"","rho",["TParallel","TPerpendicular"]]
    norm_list = [1.0e-9,1.0e+3,1.0e-9,1,1.0e+6,1.0e+6]
    ylabels = ["$\mathrm{P_{dyn}~[nPa]}$","$\mathrm{v~[kms^{-1}]}$","$\mathrm{B~[nT]}$","$\mathrm{W~[eV]}$","$\mathrm{n~[cm^{-3}]}$","$\mathrm{T~[MK]}$"]
    annot_list_list = [[""],["vx","vy","vz","v"],["Bx","By","Bz","B"],[""],[""],["TPar","TPerp"]]


    bulkpath = ja.find_bulkpath(runid)

    mmsjr = MMSJet()

    t_mms = mmsjr.read("time")
    v_mms = mmsjr.read_mult(["vx","vy","vz","v"])
    B_mms = mmsjr.read_mult(["Bx","By","Bz","B"])
    T_mms = mmsjr.read_mult(["TParallel","TPerpendicular"])
    n_mms = mmsjr.read("rho")
    pdyn_mms = mmsjr.read("Pdyn")
    ebins_mms = mmsjr.energy_bins
    flux_mms = mmsjr.read("flux").T

    time_ar,energy_ar,datamap = pt.plot.get_energy_spectrum(bulkpath,"bulk","proton",filenr-htw-100,filenr+htw+100,cellid,0.03,20,enum=32,fluxout=True,numproc=8)

    data_mms = [pdyn_mms,v_mms.T,B_mms.T,n_mms,n_mms,T_mms.T]
    time_mms = [t_mms,np.array([t_mms,t_mms,t_mms,t_mms]).T,np.array([t_mms,t_mms,t_mms,t_mms]).T,t_mms,t_mms,np.array([t_mms,t_mms]).T]

    mms_max_time = t_mms[np.argmax(pdyn_mms)]

    fig,ax_list = plt.subplots(6,2,figsize=(10,12))

    for col in range(2):
        for row in range(6):
            if col == 0:
                print(row)
                ax = ax_list[row][col]
                ax.set_xlim(float(filenr-htw-100)/2.0,float(filenr+htw+100)/2.0)
                if row == 3:
                    im = ax.pcolormesh(time_ar,energy_ar,np.log10(datamap),cmap="jet",vmin=3.5,vmax=7.5)
                    ax.set_yscale("log")
                    cbar = colorbar(im,ax_list[:,0].tolist())
                    #cbar.set_label("log Diff. energy flux\n$keV / (cm^2~s~sr~keV)$")
                    cbar.set_ticks([4,5,6,7])
                    ax.set_ylim(energy_ar[0],energy_ar[-1])
                else:
                    time,data = get_timeseries(runid,filenr-htw-100,filenr+htw+1+100,var_list_list[row],cellids=cellid)
                    data = data.T
                    time = time.T

                    ax.plot(time,data,linewidth=1.0)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune="lower"))
                ax.axvline(float(filenr)/2.0,linestyle="dashed",linewidth=0.8,color="black")
                #ax.axvline(380,linestyle="dashed",linewidth=0.8,color="black")
                ax.set_ylabel(ylabels[row],labelpad=10,fontsize=12)
                if row == 5:
                    ax.set_xlabel("Simulation time [s]",labelpad=10,fontsize=15)
                ann_list = annot_list_list[row]
                for m in range(len(ann_list)):
                    ax.annotate(ann_list[m],xy=(0.7+m*0.3/len(ann_list),0.05),xycoords="axes fraction",color=color_list[m])
            if col == 1:
                print(row)
                ax = ax_list[row][col]
                if row == 3:
                    im_mms = ax.pcolormesh(t_mms,ebins_mms,np.log10(flux_mms),cmap="jet",vmin=4.5,vmax=7.5)
                    ax.set_yscale("log")
                    cbar_mms = colorbar(im_mms,ax_list[:,1].tolist())
                    cbar_mms.set_label("log Diff. energy flux\n$\mathrm{keV / (cm^2~s~sr~keV)}$")
                    cbar_mms.set_ticks([5,6,7])
                    ax.set_yticks([1e2,1e3,1e4])
                    ax.set_ylim(30,20000)

                else:
                    data = data_mms[row]
                    time = time_mms[row]

                    ax.plot(time,data,linewidth=1.0)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.axvline(mms_max_time,linestyle="dashed",linewidth=0.8,color="black")
                #ax.set_xlim(5+27.5/60,5+41.0/60)
                ax.set_xlim(mms_min,mms_max)
                #ax.set_xticks(5+np.arange(28,42,1)/60.0)
                #ax.set_xticklabels(['', '', '05:30', '', '', '', '', '05:35', '', '', '', '', '05:40',''])
                ax.set_xticks(5+np.arange(31,39,1)/60.0)
                ax.set_xticklabels(["05:31","","05:33","","05:35","","05:37",""])
                if row == 5:
                    ax.set_xlabel("2015-12-03 UTC",labelpad=10,fontsize=15)
                ann_list = annot_list_list[row]
                for m in range(len(ann_list)):
                    ax.annotate(ann_list[m],xy=(0.7+m*0.3/len(ann_list),0.05),xycoords="axes fraction",color=color_list[m])

    ax_list[0][0].set_title("Run AEA\nX,Y,Z = [11.8, -0.9, 0.0]$R_e$",fontsize=20)
    ax_list[0][1].set_title("MMS\nX,Y,Z = [11.9, -0.9, -0.9]$R_e$",fontsize=20)
    ax_list[2][0].set_ylim(bottom=-20)
    ax_list[5][0].set_ylim(bottom=2)
    plt.tight_layout()

    fig.savefig(outputfolder+outpfn)
    plt.close(fig)


def hack_2019_fig1():

    outputdir = homedir+"Figures/hackathon_paper/"

    # Initialise required global variables
    global jet_cells,full_cells
    global xmean_list,ymean_list
    global xvmax_list,yvmax_list
    global vl_xy,mms_xy
    global runid_g,filenr_g,boxre_g

    #runids = ["AEA","AEC"]
    runids = ["AEA"]
    outpfn = ["fig1_AEA.png","fig1_AEC.png"]
    outpfn_2 = ["fig1_AEA_zoom.png","fig1_AEC_zoom.png"]
    cellid = [1301051,1700451]
    filenr = [820,760]
    boxre = [[6,18,-8,6],[6,18,-6,6]]
    boxre_2 = [[10,14,-3,1],[7,11,-3,1]]
    vmax = [1.5,4.5]
    pos_mms = [np.array([11.8709 , -0.8539, -0.9172]),np.array([ 9.1060, -0.9715,   -0.8508])]
    pos_vl = [np.array([ 11.81803436,  -0.87607214, 0]),np.array([8.24222971, -0.87607214, 0])]

    for n in range(len(runids)):

        runid_g = runids[n]
        filenr_g = filenr[n]
        boxre_g = boxre[n]

        vl_xy = pos_vl[n][:-1]
        mms_xy = pos_mms[n][:-1]

        # Initialise lists of coordinates
        xmean_list = []
        ymean_list = []
        xvmax_list = []
        yvmax_list = []

        event_props = np.array(jio.eventprop_read(runids[n],filenr[n]))
        xmean_list = event_props[:,1]
        ymean_list = event_props[:,2]
        xvmax_list = event_props[:,11]
        yvmax_list = event_props[:,12]


        # Try reading events file
        try:
            fileobj = open(wrkdir_DNR+"working/events/{}/{}.events".format(runids[n],filenr[n]),"r")
            contents = fileobj.read()
            fileobj.close()
            jet_cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            jet_cells = []

        # Try reading mask file
        try:
            full_cells = np.loadtxt(wrkdir_DNR+"working/Masks/{}/{}.mask".format(runids[n],filenr[n])).astype(int)
        except IOError:
            full_cells = []

        # Find correct file path
        bulkpath = ja.find_bulkpath(runids[n])

        bulkname = "bulk.{}.vlsv".format(str(filenr[n]).zfill(7))

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file {} not found, continuing".format(str(filenr[n])))
            continue

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputfile=outputdir+outpfn[n],usesci=0,lin=1,boxre=boxre[n],expression=pc.expr_pdyn,vmin=0,vmax=vmax[n],colormap="parula",cbtitle="nPa",external=h19_fig1_ext,pass_vars=["rho","v","CellID","Pdyn"])

        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputfile=outputdir+outpfn_2[n],usesci=0,lin=1,boxre=boxre_2[n],expression=pc.expr_pdyn,vmin=0,vmax=vmax[n],colormap="parula",cbtitle="nPa",external=h19_fig1_ext,pass_vars=["rho","v","CellID"])


def h19_fig1_ext(ax,XmeshXY,YmeshXY,pass_maps):

    # External function for jet_plotter

    cellids = pass_maps["CellID"]
    rho = pass_maps["rho"]
    vx = pass_maps["v"][:,:,0]
    pdyn = m_p*rho*vx*vx
    sw_pars = ja.sw_par_dict(runid_g)
    pd_sw = sw_pars[3]

    bs_y = np.arange(boxre_g[2],boxre_g[3],0.01)
    #bs_p = ja.bow_shock_markus(runid_g,filenr_g)[::-1]
    mp_p,bs_p = ja.bs_mp_fit(runid_g,filenr_g,boxre_g)
    bs_x = np.polyval(bs_p,bs_y)
    mp_x = np.polyval(mp_p,bs_y)+1

    # Mask jets
    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    # Mask Plaschke

    plas_mask = (pdyn >= 0.25*pd_sw).astype(int)
    plas_mask = np.reshape(plas_mask,cellids.shape)

    # Mask full mask
    full_mask = np.in1d(cellids,full_cells).astype(int)
    full_mask = np.reshape(full_mask,cellids.shape)

    #full_cont = ax.contour(XmeshXY,YmeshXY,full_mask,[0.5],linewidths=0.8,colors="magenta") # Contour of full mask
    p_cont = ax.contour(XmeshXY,YmeshXY,plas_mask,[0.5],linewidths=0.6,colors="magenta")
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="black") # Contour of jets

    line1, = ax.plot(xmean_list,ymean_list,"o",color="red",markersize=2) # Mean positions
    line2, = ax.plot(xvmax_list,yvmax_list,"o",color="white",markersize=2) # v_max positions

    vlas, = ax.plot(vl_xy[0],vl_xy[1],"*",markersize=5,color="black")
    mms, = ax.plot(mms_xy[0],mms_xy[1],"*",markersize=5,color="green")
    bs_cont = ax.plot(bs_x,bs_y,color="red")
    mp_cont = ax.plot(mp_x,bs_y,color="red")

def get_SEA(var_list,centering="A",runids=["ABA","ABC","AEA","AEC"],time_thresh=5):

    var_list = np.array(var_list,ndmin=1).tolist()

    epoch_arr = np.arange(-60.0,60.1,0.5)
    SEA_arr_list = [np.zeros_like(epoch_arr) for var in var_list]
    SEA_mean_list = [np.zeros_like(epoch_arr) for var in var_list]
    SEA_std_list = [np.zeros_like(epoch_arr) for var in var_list]

    runids_list = ["ABA","ABC","AEA","AEC"]
    cutoff_list = [10,8,10,8]
    cutoff_dict = dict(zip(runids_list,cutoff_list))

    data_list = [[] for var in var_list]

    for n in range(1,2500):
        for runid in runids:
            try:
                props = jio.PropReader(str(n).zfill(5),runid,580)
            except:
                continue

            if props.read("duration")[0] < time_thresh or max(props.read("r_mean")) < cutoff_dict[runid]:
                continue

            time_arr = props.read("time")
            cent_arr = props.read(centering)

            for var in var_list:
                idx = var_list.index(var)
                var_arr = props.read(var)
                if var not in ["TPar_avg","TPerp_avg"]:
                    var_arr /= ja.sw_normalisation(runid,var)
                res_arr = np.interp(epoch_arr,time_arr-time_arr[np.argmax(cent_arr)],var_arr,left=0.0,right=0.0)
                SEA_arr_list[idx] = np.vstack((SEA_arr_list[idx],res_arr))

    SEA_arr_list = [SEA_arr[1:] for SEA_arr in SEA_arr_list]

    SEA_mean_list = [np.mean(SEA_arr,axis=0) for SEA_arr in SEA_arr_list]
    SEA_std_list = [np.std(SEA_arr,ddof=1,axis=0) for SEA_arr in SEA_arr_list]

    return (epoch_arr,SEA_mean_list,SEA_std_list)


def hack_2019_fig78(time_thresh=5):
    # Creates Superposed Epoch Analysis of jets in specified run, centering specified var around maximum of
    # specified centering variable

    var_list_7 = ["size_rad","size_tan","size_ratio"]
    var_list_8 = ["Dn","Dv","Dpd","DB","DTPerp","DTPar"]

    lab_list_7 = ["$\mathrm{Extent~[R_e]}$","$\mathrm{Tangential~Size~[R_e]}$","$\mathrm{Size~Ratio}$"]
    lab_list_8 = ["$\mathrm{\\Delta n~[n_{sw}]}$","$\mathrm{\\Delta |v|~[v_{sw}]}$","$\mathrm{\\Delta P_{dyn}~[P_{dyn,sw}]}$","$\mathrm{\\Delta |B|~[B_{IMF}]}$","$\mathrm{\\Delta T_{perp}~[MK]}$","$\mathrm{\\Delta T_{par}~[MK]}$"]

    epoch_arr,SEA_mean_list_7_all,SEA_std_list_7_all = get_SEA(var_list_7,time_thresh=time_thresh)
    epoch_arr,SEA_mean_list_8_all,SEA_std_list_8_all = get_SEA(var_list_8,time_thresh=time_thresh)

    epoch_arr,SEA_mean_list_7_ABA,SEA_std_list_7_ABA = get_SEA(var_list_7,time_thresh=time_thresh,runids=["ABA"])
    epoch_arr,SEA_mean_list_7_ABC,SEA_std_list_7_ABC = get_SEA(var_list_7,time_thresh=time_thresh,runids=["ABC"])
    epoch_arr,SEA_mean_list_7_AEA,SEA_std_list_7_AEA = get_SEA(var_list_7,time_thresh=time_thresh,runids=["AEA"])
    epoch_arr,SEA_mean_list_7_AEC,SEA_std_list_7_AEC = get_SEA(var_list_7,time_thresh=time_thresh,runids=["AEC"])

    epoch_arr,SEA_mean_list_8_ABA,SEA_std_list_8_ABA = get_SEA(var_list_8,time_thresh=time_thresh,runids=["ABA"])
    epoch_arr,SEA_mean_list_8_ABC,SEA_std_list_8_ABC = get_SEA(var_list_8,time_thresh=time_thresh,runids=["ABC"])
    epoch_arr,SEA_mean_list_8_AEA,SEA_std_list_8_AEA = get_SEA(var_list_8,time_thresh=time_thresh,runids=["AEA"])
    epoch_arr,SEA_mean_list_8_AEC,SEA_std_list_8_AEC = get_SEA(var_list_8,time_thresh=time_thresh,runids=["AEC"])

    fig_7,ax_list_7 = plt.subplots(3,1,figsize=(10,10),sharex=True)

    for col in range(3):
        ax = ax_list_7[col]

        SEA_mean = SEA_mean_list_7_all[col]
        SEA_std = SEA_std_list_7_all[col]

        SEA_mean_ABA = SEA_mean_list_7_ABA[col]
        SEA_mean_ABC = SEA_mean_list_7_ABC[col]
        SEA_mean_AEA = SEA_mean_list_7_AEA[col]
        SEA_mean_AEC = SEA_mean_list_7_AEC[col]

        ax.plot(epoch_arr,SEA_mean_ABA,color="black",label="ABA")
        ax.plot(epoch_arr,SEA_mean_ABC,color="blue",label="ABC")
        ax.plot(epoch_arr,SEA_mean_AEA,color="red",label="AEA")
        ax.plot(epoch_arr,SEA_mean_AEC,color="green",label="AEC")

        ax.fill_between(epoch_arr,SEA_mean-SEA_std,SEA_mean+SEA_std,alpha=0.25)
        ax.set_ylabel(lab_list_7[col],fontsize=15)
        ax.set_xlim(-60,60)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if col == 2:
            ax.set_xlabel("Epoch Time [s]",fontsize=20)

    ax_list_7[0].annotate("ABA",xy=(0.5-0.2,1.05),xycoords="axes fraction",color="black",fontsize=20)
    ax_list_7[0].annotate("ABC",xy=(0.5-0.1,1.05),xycoords="axes fraction",color="blue",fontsize=20)
    ax_list_7[0].annotate("AEA",xy=(0.5,1.05),xycoords="axes fraction",color="red",fontsize=20)
    ax_list_7[0].annotate("AEC",xy=(0.5+0.1,1.05),xycoords="axes fraction",color="green",fontsize=20)

    #plt.tight_layout()

    fig_7.savefig(homedir+"Figures/hackathon_paper/fig7.png")
    plt.close(fig_7)

    fig_8,ax_list_8 = plt.subplots(6,1,figsize=(10,12),sharex=True)

    for col in range(6):
        ax = ax_list_8[col]
        SEA_mean = SEA_mean_list_8_all[col]
        SEA_std = SEA_std_list_8_all[col]

        SEA_mean_ABA = SEA_mean_list_8_ABA[col]
        SEA_mean_ABC = SEA_mean_list_8_ABC[col]
        SEA_mean_AEA = SEA_mean_list_8_AEA[col]
        SEA_mean_AEC = SEA_mean_list_8_AEC[col]

        ax.plot(epoch_arr,SEA_mean_ABA,color="black",label="ABA")
        ax.plot(epoch_arr,SEA_mean_ABC,color="blue",label="ABC")
        ax.plot(epoch_arr,SEA_mean_AEA,color="red",label="AEA")
        ax.plot(epoch_arr,SEA_mean_AEC,color="green",label="AEC")

        ax.fill_between(epoch_arr,SEA_mean-SEA_std,SEA_mean+SEA_std,alpha=0.25)
        ax.set_ylabel(lab_list_8[col],fontsize=15)
        ax.set_xlim(-60,60)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if col == 5:
            ax.set_xlabel("Epoch Time [s]",fontsize=20)

    ax_list_8[0].annotate("ABA",xy=(0.5-0.2,1.1),xycoords="axes fraction",color="black",fontsize=20)
    ax_list_8[0].annotate("ABC",xy=(0.5-0.1,1.1),xycoords="axes fraction",color="blue",fontsize=20)
    ax_list_8[0].annotate("AEA",xy=(0.5,1.1),xycoords="axes fraction",color="red",fontsize=20)
    ax_list_8[0].annotate("AEC",xy=(0.5+0.1,1.1),xycoords="axes fraction",color="green",fontsize=20)

    #plt.tight_layout()

    fig_8.savefig(homedir+"Figures/hackathon_paper/fig8.png")
    plt.close(fig_8)

    return None

def hack_2019_fig9(time_thresh=5,nbins=10):

    var_list = ["B_avg","n_avg","v_avg"]
    xlabels = ["$\mathrm{|B|_{mean}~[B_{IMF}]}$","$\mathrm{n_{mean}~[n_{sw}]}$","$\mathrm{|v|_{mean}~[v_{sw}]}$"]
    ylabels = ["Vlasiator\nFraction of jets","MMS\nFraction of jets"]
    bin_list = [np.linspace(0,8,nbins+1),np.linspace(0,7,nbins+1),np.linspace(0,1,nbins+1)]

    vlas_high = [read_mult_runs(var,time_thresh,runids=["ABA","ABC"],amax=False) for var in var_list]
    vlas_low = [read_mult_runs(var,time_thresh,runids=["AEA","AEC"],amax=False) for var in var_list]

    mms_props_low = MMSReader("LowMachJets.txt")
    mms_props_high = MMSReader("HighMachJets.txt")

    MMS_low = [mms_props_low.read(var) for var in var_list]
    MMS_high = [mms_props_high.read(var) for var in var_list]

    data_low = [vlas_low,MMS_low]
    data_high = [vlas_high,MMS_high]

    fig,ax_list = plt.subplots(2,3,figsize=(10,10),sharey=True)

    for row in range(2):
        for col in range(3):
            ax = ax_list[row][col]
            var_low = data_low[row][col]
            var_high = data_high[row][col]
            weights_low = np.ones(var_low.shape,dtype=float)/var_low.size
            weights_high = np.ones(var_high.shape,dtype=float)/var_high.size
            if row == 0 and col == 0:
                label = ["Low\nmed:{:.2f}\nstd:{:.2f}".format(np.median(var_low),np.std(var_low,ddof=1)),"High\nmed:{:.2f}\nstd:{:.2f}".format(np.median(var_high),np.std(var_high,ddof=1))]
            else:
                label = ["med:{:.2f}\nstd:{:.2f}".format(np.median(var_low),np.std(var_low,ddof=1)),"med:{:.2f}\nstd:{:.2f}".format(np.median(var_high),np.std(var_high,ddof=1))]

            ax.hist([var_low,var_high],weights=[weights_low,weights_high],color=["blue","red"],label=label,histtype="step",bins=bin_list[col])
            ax.legend(fontsize=15,frameon=False)
            ax.set_ylim(0,1)
            if col == 0:
                ax.set_ylabel(ylabels[row],fontsize=20,labelpad=10)
            if row == 1:
                ax.set_xlabel(xlabels[col],fontsize=20,labelpad=10)

    plt.tight_layout()
    fig.savefig(homedir+"Figures/hackathon_paper/fig9.png")
    plt.close(fig)

    return None

###PLOT MAKER HERE###



###CONTOUR MAKER HERE###



###VIRTUAL SPACECRAFT MAKER HERE###


# MOVED TO vspacecraft.py




###MULTI FILE SCRIPTS HERE###



### MISC SCRIPTS ###
