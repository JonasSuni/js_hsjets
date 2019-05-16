import numpy as np
import pytools as pt
import scipy
import pandas as pd
import jet_analyser as ja
import os
import jet_scripts as js
import copy
import matplotlib.pyplot as plt
import plot_contours as pc
import scipy.constants as sc

m_p = 1.672621898e-27
r_e = 6.371e+6

class PropReader:
    # Class for reading jet property files

    def __init__(self,ID,runid,start=580,fname=None,transient="jet"):

        if transient == "jet":
            inputdir = "jets"
        elif transient == "slamsjet":
            inputdir = "SLAMSJETS/slamsjets"
        elif transient == "slams":
            inputdir = "SLAMS/slams"

        self.ID = ID
        self.runid = runid
        self.start = start
        self.sw_pars = ja.sw_par_dict(runid)
        self.sw_pars[0] /= 1.0e+6
        self.sw_pars[1] /= 1.0e+3
        self.sw_pars[2] /= 1.0e-9
        self.sw_pars[3] /= 1.0e-9

        if type(fname) is not str:
            self.fname = str(start)+"."+ID+".props"
        else:
            self.fname = fname

        try:
            self.props = pd.read_csv(inputdir+"/"+runid+"/"+self.fname).as_matrix()
        except IOError:
            raise IOError("File not found!")

        var_list = ["time","x_mean","y_mean","z_mean","A","Nr_cells","r_mean","theta_mean","phi_mean","size_rad","size_tan","x_vmax","y_vmax","z_vmax","n_avg","n_med","n_max","v_avg","v_med","v_max","B_avg","B_med","B_max","T_avg","T_med","T_max","TPar_avg","TPar_med","TPar_max","TPerp_avg","TPerp_med","TPerp_max","beta_avg","beta_med","beta_max","x_min","rho_vmax","b_vmax","pd_avg","pd_med","pd_max"]
        n_list = list(xrange(len(var_list)))
        self.var_dict = dict(zip(var_list,n_list))

    def read(self,name):
        if name == "pdyn_vmax":
            return 1.0e+21*m_p*self.props[:,self.var_dict["rho_vmax"]]*self.props[:,self.var_dict["v_max"]]**2
        elif name not in self.var_dict:
            print("Variable not found!")
            return None
        else:
            return self.props[:,self.var_dict[name]]

    def amax_index(self):
        return self.read("A").argmax()

    def time_index(self,time):
        time_arr = self.read("time")
        if time not in time_arr:
            raise IOError("Time not found!")
        else:
            return time_arr.tolist().index(time)

    def read_at_time(self,var,time):
        return self.read(var)[self.time_index(time)]


    def read_at_amax(self,name):
        return self.read(name)[self.amax_index()]

class Jet:
    # Class for identifying and handling individual jets and their properties

    def __init__(self,ID,runid,birthday):

        self.ID = ID
        self.runid = runid
        self.birthday = birthday
        self.cellids = []
        self.times = [birthday]

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):

        return "\n".join([",".join(map(str,l)) for l in self.cellids])

    def return_time_string(self):

        return "\n".join(map(str,self.times))

def jet_maker(runid,start,stop,boxre=[6,18,-8,6],maskfile=False,avgfile=False):

    outputdir = "/homeappl/home/sunijona/events/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for file_nr in xrange(start,stop+1):

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
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt("Masks/"+runid+"/"+str(file_nr)+".mask").astype(int)
        else:
            msk = ja.make_cust_mask(file_nr,runid,180,boxre,avgfile)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        jets = ja.sort_jets(vlsvobj,msk,25,4500,[2,2])

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for jet in jets:

            fileobj.write(",".join(map(str,jet))+"\n")

        fileobj.close()

    return None

def timefile_read(runid,filenr,key,transient="jet"):
    # Read array of times from file

    if transient == "jet":
        inputdir = "jets"
    elif transient == "slamsjet":
        inputdir = "SLAMSJETS/slamsjets"

    tf = open("{}/{}/{}.{}.times".format(inputdir,runid,str(filenr),key),"r")
    contents = tf.read().split("\n")
    tf.close()

    return map(float,contents)

def jetfile_read(runid,filenr,key,transient="jet"):
    # Read array of cellids from file

    if transient == "jet":
        inputdir = "jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = "SLAMSJETS/slamsjets"
        extension = "slamsjet"

    outputlist = []

    jf = open("{}/{}/{}.{}.{}".format(inputdir,runid,str(filenr),key,extension),"r")
    contents = jf.read()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def eventfile_read(runid,filenr,transient="jet"):
    # Read array of arrays of cellids from file

    if transient == "jet":
        inputdir = "events"
    elif transient == "slams":
        inputdir = "SLAMS/events"

    outputlist = []

    ef = open("{}/{}/{}.events".format(inputdir,runid,str(filenr)),"r")
    contents = ef.read().strip("\n")
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def propfile_write(runid,filenr,key,props,transient="jet"):
    # Write jet properties to file

    if transient == "jet":
        outputdir = "jets"
    elif transient == "slamsjet":
        outputdir = "SLAMSJETS/slamsjets"

    open(outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open(outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write("time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa]"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to "+outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props")

def jio_figmake(runid,start,jetid,figname,tp_files=False):
    # Create time series figures of specified jet

    props = calc_jet_properties(runid,start,jetid,tp_files=tp_files)

    if type(props) is not np.ndarray:
        return 1
    else:
        jetsize_fig(runid,start,jetid,figname=figname,props_arr=props)

def figmake_script(runid,start,ids,tp_files=False):

    for ID in ids:
        jio_figmake(runid,start,ID,figname=ID,tp_files=tp_files)

def plotmake_script_BFD(start,stop,runid="BFD",vmax=1.5,boxre=[4,20,-10,4]):

    if not os.path.exists("Contours/jetfigs/"+runid):
        try:
            os.makedirs("Contours/jetfigs/"+runid)
        except OSError:
            pass

    # Find names of property files
    filenames = os.listdir("jets/"+runid)
    prop_fns = []
    for filename in filenames:
        if ".props" in filename:
            prop_fns.append(filename)
    prop_fns.sort()

    xmean_dict = dict()
    ymean_dict = dict()
    xmax_dict = dict()
    ymax_dict = dict()

    for fname in prop_fns:
        jet_id = fname[4:-6]
        props = PropReader(ID=jet_id,runid=runid)
        time = props.read("time")
        x_mean = props.read("x_mean")
        y_mean = props.read("y_mean")
        z_mean = props.read("z_mean")
        x_vmax = props.read("x_vmax")
        y_vmax = props.read("y_vmax")
        z_vmax = props.read("z_vmax")
        if runid in ["BFD"]:
            y_mean = z_mean
            y_vmax = z_vmax
        for itr in xrange(time.size):
            if time[itr] not in xmean_dict:
                xmean_dict[time[itr]] = [x_mean[itr]]
                ymean_dict[time[itr]] = [y_mean[itr]]
                xmax_dict[time[itr]] = [x_vmax[itr]]
                ymax_dict[time[itr]] = [y_vmax[itr]]
            else:
                xmean_dict[time[itr]].append(x_mean[itr])
                ymean_dict[time[itr]].append(y_mean[itr])
                xmax_dict[time[itr]].append(x_vmax[itr])
                ymax_dict[time[itr]].append(y_vmax[itr])

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for itr2 in xrange(start,stop+1):

        t = float(itr2)/2

        bulkname = "bulk."+str(itr2).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(itr2)+" not found, continuing")
            continue

        if runid == "BFD" and itr2 == 961:
            print("Broken file!")
            continue

        if runid in ["BFD"]:
            pass_vars = ["proton/rho","proton/V","CellID"]
        else:
            pass_vars = ["rho","v","CellID"]

        try:
            fullmask = np.loadtxt("Masks/"+runid+"/"+str(itr2)+".mask").astype(int)
        except IOError:
            fullmask = np.array([])

        try:
            fileobj = open("events/"+runid+"/"+str(itr2)+".events","r")
            contents = fileobj.read()
            cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            cells = []

        # Create plot
        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir="Contours/jetfigs/"+runid+"/",step=itr2,run=runid,usesci=0,lin=1,boxre=boxre,vmin=0,vmax=vmax,colormap="parula",cbtitle="",external=pms_ext,expression=pc.expr_pdyn,pass_vars=pass_vars,ext_pars=[xmean_dict[t],ymean_dict[t],cells,fullmask,xmax_dict[t],ymax_dict[t]])

    return None

def pms_ext(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho,v,cellids = extmaps["rho"],extmaps["v"],extmaps["CellID"]

    x_list,y_list,cells,fullmask,xmax_list,ymax_list = ext_pars

    # Create mask
    msk = np.in1d(cellids,cells).astype(int)
    msk = np.reshape(msk,rho.shape)

    fullmsk = np.in1d(cellids,fullmask).astype(int)
    fullmsk = np.reshape(fullmsk,rho.shape)

    # Draw contours
    fullcont = ax.contour(XmeshXY,YmeshXY,fullmsk,[0.5],linewidths=1.0,colors="magenta")
    cont = ax.contour(XmeshXY,YmeshXY,msk,[0.5],linewidths=1.0,colors="black")

    # Plot jet positions
    ax.plot(x_list,y_list,"o",color="red",markersize=4)
    ax.plot(xmax_list,ymax_list,"o",color="white",markersize=4)

def jetsize_fig(runid,start,jetid,figsize=(15,10),figname="sizefig",props_arr=None):
    # script for creating time series of jet linear sizes and area

    # Decide whether to read properties from file or input variable
    if props_arr == None:
        linsizes = pd.read_csv("/homeappl/home/sunijona/jets/"+runid+"/"+str(start)+"."+jetid+".props").as_matrix()
    else:
        linsizes = props_arr

    # Create variable value arrays
    time_arr = linsizes[:,0]
    area_arr = linsizes[:,4]
    rad_size_arr = linsizes[:,9]
    tan_size_arr = linsizes[:,10]
    x_arr = linsizes[:,1]
    y_arr = linsizes[:,2]
    z_arr = linsizes[:,3]

    if runid == "BFD":
        y_arr = z_arr

    # Minimum and maximum values
    minmax_list = [min(time_arr),max(time_arr),min(area_arr),max(area_arr),min(rad_size_arr),max(rad_size_arr),min(tan_size_arr),max(tan_size_arr),min(x_arr),max(x_arr),min(y_arr),max(y_arr)]

    for n in xrange(0,len(minmax_list),2):
        if np.abs((minmax_list[n]-minmax_list[n+1])/float(minmax_list[n])) < 1.0e-5:
            minmax_list[n+1] += 1
            minmax_list[n] -= 1

    tmin,tmax,Amin,Amax,rsmin,rsmax,psmin,psmax,xmin,xmax,ymin,ymax = minmax_list

    # Create figure
    plt.ioff()
    fig = plt.figure(figsize=figsize)

    # Add subplots
    area_ax = fig.add_subplot(321)
    rad_size_ax = fig.add_subplot(323)
    tan_size_ax = fig.add_subplot(325)
    x_ax = fig.add_subplot(322)
    y_ax = fig.add_subplot(324)

    # Draw grids
    area_ax.grid()
    rad_size_ax.grid()
    tan_size_ax.grid()
    x_ax.grid()
    y_ax.grid()

    # Set x-limits
    area_ax.set_xlim(tmin,tmax)
    rad_size_ax.set_xlim(tmin,tmax)
    tan_size_ax.set_xlim(tmin,tmax)
    x_ax.set_xlim(tmin,tmax)
    y_ax.set_xlim(tmin,tmax)

    # Set y-limits
    area_ax.set_ylim(Amin,Amax)
    rad_size_ax.set_ylim(rsmin,rsmax)
    tan_size_ax.set_ylim(psmin,psmax)
    x_ax.set_ylim(xmin,xmax)
    y_ax.set_ylim(ymin,ymax)

    # Set x-ticklabels
    area_ax.set_xticklabels([])
    rad_size_ax.set_xticklabels([])
    x_ax.set_xticklabels([])
    y_ax.set_xticklabels([])

    # Set y-labels
    area_ax.set_ylabel("Area [R$_{e}^{2}$]",fontsize=20)
    rad_size_ax.set_ylabel("Radial size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_ylabel("Tangential size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_xlabel("Time [s]",fontsize=20)
    x_ax.set_ylabel("X [R$_{e}$]",fontsize=20)
    y_ax.set_ylabel("Y [R$_{e}$]",fontsize=20)
    y_ax.set_xlabel("Time [s]",fontsize=20)

    if runid == "BFD":
        y_ax.set_ylabel("Z [R$_{e}$]",fontsize=20)

    # Set tick label sizes
    area_ax.tick_params(labelsize=16)
    rad_size_ax.tick_params(labelsize=16)
    tan_size_ax.tick_params(labelsize=16)
    x_ax.tick_params(labelsize=16)
    y_ax.tick_params(labelsize=16)

    # Plot variables
    area_ax.plot(time_arr,area_arr,color="black",linewidth=2)
    rad_size_ax.plot(time_arr,rad_size_arr,color="black",linewidth=2)
    tan_size_ax.plot(time_arr,tan_size_arr,color="black",linewidth=2)
    x_ax.plot(time_arr,x_arr,color="black",linewidth=2)
    y_ax.plot(time_arr,y_arr,color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    # Create outputdir if it doesn't already exist
    if not os.path.exists("jet_sizes/"+runid):
        try:
            os.makedirs("jet_sizes/"+runid)
        except OSError:
            pass

    # Save figure
    plt.savefig("jet_sizes/"+runid+"/"+figname+".png")
    print("jet_sizes/"+runid+"/"+figname+".png")

    plt.close(fig)

    return None

def tpar_reader(runid,filenumber,cellids,cells):
    # Read parallel temperatures of specific cells

    TPar = np.loadtxt("/wrk/sunijona/DONOTREMOVE/TP/"+runid+"/"+str(filenumber)+".tpar")
    TPar = TPar[np.in1d(cellids,cells)]

    return TPar

def tperp_reader(runid,filenumber,cellids,cells):
    # Read perpendicular temperatures of specific cells

    TPerp = np.loadtxt("/wrk/sunijona/DONOTREMOVE/TP/"+runid+"/"+str(filenumber)+".tperp")
    TPerp = TPerp[np.in1d(cellids,cells)]

    return TPerp

def calc_jet_properties(runid,start,jetid,tp_files=False,transient="jet"):

    if transient == "jet":
        inputdir = "jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = "SLAMSJETS/slamsjets"
        extension = "slamsjet"

    if "{}.{}.{}".format(str(start),jetid,extension) not in os.listdir("{}/{}".format(inputdir,runid)):
        print("Transient with ID "+jetid+" does not exist, exiting.")
        return 1

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid,transient)
    time_list = timefile_read(runid,start,jetid,transient)

    # Discard jet if it's very short-lived
    if len(time_list) < 5:
        print("Jet not sufficiently long-lived, exiting.")
        return 1

    # Discard jet if it has large gaps in the times
    dt = np.ediff1d(time_list)
    if max(dt) > 5:
        print("Jet not sufficiently continuous, exiting.")
        return 1

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # Convert times to file numbers
    nr_list = [int(t*2) for t in time_list]

    # Initialise property array
    prop_arr = np.array([])

    for n in xrange(len(nr_list)):

        curr_list = jet_list[n]
        curr_list.sort()

        # Find correct file name
        if runid == "AED":
            bulkname = "bulk.old."+str(nr_list[n]).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(nr_list[n]).zfill(7)+".vlsv"

        # Open VLSV file
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        origid = vlsvobj.read_variable("CellID")
        sorigid = origid[origid.argsort()]

        # read variables
        if vlsvobj.check_variable("X"):
            X = vlsvobj.read_variable("X",cellids=curr_list)
            Y = vlsvobj.read_variable("Y",cellids=curr_list)
            Z = vlsvobj.read_variable("Z",cellids=curr_list)
        else:
            X,Y,Z = ja.xyz_reconstruct(vlsvobj,cellids=curr_list)

        # Calculate area of one cell
        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = ja.get_cell_volume(vlsvobj)

        # If file has more than one population, choose proton population
        var_list = ["rho","v","B","Temperature","CellID","beta","TParallel","TPerpendicular"]
        var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","CellID","proton/beta","proton/TParallel","proton/TPerpendicular"]
        if vlsvobj.check_population("proton"):
            try:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list_alt]
            except:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list]
        else:
            rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list]

        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        # Scale variables
        rho /= 1.0e+6
        v /= 1.0e+3
        B /= 1.0e-9
        pdyn /= 1.0e-9
        T /= 1.0e+6
        TParallel /= 1.0e+6
        TPerpendicular /= 1.0e+6

        # Calculate magnitudes of v and B
        vmag = np.linalg.norm(v,axis=-1)
        Bmag = np.linalg.norm(B,axis=-1)

        # Calculate means, medians and maximums for rho,vmag,Bmag,Pdyn,T,TParallel,TPerpendicular,beta
        n_avg = np.nanmean(rho)
        n_med = np.median(rho)
        n_max = np.max(rho)

        v_avg = np.nanmean(vmag)
        v_med = np.median(vmag)
        v_max = np.max(vmag)

        B_avg = np.nanmean(Bmag)
        B_med = np.median(Bmag)
        B_max = np.max(Bmag)

        pd_avg = np.nanmean(pdyn)
        pd_med = np.median(pdyn)
        pd_max = np.max(pdyn)

        T_avg = np.nanmean(T)
        T_med = np.median(T)
        T_max = np.max(T)

        TPar_avg = np.nanmean(TParallel)
        TPar_med = np.median(TParallel)
        TPar_max = np.max(TParallel)

        TPerp_avg = np.nanmean(TPerpendicular)
        TPerp_med = np.median(TPerpendicular)
        TPerp_max = np.max(TPerpendicular)

        beta_avg = np.nanmean(beta)
        beta_med = np.median(beta)
        beta_max = np.max(beta)

        # Convert X,Y,Z to spherical coordinates
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
        theta = np.rad2deg(np.arccos(Z/r))
        phi = np.rad2deg(np.arctan(Y/X))

        # calculate geometric center of jet
        r_mean = np.mean(r)/r_e
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)

        # Geometric center of jet in cartesian coordinates
        x_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.cos(np.deg2rad(phi_mean))
        y_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.sin(np.deg2rad(phi_mean))
        z_mean = r_mean*np.cos(np.deg2rad(theta_mean))

        # Position of maximum velocity in cartesian coordinates
        x_max = X[vmag==max(vmag)][0]/r_e
        y_max = Y[vmag==max(vmag)][0]/r_e
        z_max = Z[vmag==max(vmag)][0]/r_e

        # Minimum x and density at maximum velocity
        x_min = min(X)/r_e
        rho_vmax = rho[vmag==max(vmag)]
        b_vmax = beta[vmag==max(vmag)]

        #r_max = np.linalg.norm(np.array([x_mean,y_mean,z_mean]))
        #theta_max = np.rad2deg(np.arccos(z_mean/r_mean))
        #phi_max = np.rad2deg(np.arctan(y_mean/x_mean))

        # calculate jet size
        A = dA*len(curr_list)/(r_e**2)
        Nr_cells = len(curr_list)

        # calculate linear sizes of jet
        size_rad = (max(r)-min(r))/r_e
        size_tan = A/size_rad

        # current time
        time = time_list[n]

        ''' 
        0: time [s],
        1: x_mean [R_e],        2: y_mean [R_e],        3: z_mean [R_e],
        4: A [R_e^2],           5: Nr_cells,
        6: r_mean [R_e],        7: theta_mean [deg],    8: phi_mean [deg],
        9: size_rad [R_e],      10: size_tan [R_e],
        11: x_max [R_e],        12: y_max [R_e],        13: z_max [R_e],
        14: n_avg [1/cm^3],     15: n_med [1/cm^3],     16: n_max [1/cm^3],
        17: v_avg [km/s],       18: v_med [km/s],       19: v_max [km/s],
        20: B_avg [nT],         21: B_med [nT],         22: B_max [nT],
        23: T_avg [MK],         24: T_med [MK],         25: T_max [MK],
        26: TPar_avg [MK],      27: TPar_med [MK],      28: TPar_max [MK],
        29: TPerp_avg [MK],     30: TPerp_med [MK],     31: TPerp_max [MK],
        32: beta_avg,           33: beta_med,           34: beta_max
        35: x_min [R_e],        36: rho_vmax [1/cm^3],  37: b_vmax
        38: pd_avg [nPa],       39: pd_med [nPa],       40: pd_max [nPa]
        '''

        # Create temporary property array
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr,transient)

    return prop_arr

def track_jets(runid,start,stop,threshold=0.3,track_splinters = True):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists("/homeappl/home/sunijona/jets/"+runid):
        try:
            os.makedirs("/homeappl/home/sunijona/jets/"+runid)
        except OSError:
            pass

    # Get solar wind parameters
    sw_pars = ja.sw_par_dict(runid)
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file, get Cell IDs and sort them
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    sorigid = vlsvobj.read_variable("CellID")
    sorigid = sorigid[sorigid.argsort()]

    # Reconstruct X,Y,Z
    #fX,fY,fZ = ja.xyz_reconstruct(vlsvobj)
    
    # Find bow shock cells and area of one cell
    bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = ja.get_cell_volume(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # remove events that are not initially at the bow shock
    bs_events = []
    for old_event in events_old:
        if np.intersect1d(bs_cells,ja.get_neighbors(vlsvobj,old_event,[3,3])).size > 0:
            bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for jets at bow shock
    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(Jet(curr_id,runid,float(start)/2))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in xrange(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] > 5:
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            events = []
            continue

        # Open bulkfile and get bow shock cells
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # Filtered list of events that are at the bow shock at the current time
        bs_events = []
        for old_event in events:
            if np.intersect1d(bs_cells,ja.get_neighbors(vlsvobj,old_event,[3,3])).size > 0:
                bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID in flags:
                    
                    if np.intersect1d(jetobj.cellids[-2],event).size > threshold*len(event):

                        if track_splinters:

                            curr_id = str(counter).zfill(5)

                            # Create new jet
                            jetobj_new = Jet(curr_id,runid,float(n)/2)
                            jetobj_new.cellids.append(event)
                            jetobj_list.append(jetobj_new)
                            curr_jet_temp_list.append(event)

                            # Iterate counter
                            counter += 1

                            break

                        else:

                            break

                else:

                    if np.intersect1d(jetobj.cellids[-1],event).size > threshold*len(event):

                        # Append event to jet object properties
                        jetobj.cellids.append(event)
                        jetobj.times.append(float(n)/2)
                        print("Updated jet with ID "+jetobj.ID)

                        # Flag jet object
                        flags.append(jetobj.ID)
                        curr_jet_temp_list.append(event)

                        break

        # Look for new jets at bow shock
        for event in events:

            if event not in curr_jet_temp_list:

                for bs_event in bs_events:

                    if np.intersect1d(bs_event,event).size > threshold*len(event):

                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_list.append(Jet(curr_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".jet","w")
        timefile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None

def slams_eventfile_read(runid,filenr):
    # Read array of arrays of cellids from file

    outputlist = []

    ef = open("/homeappl/home/sunijona/SLAMS/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read().strip("\n")
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def track_slamsjets(runid,start,stop,threshold=0.3, track_splinters = True):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists("/homeappl/home/sunijona/SLAMSJETS/slamsjets/"+runid):
        try:
            os.makedirs("/homeappl/home/sunijona/SLAMSJETS/slamsjets/"+runid)
        except OSError:
            pass

    # Open file, get Cell IDs and sort them
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    # Read initial event files
    events = eventfile_read(runid,start+1)

    slams_events_old = slams_eventfile_read(runid,start)
    slams_events = slams_eventfile_read(runid,start+1)

    slamsobj_list = []
    dead_slamsobj_list = []

    slamsjet_list = []
    dead_slamsjet_list = []

    # Initialise unique ID counter
    slams_counter = 1
    slamsjet_counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for slams
    for slams_event in slams_events:

        for slams_old_event in slams_events_old:

            if np.intersect1d(slams_old_event,slams_event).size > threshold*len(slams_event):

                # Create unique ID
                curr_slams_id = str(slams_counter).zfill(5)

                # Create new jet object
                slamsobj_list.append(Jet(curr_slams_id,runid,float(start)/2))

                # Append current events to jet object properties
                slamsobj_list[-1].cellids.append(slams_old_event)
                slamsobj_list[-1].cellids.append(slams_event)
                slamsobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                slams_counter += 1

                break

    # Look for slamsjets
    for slamsobj in slamsobj_list:

        for event in events:

            if np.intersect1d(slamsobj.cellids[-1],ja.get_neighbors(vlsvobj,event,[3,3])).size > 0:

                curr_slamsjet_id = str(slamsjet_counter).zfill(5)
                slamsjet_obj = copy.deepcopy(slamsobj)
                slamsjet_obj.ID = curr_slamsjet_id
                slamsjet_obj.cellids[-1] = np.unique(slamsjet_obj.cellids[-1]+event).tolist()
                slamsjet_list.append(slamsjet_obj)
                slamsobj_list.remove(slamsobj)

                slamsjet_counter += 1

                break

    # Track jets, slams and slamjets
    for n in xrange(start+2,stop+1):

        for slamsobj in slamsobj_list:
            if float(n)/2 - slamsobj.times[-1] > 5:
                print("Killed SLAMS {}".format(slamsobj.ID))
                dead_slamsobj_list.append(slamsobj)
                slamsobj_list.remove(slamsobj)

        for slamsjet in slamsjet_list:
            if float(n)/2 - slamsjet.times[-1] > 5:
                print("Killed SLAMSJET {}".format(slamsjet.ID))
                dead_slamsjet_list.append(slamsjet)
                slamsjet_list.remove(slamsjet)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            events = []
            continue

        # Open bulkfile and get bow shock cells
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # Initialise flags for finding splintering jets
        slams_flags = []
        slamsjet_flags = []

        old_slams_events = []
        for slams_event in slams_events:
            old_slams_events.append(slams_event)

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        slams_events = slams_eventfile_read(runid,n)
        slams_events.sort(key=len)
        slams_events = slams_events[::-1]        

        # Iniatilise list of cells currently being tracked
        curr_slams_temp_list = []
        curr_slamsjet_temp_list = []

        # Update existing slams
        for slams_event in slams_events:

            for slamsobj in slamsobj_list:

                if slamsobj.ID in slams_flags:
                    
                    if np.intersect1d(slamsobj.cellids[-2],slams_event).size > threshold*len(slams_event):

                        if track_splinters:

                            curr_slams_id = str(slams_counter).zfill(5)

                            # Create new jet
                            slamsobj_new = Jet(curr_slams_id,runid,float(n)/2)
                            slamsobj_new.cellids.append(slams_event)
                            slamsobj_list.append(slamsobj_new)
                            curr_slams_temp_list.append(slams_event)

                            # Iterate counter
                            slams_counter += 1

                            break

                        else:

                            break

                else:

                    if np.intersect1d(slamsobj.cellids[-1],slams_event).size > threshold*len(slams_event):

                        # Append event to jet object properties
                        slamsobj.cellids.append(slams_event)
                        slamsobj.times.append(float(n)/2)
                        print("Updated SLAMS with ID "+slamsobj.ID)

                        # Flag jet object
                        slams_flags.append(slamsobj.ID)
                        curr_slams_temp_list.append(slams_event)

                        break

        # Update existing slamsjets
        for event in events:

            for slamsjet in slamsjet_list:

                if slamsjet.ID in slamsjet_flags:
                    
                    if np.intersect1d(slamsjet.cellids[-2],event).size > threshold*len(event):

                        if track_splinters:
                            curr_slamsjet_id = str(slamsjet_counter).zfill(5)

                            # Create new jet
                            slamsjet_new = Jet(curr_slamsjet_id,runid,float(n)/2)
                            slamsjet_new.cellids.append(event)
                            slamsjet_list.append(slamsjet_new)
                            curr_slamsjet_temp_list.append(event)

                            # Iterate counter
                            slamsjet_counter += 1

                            break

                        else:

                            break

                else:

                    if np.intersect1d(slamsjet.cellids[-1],event).size > threshold*len(event):

                        # Append event to jet object properties
                        slamsjet.cellids.append(event)
                        slamsjet.times.append(float(n)/2)
                        print("Updated SLAMSJET with ID "+slamsjet.ID)

                        # Flag jet object
                        slamsjet_flags.append(slamsjet.ID)
                        curr_slamsjet_temp_list.append(event)

                        break

        # Look for new slamsjets at bow shock
        for slamsobj in slamsobj_list:

            for event in events:

                if event not in curr_slamsjet_temp_list:

                    if np.intersect1d(slamsobj.cellids[-1],ja.get_neighbors(vlsvobj,event,[3,3])).size > 0:

                        curr_slamsjet_id = str(slamsjet_counter).zfill(5)

                        slamsjet_obj = copy.deepcopy(slamsobj)
                        slamsjet_obj.ID = curr_slamsjet_id
                        if slamsobj.ID in slams_flags:
                            slamsjet_obj.cellids[-1] = np.unique(slamsjet_obj.cellids[-1]+event).tolist()
                        else:
                            slamsjet_obj.cellids.append(event)
                        slamsjet_list.append(slamsjet_obj)
                        slamsobj_list.remove(slamsobj)

                        slamsjet_counter += 1

                        break

        # Look for new slams
        for slams_event in slams_events:

            if slams_event not in curr_slams_temp_list:

                for old_slams_event in old_slams_events:

                    if np.intersect1d(old_slams_event,slams_event).size > threshold*len(slams_event):

                        # Create unique ID
                        curr_slams_id = str(slams_counter).zfill(5)

                        # Create new jet object
                        slamsobj_list.append(Jet(curr_slams_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        slamsobj_list[-1].cellids.append(old_slams_event)
                        slamsobj_list[-1].cellids.append(slams_event)
                        slamsobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        slams_counter += 1

                        break


    slamsjet_list = slamsjet_list + dead_slamsjet_list

    for slamsjet in slamsjet_list:

        # Write jet object cellids and times to files
        slamsjetfile = open("/homeappl/home/sunijona/SLAMSJETS/slamsjets/"+slamsjet.runid+"/"+str(start)+"."+slamsjet.ID+".slamsjet","w")
        timefile = open("/homeappl/home/sunijona/SLAMSJETS/slamsjets/"+slamsjet.runid+"/"+str(start)+"."+slamsjet.ID+".times","w")

        slamsjetfile.write(slamsjet.return_cellid_string())
        timefile.write(slamsjet.return_time_string())

        slamsjetfile.close()
        timefile.close()

    return None