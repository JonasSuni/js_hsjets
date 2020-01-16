import numpy as np
import pytools as pt
import scipy
import jet_analyser as ja
import os
import jet_scripts as js
import copy
import matplotlib.pyplot as plt
import plot_contours as pc
import scipy.constants as sc
import random
import jet_aux as jx

m_p = 1.672621898e-27
r_e = 6.371e+6

#wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"
wrkdir_DNR = os.environ["WRK"]+"/"
propfile_var_list = ["time","x_mean","y_mean","z_mean","A","Nr_cells","r_mean","theta_mean","phi_mean","size_rad","size_tan","x_vmax","y_vmax","z_vmax","n_avg","n_med","n_max","v_avg","v_med","v_max","B_avg","B_med","B_max","T_avg","T_med","T_max","TPar_avg","TPar_med","TPar_max","TPerp_avg","TPerp_med","TPerp_max","beta_avg","beta_med","beta_max","x_min","rho_vmax","b_vmax","pd_avg","pd_med","pd_max","B_sheath","TPar_sheath","TPerp_sheath","T_sheath","n_sheath","v_sheath","pd_sheath","is_upstream"]
propfile_header_list = "time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa],B_sheath [nT],TPar_sheath [MK],TPerp_sheath [MK],T_sheath [MK],n_sheath [1/cm^3],v_sheath [km/s],pd_sheath [nPa],bool"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir="/proj/vlasov"

class PropReader:
    # Class for reading jet property files

    def __init__(self,ID,runid,start=580,fname=None,transient="jet"):

        # Check for transient type
        if transient == "jet":
            inputdir = wrkdir_DNR+"working/jets"
        elif transient == "slamsjet":
            inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
        elif transient == "slams":
            inputdir = wrkdir_DNR+"working/SLAMS/slams"

        self.ID = ID # Should be a string of 5 digits
        self.runid = runid # Should be a string of 3 letters
        self.start = start # Should be a float of accuracy to half a second
        self.transient = transient
        self.meta = []
        self.sw_pars = ja.sw_par_dict(runid) # Solar wind parameters for run
        self.sw_pars[0] /= 1.0e+6 # rho in 1/cm^3
        self.sw_pars[1] /= 1.0e+3 # v in km/s
        self.sw_pars[2] /= 1.0e-9 # Pdyn in nPa
        self.sw_pars[3] /= 1.0e-9 # B in nT

        # Check if passing free-form filename to function
        if type(fname) is not str:
            self.fname = str(start)+"."+ID+".props"
        else:
            self.fname = fname

        # Try opening file
        try:
            props_f = open(inputdir+"/"+runid+"/"+self.fname)
        except IOError:
            raise IOError("File not found!")

        props = props_f.read()
        props_f.close()
        props = props.split("\n")
        if "META" in props[0]:
            self.meta = props[0].split(",")[1:]
            props = props[2:]
        else:
            props = props[1:]
        props = [line.split(",") for line in props]
        self.props = np.asarray(props,dtype="float")
        #self.times = timefile_read(self.runid,self.start,self.ID,transient=self.transient)
        #self.cells = jetfile_read(self.runid,self.start,self.ID,transient=self.transient)

        # Initialise list of variable names and associated dictionary
        var_list = propfile_var_list
        n_list = list(range(len(var_list)))
        self.var_dict = dict(zip(var_list,n_list))

        self.delta_list = ["DT","Dn","Dv","Dpd","DB","DTPar","DTPerp"]
        self.davg_list = ["T_avg","n_max","v_max","pd_max","B_max","TPar_avg","TPerp_avg"]
        self.sheath_list = ["T_sheath","n_sheath","v_sheath","pd_sheath","B_sheath","TPar_sheath","TPerp_sheath"]

    def get_times(self):
        return timefile_read(self.runid,self.start,self.ID,transient=self.transient)

    def get_cells(self):
        return jetfile_read(self.runid,self.start,self.ID,transient=self.transient)

    def read(self,name):
        # Read data of specified variable

        if name in self.var_dict:
            return self.props[:,self.var_dict[name]]
        elif name in self.delta_list:
            return self.read(self.davg_list[self.delta_list.index(name)])-self.read(self.sheath_list[self.delta_list.index(name)])
        elif name == "pdyn_vmax":
            return 1.0e+21*m_p*self.read("rho_vmax")*self.read("v_max")**2
        elif name == "duration":
            t = self.read("time")
            return np.ones(t.shape)*(t[-1]-t[0] + 0.5)
        elif name == "size_ratio":
            return self.read("size_rad")/self.read("size_tan")
        elif name == "death_distance":
            x,y,z = self.read("x_vmax")[-1],self.read("y_vmax")[-1],self.read("z_vmax")[-1]
            t = self.read("time")[-1]
            outp = np.ones(t.shape)
            pfit = ja.bow_shock_markus(self.runid,int(t*2))[::-1]
            x_bs = np.polyval(pfit,np.linalg.norm([y,z]))
            return outp*(x-x_bs)
        elif name == "bs_distance":
            y,t = self.read("y_mean"),self.read("time")
            x_mp = np.zeros_like(y)
            for n in range(y.size):
                p = ja.bow_shock_markus(self.runid,int(t[n]*2))[::-1]
                x_mp[n] = np.polyval(p,y[n])
            return x_mp
        else:
            print("Variable not found!")
            return None

    def amax_index(self):
        # Return list index of time when area is largest

        return self.read("A").argmax()

    def time_index(self,time):
        # Return list index of specified time

        time_arr = self.read("time")
        if time not in time_arr:
            raise IOError("Time not found!")
        else:
            return time_arr.tolist().index(time)

    def read_at_time(self,var,time):
        # Return variable data at specified time

        return self.read(var)[self.time_index(time)]

    def read_at_randt(self,var):

        time_arr = self.read("time")
        randt = random.choice(time_arr)

        return self.read_at_time(var,randt)

    def read_at_amax(self,name):
        # Return variable data at time when area is largest

        return self.read(name)[self.amax_index()]

class Transient:
    # Class for identifying and handling individual jets and their properties

    def __init__(self,ID,runid,birthday,transient="jet"):

        self.ID = ID # Should be a string of 5 digits
        self.runid = runid # Should be a string of 3 letters
        self.birthday = birthday # Should be a float of accuracy to half a second
        self.cellids = []
        self.times = [birthday]
        self.props = []
        self.meta = ["META"]
        self.transient = transient

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):
        # Return string of lists of cellids for printing to file

        return "\n".join([",".join(list(map(str,l))) for l in self.cellids])

    def return_time_string(self):
        # Return string of times for printing to file

        return "\n".join(list(map(str,self.times)))

    def jetprops_write(self,start):

        if self.transient != "slamsjet":
            if self.times[-1]-self.times[0] >= 4.5:
                propfile_write(self.runid,start,self.ID,self.props,self.meta,transient=self.transient)
            else:
                print("Transient {} too short-lived, propfile not written!".format(self.ID))
        else:
            if self.times[-1]-self.times[0] < 4.5:
                print("Transient {} is not SLAMSJET, propfile not written!".format(self.ID))
                return None
            x = np.array(self.props)[:,1]
            y = np.array(self.props)[:,2]
            t = self.times
            x_birth,y_birth = x[0],y[0]
            x_death,y_death = x[-1],y[-1]
            bsp_birth,bsp_death = [ja.bow_shock_markus(self.runid,int(t[0]*2))[::-1],ja.bow_shock_markus(self.runid,int(t[-1]*2))[::-1]]
            x_bs = [np.polyval(ja.bow_shock_markus(self.runid,int(t[n]*2))[::-1],y[n]) for n in range(len(y))]
            t_crossing = t[np.argmin(np.abs(np.array(x)-np.array(x_bs)))]
            bsx_birth,bsx_death = [np.polyval(bsp_birth,y_birth),np.polyval(bsp_death,y_death)]
            if t_crossing-t[0] >= 4.5 and t[-1]-t_crossing >= 4.5 and x_birth >= bsx_birth and x_death <= bsx_death:
                propfile_write(self.runid,start,self.ID,self.props,self.meta,transient=self.transient)
            else:
                print("Transient {} is not SLAMSJET, propfile not written!".format(self.ID))
                return None


        return None

def jet_maker(runid,start,stop,boxre=[6,18,-8,6],maskfile=False,avgfile=False,nbrs=[2,2,0],transient="jet"):

    if transient == "jet":
        outputdir = wrkdir_DNR+"working/events/"+runid+"/"
        maskdir = wrkdir_DNR+"working/Masks/"+runid+"/"
        nmin=2
        nmax=4500
    elif transient == "slams":
        outputdir = wrkdir_DNR+"working/SLAMS/events/"+runid+"/"
        maskdir = wrkdir_DNR+"working/SLAMS/Masks/"+runid+"/"
        nmin=2
        nmax=6000
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/events/"+runid+"/"
        maskdir = wrkdir_DNR+"working/SLAMSJETS/Masks/"+runid+"/"
        nmin=2
        nmax=6000

    global rho_sw_g

    rho_sw_g = jx.sw_par_dict(runid)[0]

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = ja.find_bulkpath(runid)

    for file_nr in range(start,stop+1):

        # find correct file based on file number and run id

        bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(file_nr)+" not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        vlsvobj.optimize_open_file()

        # create mask
        if maskfile:
            msk = np.loadtxt(maskdir+str(file_nr)+".mask").astype(int)
        else:
            msk = ja.make_cust_mask_opt(file_nr,runid,180,boxre,avgfile,transient=transient)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        jets,props_inc = sort_jets_2(vlsvobj,msk,nmin,nmax,nbrs)

        props = [[float(file_nr)/2.0]+line for line in props_inc]

        eventprop_write(runid,file_nr,props,transient=transient)

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for jet in jets:

            fileobj.write(",".join(list(map(str,jet)))+"\n")

        fileobj.close()
        vlsvobj.optimize_close_file()

    return None

def timefile_read(runid,filenr,key,transient="jet"):
    # Read array of times from file

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
    elif transient == "slams":
        inputdir = wrkdir_DNR+"working/SLAMS/slams"

    tf = open("{}/{}/{}.{}.times".format(inputdir,runid,str(filenr),key),"r")
    contents = tf.read().split("\n")
    tf.close()

    return list(map(float,contents))

def jetfile_read(runid,filenr,key,transient="jet"):
    # Read array of cellids from file

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
        extension = "slamsjet"
    elif transient == "slams":
        inputdir = wrkdir_DNR+"working/SLAMS/slams"
        extension = "slams"

    outputlist = []

    jf = open("{}/{}/{}.{}.{}".format(inputdir,runid,str(filenr),key,extension),"r")
    contents = jf.read()
    jf.close()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(list(map(int,line.split(","))))

    return outputlist

def eventfile_read(runid,filenr,transient="jet"):
    # Read array of arrays of cellids from file

    if transient == "jet":
        inputdir = wrkdir_DNR+"working/events"
    elif transient == "slams":
        inputdir = wrkdir_DNR+"working/SLAMS/events"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/events"

    outputlist = []

    ef = open("{}/{}/{}.events".format(inputdir,runid,str(filenr)),"r")
    contents = ef.read().strip("\n")
    ef.close()
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(list(map(int,line.split(","))))

    return outputlist

def eventprop_write(runid,filenr,props,transient="jet"):

    if transient == "jet":
        outputdir = wrkdir_DNR+"working/event_props/"+runid
    elif transient == "slams":
        outputdir = wrkdir_DNR+"working/SLAMS/event_props/"+runid
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/event_props/"+runid

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    open(outputdir+"/{}.eventprops".format(str(filenr)),"w").close()
    epf = open(outputdir+"/{}.eventprops".format(str(filenr)),"w")

    epf.write(propfile_header_list+"\n")

    epf.write("\n".join([",".join(list(map(str,line))) for line in props]))
    epf.close()
    print("Wrote to "+outputdir+"/{}.eventprops".format(str(filenr)))

def eventprop_read(runid,filenr,transient="jet"):

    if transient == "jet":
        inputname = wrkdir_DNR+"working/event_props/{}/{}.eventprops".format(runid,str(filenr))
    elif transient == "slams":
        inputname = wrkdir_DNR+"working/SLAMS/event_props/{}/{}.eventprops".format(runid,str(filenr))
    elif transient == "slamsjet":
        inputname = wrkdir_DNR+"working/SLAMSJETS/event_props/{}/{}.eventprops".format(runid,str(filenr))

    try:
        props_f = open(inputname)
    except IOError:
        raise IOError("File not found!")

    props = props_f.read()
    props_f.close()
    props = props.split("\n")[1:]
    props = [list(map(float,line.split(","))) for line in props]

    return props

def propfile_write(runid,filenr,key,props,meta,transient="jet"):
    # Write jet properties to file

    if transient == "jet":
        outputdir = wrkdir_DNR+"working/jets"
    elif transient == "slams":
        outputdir = wrkdir_DNR+"working/SLAMS/slams"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"

    open(outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open(outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write(",".join(meta)+"\n")
    pf.write(propfile_header_list+"\n")
    pf.write("\n".join([",".join(list(map(str,line))) for line in props]))
    pf.close()
    print("Wrote to "+outputdir+"/"+runid+"/"+str(filenr)+"."+key+".props")

def tpar_reader(runid,filenumber,cellids,cells):
    # Read parallel temperatures of specific cells

    TPar = np.loadtxt(wrkdir_DNR+"TP/"+runid+"/"+str(filenumber)+".tpar")
    TPar = TPar[np.in1d(cellids,cells)]

    return TPar

def tperp_reader(runid,filenumber,cellids,cells):
    # Read perpendicular temperatures of specific cells

    TPerp = np.loadtxt(wrkdir_DNR+"TP/"+runid+"/"+str(filenumber)+".tperp")
    TPerp = TPerp[np.in1d(cellids,cells)]

    return TPerp

def calc_event_props(vlsvobj,cells):

    if np.argmin(vlsvobj.get_spatial_mesh_size())==1:
        sheath_cells = get_sheath_cells(vlsvobj,cells,neighborhood_reach=[2,0,2])
        ssh_cells = get_sheath_cells(vlsvobj,cells,neighborhood_reach=[1,0,1])
    else:
        sheath_cells = get_sheath_cells(vlsvobj,cells)
        ssh_cells = get_sheath_cells(vlsvobj,cells,neighborhood_reach=[1,1,0])

    # read variables
    if vlsvobj.check_variable("X"):
        X = np.array(vlsvobj.read_variable("X",cellids=cells),ndmin=1)
        Y = np.array(vlsvobj.read_variable("Y",cellids=cells),ndmin=1)
        Z = np.array(vlsvobj.read_variable("Z",cellids=cells),ndmin=1)
        dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
    else:
        X,Y,Z = ja.xyz_reconstruct(vlsvobj,cellids=cells)
        X = np.array(X,ndmin=1)
        Y = np.array(Y,ndmin=1)
        Z = np.array(Z,ndmin=1)
        dA = ja.get_cell_volume(vlsvobj)

    var_list = ["rho","v","B","Temperature","CellID","beta","TParallel","TPerpendicular"]
    var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","CellID","proton/beta","proton/TParallel","proton/TPerpendicular"]
    sheath_list = ["rho","v","B","Temperature","TParallel","TPerpendicular","Pdyn"]
    sheath_list_alt = ["proton/rho","proton/V","B","proton/Temperature","proton/TParallel","proton/TPerpendicular","proton/Pdyn"]

    if vlsvobj.check_variable("proton/rho"):
        rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=cells),ndmin=1) for s in var_list_alt]
        rho_sheath,v_sheath,B_sheath,T_sheath,TPar_sheath,TPerp_sheath,pd_sheath = [np.array(vlsvobj.read_variable(s,cellids=sheath_cells),ndmin=1) for s in sheath_list_alt]
        rho_ssh = np.array(vlsvobj.read_variable("proton/rho",cellids=ssh_cells),ndmin=1)
        pr_rhonbs = np.array(vlsvobj.read_variable("RhoNonBackstream", cellids=ssh_cells),ndmin=1)
        pr_PTDNBS = np.array(vlsvobj.read_variable("PTensorNonBackstreamDiagonal", cellids=ssh_cells),ndmin=1)
    else:
        rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=cells),ndmin=1) for s in var_list]
        rho_sheath,v_sheath,B_sheath,T_sheath,TPar_sheath,TPerp_sheath,pd_sheath = [np.array(vlsvobj.read_variable(s,cellids=sheath_cells),ndmin=1) for s in sheath_list]
        rho_ssh = np.array(vlsvobj.read_variable("rho",cellids=ssh_cells),ndmin=1)
        pr_rhonbs = np.array(vlsvobj.read_variable("RhoNonBackstream", cellids=ssh_cells),ndmin=1)
        pr_PTDNBS = np.array(vlsvobj.read_variable("PTensorNonBackstreamDiagonal", cellids=ssh_cells),ndmin=1)

    #rho_sw = rho_sw_g
    T_sw = 0.5e+6

    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    #is_upstream = int(np.all(rho_ssh < 2*rho_sw))
    is_upstream = int(np.all(pr_TNBS < 3*T_sw))

    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    # Scale variables
    rho /= 1.0e+6
    v /= 1.0e+3
    B /= 1.0e-9
    B_sheath /= 1.0e-9
    pdyn /= 1.0e-9
    T /= 1.0e+6
    TParallel /= 1.0e+6
    TPerpendicular /= 1.0e+6
    TPar_sheath /= 1.0e+6
    TPerp_sheath /= 1.0e+6
    T_sheath /= 1.0e+6
    rho_sheath /= 1.0e+6
    v_sheath /= 1.0e+3
    pd_sheath /= 1.0e-9

    # Calculate magnitudes of v and B
    vmag = np.linalg.norm(v,axis=-1)
    Bmag = np.linalg.norm(B,axis=-1)
    B_sheath_mag = np.linalg.norm(B_sheath,axis=-1)
    v_sheath_mag = np.linalg.norm(v_sheath,axis=-1)

    if type(vmag) == float:
        vmag = np.array(vmag)
    if type(Bmag) == float:
        Bmag = np.array(Bmag)
    if type(B_sheath_mag) == float:
        B_sheath_mag = np.array(B_sheath_mag)

    n_avg,n_med,n_max = mean_med_max(rho)

    v_avg,v_med,v_max = mean_med_max(vmag)

    B_avg,B_med,B_max = mean_med_max(Bmag)

    pd_avg,pd_med,pd_max = mean_med_max(pdyn)

    T_avg,T_med,T_max = mean_med_max(T)

    TPar_avg,TPar_med,TPar_max = mean_med_max(TParallel)

    TPerp_avg,TPerp_med,TPerp_max = mean_med_max(TPerpendicular)

    beta_avg,beta_med,beta_max = mean_med_max(beta)

    # Convert X,Y,Z to spherical coordinates
    r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
    theta = np.rad2deg(np.arccos(Z/r))
    phi = np.rad2deg(np.arctan(Y/X))

    # calculate geometric center of jet
    r_mean = np.mean(r)/r_e
    theta_mean = np.mean(theta)
    phi_mean = np.mean(phi)

    # Geometric center of jet in cartesian coordinates
    x_mean = np.nanmean(X)/r_e
    y_mean = np.nanmean(Y)/r_e
    z_mean = np.nanmean(Z)/r_e

    # Position of maximum velocity in cartesian coordinates
    x_max = X[vmag==max(vmag)][0]/r_e
    y_max = Y[vmag==max(vmag)][0]/r_e
    z_max = Z[vmag==max(vmag)][0]/r_e

    # Minimum x and density at maximum velocity
    x_min = min(X)/r_e
    rho_vmax = rho[vmag==max(vmag)][0]
    b_vmax = beta[vmag==max(vmag)][0]

    # calculate jet size
    A = dA*len(cells)/(r_e**2)
    Nr_cells = len(cells)

    # calculate linear sizes of jet
    size_rad = (max(X)-min(X))/r_e+np.sqrt(dA)/r_e
    size_tan = A/size_rad

    [B_sheath_avg,TPar_sheath_avg,TPerp_sheath_avg,T_sheath_avg,n_sheath_avg,v_sheath_avg,pd_sheath_avg] = [np.nanmean(v) for v in [B_sheath_mag,TPar_sheath,TPerp_sheath,T_sheath,rho_sheath,v_sheath_mag,pd_sheath]]

    temp_arr = [x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max,B_sheath_avg,TPar_sheath_avg,TPerp_sheath_avg,T_sheath_avg,n_sheath_avg,v_sheath_avg,pd_sheath_avg,is_upstream]

    return temp_arr

def get_sheath_cells(vlsvobj,cells,neighborhood_reach=[2,2,0]):

    plus_sheath_cells = get_neighbors(vlsvobj,cells,neighborhood_reach)
    sheath_cells = plus_sheath_cells[~np.in1d(plus_sheath_cells,cells)]

    return sheath_cells

def get_neighbors(vlsvobj,c_i,neighborhood_reach=[1,1,0]):
    # finds the neighbors of the specified cells within the maximum offsets in neighborhood_reach

    # initialise array of neighbors
    neighbors = np.array([],dtype=int)

    # range of offsets to take into account
    x_r = range(-1*neighborhood_reach[0],neighborhood_reach[0]+1)
    y_r = range(-1*neighborhood_reach[1],neighborhood_reach[1]+1)
    z_r = range(-1*neighborhood_reach[2],neighborhood_reach[2]+1)

    for n in c_i:

        # append cellids of neighbors, cast as int, to array of neighbors
        for a in x_r:
            for b in y_r:
                for c in z_r:
                    neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,b,c],periodic=[0,0,0])))

    # discard invalid cellids
    neighbors = neighbors[neighbors != 0]

    # discard duplicate cellids
    neighbors = np.unique(neighbors)

    return neighbors

def sort_jets_2(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1,0]):

    cells = np.array(cells,ndmin=1,dtype=int)

    events = []
    curr_event = np.array([],dtype=int)

    while cells.size != 0:
        curr_event = np.array([cells[0]],dtype=int)
        curr_event_size = curr_event.size

        curr_event = np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))

        while curr_event.size != curr_event_size:

            curr_event_size = curr_event.size

            curr_event = np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))

        events.append(curr_event.astype(int))
        cells = cells[~np.in1d(cells,curr_event)]

    events_culled = [jet for jet in events if jet.size>=min_size and jet.size<=max_size]

    props = [calc_event_props(vlsvobj,event) for event in events_culled]

    return [events_culled,props]



def mean_med_max(var):

    var_mean = np.nanmean(var)
    var_med = np.median(var)
    var_max = np.max(var)

    return [var_mean,var_med,var_max]



def check_threshold(A,B,thresh):

    return np.intersect1d(A,B).size > thresh*min(len(A),len(B))

def track_jets(runid,start,stop,threshold=0.3,nbrs_bs=[3,3,0],transient="jet"):

    if transient == "jet":
        outputdir = wrkdir_DNR+"working/jets/"+runid
        extension = ".jet"
    elif transient == "slams":
        outputdir = wrkdir_DNR+"working/SLAMS/slams/"+runid
        extension = ".slams"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets/"+runid
        extension = ".slamsjet"


    # bulkpath = ja.find_bulkpath(runid)
    #
    # bulkname = "bulk."+str(start).zfill(7)+".vlsv"
    #
    # if bulkname not in os.listdir(bulkpath):
    #     print("Bulk file "+str(start)+" not found, exiting")
    #     return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # Get solar wind parameters
    sw_pars = ja.sw_par_dict(runid)
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file, get Cell IDs and sort them
    #vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    # Find bow shock cells and area of one cell
    #bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    #dA = ja.get_cell_volume(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start,transient=transient)
    bs_props = eventprop_read(runid,start,transient=transient)
    events_unsrt = eventfile_read(runid,start+1,transient=transient)
    props_unsrt = eventprop_read(runid,start+1,transient=transient)

    # remove events that are not initially at the bow shock
    bs_events = []
    for old_event in events_old:
        #if np.intersect1d(bs_cells,get_neighbors(vlsvobj,old_event,nbrs_bs)).size > 0:
            #bs_events.append(old_event)
        bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for jets at bow shock
    for event in events_unsrt:

        for bs_event in bs_events:

            if check_threshold(bs_event,event,threshold):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(Transient(curr_id,runid,float(start)/2,transient=transient))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].props.append(bs_props[bs_events.index(bs_event)])
                jetobj_list[-1].props.append(props_unsrt[events_unsrt.index(event)])
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in range(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] + 0.5 > 10:
                print("Killing jet {}".format(jetobj.ID))
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        # bulkname = "bulk."+str(n).zfill(7)+".vlsv"
        #
        # if bulkname not in os.listdir(bulkpath):
        #     print("Bulk file "+str(n)+" not found, continuing")
        #     events = []
        #     continue

        # Open bulkfile and get bow shock cells
        #vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # Filtered list of events that are at the bow shock at the current time
        bs_events = []
        bs_props = props_unsrt
        for old_event in events_unsrt:
            bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events_unsrt = eventfile_read(runid,n,transient=transient)
        props_unsrt = eventprop_read(runid,n,transient=transient)
        events = sorted(events_unsrt,key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID not in flags:

                    if event not in curr_jet_temp_list:

                        if check_threshold(jetobj.cellids[-1],event,threshold):

                            # Append event to jet object properties
                            jetobj.cellids.append(event)
                            jetobj.props.append(props_unsrt[events_unsrt.index(event)])
                            jetobj.times.append(float(n)/2)
                            print("Updated jet "+jetobj.ID)

                            # Flag jet object
                            flags.append(jetobj.ID)
                            curr_jet_temp_list.append(event)

                        else:
                            continue

                    else:
                        if check_threshold(jetobj.cellids[-1],event,threshold):
                            jetobj.meta.append("merger")
                            print("Killing jet {}".format(jetobj.ID))
                            dead_jetobj_list.append(jetobj)
                            jetobj_list.remove(jetobj)
                        else:
                            continue

                else:
                    if event not in curr_jet_temp_list:

                        if check_threshold(jetobj.cellids[-2],event,threshold):

                            curr_id = str(counter).zfill(5)

                            # Create new jet
                            jetobj_new = Transient(curr_id,runid,float(n)/2,transient=transient)
                            jetobj_new.meta.append("splinter")
                            jetobj_new.cellids.append(event)
                            jetobj_new.props.append(props_unsrt[events_unsrt.index(event)])
                            jetobj_list.append(jetobj_new)
                            curr_jet_temp_list.append(event)

                            # Iterate counter
                            counter += 1

                            break
                        else:
                            continue

                    else:
                        continue

        # Look for new jets at bow shock
        for event in events:

            if event not in curr_jet_temp_list:

                for bs_event in bs_events:

                    if check_threshold(bs_event,event,threshold):

                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_list.append(Transient(curr_id,runid,float(n-1)/2,transient=transient))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].props.append(bs_props[bs_events.index(bs_event)])
                        jetobj_list[-1].props.append(props_unsrt[events_unsrt.index(event)])
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open(outputdir+"/"+str(start)+"."+jetobj.ID+extension,"w")
        timefile = open(outputdir+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())
        jetobj.jetprops_write(start)

        jetfile.close()
        timefile.close()

    return None

def slams_eventfile_read(runid,filenr):
    # Read array of arrays of cellids from file

    outputlist = []

    ef = open(wrkdir_DNR+"working/SLAMS/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read().strip("\n")
    ef.close()
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(list(map(int,line.split(","))))

    return outputlist
