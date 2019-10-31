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

m_p = 1.672621898e-27
r_e = 6.371e+6

wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"
propfile_var_list = ["time","x_mean","y_mean","z_mean","A","Nr_cells","r_mean","theta_mean","phi_mean","size_rad","size_tan","x_vmax","y_vmax","z_vmax","n_avg","n_med","n_max","v_avg","v_med","v_max","B_avg","B_med","B_max","T_avg","T_med","T_max","TPar_avg","TPar_med","TPar_max","TPerp_avg","TPerp_med","TPerp_max","beta_avg","beta_med","beta_max","x_min","rho_vmax","b_vmax","pd_avg","pd_med","pd_max","B_sheath","TPar_sheath","TPerp_sheath","T_sheath","n_sheath","v_sheath","pd_sheath"]
propfile_header_list = "time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa],B_sheath [nT],TPar_sheath [MK],TPerp_sheath [MK],T_sheath [MK],n_sheath [1/cm^3],v_sheath [km/s],pd_sheath [nPa],B_sheath [nT]"

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

        # Initialise list of variable names and associated dictionary
        var_list = propfile_var_list
        n_list = list(xrange(len(var_list)))
        self.var_dict = dict(zip(var_list,n_list))

        self.delta_list = ["DT","Dn","Dv","Dpd","DB","DTPar","DTPerp"]
        self.davg_list = ["T_avg","n_max","v_max","pd_max","B_max","TPar_avg","TPerp_avg"]
        self.sheath_list = ["T_sheath","n_sheath","v_sheath","pd_sheath","B_sheath","TPar_sheath","TPerp_sheath"]

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

    def __init__(self,ID,runid,birthday):

        self.ID = ID # Should be a string of 5 digits
        self.runid = runid # Should be a string of 3 letters
        self.birthday = birthday # Should be a float of accuracy to half a second
        self.cellids = []
        self.times = [birthday]
        self.props = []
        self.meta = ["META"]

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):
        # Return string of lists of cellids for printing to file

        return "\n".join([",".join(map(str,l)) for l in self.cellids])

    def return_time_string(self):
        # Return string of times for printing to file

        return "\n".join(map(str,self.times))

    def jetprops_write(self,start):

        if self.times[-1]-self.times[0] >= 4.5:
            propfile_write(self.runid,start,self.ID,self.props,self.meta)
        else:
            print("Transient {} too short-lived, propfile not written!".format(self.ID))

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

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for file_nr in xrange(start,stop+1):

        # find correct file based on file number and run id
        bulkpath = ja.find_bulkpath(runid)

        bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(file_nr)+" not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt(maskdir+str(file_nr)+".mask").astype(int)
        else:
            msk = ja.make_cust_mask_opt(file_nr,runid,180,boxre,avgfile,transient=transient)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        jets,props_inc = sort_jets_new(vlsvobj,msk,nmin,nmax,nbrs)

        props = [[float(file_nr)/2.0]+line for line in props_inc]

        eventprop_write(runid,file_nr,props,transient=transient)

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

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"

    tf = open("{}/{}/{}.{}.times".format(inputdir,runid,str(filenr),key),"r")
    contents = tf.read().split("\n")
    tf.close()

    return map(float,contents)

def jetfile_read(runid,filenr,key,transient="jet"):
    # Read array of cellids from file

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
        extension = "slamsjet"

    outputlist = []

    jf = open("{}/{}/{}.{}.{}".format(inputdir,runid,str(filenr),key,extension),"r")
    contents = jf.read()
    jf.close()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def eventfile_read(runid,filenr,transient="jet"):
    # Read array of arrays of cellids from file

    if transient == "jet":
        inputdir = wrkdir_DNR+"working/events"
    elif transient == "slams":
        inputdir = wrkdir_DNR+"working/SLAMS/events"

    outputlist = []

    ef = open("{}/{}/{}.events".format(inputdir,runid,str(filenr)),"r")
    contents = ef.read().strip("\n")
    ef.close()
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def eventprop_write(runid,filenr,props,transient="jet"):

    if transient == "jet":
        outputdir = wrkdir_DNR+"working/event_props/"+runid
    elif transient == "slams":
        outputdir = wrkdir_DNR+"working/SLAMS/event_props"+runid

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    open(outputdir+"/{}.eventprops".format(str(filenr)),"w").close()
    epf = open(outputdir+"/{}.eventprops".format(str(filenr)),"w")

    epf.write(propfile_header_list+"\n")

    epf.write("\n".join([",".join(map(str,line)) for line in props]))
    epf.close()
    print("Wrote to "+outputdir+"/{}.eventprops".format(str(filenr)))

def eventprop_read(runid,filenr,transient="jet"):

    if transient == "jet":
        inputname = wrkdir_DNR+"working/event_props/{}/{}.eventprops".format(runid,str(filenr))
    elif transient == "slams":
        inputname = wrkdir_DNR+"working/SLAMS/event_props/{}/{}.eventprops".format(runid,str(filenr))

    try:
        props_f = open(inputname)
    except IOError:
        raise IOError("File not found!")

    props = props_f.read()
    props_f.close()
    props = props.split("\n")[1:]
    props = [map(float,line.split(",")) for line in props]

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
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
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

    if np.argmin(vlsvobj.get_spatial_mesh_size()==1):
        sheath_cells = get_sheath_cells(vlsvobj,cells,neighborhood_reach=[2,0,2])
    else:
        sheath_cells = get_sheath_cells(vlsvobj,cells)

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

    try:
        rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=cells),ndmin=1) for s in var_list_alt]
        rho_sheath,v_sheath,B_sheath,T_sheath,TPar_sheath,TPerp_sheath,pd_sheath = [np.array(vlsvobj.read_variable(s,cellids=sheath_cells),ndmin=1) for s in sheath_list_alt]
    except:
        rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=cells),ndmin=1) for s in var_list]
        rho_sheath,v_sheath,B_sheath,T_sheath,TPar_sheath,TPerp_sheath,pd_sheath = [np.array(vlsvobj.read_variable(s,cellids=sheath_cells),ndmin=1) for s in sheath_list]

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

    temp_arr = [x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max,B_sheath_avg,TPar_sheath_avg,TPerp_sheath_avg,T_sheath_avg,n_sheath_avg,v_sheath_avg,pd_sheath_avg]

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
    x_r = xrange(-1*neighborhood_reach[0],neighborhood_reach[0]+1)
    y_r = xrange(-1*neighborhood_reach[1],neighborhood_reach[1]+1)
    z_r = xrange(-1*neighborhood_reach[2],neighborhood_reach[2]+1)

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

def sort_jets_new(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1,0]):
    # sort masked cells into events based on proximity in X,Y-space

    # initialise list of events and current event
    events = []
    curr_event = np.array([],dtype=int)

    for cell in cells:

        # check if cell already in list of events
        bl_a = False
        for event in events:
            if cell in event:
                bl_a = True
        if bl_a:
            continue

        # number of times to search for more neighbors
        it_range = xrange(200)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_size = curr_event.size

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [jet for jet in events if jet.size>=min_size and jet.size<=max_size]

    props = [calc_event_props(vlsvobj,event) for event in events_culled]

    return [events_culled,props]

def mean_med_max(var):

    var_mean = np.nanmean(var)
    var_med = np.median(var)
    var_max = np.max(var)

    return [var_mean,var_med,var_max]

def calc_jet_properties(runid,start,jetid,tp_files=False,transient="jet"):

    # Check transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
        extension = "slamsjet"

    # Check if transient with specified ID exists
    if "{}.{}.{}".format(str(start),jetid,extension) not in os.listdir("{}/{}".format(inputdir,runid)):
        print("Transient with ID "+jetid+" does not exist, exiting.")
        return 1

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid,transient)
    time_list = timefile_read(runid,start,jetid,transient)

    # Discard jet if it's very short-lived
    if time_list[-1] - time_list[0] + 0.5 < 5:
        print("Transient not sufficiently long-lived, exiting.")
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
            X = np.array(vlsvobj.read_variable("X",cellids=curr_list),ndmin=1)
            Y = np.array(vlsvobj.read_variable("Y",cellids=curr_list),ndmin=1)
            Z = np.array(vlsvobj.read_variable("Z",cellids=curr_list),ndmin=1)
        else:
            X,Y,Z = ja.xyz_reconstruct(vlsvobj,cellids=curr_list)
            X = np.array(X,ndmin=1)
            Y = np.array(Y,ndmin=1)
            Z = np.array(Z,ndmin=1)

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
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list_alt]
            except:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list]
        else:
            rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list]

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

        if type(vmag) == float:
            vmag = np.array(vmag)
        if type(Bmag) == float:
            Bmag = np.array(Bmag)

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

def check_threshold(A,B,thresh):

    return np.intersect1d(A,B).size > thresh*min(len(A),len(B))

def track_jets(runid,start,stop,threshold=0.3,nbrs_bs=[3,3,0]):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    bulkpath = ja.find_bulkpath(runid)

    bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists(wrkdir_DNR+"working/jets/"+runid):
        try:
            os.makedirs(wrkdir_DNR+"working/jets/"+runid)
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

    # Find bow shock cells and area of one cell
    #bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = ja.get_cell_volume(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    bs_props = eventprop_read(runid,start)
    events_unsrt = eventfile_read(runid,start+1)
    props_unsrt = eventprop_read(runid,start+1)

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
                jetobj_list.append(Transient(curr_id,runid,float(start)/2))

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
    for n in xrange(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] + 0.5 > 10:
                print("Killing jet {}".format(jetobj.ID))
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
        #bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # Filtered list of events that are at the bow shock at the current time
        bs_events = []
        bs_props = props_unsrt
        for old_event in events_unsrt:
            #if np.intersect1d(bs_cells,get_neighbors(vlsvobj,old_event,nbrs_bs)).size > 0:
            #    bs_events.append(old_event)
            bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events_unsrt = eventfile_read(runid,n)
        props_unsrt = eventprop_read(runid,n)
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
                            jetobj_new = Transient(curr_id,runid,float(n)/2)
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
                        jetobj_list.append(Transient(curr_id,runid,float(n-1)/2))

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
        jetfile = open(wrkdir_DNR+"working/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".jet","w")
        timefile = open(wrkdir_DNR+"working/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

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

        outputlist.append(map(int,line.split(",")))

    return outputlist
