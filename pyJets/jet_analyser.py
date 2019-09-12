import pytools as pt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.constants as sc
import scipy.ndimage
import scipy.optimize as so

m_p = 1.672621898e-27
r_e = 6.371e+6

wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"

def bs_fitter_new(runid,start,stop,step,angle_offset=0):

    filenr_arr = np.array(range(start,stop+1,step))
    bsr_arr = np.zeros_like(filenr_arr).astype(float)

    for n in range(filenr_arr.size):
        bsr_arr[n] = bs_finder_new(runid,filenr_arr[n],angle_offset=angle_offset)

    time_arr = filenr_arr.astype(float)/2

    p = np.polyfit(time_arr,bsr_arr,deg=1)

    print("bsr={:.3g} t + {:.3g}".format(p[0],p[1]))

def bs_finder_new(runid,file_nr,angle_offset=0):

    rho_sw = sw_par_dict(runid)[0]

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

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    rho = vlsvobj.read_variable("rho")
    cellids = vlsvobj.read_variable("CellID")
    X,Y,Z = xyz_reconstruct(vlsvobj,cellids)

    if vlsvobj.get_spatial_mesh_size()[2]==1:
        r_angle = np.rad2deg(np.arctan(Y/X))
    elif vlsvobj.get_spatial_mesh_size()[3]==1:
        r_angle = np.rad2deg(np.arctan(Z/X))
    else:
        r_angle = np.rad2deg(np.arctan(np.linalg.norm([Y,Z],axis=0)/X))

    mask1 = (rho>=2*rho_sw)
    mask2 = (X>=0)
    mask3 = (r_angle<=45-angle_offset)
    mask4 = (r_angle>=-45-angle_offset)

    mask = np.logical_and(np.logical_and(mask1,mask2),np.logical_and(mask3,mask4))

    R = np.linalg.norm(np.array([X,Y,Z]),axis=0)/r_e

    R_masked = R[mask]

    h = plt.hist(R_masked,bins=np.arange(0.0,20.0,0.1))

    hg = np.gradient(h[0])

    print(file_nr,h[1][hg==np.min(hg)][-1])
    return h[1][hg==np.min(hg)][-1]

def rho_r_script():

    runids = ["ABA","ABC","AEA","AEC","BFD"]

    print("r = r(290) + k*(t-290)")

    for runid in runids:
        p = rho_r_auto(runid,580,800,[8,0,0],10)
        print(runid+": k = "+str(p[0]))
        print(runid+": r(290) = "+str(p[1]+290*p[0]))

    return 0

def rho_r_auto(runid,start,stop,start_p,len_line):

    n_list = np.array(xrange(start,stop+1))
    time_list = n_list/2.0
    r_list = np.zeros(n_list.shape)

    for i in xrange(n_list.size):
        r_list[i] = rho_r_simple(runid,n_list[i],start_p,len_line)

    p = np.polyfit(time_list,r_list,deg=1)

    return p

def rho_r_simple(runid,filenumber,start_p,len_line):

    if type(start_p) == list and len(start_p) == 3:
        pass
    elif type(start_p) in [int,float]:
        start_p = [start_p,0,0]
    else:
        print("start_p has invalid format, please use float, int or list!")
        return 1

    # find correct file based on file number and run id
    if runid in ["AEC"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    angles_dict = {"ABA":[30,0],"ABC":[5,0],"AEA":[30,0],"AEC":[5,0],"BFD":[0,45]}

    a = map(np.deg2rad,angles_dict[runid])

    p1 = np.array([c*r_e for c in start_p])
    len_line = len_line*r_e
    p2 = p1+len_line*np.array([np.cos(a[0])*np.cos(a[1]),-np.sin(a[0])*np.cos(a[1]),-np.sin(a[1])])
    
    cut_thru = pt.calculations.cut_through(vlsvobj,p1,p2)

    cells = cut_thru[0].data.data.astype(int)[1:]
    cellids = vlsvobj.read_variable("CellID")
    if vlsvobj.check_variable("rho"):
        rho = vlsvobj.read_variable("rho")[cellids.argsort()]
    else:
        rho = vlsvobj.read_variable("proton/rho")[cellids.argsort()]

    cellids.sort()

    r_data = np.linalg.norm(cut_thru[2].data.data,axis=-1)/r_e
    r_data = r_data[cells.argsort()]
    
    rho_data = rho[np.in1d(cellids,cells)]
    rho_data = rho_data[r_data.argsort()]
    r_data.sort()

    rho_der = np.gradient(rho_data)

    return r_data[rho_der == np.min(rho_der)][0]

def bow_shock_markus(runid,filenr):

    runids = ["ABA","ABC","AEA","AEC"]
    start_time = [580,580,580,580]
    stop_time = [839,1179,1339,879]
    poly_start = [np.array([1.18421784e+01,5.63644824e-02,-1.89766867e-02,-1.32058567e-05,-4.77323693e-05]),np.array([1.04110415e+01,3.32422851e-02,-3.37451899e-02,-1.98441704e-03,-1.70630123e-04]),np.array([1.20355620e+01,4.61446000e-02,-1.93338601e-02,7.60320584e-04,2.53691977e-05]),np.array([1.01305160e+01,1.25696460e-02,-3.92416704e-02,-3.34828851e-04,3.52869359e-05])]
    poly_stop = [np.array([1.31328718e+01,2.34156918e-02,-4.52496795e-02,7.14611033e-04,4.41093590e-04]),np.array([1.16623972e+01,6.90177048e-03,-2.39601957e-02,-4.66990093e-04,-1.54057259e-04]),np.array([1.54588619e+01,6.45523782e-02,-1.60969129e-02,1.28774254e-04,-7.24487366e-05]),np.array([1.08577750e+01,6.67598389e-02,-3.11619040e-02,-7.65761773e-04,1.44480631e-05])]


    runid_index = runids.index(runid)
    interp_dist = (filenr-start_time[runid_index])/float(stop_time[runid_index] - start_time[runid_index])
    bs_fit_array = (1.-interp_dist)*poly_start[runid_index] + interp_dist*poly_stop[runid_index]

    return bs_fit_array

def bow_shock_auto_r(runid,t):

    r0_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[12.0050199853,10.498081067,12.3991595248,10.2195045081,12.5685172417]))
    v_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[6.80178857e-03,3.74846530e-03,8.73143012e-03,4.70099989e-03,3.67880773e-03]))

    return r0_dict[runid]+v_dict[runid]*(t-290)

def bow_shock_auto_2(runid,t):

    r0_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[11.7021649174,10.3497476216,11.7493270578,9.94235078718,12.2559321215]))
    v_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[0.00780715908985,0.00455983873726,0.00972525860489,0.00553000332236,0.013098391388]))

    return r0_dict[runid]+v_dict[runid]*(t-290)

def bow_shock_r(runid,t):

    r0_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[11.7851239669,10.3130434783,11.9669421488,9.9652173913,12.4938271605]))
    v_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[0.0089345544,0.0044131524,0.0089722231,0.0054675004,0.0053351551]))

    return r0_dict[runid]+v_dict[runid]*(t-290)

def bow_shock_finder(vlsvobj,rho_sw=1.0e+6,v_sw=750e+3):
    # returns cells outside the bow shock

    # If file has separate populations, find proton population
    if vlsvobj.check_variable("rho"):
        rho = vlsvobj.read_variable("rho")
    else:
        rho = vlsvobj.read_variable("proton/rho")
        
    cellids = vlsvobj.read_variable("CellID")

    simdim = vlsvobj.get_spatial_mesh_size()

    # Create mask
    bs = np.ma.masked_less(rho,1.85*rho_sw)

    # Find IDs of masked cells
    masked_ci = np.ma.array(cellids,mask=~bs.mask).compressed()

    return masked_ci

def sw_normalisation(runid,var):
    # Normalisation values for specified var for specified runid

    sw_pars = sw_par_dict(runid)
    key_list = ["time",
    "x_mean","y_mean","z_mean",
    "A","Nr_cells",
    "r_mean","theta_mean","phi_mean",
    "size_rad","size_tan",
    "x_vmax","y_vmax","z_vmax",
    "n_avg","n_med","n_max",
    "v_avg","v_med","v_max",
    "B_avg","B_med","B_max",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "beta_avg","beta_med","beta_max",
    "x_min","rho_vmax","b_vmax",
    "pd_avg","pd_med","pd_max","pdyn_vmax",
    "duration","size_ratio"]
    
    if var not in key_list:
        return 1
    
    norm_list = [1,
    1,1,1,
    1,1,
    1,1,1,
    1,1,
    1,1,1,
    sw_pars[0]/1.0e+6,sw_pars[0]/1.0e+6,sw_pars[0]/1.0e+6,
    sw_pars[1]/1.0e+3,sw_pars[1]/1.0e+3,sw_pars[1]/1.0e+3,
    sw_pars[2]/1.0e-9,sw_pars[2]/1.0e-9,sw_pars[2]/1.0e-9,
    sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,
    sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,
    sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,sw_pars[5]/1.0e+6,
    sw_pars[4],sw_pars[4],sw_pars[4],
    1,sw_pars[0]/1.0e+6,sw_pars[4],
    sw_pars[3]/1.0e-9,sw_pars[3]/1.0e-9,sw_pars[3]/1.0e-9,sw_pars[3]/1.0e-9,
    1,1]

    return norm_list[key_list.index(var)]

def sw_par_dict(runid):
    # Returns solar wind parameters for specified run
    # Output is 0: density, 1: velocity, 2: IMF strength 3: dynamic pressure 4: plasma beta

    runs = ["ABA","ABC","AEA","AEC","BFD"]
    sw_rho = [1e+6,3.3e+6,1.0e+6,3.3e+6,1.0e+6]
    sw_v = [750e+3,600e+3,750e+3,600e+3,750e+3]
    sw_B = [5.0e-9,5.0e-9,10.0e-9,10.0e-9,5.0e-9]
    sw_T = [500e+3,500e+3,500e+3,500e+3,500e+3]
    sw_pdyn = [m_p*sw_rho[n]*(sw_v[n]**2) for n in xrange(len(runs))]
    sw_beta = [2*sc.mu_0*sw_rho[n]*sc.k*sw_T[n]/(sw_B[n]**2) for n in xrange(len(runs))]

    return [sw_rho[runs.index(runid)],sw_v[runs.index(runid)],sw_B[runs.index(runid)],sw_pdyn[runs.index(runid)],sw_beta[runs.index(runid)],sw_T[runs.index(runid)]]

def sort_jets(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1,0]):
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

    return events_culled


def get_neighbors_old(vlsvobj,c_i,neighborhood_reach=[1,1]):
    # finds the neighbors of the specified cells within the maximum offsets in neighborhood_reach

    if type(c_i) is int:
        c_i = [c_i]

    simsize = vlsvobj.get_spatial_mesh_size()

    # initialise array of neighbors
    neighbors = np.array([],dtype=int)

    # range of offsets to take into account
    x_r = xrange(-1*neighborhood_reach[0],neighborhood_reach[0]+1)
    y_r = xrange(-1*neighborhood_reach[1],neighborhood_reach[1]+1)

    for n in c_i:

        # append cellids of neighbors, cast as int, to array of neighbors
        for a in x_r:
            for b in y_r:
                if simsize[1] == 1:
                    neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,0,b],periodic=[0,0,0])))
                else:
                    neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,b,0],periodic=[0,0,0])))

    # discard invalid cellids
    neighbors = neighbors[neighbors != 0]

    # discard duplicate cellids
    neighbors = np.unique(neighbors)

    return neighbors

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


def ci2vars(vlsvobj,input_vars,cells):
    # find the values for the input variables corresponding to the masked cellids.
    # reads the specified variables from the vlsvobject

    # find the indices of the masked cells
    cellids = vlsvobj.read_variable("CellID")
    n_i = np.in1d(cellids,cells)

    # initialise list of output vars
    output_vars = []

    for input_var in input_vars:

        # find values for the masked cells
        variable = vlsvobj.read_variable(input_var)[n_i]

        # append variable values to list of output vars
        output_vars.append(variable)

    return output_vars


def ci2vars_nofile(input_vars,cellids,cells):
    # find the values for the input variables corresponding to the masked cellids.
    # uses variables that have been read from vlsvobject beforehand

    # initialise list of output vars
    output_vars = []

    # find the indices of the masked cells
    n_i = np.in1d(cellids,cells)

    for input_var in input_vars:

        # find values of the variable for the masked cells
        n_var = input_var[n_i]

        # append variable values to list of output vars
        output_vars.append(n_var)

    return output_vars

def get_cell_volume(vlsvobj):
    # returns volume or area of one cell

    # get spatial extent of simulation and the number of cells in each direction
    simextent = vlsvobj.get_spatial_mesh_extent().reshape((2,3))
    simsize = vlsvobj.get_spatial_mesh_size()

    # calculate DX,DY,DZ
    cell_sizes = (simextent[1]-simextent[0])/simsize
    
    # calculate volume or area of one cell
    if (simsize==1).any():
        dV = cell_sizes[0]*cell_sizes[1]
    else:
        dV = cell_sizes[0]*cell_sizes[1]*cell_sizes[2]

    return dV

def read_mult_vars(vlsvobj,input_vars,cells=-1):
    # reads multiple variables from vlsvobject

    # initialise list of output vars
    output_vars = []

    for input_var in input_vars:

        # read variable from vlsvobject and append it to list of output_vars
        variable = vlsvobj.read_variable(input_var,cellids=cells)
        output_vars.append(variable)

    return output_vars

def xyz_reconstruct_old(vlsvobj):
    # reconstructs coordinates based on spatial mesh parameters

    # get simulation extents and dimension sizes
    simextent = vlsvobj.get_spatial_mesh_extent()
    simsize = vlsvobj.get_spatial_mesh_size()

    # discard 3rd dimension
    simdim = simsize[simsize!=1]

    # reconstruct X
    X = np.linspace(simextent[0],simextent[3],simdim[0]+1)[:-1]
    X = np.pad(X,(0,simdim[0]*(simdim[1]-1)),"wrap")

    # reconstruct Y
    Y = np.linspace(simextent[1],simextent[4],simdim[1]+1)[:-1]
    Y = np.pad(Y,(0,simdim[1]*(simdim[0]-1)),"wrap")
    Y = np.reshape(Y,(simdim[0],simdim[1]))
    Y = Y.T
    Y = Y.flatten()

    # reconstruct Z
    Z = np.linspace(simextent[2],simextent[5],simdim[1]+1)[:-1]
    Z = np.pad(Z,(0,simdim[1]*(simdim[0]-1)),"wrap")
    Z = np.reshape(Z,(simdim[0],simdim[1]))
    Z = Z.T
    Z = Z.flatten()

    return np.array([X,Y,Z])

def xyz_reconstruct(vlsvobj,cellids=-1):

    if type(cellids) == int and cellids == -1:
        ci = vlsvobj.read_variable("CellID")
    else:
        ci = np.asarray([cellids]).flatten()

    try:
        coords = vlsvobj.get_cell_coordinates_multi(ci)
    except:
        coords = np.array([vlsvobj.get_cell_coordinates(cell) for cell in ci])
        
    coords = coords.T

    return coords

def restrict_area(vlsvobj,boxre):

    if len(boxre) == 4:
        boxre = [boxre[0],boxre[1],boxre[2],boxre[3],0,0]

    cellids = vlsvobj.read_variable("CellID")

    # If X doesn't exist, reconstruct X,Y,Z, otherwise read X,Y,Z
    if vlsvobj.check_variable("X"):
        X,Y,Z = vlsvobj.read_variable("X"),vlsvobj.read_variable("Y"),vlsvobj.read_variable("Z")
    else:
        X,Y,Z = xyz_reconstruct(vlsvobj)

    Xmin = X[np.abs(X-boxre[0]*r_e)==np.min(np.abs(X-boxre[0]*r_e))][0]
    Xmax = X[np.abs(X-boxre[1]*r_e)==np.min(np.abs(X-boxre[1]*r_e))][0]

    Ymin = Y[np.abs(Y-boxre[2]*r_e)==np.min(np.abs(Y-boxre[2]*r_e))][0]
    Ymax = Y[np.abs(Y-boxre[3]*r_e)==np.min(np.abs(Y-boxre[3]*r_e))][0]

    Zmin = Z[np.abs(Z-boxre[4]*r_e)==np.min(np.abs(Z-boxre[4]*r_e))][0]
    Zmax = Z[np.abs(Z-boxre[5]*r_e)==np.min(np.abs(Z-boxre[5]*r_e))][0]

    # X_cells = cellids[np.logical_and(X>=Xmin,X<=Xmax)]
    # Y_cells = cellids[np.logical_and(Y>=Ymin,Y<=Ymax)]
    # Z_cells = cellids[np.logical_and(Z>=Zmin,Z<=Zmax)]

    # masked_cells = np.intersect1d(X_cells,Y_cells)
    # mesked_cells = np.intersect1d(masked_cells,Z_cells)

    # return masked_cells

    # mask the cellids within the specified limits
    msk = np.ma.masked_greater_equal(X,Xmin)
    msk.mask[X > Xmax] = False
    msk.mask[Y < Ymin] = False
    msk.mask[Y > Ymax] = False
    msk.mask[Z < Zmin] = False
    msk.mask[Z > Zmax] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(cellids,mask=~msk.mask).compressed()

    return masked_ci


def restrict_area_old(vlsvobj,boxre):
    # find cellids of cells that correspond to X,Y-positions within the specified limits

    cellids = vlsvobj.read_variable("CellID")

    # If X doesn't exist, reconstruct X,Y,Z, otherwise read X,Y,Z
    if vlsvobj.check_variable("X"):
        X,Y,Z = vlsvobj.read_variable("X"),vlsvobj.read_variable("Y"),vlsvobj.read_variable("Z")
    else:
        X,Y,Z = xyz_reconstruct(vlsvobj)

    # Get the simulation size
    simsize = vlsvobj.get_spatial_mesh_size()
    
    # if polar run, equate replace Y with Z
    if simsize[1] == 1:
        Y = Z

    # mask the cellids within the specified limits
    msk = np.ma.masked_greater_equal(X,boxre[0]*r_e)
    msk.mask[X > boxre[1]*r_e] = False
    msk.mask[Y < boxre[2]*r_e] = False
    msk.mask[Y > boxre[3]*r_e] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(cellids,mask=~msk.mask).compressed()

    return masked_ci

def make_cust_mask(filenumber,runid,halftimewidth=180,boxre=[6,18,-8,6],avgfile=False):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("proton/V")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]

    npdynx = spdynx/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = xrange(filenumber-halftimewidth,filenumber+halftimewidth+1)

    missing_file_counter = 0

    for n_t in timerange:

        if avgfile:
            continue

        # exclude the main timestep
        if n_t == filenumber:
            continue

        # find correct file path for current time step
        if runid == "AED":
            tfile_name = "bulk.old."+str(n_t).zfill(7)+".vlsv"
        else:
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

        if tfile_name not in os.listdir(bulkpath):
            missing_file_counter += 1
            print("Bulk file "+str(n_t)+" not found, continuing")
            continue

        # open file for current time step
        f = pt.vlsvfile.VlsvReader(bulkpath+tfile_name)
        f.optimize_open_file()
        
        # if file has separate populations, read proton population
        if type(f.read_variable("rho")) is not np.ndarray:
            trho = f.read_variable("proton/rho")
            tv = f.read_variable("proton/V")
        else:
            trho = f.read_variable("rho")
            tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")

        f.optimize_close_file()
        
        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]
        
        tpdynavg = np.add(tpdynavg,otpdyn)

    # calculate time average of dynamic pressure
    tpdynavg /= len(timerange)-1-missing_file_counter

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    if avgfile:
        tpdynavg = np.loadtxt(wrkdir_DNR+"tavg/"+runid+"/"+str(filenumber)+"_pdyn.tavg")

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom jet mask
    jet = np.ma.masked_greater(npdynx,0.25)
    jet.mask[nrho < 3.5] = False
    jet.mask[tapdyn > 2] = True

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~jet.mask).compressed()

    if not os.path.exists(wrkdir_DNR+"working/Masks/"+runid+"/"):
        os.makedirs(wrkdir_DNR+"working/Masks/"+runid+"/")

    print("Writing to "+wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask")

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre))
        np.savetxt(wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt(wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci

def make_cust_mask_opt(filenumber,runid,halftimewidth=180,boxre=[6,18,-8,6],avgfile=False):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("proton/V")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]

    if vlsvreader.check_variable("X"):
        X,Y,Z = [vlsvreader.read_variable("X"),vlsvreader.read_variable("Y"),vlsvreader.read_variable("Z")]
    else:
        X,Y,Z = xyz_reconstruct(vlsvreader)

    X,Y,Z = [X[np.argsort(origid)]/r_e,Y[np.argsort(origid)]/r_e,Z[np.argsort(origid)]/r_e]

    p = bow_shock_markus(runid,filenumber) #PLACEHOLDER

    bs_cond = X-p[0]-p[1]*Y-p[2]*(Y**2)-p[3]*(Y**3)-p[4]*(Y**4)

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]

    npdynx = spdynx/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = xrange(filenumber-halftimewidth,filenumber+halftimewidth+1)

    missing_file_counter = 0

    vlsvobj_list = []

    for n_t in timerange:

        if avgfile:
            continue

        # exclude the main timestep
        if n_t == filenumber:
            continue

        # find correct file path for current time step
        if runid == "AED":
            tfile_name = "bulk.old."+str(n_t).zfill(7)+".vlsv"
        else:
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

        if tfile_name not in os.listdir(bulkpath):
            missing_file_counter += 1
            print("Bulk file "+str(n_t)+" not found, continuing")
            continue

        # open file for current time step
        vlsvobj_list.append(pt.vlsvfile.VlsvReader(bulkpath+tfile_name))

    for f in vlsvobj_list:

        f.optimize_open_file()
        
        # if file has separate populations, read proton population
        if type(f.read_variable("rho")) is not np.ndarray:
            trho = f.read_variable("proton/rho")
            tv = f.read_variable("proton/V")
        else:
            trho = f.read_variable("rho")
            tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")
        
        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]
        
        tpdynavg = np.add(tpdynavg,otpdyn)

        f.optimize_clear_fileindex_for_cellid()
        f.optimize_close_file()

    # calculate time average of dynamic pressure
    tpdynavg /= (len(timerange)-1-missing_file_counter)

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    if avgfile:
        tpdynavg = np.loadtxt(wrkdir_DNR+"tavg/"+runid+"/"+str(filenumber)+"_pdyn.tavg")

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom jet mask
    #jet = np.ma.masked_greater(npdynx,0.25)
    #jet.mask[nrho < 3.5] = False
    #jet.mask[tapdyn > 2] = True
    jet = np.ma.masked_greater_equal(tapdyn,2.0)
    #jet.mask[nrho < 2.0] = False
    jet.mask[bs_cond > 0] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~jet.mask).compressed()

    if not os.path.exists(wrkdir_DNR+"working/Masks/"+runid+"/"):
        os.makedirs(wrkdir_DNR+"working/Masks/"+runid+"/")

    print("Writing to "+wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask")

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre))
        np.savetxt(wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt(wrkdir_DNR+"working/Masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci