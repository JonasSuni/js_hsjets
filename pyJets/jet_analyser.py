import pytools as pt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.constants as sc
import scipy.ndimage
import scipy.optimize as so
import time

m_p = 1.672621898e-27
r_e = 6.371e+6

#wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"
wrkdir_DNR = os.environ["WRK"]+"/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir="/proj/vlasov"

try:
    tavgdir = os.environ["TAVG"]
except:
    tavgdir = wrkdir_DNR

def find_bulkpath(runid):

    runid_list = ["ABA","ABC","AEA","AEC","BFD"]
    path_list = ["bulk/","bulk/","round_3_boundary_sw/","bulk/","bulk/"]

    vlpath = "{}/2D/{}/".format(vlasdir,runid)

    if runid in runid_list:
        bulkpath = vlpath+path_list[runid_list.index(runid)]
    else:
        bulkpath = vlpath+"bulk/"

    return bulkpath

def bs_mp_fit(runid,file_nr,boxre=[6,18,-8,6]):

    rho_sw = sw_par_dict(runid)[0]

    bulkpath = find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(file_nr).zfill(7))

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    cellids = restrict_area(vlsvobj,boxre)
    rho = vlsvobj.read_variable("rho",cellids=cellids)
    X,Y,Z = xyz_reconstruct(vlsvobj,cellids)

    mask = (rho>=2*rho_sw)

    X_masked = X[mask]
    Y_masked = Y[mask]

    Y_unique = np.unique(Y_masked)
    Yun2 = Y_unique[np.logical_and(Y_unique>=-2*r_e,Y_unique<=2*r_e)]
    X_min = np.array([np.min(X_masked[Y_masked == y]) for y in Yun2])
    X_max = np.array([np.max(X_masked[Y_masked == y]) for y in Y_unique])

    bs_fit = np.polyfit(Y_unique/r_e,X_max/r_e,deg=5)
    mp_fit = np.polyfit(Yun2/r_e,X_min/r_e,deg=2)

    return (mp_fit,bs_fit)

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
    "duration","size_ratio",
    "DT","Dn","Dv","Dpd","DB",
    "DTPar","DTPerp"]

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
    1,1,
    sw_pars[5]/1.0e+6,sw_pars[0]/1.0e+6,sw_pars[1]/1.0e+3,sw_pars[3]/1.0e-9,sw_pars[2]/1.0e-9,
    1,1]

    return norm_list[key_list.index(var)]

def spat_res(runid):

    runs = ["ABA","ABC","AEA","AEC","BFD"]
    res = [227000/r_e,227000/r_e,227000/r_e,227000/r_e,300000/r_e]

    return res[runs.index(runid)]

def sw_par_dict(runid):
    # Returns solar wind parameters for specified run
    # Output is 0: density, 1: velocity, 2: IMF strength 3: dynamic pressure 4: plasma beta

    runs = ["ABA","ABC","AEA","AEC","BFD"]
    sw_rho = [1e+6,3.3e+6,1.0e+6,3.3e+6,1.0e+6]
    sw_v = [750e+3,600e+3,750e+3,600e+3,750e+3]
    sw_B = [5.0e-9,5.0e-9,10.0e-9,10.0e-9,5.0e-9]
    sw_T = [500e+3,500e+3,500e+3,500e+3,500e+3]
    sw_pdyn = [m_p*sw_rho[n]*(sw_v[n]**2) for n in range(len(runs))]
    sw_beta = [2*sc.mu_0*sw_rho[n]*sc.k*sw_T[n]/(sw_B[n]**2) for n in range(len(runs))]

    return [sw_rho[runs.index(runid)],sw_v[runs.index(runid)],sw_B[runs.index(runid)],sw_pdyn[runs.index(runid)],sw_beta[runs.index(runid)],sw_T[runs.index(runid)]]

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

def mask_maker(runid,filenr,boxre=[6,18,-8,6],avgfile=True):

    bulkpath = find_bulkpath(runid)
    bulkname = "bulk."+str(filenr).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenr)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]
    proton_bool = False

    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        pdyn = vlsvreader.read_variable("proton/Pdyn")[np.argsort(origid)]
        B = vlsvreader.read_variable("B")[np.argsort(origid)]
        pr_rhonbs = vlsvreader.read_variable("RhoNonBackstream")[np.argsort(origid)]
        pr_PTDNBS = vlsvreader.read_variable("PTensorNonBackstreamDiagonal")[np.argsort(origid)]
        proton_bool = True
    else:
        pdyn = vlsvreader.read_variable("Pdyn")[np.argsort(origid)]
        B = vlsvreader.read_variable("B")[np.argsort(origid)]
        pr_rhonbs = vlsvreader.read_variable("RhoNonBackstream")[np.argsort(origid)]
        pr_PTDNBS = vlsvreader.read_variable("PTensorNonBackstreamDiagonal")[np.argsort(origid)]

    T_sw = 0.5e+6
    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    Bmag = np.linalg.norm(B,axis=-1)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]
    B_sw = sw_pars[2]

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = range(filenr-180,filenr+180+1)

    missing_file_counter = 0

    vlsvobj_list = []

    if avgfile:
        tpdynavg = np.load(tavgdir+"/"+runid+"/"+str(filenr)+"_pdyn.npy")
    else:

        for n_t in timerange:

            # exclude the main timestep
            if n_t == filenumber:
                continue

            # find correct file path for current time step
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
            if proton_bool:
                tpdyn = f.read_variable("proton/Pdyn")
            else:
                tpdyn = f.read_variable("Pdyn")

            # read cellids for current time step
            cellids = f.read_variable("CellID")

            # sort dynamic pressures
            otpdyn = tpdyn[cellids.argsort()]

            tpdynavg = np.add(tpdynavg,otpdyn)

            f.optimize_clear_fileindex_for_cellid()
            f.optimize_close_file()

        # calculate time average of dynamic pressure
        tpdynavg /= (len(timerange)-1-missing_file_counter)

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    slams = np.ma.masked_greater_equal(Bmag,1.4*B_sw)
    slams.mask[pr_TNBS>3.0*T_sw] = False
    slams.mask[pdyn<1.2*pd_sw] = False
    jet = np.ma.masked_greater_equal(pdyn,2.0*tpdynavg)
    jet.mask[pr_TNBS<3.0*T_sw] = False
    slamsjet = np.logical_or(slams,jet)

    jet_ci = np.ma.array(sorigid,mask=~jet.mask).compressed()
    slams_ci = np.ma.array(sorigid,mask=~slams.mask).compressed()
    slamsjet_ci = np.ma.array(sorigid,mask=~slamsjet.mask).compressed()

    restr_ci = restrict_area(vlsvreader,boxre)

    if not os.path.exists(wrkdir_DNR+"working/jets/Masks/"+runid):
        os.makedirs(wrkdir_DNR+"working/jets/Masks/"+runid)
        os.makedirs(wrkdir_DNR+"working/SLAMS/Masks/"+runid)
        os.makedirs(wrkdir_DNR+"working/SLAMSJETS/Masks/"+runid)

    np.savetxt(wrkdir_DNR+"working/jets/Masks/"+runid+"/{}.mask".format(str(filenr)),np.intersect1d(jet_ci,restr_ci))
    np.savetxt(wrkdir_DNR+"working/SLAMS/Masks/"+runid+"/{}.mask".format(str(filenr)),np.intersect1d(slams_ci,restr_ci))
    np.savetxt(wrkdir_DNR+"working/SLAMSJETS/Masks/"+runid+"/{}.mask".format(str(filenr)),np.intersect1d(slamsjet_ci,restr_ci))

    return (np.intersect1d(jet_ci,restr_ci),np.intersect1d(slams_ci,restr_ci),np.intersect1d(slamsjet_ci,restr_ci))

def make_cust_mask_opt(filenumber,runid,halftimewidth=180,boxre=[6,18,-8,6],avgfile=False,transient="jet"):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    if transient == "jet":
        trans_folder = "jets/"
    elif transient == "slams":
        trans_folder = "SLAMS/"
    elif transient == "slamsjet":
        trans_folder = "SLAMSJETS/"

    bulkpath = find_bulkpath(runid)

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
        B = vlsvreader.read_variable("B")[np.argsort(origid)]
        pr_rhonbs = vlsvreader.read_variable("RhoNonBackstream")[np.argsort(origid)]
        pr_PTDNBS = vlsvreader.read_variable("PTensorNonBackstreamDiagonal")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]
        B = vlsvreader.read_variable("B")[np.argsort(origid)]
        pr_rhonbs = vlsvreader.read_variable("RhoNonBackstream")[np.argsort(origid)]
        pr_PTDNBS = vlsvreader.read_variable("PTensorNonBackstreamDiagonal")[np.argsort(origid)]

    if vlsvreader.check_variable("X"):
        X,Y,Z = [vlsvreader.read_variable("X"),vlsvreader.read_variable("Y"),vlsvreader.read_variable("Z")]
    else:
        X,Y,Z = xyz_reconstruct(vlsvreader)

    X,Y,Z = [X[np.argsort(origid)]/r_e,Y[np.argsort(origid)]/r_e,Z[np.argsort(origid)]/r_e]

    T_sw = 0.5e+6
    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    mask = (pr_TNBS>=3*T_sw)

    X_masked = X[mask]
    Y_masked = Y[mask]

    Y_unique = np.unique(Y_masked)
    X_max = np.array([np.max(X_masked[Y_masked == y]) for y in Y_unique])
    bs_fit = np.polyfit(Y_unique,X_max,deg=5)

    p = bs_fit[::-1]

    #p = bow_shock_markus(runid,filenumber) #PLACEHOLDER

    x_res = spat_res(runid)

    bs_cond = X-p[0]-p[1]*Y-p[2]*(Y**2)-p[3]*(Y**3)-p[4]*(Y**4)

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    Bmag = np.linalg.norm(B,axis=-1)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]
    B_sw = sw_pars[2]

    npdynx = spdynx/pdyn_sw
    npdyn = pdyn/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = range(filenumber-halftimewidth,filenumber+halftimewidth+1)

    missing_file_counter = 0

    vlsvobj_list = []

    if avgfile:
        tpdynavg = np.load(tavgdir+"/"+runid+"/"+str(filenumber)+"_pdyn.npy")
    else:

        for n_t in timerange:

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

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom jet mask
    if transient == "jet":
        jet = np.ma.masked_greater_equal(tapdyn,2.0)
        jet.mask[bs_cond-2*x_res-0.5 > 0] = False
    elif transient == "slams":
        jet = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet.mask[nrho < 1.5] = False
        jet.mask[npdyn < 1.25] = False
    elif transient == "slamsjet":
        jet1 = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet1.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet1.mask[nrho < 1.5] = False
        jet1.mask[npdyn < 1.25] = False
        jet2 = np.ma.masked_greater_equal(tapdyn,2.0)
        jet2.mask[bs_cond-2*x_res-0.5 > 0] = False
        jet = np.logical_or(jet1,jet2)

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~jet.mask).compressed()

    if not os.path.exists("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)):
        os.makedirs("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid))
    if transient == "slamsjet":
        if not os.path.exists("{}working/{}Masks/{}/".format(wrkdir_DNR,"jets/",runid)):
            os.makedirs("{}working/{}Masks/{}/".format(wrkdir_DNR,"jets/",runid))
        if not os.path.exists("{}working/{}Masks/{}/".format(wrkdir_DNR,"SLAMS/",runid)):
            os.makedirs("{}working/{}Masks/{}/".format(wrkdir_DNR,"SLAMS/",runid))
        masked_ci_jet = np.ma.array(sorigid,mask=~jet2.mask).compressed()
        masked_ci_slams = np.ma.array(sorigid,mask=~jet1.mask).compressed()


    print("Writing to "+"{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask")

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre))
        np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask",masked_ci)
        #print(masked_ci[69])
        if transient == "slamsjet":
            masked_ci_jet = np.intersect1d(masked_ci_jet,restrict_area(vlsvreader,boxre))
            masked_ci_slams = np.intersect1d(masked_ci_slams,restrict_area(vlsvreader,boxre))
            np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,"jets/",runid)+str(filenumber)+".mask",masked_ci_jet)
            print("Writing to "+"{}working/{}Masks/{}/".format(wrkdir_DNR,"jets/",runid)+str(filenumber)+".mask")
            np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,"SLAMS/",runid)+str(filenumber)+".mask",masked_ci_slams)
            print("Writing to "+"{}working/{}Masks/{}/".format(wrkdir_DNR,"SLAMS/",runid)+str(filenumber)+".mask")
        return masked_ci
    else:
        np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask",masked_ci)
        #print(masked_ci[69])
        return masked_ci

def make_cust_mask_opt_new(filenumber,runid,halftimewidth=180,boxre=[6,18,-8,6],avgfile=False,transient="jet"):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    if transient == "jet":
        trans_folder = "jets/"
    elif transient == "slams":
        trans_folder = "SLAMS/"
    elif transient == "slamsjet":
        trans_folder = "SLAMSJETS/"

    bulkpath = find_bulkpath(runid)

    bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    if not not boxre:
        cells = restrict_area(vlsvreader,boxre)
        cells.sort()
    else:
        cells = -1

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho",cellids=cells)
        v = vlsvreader.read_variable("proton/V",cellids=cells)
        B = vlsvreader.read_variable("B",cellids=cells)
        tvars = ["proton/rho","proton/V"]
    else:
        rho = vlsvreader.read_variable("rho",cellids=cells)
        v = vlsvreader.read_variable("v",cellids=cells)
        B = vlsvreader.read_variable("B",cellids=cells)
        tvars = ["rho","v"]

    if vlsvreader.check_variable("X"):
        X,Y,Z = [vlsvreader.read_variable("X",cellids=cells),vlsvreader.read_variable("Y",cellids=cells),vlsvreader.read_variable("Z",cellids=cells)]
    else:
        X,Y,Z = xyz_reconstruct(vlsvreader,cellids=cells)

    X,Y,Z = [X/r_e,Y/r_e,Z/r_e]

    x_res = spat_res(runid)

    p = bow_shock_markus(runid,filenumber) #PLACEHOLDER

    bs_cond = X-np.polyval(p[::-1],Y)

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    Bmag = np.linalg.norm(B,axis=-1)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]
    B_sw = sw_pars[2]

    npdyn = pdyn/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = range(filenumber-halftimewidth,filenumber+halftimewidth+1)

    missing_file_counter = 0

    vlsvobj_list = []

    if avgfile:
        tpdynavg = np.load(tavgdir+"/"+runid+"/"+str(filenumber)+"_pdyn.npy")
        tpdynavg = tpdynavg[cells-1]
    else:

        for n_t in timerange:

            # exclude the main timestep
            if n_t == filenumber:
                continue

            # find correct file path for current time step
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

            if tfile_name not in os.listdir(bulkpath):
                print("Bulk file "+str(n_t)+" not found, continuing")
                continue

            # open file for current time step
            vlsvobj_list.append(pt.vlsvfile.VlsvReader(bulkpath+tfile_name))

        for f in vlsvobj_list:

            f.optimize_open_file()

            trho = f.read_variable(tvars[0],cellids = cells)
            tv = f.read_variable(tvars[1],cellids = cells)

            # dynamic pressure for current time step
            tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

            tpdynavg = np.add(tpdynavg,tpdyn)

            f.optimize_clear_fileindex_for_cellid()
            f.optimize_close_file()

        # calculate time average of dynamic pressure
        tpdynavg /= len(vlsvobj_list)

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom jet mask
    if transient == "jet":
        jet = np.ma.masked_greater_equal(tapdyn,2.0)
        jet.mask[bs_cond-2*x_res-0.5 > 0] = False
    elif transient == "slams":
        jet = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet.mask[nrho < 1.5] = False
        jet.mask[npdyn < 1.25] = False
    elif transient == "slamsjet":
        jet1 = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet1.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet1.mask[nrho < 1.5] = False
        jet1.mask[npdyn < 1.25] = False
        jet2 = np.ma.masked_greater_equal(tapdyn,2.0)
        jet2.mask[bs_cond-2*x_res-0.5 > 0] = False
        jet = np.logical_or(jet1,jet2)

    # discard unmasked cellids
    masked_ci = np.ma.array(cells,mask=~jet.mask).compressed()

    if not os.path.exists("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)):
        os.makedirs("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid))

    print("Writing to "+"{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask")
    #print(masked_ci[69])

    np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask",masked_ci)
    return masked_ci
