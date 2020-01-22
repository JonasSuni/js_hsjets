import numpy as np
from Merka_BS_model import BS_distance_Merka2005
from Shue_Mpause_model import Shue_Mpause_model
import scipy.constants as sc
import os
import pytools as pt

medium_blue = '#006DDB'
crimson = '#920000'
violet = '#B66DFF'
dark_blue = '#490092'
orange = '#db6d00'
m_p = 1.672621898e-27
r_e = 6.371e+6

wrkdir_DNR = os.environ["WRK"]+"/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir="/proj/vlasov"

try:
    tavgdir = os.environ["TAVG"]
except:
    tavgdir = wrkdir_DNR

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

def BS_xy():
    #theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-60.25,60,0.5))
    R_bs = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta==a)[0][0]
        R_bs[index] = BS_distance_Merka2005(np.pi/2,a,6,400,8,[])

    #x_bs = R_bs*np.cos(np.deg2rad(theta))
    #y_bs = R_bs*np.sin(np.deg2rad(theta))
    x_bs = R_bs*np.cos(theta)
    y_bs = R_bs*np.sin(theta)

    return [x_bs,y_bs]


def MP_xy():
    #theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-60.25,60,0.5))
    R_mp = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta==a)[0][0]
        R_mp[index] = Shue_Mpause_model(m_p*400e3*400e3*6e6*1.e9,0.0,[a],[0])

    #x_mp = R_mp*np.cos(np.deg2rad(theta))
    #y_mp = R_mp*np.sin(np.deg2rad(theta))
    x_mp = R_mp*np.cos(theta)
    y_mp = R_mp*np.sin(theta)

    return [x_mp,y_mp]

def bs_mp_fit(runid,file_nr,boxre=[6,18,-8,6]):

    bulkpath = find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(file_nr).zfill(7))

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    cellids = restrict_area(vlsvobj,boxre)
    rho = vlsvobj.read_variable("rho",cellids=cellids)
    X,Y,Z = xyz_reconstruct(vlsvobj,cellids)

    T_sw = 0.5e+6

    pr_rhonbs = vlsvobj.read_variable("RhoNonBackstream",cellids=cellids)
    pr_PTDNBS = vlsvobj.read_variable("PTensorNonBackstreamDiagonal",cellids=cellids)

    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    mask = (pr_TNBS>=3*T_sw)

    X_masked = X[mask]
    Y_masked = Y[mask]

    Y_unique = np.unique(Y_masked)
    Yun2 = Y_unique[np.logical_and(Y_unique>=-2*r_e,Y_unique<=2*r_e)]
    X_min = np.array([np.min(X_masked[Y_masked == y]) for y in Yun2])
    X_max = np.array([np.max(X_masked[Y_masked == y]) for y in Y_unique])

    bs_fit = np.polyfit(Y_unique/r_e,X_max/r_e,deg=5)
    mp_fit = np.polyfit(Yun2/r_e,X_min/r_e,deg=2)

    return (mp_fit,bs_fit)

def make_bs_fit(runid,start,stop):

    bs_fit_arr = np.zeros(6)
    for n in range(start,stop+1):
        mp_fit,bs_fit = bs_mp_fit(runid,n,boxre=[6,18,-8,6])
        bs_fit_arr = np.vstack((bs_fit_arr,bs_fit))

    bs_fit_arr = bs_fit_arr[1:]

    if not os.path.exists(wrkdir_DNR+"bsfit/{}".format(runid)):
        try:
            os.makedirs(wrkdir_DNR+"bsfit/{}".format(runid))
        except OSError:
            pass

    np.savetxt(wrkdir_DNR+"bsfit/{}/{}_{}".format(runid,start,stop),bs_fit_arr)

def bow_shock_jonas(runid,filenr):

    runids = ["ABA","ABC","AEA","AEC"]
    r_id = runids.index(runid)
    maxtime_list = [839,1179,1339,879]
    bs_fit_arr = np.loadtxt(wrkdir_DNR+"bsfit/{}/580_{}".format(runid,maxtime_list[r_id]))

    return bs_fit_arr[filenr-580][::-1]

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

def bs_rd_jonas(runid,time_arr,x_arr,y_arr):

    filenr_arr = (time_arr*2).astype(int)

    bs_rd_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_jonas(runid,filenr_arr[n])[::-1]
        Y = y_arr[n]
        X = x_arr[n]
        #X_bs = np.polyval(bs_fit,Y)
        rv_fit = np.polyfit([0,X],[0,Y],deg=1)
        x_range = np.arange(X-1.0,X+1.0,0.01)
        y_range = np.polyval(rv_fit,x_range)
        x_bs_range = np.polyval(bs_fit,y_range)
        x_bs,y_bs = x_range[np.argmin(np.abs(x_range-x_bs_range))],y_range[np.argmin(np.abs(x_range-x_bs_range))]
        bs_rd_arr[n] = np.sign(X-x_bs)*np.linalg.norm([x_bs-X,y_bs-Y])

    return bs_rd_arr

def bs_rd_markus(runid,time_arr,x_arr,y_arr):

    filenr_arr = (time_arr*2).astype(int)

    bs_rd_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_markus(runid,filenr_arr[n])[::-1]
        Y = y_arr[n]
        X = x_arr[n]
        #X_bs = np.polyval(bs_fit,Y)
        rv_fit = np.polyfit([0,X],[0,Y],deg=1)
        x_range = np.arange(X-1.0,X+1.0,0.01)
        y_range = np.polyval(rv_fit,x_range)
        x_bs_range = np.polyval(bs_fit,y_range)
        x_bs,y_bs = x_range[np.argmin(np.abs(x_range-x_bs_range))],y_range[np.argmin(np.abs(x_range-x_bs_range))]
        bs_rd_arr[n] = np.sign(X-x_bs)*np.linalg.norm([x_bs-X,y_bs-Y])

    return bs_rd_arr

def bs_dist_markus(runid,time_arr,x_arr,y_arr):

    filenr_arr = (time_arr*2).astype(int)

    bs_x_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_markus(runid,filenr_arr[n])[::-1]
        Y = y_arr[n]
        X_bs = np.polyval(bs_fit,Y)
        bs_x_arr[n] = X_bs

    return x_arr - bs_x_arr


def get_cell_coordinates(runid,cellid):

    spatmesh = spatmesh_get(runid)

    xmin,ymin,zmin,xmax,ymax,zmax = spatmesh[0]
    xcells,ycells,zcells = spatmesh[1]

    # Get cell lengths:
    cell_lengths = np.array([(xmax - xmin)/xcells, (ymax - ymin)/ycells, (zmax - zmin)/zcells])
    # Get cell indices:
    cellid = cellid - 1
    cellindices = np.zeros(3)
    cellindices[0] = cellid%xcells
    cellindices[1] = (cellid//xcells)%ycells
    cellindices[2] = cellid//(xcells*ycells)

    # Get cell coordinates:
    cellcoordinates = np.zeros(3)
    cellcoordinates[0] = xmin + (cellindices[0] + 0.5) * cell_lengths[0]
    cellcoordinates[1] = ymin + (cellindices[1] + 0.5) * cell_lengths[1]
    cellcoordinates[2] = zmin + (cellindices[2] + 0.5) * cell_lengths[2]
    # Return the coordinates:
    return np.array(cellcoordinates)

def spatmesh_get(runid):

    runids = ["ABA","ABC","AEA","AEC"]

    spat_extent = [np.array([-5.01191931e+07, -1.99337700e+08, -1.13907257e+05,  2.98437013e+08,  1.99337700e+08,  1.13907257e+05]),np.array([-5.01191931e+07, -1.99337700e+08, -1.13907257e+05,  4.05509835e+08,  1.99337700e+08,  1.13907257e+05]),np.array([-5.01191931e+07, -1.99337700e+08, -1.13907257e+05,  2.98437013e+08,  1.99337700e+08,  1.13907257e+05]),np.array([-5.01191931e+07, -1.99337700e+08, -1.13907257e+05,  4.05509835e+08,  1.99337700e+08,  1.13907257e+05])]
    spat_size = [np.array([1530, 1750,    1],dtype=np.uint64),np.array([2000, 1750,    1], dtype=np.uint64),np.array([1530, 1750,    1], dtype=np.uint64),np.array([2000, 1750,    1], dtype=np.uint64)]

    return (spat_extent[runids.index(runid)],spat_size[runids.index(runid)])

def find_bulkpath(runid):

    runid_list = ["ABA","ABC","AEA","AEC","BFD"]
    path_list = ["bulk/","bulk/","round_3_boundary_sw/","bulk/","bulk/"]

    vlpath = "{}/2D/{}/".format(vlasdir,runid)

    if runid in runid_list:
        bulkpath = vlpath+path_list[runid_list.index(runid)]
    else:
        bulkpath = vlpath+"bulk/"

    return bulkpath

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
