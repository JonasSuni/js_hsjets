import numpy as np
from Merka_BS_model import BS_distance_Merka2005
from Shue_Mpause_model import Shue_Mpause_model
import scipy.constants as sc

medium_blue = '#006DDB'
crimson = '#920000'
violet = '#B66DFF'
dark_blue = '#490092'
orange = '#db6d00'
m_p = 1.672621898e-27
r_e = 6.371e+6

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

def bs_dist_markus(runid,time_arr,x_arr,y_arr):

    filenr_arr = (time_arr*2).astype(int)

    bs_x_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_markus(runid,filenr_arr[n])[::-1]
        Y = y_arr[n]
        X_bs = np.polyval(bs_fit,Y)
        bs_x_arr[n] = X_bs

    return x_arr - bs_x_arr

def find_bulkpath(runid):

    runid_list = ["ABA","ABC","AEA","AEC","BFD"]
    path_list = ["bulk/","bulk/","round_3_boundary_sw/","","bulk/"]

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
