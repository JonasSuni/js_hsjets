import numpy as np
from Merka_BS_model import BS_distance_Merka2005
from Shue_Mpause_model import Shue_Mpause_model
import scipy.constants as sc
import os
import pytools as pt
import time

medium_blue = "#006DDB"
crimson = "#920000"
violet = "#B66DFF"
dark_blue = "#490092"
orange = "#db6d00"
green = "#24ff24"
m_p = 1.672621898e-27
r_e = 6.371e6

wrkdir_DNR = os.environ["WRK"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"

try:
    tavgdir = os.environ["TAVG"]
except:
    tavgdir = wrkdir_DNR

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#dede00",
]


def ext_magpause(ax, XmeshXY, YmeshXY, pass_maps):
    # if reqvars:
    #    return ["3d"]

    vlsvobj = vlsvobj_g
    fileindex = fileindex_g

    xarr = np.arange(11.0 * r_e, 7.0 * r_e, -1e6)
    slice = np.arange(xarr.size, dtype=float)
    theta = np.arange(0, 2 * np.pi + 0.001, np.pi / 30)
    Slice, Theta = np.meshgrid(slice, theta)
    psi = np.zeros_like(Slice[:-1, :-1], dtype=float)

    for irx, x in enumerate(xarr[:-1]):
        coords = np.array([x, 0, 0], dtype=float)
        for itx, t in enumerate(theta[:-1]):
            for r in np.arange(0, 10 * r_e, 1e6):
                y = np.cos(t) * r
                z = np.sin(t) * r
                coords[1] = y
                coords[2] = z
                cellid = vlsvobj.get_cellid(coords)
                bz = vlsvobj.read_variable("vg_b_vol", operator="z", cellids=cellid)
                if bz <= 0:
                    vz = vlsvobj.read_variable("vg_v", operator="z", cellids=cellid)
                    psi[irx, itx] = vz
                    break

    fig, ax = plt.subplots(1, 1)
    Xmesh = Slice * np.cos(Theta)
    Ymesh = Slice * np.sin(Theta)

    ax.pcolormesh(Xmesh, Ymesh, psi)
    fig.savefig("/wrk/users/jesuni/jh21_mov/magpause/magplot_{}".format(fileindex))
    plt.close(fig)


def legend_compact(leg):

    for n, item in enumerate(leg.legendHandles):
        try:
            color = item.get_color()
        except:
            color = item.get_ec()
        item.set_visible(False)
        leg.texts[n].set_color(color)


def transfer_tavg(runid, start, stop):

    time_s = time.time()

    inputdir = "/scratch/project_2000203/sunijona/tavg/{}/".format(runid)
    outputdir = "/scratch/project_2000506/sunijona/tavg/{}/".format(runid)

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for n in range(start, stop + 1):
        print("Filenr: {}, Time: {}".format(n, time.time() - time_s))

        rho_arr = np.loadtxt(inputdir + "{}_rho.tavg".format(n))
        pdyn_arr = np.loadtxt(inputdir + "{}_pdyn.tavg".format(n))

        np.save(outputdir + "{}_rho.npy".format(n), rho_arr)
        np.save(outputdir + "{}_pdyn.npy".format(n), pdyn_arr)

    return None


def xyz_reconstruct(vlsvobj, cellids=-1):

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


def restrict_area(vlsvobj, boxre):

    if len(boxre) == 4:
        boxre = [boxre[0], boxre[1], boxre[2], boxre[3], 0, 0]

    cellids = vlsvobj.read_variable("CellID")

    # If X doesn't exist, reconstruct X,Y,Z, otherwise read X,Y,Z
    if vlsvobj.check_variable("X"):
        X, Y, Z = (
            vlsvobj.read_variable("X"),
            vlsvobj.read_variable("Y"),
            vlsvobj.read_variable("Z"),
        )
    else:
        X, Y, Z = xyz_reconstruct(vlsvobj)

    Xmin = X[np.abs(X - boxre[0] * r_e) == np.min(np.abs(X - boxre[0] * r_e))][0]
    Xmax = X[np.abs(X - boxre[1] * r_e) == np.min(np.abs(X - boxre[1] * r_e))][0]

    Ymin = Y[np.abs(Y - boxre[2] * r_e) == np.min(np.abs(Y - boxre[2] * r_e))][0]
    Ymax = Y[np.abs(Y - boxre[3] * r_e) == np.min(np.abs(Y - boxre[3] * r_e))][0]

    Zmin = Z[np.abs(Z - boxre[4] * r_e) == np.min(np.abs(Z - boxre[4] * r_e))][0]
    Zmax = Z[np.abs(Z - boxre[5] * r_e) == np.min(np.abs(Z - boxre[5] * r_e))][0]

    # X_cells = cellids[np.logical_and(X>=Xmin,X<=Xmax)]
    # Y_cells = cellids[np.logical_and(Y>=Ymin,Y<=Ymax)]
    # Z_cells = cellids[np.logical_and(Z>=Zmin,Z<=Zmax)]

    # masked_cells = np.intersect1d(X_cells,Y_cells)
    # mesked_cells = np.intersect1d(masked_cells,Z_cells)

    # return masked_cells

    # mask the cellids within the specified limits
    msk = np.ma.masked_greater_equal(X, Xmin)
    msk.mask[X > Xmax] = False
    msk.mask[Y < Ymin] = False
    msk.mask[Y > Ymax] = False
    msk.mask[Z < Zmin] = False
    msk.mask[Z > Zmax] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(cellids, mask=~msk.mask).compressed()

    return masked_ci


def BS_xy():
    # theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-60.25, 60, 0.5))
    R_bs = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta == a)[0][0]
        R_bs[index] = BS_distance_Merka2005(np.pi / 2, a, 6, 400, 8, [])

    # x_bs = R_bs*np.cos(np.deg2rad(theta))
    # y_bs = R_bs*np.sin(np.deg2rad(theta))
    x_bs = R_bs * np.cos(theta)
    y_bs = R_bs * np.sin(theta)

    return [x_bs, y_bs]


def MP_xy():
    # theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-60.25, 60, 0.5))
    R_mp = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta == a)[0][0]
        R_mp[index] = Shue_Mpause_model(
            m_p * 400e3 * 400e3 * 6e6 * 1.0e9, 0.0, [a], [0]
        )

    # x_mp = R_mp*np.cos(np.deg2rad(theta))
    # y_mp = R_mp*np.sin(np.deg2rad(theta))
    x_mp = R_mp * np.cos(theta)
    y_mp = R_mp * np.sin(theta)

    return [x_mp, y_mp]


def bs_norm(runid="ABC", filenr=825, vlsvobj=None, boxre=[6, 18, -8, 8]):

    if vlsvobj is None:
        bulkpath = find_bulkpath(runid)
        bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

    cellids = restrict_area(vlsvobj, boxre)
    core_heating = vlsvobj.read_variable("core_heating", cellids=cellids)
    X, Y, Z = xyz_reconstruct(vlsvobj, cellids)
    T_sw = 0.5e6

    mask_ch = core_heating >= 3 * T_sw

    X_ch = X[mask_ch]
    Y_ch = Y[mask_ch]
    Y_un_ch = np.unique(Y_ch)
    X_max_ch = np.array([np.max(X_ch[Y_ch == y]) for y in Y_un_ch])

    Xnew = X_max_ch / r_e
    Ynew = Y_un_ch / r_e

    Xnew = Xnew[np.argsort(Ynew)]
    Ynew.sort()

    dx = np.gradient(Xnew)
    dy = np.gradient(Ynew)

    shock_vector = np.array([dx, dy, np.zeros_like(dx)])
    zvec = np.array([np.zeros_like(dx), np.zeros_like(dx), np.ones_like(dx)])
    cross_vec = np.cross(shock_vector.T, zvec.T)
    norm_vec = np.array([v / np.linalg.norm(v) for v in cross_vec])

    nx = norm_vec.T[0]
    ny = norm_vec.T[1]

    return np.array([Xnew, Ynew, nx, ny]).T


def bs_nonloc(vlsvobj, rho_sw, boxre=[6, 18, -8, 8]):

    cellids = restrict_area(vlsvobj, boxre)
    rho = vlsvobj.read_variable("rho", cellids=cellids)
    pr_rhonbs = vlsvobj.read_variable("RhoNonBackstream", cellids=cellids)
    pr_PTDNBS = vlsvobj.read_variable("PTensorNonBackstreamDiagonal", cellids=cellids)
    mms = vlsvobj.read_variable("Mms", cellids=cellids)
    X, Y, Z = xyz_reconstruct(vlsvobj, cellids)

    T_sw = 0.5e6
    epsilon = 1.0e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0 / 3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs / ((pr_rhonbs + epsilon) * kb)

    mask_ch = pr_TNBS >= 3 * T_sw
    mask_rho = rho >= 2 * rho_sw
    mask_mms = mms <= 1

    X_ch = X[mask_ch]
    Y_ch = Y[mask_ch]
    Y_un_ch = np.unique(Y_ch)
    X_max_ch = np.array([np.max(X_ch[Y_ch == y]) for y in Y_un_ch])
    bs_ch = np.polyfit(Y_un_ch / r_e, X_max_ch / r_e, deg=5)

    X_rho = X[mask_rho]
    Y_rho = Y[mask_rho]
    Y_un_rho = np.unique(Y_rho)
    X_max_rho = np.array([np.max(X_rho[Y_rho == y]) for y in Y_un_rho])
    bs_rho = np.polyfit(Y_un_rho / r_e, X_max_rho / r_e, deg=5)

    X_mms = X[mask_mms]
    Y_mms = Y[mask_mms]
    Y_un_mms = np.unique(Y_mms)
    X_max_mms = np.array([np.max(X_mms[Y_mms == y]) for y in Y_un_mms])
    bs_mms = np.polyfit(Y_un_mms / r_e, X_max_mms / r_e, deg=5)

    return (bs_ch, bs_rho, bs_mms)


def bs_mp_fit(runid, file_nr, boxre=[6, 18, -8, 6]):

    bulkpath = find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(file_nr).zfill(7))

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

    cellids = restrict_area(vlsvobj, boxre)
    rho = vlsvobj.read_variable("rho", cellids=cellids)
    X, Y, Z = xyz_reconstruct(vlsvobj, cellids)

    T_sw = 0.5e6

    pr_rhonbs = vlsvobj.read_variable("RhoNonBackstream", cellids=cellids)
    pr_PTDNBS = vlsvobj.read_variable("PTensorNonBackstreamDiagonal", cellids=cellids)

    epsilon = 1.0e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0 / 3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs / ((pr_rhonbs + epsilon) * kb)

    mask = pr_TNBS >= 3 * T_sw

    X_masked = X[mask]
    Y_masked = Y[mask]

    Y_unique = np.unique(Y_masked)
    Yun2 = Y_unique[np.logical_and(Y_unique >= -4 * r_e, Y_unique <= 4 * r_e)]
    X_min = np.array([np.min(X_masked[Y_masked == y]) for y in Yun2])
    X_max = np.array([np.max(X_masked[Y_masked == y]) for y in Y_unique])

    bs_fit = np.polyfit(Y_unique / r_e, X_max / r_e, deg=5)
    mp_fit = np.polyfit(Yun2 / r_e, X_min / r_e, deg=2)

    return (mp_fit, bs_fit)


def make_bs_fit(runid, start, stop):

    bs_fit_arr = np.zeros(6)
    mp_fit_arr = np.zeros(3)
    for n in range(start, stop + 1):
        mp_fit, bs_fit = bs_mp_fit(runid, n, boxre=[6, 18, -8, 6])
        bs_fit_arr = np.vstack((bs_fit_arr, bs_fit))
        mp_fit_arr = np.vstack((mp_fit_arr, mp_fit))

    bs_fit_arr = bs_fit_arr[1:]
    mp_fit_arr = mp_fit_arr[1:]

    if not os.path.exists(wrkdir_DNR + "bsfit/{}".format(runid)):
        try:
            os.makedirs(wrkdir_DNR + "bsfit/{}".format(runid))
        except OSError:
            pass

    if not os.path.exists(wrkdir_DNR + "mpfit/{}".format(runid)):
        try:
            os.makedirs(wrkdir_DNR + "mpfit/{}".format(runid))
        except OSError:
            pass

    np.savetxt(wrkdir_DNR + "bsfit/{}/{}_{}".format(runid, start, stop), bs_fit_arr)
    np.savetxt(wrkdir_DNR + "mpfit/{}/{}_{}".format(runid, start, stop), mp_fit_arr)


def bow_shock_jonas(runid, filenr):

    runids = ["ABA", "ABC", "AEA", "AEC"]
    r_id = runids.index(runid)
    maxtime_list = [839, 1179, 1339, 879]
    try:
        bs_fit_arr = np.loadtxt(
            wrkdir_DNR + "bsfit/{}/580_{}".format(runid, maxtime_list[r_id])
        )

        return bs_fit_arr[filenr - 580][::-1]
    except:
        mp_fit, bs_fit = bs_mp_fit(runid, filenr)

        return bs_fit[::-1]


def mag_pause_jonas(runid, filenr):

    runids = ["ABA", "ABC", "AEA", "AEC"]
    r_id = runids.index(runid)
    maxtime_list = [839, 1179, 1339, 879]
    mp_fit_arr = np.loadtxt(
        wrkdir_DNR + "mpfit/{}/580_{}".format(runid, maxtime_list[r_id])
    )

    return mp_fit_arr[filenr - 580][::-1]


def bow_shock_markus(runid, filenr):

    runids = ["ABA", "ABC", "AEA", "AEC"]
    start_time = [580, 580, 580, 580]
    stop_time = [839, 1179, 1339, 879]
    poly_start = [
        np.array(
            [
                1.18421784e01,
                5.63644824e-02,
                -1.89766867e-02,
                -1.32058567e-05,
                -4.77323693e-05,
            ]
        ),
        np.array(
            [
                1.04110415e01,
                3.32422851e-02,
                -3.37451899e-02,
                -1.98441704e-03,
                -1.70630123e-04,
            ]
        ),
        np.array(
            [
                1.20355620e01,
                4.61446000e-02,
                -1.93338601e-02,
                7.60320584e-04,
                2.53691977e-05,
            ]
        ),
        np.array(
            [
                1.01305160e01,
                1.25696460e-02,
                -3.92416704e-02,
                -3.34828851e-04,
                3.52869359e-05,
            ]
        ),
    ]
    poly_stop = [
        np.array(
            [
                1.31328718e01,
                2.34156918e-02,
                -4.52496795e-02,
                7.14611033e-04,
                4.41093590e-04,
            ]
        ),
        np.array(
            [
                1.16623972e01,
                6.90177048e-03,
                -2.39601957e-02,
                -4.66990093e-04,
                -1.54057259e-04,
            ]
        ),
        np.array(
            [
                1.54588619e01,
                6.45523782e-02,
                -1.60969129e-02,
                1.28774254e-04,
                -7.24487366e-05,
            ]
        ),
        np.array(
            [
                1.08577750e01,
                6.67598389e-02,
                -3.11619040e-02,
                -7.65761773e-04,
                1.44480631e-05,
            ]
        ),
    ]

    runid_index = runids.index(runid)
    interp_dist = (filenr - start_time[runid_index]) / float(
        stop_time[runid_index] - start_time[runid_index]
    )
    bs_fit_array = (1.0 - interp_dist) * poly_start[
        runid_index
    ] + interp_dist * poly_stop[runid_index]

    return bs_fit_array


def bs_rd(runid, time_arr, x_arr, y_arr):

    filenr_arr = (time_arr * 2).astype(int)

    bs_rd_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_jonas(runid, filenr_arr[n])[::-1]
        Y = y_arr[n]
        X = x_arr[n]
        # X_bs = np.polyval(bs_fit,Y)
        rv_fit = np.polyfit([0, X], [0, Y], deg=1)
        x_range = np.arange(X - 1.0, X + 1.0, 0.01)
        y_range = np.polyval(rv_fit, x_range)
        x_bs_range = np.polyval(bs_fit, y_range)
        x_bs, y_bs = (
            x_range[np.argmin(np.abs(x_range - x_bs_range))],
            y_range[np.argmin(np.abs(x_range - x_bs_range))],
        )
        bs_rd_arr[n] = np.sign(X - x_bs) * np.linalg.norm([x_bs - X, y_bs - Y])

    return bs_rd_arr


def bs_dist(runid, time_arr, x_arr, y_arr):

    filenr_arr = (time_arr * 2).astype(int)

    bs_x_arr = np.zeros_like(time_arr)

    for n in range(filenr_arr.size):
        bs_fit = bow_shock_jonas(runid, filenr_arr[n])[::-1]
        Y = y_arr[n]
        X_bs = np.polyval(bs_fit, Y)
        bs_x_arr[n] = X_bs

    return x_arr - bs_x_arr


def get_cell_coordinates(runid, cellid):

    spatmesh = spatmesh_get(runid)

    xmin, ymin, zmin, xmax, ymax, zmax = spatmesh[0]
    xcells, ycells, zcells = spatmesh[1]

    # Get cell lengths:
    cell_lengths = np.array(
        [(xmax - xmin) / xcells, (ymax - ymin) / ycells, (zmax - zmin) / zcells]
    )
    # Get cell indices:
    cellid = cellid - 1
    cellindices = np.zeros(3)
    cellindices[0] = cellid % xcells
    cellindices[1] = (cellid // xcells) % ycells
    cellindices[2] = cellid // (xcells * ycells)

    # Get cell coordinates:
    cellcoordinates = np.zeros(3)
    cellcoordinates[0] = xmin + (cellindices[0] + 0.5) * cell_lengths[0]
    cellcoordinates[1] = ymin + (cellindices[1] + 0.5) * cell_lengths[1]
    cellcoordinates[2] = zmin + (cellindices[2] + 0.5) * cell_lengths[2]
    # Return the coordinates:
    return np.array(cellcoordinates)


def spatmesh_get(runid):

    runids = ["ABA", "ABC", "AEA", "AEC"]

    spat_extent = [
        np.array(
            [
                -5.01191931e07,
                -1.99337700e08,
                -1.13907257e05,
                2.98437013e08,
                1.99337700e08,
                1.13907257e05,
            ]
        ),
        np.array(
            [
                -5.01191931e07,
                -1.99337700e08,
                -1.13907257e05,
                4.05509835e08,
                1.99337700e08,
                1.13907257e05,
            ]
        ),
        np.array(
            [
                -5.01191931e07,
                -1.99337700e08,
                -1.13907257e05,
                2.98437013e08,
                1.99337700e08,
                1.13907257e05,
            ]
        ),
        np.array(
            [
                -5.01191931e07,
                -1.99337700e08,
                -1.13907257e05,
                4.05509835e08,
                1.99337700e08,
                1.13907257e05,
            ]
        ),
    ]
    spat_size = [
        np.array([1530, 1750, 1], dtype=np.uint64),
        np.array([2000, 1750, 1], dtype=np.uint64),
        np.array([1530, 1750, 1], dtype=np.uint64),
        np.array([2000, 1750, 1], dtype=np.uint64),
    ]

    return (spat_extent[runids.index(runid)], spat_size[runids.index(runid)])


def get_neighs(runid, cells, neighborhood_reach=[1, 1, 0]):

    xn = neighborhood_reach[0]
    yn = neighborhood_reach[1]
    zn = neighborhood_reach[2]

    return get_neighs_asym(runid, cells, neighborhood_reach=[-xn, xn, -yn, yn, -zn, zn])


def get_neighs_asym(runid, cells, neighborhood_reach=[-1, 1, -1, 1, 0, 0]):

    spat_ext, spat_size = spatmesh_get(runid)
    x_size, y_size, z_size = spat_size

    cells = np.array(cells, ndmin=1)
    out_cells = np.array(cells, ndmin=1)

    for a in range(neighborhood_reach[0], neighborhood_reach[1] + 1):
        for b in range(neighborhood_reach[2], neighborhood_reach[3] + 1):
            for c in range(neighborhood_reach[4], neighborhood_reach[5] + 1):
                new_cells = cells + a
                new_cells = new_cells[
                    (new_cells - 1) // x_size == (cells - 1) // x_size
                ]
                new_cells = new_cells[
                    np.logical_and((new_cells > 0), (new_cells <= x_size * y_size))
                ]
                new_cells = new_cells + x_size * b
                new_cells = new_cells[
                    np.logical_and((new_cells > 0), (new_cells <= x_size * y_size))
                ]
                out_cells = np.append(out_cells, new_cells)

    return np.unique(out_cells).astype(int)


def find_bulkpath(runid):

    runid_list = ["ABA", "ABC", "AEA", "AEC", "BFD"]
    path_list = ["bulk/", "bulk/", "round_3_boundary_sw/", "bulk/", "bulk/"]

    vlpath = "{}/2D/{}/".format(vlasdir, runid)

    if runid in runid_list:
        bulkpath = vlpath + path_list[runid_list.index(runid)]
    else:
        bulkpath = vlpath + "bulk/"

    return bulkpath


def sw_par_dict(runid):
    # Returns solar wind parameters for specified run
    # Output is 0: density, 1: velocity, 2: IMF strength 3: dynamic pressure 4: plasma beta

    runs = ["ABA", "ABC", "AEA", "AEC", "BFD"]
    sw_rho = [1e6, 3.3e6, 1.0e6, 3.3e6, 1.0e6]
    sw_v = [750e3, 600e3, 750e3, 600e3, 750e3]
    sw_B = [5.0e-9, 5.0e-9, 10.0e-9, 10.0e-9, 5.0e-9]
    sw_T = [500e3, 500e3, 500e3, 500e3, 500e3]
    sw_pdyn = [m_p * sw_rho[n] * (sw_v[n] ** 2) for n in range(len(runs))]
    sw_beta = [
        2 * sc.mu_0 * sw_rho[n] * sc.k * sw_T[n] / (sw_B[n] ** 2)
        for n in range(len(runs))
    ]

    return [
        sw_rho[runs.index(runid)],
        sw_v[runs.index(runid)],
        sw_B[runs.index(runid)],
        sw_pdyn[runs.index(runid)],
        sw_beta[runs.index(runid)],
        sw_T[runs.index(runid)],
    ]


def division_ste(a, b, stea, steb):

    return (a / b) * np.sqrt((stea / a) ** 2 + (steb / b) ** 2)
