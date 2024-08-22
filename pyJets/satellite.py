# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
# import jet_aux as jx
import pytools as pt
import os
import sys
from random import choice
from copy import deepcopy

from Shue_Mpause_model import Shue_Mpause_model
from Merka_BS_model import BS_distance_Merka2005

# import scipy
# import scipy.linalg
from scipy.linalg import eig
from scipy.fft import rfft2
from scipy.signal import butter, sosfilt, cwt, morlet2
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

filter_func = lambda d, size: uniform_filter1d(d, size=size)
# filter_func = lambda d, size: gaussian_filter1d(d, sigma=size)

from scipy.signal import butter, sosfilt, sosfiltfilt

import pyspedas
from datetime import datetime, timezone

mpl.rcParams["hatch.linewidth"] = 0.1

# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

# import plot_contours as pc
# import jet_analyser as ja
# import jet_io as jio
# import jet_jh2020 as jh20

# mpl.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# plt.rcParams.update(params)

plt.rcParams.update(
    {
        "ps.useafm": True,
        "pdf.use14corefonts": True,
        "text.usetex": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.fallback": None,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "text.latex.preamble": r"\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

r_e = 6.371e6
m_p = 1.672621898e-27
q_p = 1.602176634e-19
mu0 = 1.25663706212e-06
kb = 1.380649e-23

wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"

try:
    tavgdir = os.environ["TAVG"] + "/"
except:
    tavgdir = wrkdir_DNR + "tavg/"

wrkdir_DNR = wrkdir_DNR + "foreshock_bubble/"

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
    "#000000",
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#dede00",
    "#000000",
]


def BS_xy():
    # theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-90, 90, 0.5))
    R_bs = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta == a)[0][0]
        R_bs[index] = BS_distance_Merka2005(
            np.pi / 2, a, 5.569139338151405, 507.317425836262, 7.627438276232335, []
        )

    # x_bs = R_bs*np.cos(np.deg2rad(theta))
    # y_bs = R_bs*np.sin(np.deg2rad(theta))
    x_bs = R_bs * np.cos(theta)
    y_bs = R_bs * np.sin(theta)

    return [x_bs, y_bs]


def BS_xz():
    # theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(0, 180, 0.5))
    R_bs = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta == a)[0][0]
        R_bs[index] = BS_distance_Merka2005(
            a, 0, 5.569139338151405, 507.317425836262, 7.627438276232335, []
        )

    # x_bs = R_bs*np.cos(np.deg2rad(theta))
    # y_bs = R_bs*np.sin(np.deg2rad(theta))
    x_bs = R_bs * np.sin(theta)
    y_bs = -R_bs * np.cos(theta)

    return [x_bs, y_bs]


def MP_xy():
    # theta = np.arange(-60.25,60,0.5)
    theta = np.deg2rad(np.arange(-90, 90, 0.5))
    R_mp = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta == a)[0][0]
        R_mp[index] = Shue_Mpause_model(
            2.4084670491440354, -2.1407333318351043, [a], [0]
        )

    # x_mp = R_mp*np.cos(np.deg2rad(theta))
    # y_mp = R_mp*np.sin(np.deg2rad(theta))
    x_mp = R_mp * np.cos(theta)
    y_mp = R_mp * np.sin(theta)

    return [x_mp, y_mp]


def MVA(data, eigvals=False, prnt=True):

    M = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            M[i, j] = np.nanmean(data[i] * data[j]) - np.nanmean(data[i]) * np.nanmean(
                data[j]
            )

    eigenval, eigenvec = np.linalg.eig(M)
    eigenvec = eigenvec.T

    # for idx in range(3):
    #     if eigenvec[idx][0] > 0:
    #         eigenvec[idx] *= -1

    if prnt:
        print("\n")
        print("Eigenvalues: ", np.sort(eigenval))
        print("Eigenvectors: ", eigenvec[np.argsort(eigenval)])

    # return (np.sort(eigenval),eigenvec[np.argsort(eigenval)])
    if eigvals:
        return (np.sort(eigenval), eigenvec[np.argsort(eigenval), :])
    else:
        return eigenvec[np.argsort(eigenval), :]


def interpolate_nans(data):

    dummy_x = np.arange(data.size)
    mask = np.isnan(data)

    return np.interp(dummy_x, dummy_x[~mask], data[~mask])


def time_clip(time_list, data_list, t0, t1):

    time_clipped = [t for t in time_list if t >= t0 and t <= t1]
    data_clipped = [
        data_list[idx]
        for idx in range(len(time_list))
        if time_list[idx] >= t0 and time_list[idx] <= t1
    ]

    return (time_clipped, data_clipped)


def load_msh_sc_data(
    sc_ins_obj,
    sc,
    probe,
    var,
    t0,
    t1,
    intpol=True,
    dt=1,
    datarate="fast",
    datatype="h0",
    filt=None,
    lpfilt=None,
    species="i",
):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    vars_list = ["B", "rho", "v", "pos", "Tperp", "Tpar"]
    sc_list = ["mms", "themis", "cluster", "ace", "dscovr", "wind"]
    sc_var_names = [
        [
            "mms{}_fgm_b_gse_{}_l2".format(probe, datarate),
            "mms{}_d{}s_numberdensity_{}".format(probe, species, datarate),
            "mms{}_d{}s_bulkv_gse_{}".format(probe, species, datarate),
            "mms{}_mec_r_gse".format(probe),
            "mms{}_d{}s_tempperp_{}".format(probe, species, datarate),
            "mms{}_d{}s_temppara_{}".format(probe, species, datarate),
        ],
        [
            "th{}_fgs_gse".format(probe),
            "th{}_pe{}m_density".format(probe, species),
            "th{}_pe{}m_velocity_gse".format(probe, species),
            "th{}_pos_gse".format(probe),
        ],
        [
            "B_xyz_gse__C{}_UP_FGM".format(probe),
            "N_p__C{}_PP_CIS".format(probe),
            "V_p_xyz_gse__C{}_PP_CIS".format(probe),
        ],
        [
            "BGSEc",
            "Np",
            "Vp",
        ],
        [
            "dsc_h0_mag_B1GSE",
            "dsc_h1_fc_Np",
            "dsc_h1_fc_V_GSE",
        ],
        [
            "BGSE",
            "N_elec",
            "U_eGSE",
        ],
    ]

    if sc == "mms":
        sc_data = sc_ins_obj(
            trange=[t0, t1],
            probe=probe,
            notplot=True,
            time_clip=True,
            data_rate=datarate,
        )[sc_var_names[sc_list.index(sc)][vars_list.index(var)]]
    elif sc in ["ace", "dscovr", "wind"]:
        sc_data = sc_ins_obj(
            trange=[t0, t1],
            notplot=True,
            time_clip=True,
            datatype=datatype,
        )[sc_var_names[sc_list.index(sc)][vars_list.index(var)]]
    else:
        sc_data = sc_ins_obj(
            trange=[t0, t1], probe=probe, notplot=True, time_clip=True
        )[sc_var_names[sc_list.index(sc)][vars_list.index(var)]]
    time_list = np.array(sc_data["x"])
    data_list = np.array(sc_data["y"]).T

    # print(
    #     "Probe {} Var {} time res: {} {}".format(probe, var, time_list[0], time_list[1])
    # )

    mask = ~np.isnan(np.atleast_2d(data_list)[0])
    data_list = data_list.T[mask].T
    time_list = time_list[mask]

    if intpol:
        newtime = np.arange(
            t0plot.replace(tzinfo=timezone.utc).timestamp(),
            t1plot.replace(tzinfo=timezone.utc).timestamp(),
            dt,
        )
        if type(time_list[0]) == datetime:
            time_list = [t.replace(tzinfo=timezone.utc).timestamp() for t in time_list]
        if len(data_list.shape) > 1:
            newdata = np.array(
                [np.interp(newtime, time_list, data) for data in data_list[:3]]
            )
        else:
            newdata = np.interp(newtime, time_list, data_list)
        newtime = np.array([datetime.utcfromtimestamp(t) for t in newtime])
        if filt:
            newdata = filter_func(newdata, size=filt)
        elif lpfilt:
            sos = butter(10, lpfilt, "lowpass", fs=int(1 / dt), output="sos")
            newdata = sosfiltfilt(sos, newdata)
            # filtered = signal.sosfilt(sos, sig)
        return (newtime, newdata)
    else:
        if type(time_list[0]) != datetime:
            time_list = np.array([datetime.utcfromtimestamp(t) for t in time_list])
        if filt:
            data_list = filter_func(data_list, size=filt)
        elif lpfilt:
            sos = butter(10, lpfilt, "lowpass", fs=int(1 / dt), output="sos")
            data_list = sosfiltfilt(sos, data_list)
        return (time_list, data_list)


def intpol_data(time_arr, data_arr, t0, t1, dt=1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    mask = ~np.isnan(np.atleast_2d(data_arr)[0])
    data_arr = data_arr.T[mask].T
    time_arr = time_arr[mask]

    newtime = np.arange(
        t0plot.replace(tzinfo=timezone.utc).timestamp(),
        t1plot.replace(tzinfo=timezone.utc).timestamp(),
        dt,
    )
    if type(time_arr[0]) == datetime:
        time_arr = [t.replace(tzinfo=timezone.utc).timestamp() for t in time_arr]
    if len(data_arr.shape) > 1:
        newdata = np.array(
            [np.interp(newtime, time_arr, data) for data in data_arr[:3]]
        )
    else:
        newdata = np.interp(newtime, time_arr, data_arr)
    newtime = np.array([datetime.utcfromtimestamp(t) for t in newtime])

    return (newtime, newdata)


def avg_sw_data(t0, t1, dt=1):

    omnidata = pyspedas.omni.data(trange=[t0, t1], notplot=True, time_clip=True)
    time_arr, n_arr = intpol_data(
        omnidata["proton_density"]["x"], omnidata["proton_density"]["y"], t0, t1, dt
    )
    time_arr, v_arr = intpol_data(
        omnidata["flow_speed"]["x"], omnidata["flow_speed"]["y"], t0, t1, dt
    )
    time_arr, Bz_arr = intpol_data(
        omnidata["BZ_GSE"]["x"], omnidata["BZ_GSE"]["y"], t0, t1, dt
    )
    time_arr, Bx_arr = intpol_data(
        omnidata["BX_GSE"]["x"], omnidata["BX_GSE"]["y"], t0, t1, dt
    )
    time_arr, By_arr = intpol_data(
        omnidata["BY_GSE"]["x"], omnidata["BY_GSE"]["y"], t0, t1, dt
    )
    time_arr, MA_arr = intpol_data(
        omnidata["Mach_num"]["x"], omnidata["Mach_num"]["y"], t0, t1, dt
    )
    time_arr, timeshift_arr = intpol_data(
        omnidata["Timeshift"]["x"], omnidata["Timeshift"]["y"], t0, t1, dt
    )

    pd_arr = m_p * n_arr * 1e6 * v_arr * v_arr * 1e6 * 1e9

    B = [Bx_arr, By_arr, Bz_arr]
    ycone_arr = np.rad2deg(np.arctan2(By_arr, Bx_arr))
    zcone_arr = np.rad2deg(np.arctan2(Bz_arr, Bx_arr))
    data_arr = [Bx_arr, By_arr, Bz_arr, v_arr, n_arr, pd_arr, ycone_arr, zcone_arr]

    fig, ax_list = plt.subplots(8, 1, figsize=(12, 18), constrained_layout=True)
    for idx in range(len(ax_list)):
        ax = ax_list[idx]
        ax.plot(time_arr, data_arr[idx])
        ax.grid()
        ax.set_xlim(time_arr[0], time_arr[-1])
        ax.set_ylabel(["Bx", "By", "Bz", "v", "n", "Pdyn", "Ycone", "Zcone"][idx])

    fig.savefig(wrkdir_DNR + "Figs/satellite/SW_plot.png", dpi=150)

    return (
        np.nanmean(n_arr),
        np.nanmean(v_arr),
        np.nanmean(pd_arr),
        np.nanmean(Bx_arr),
        np.nanmean(By_arr),
        np.nanmean(Bz_arr),
        np.nanmean(MA_arr),
        np.nanmean(timeshift_arr),
        np.nanmean(ycone_arr),
        np.nanmean(zcone_arr),
    )


def plot_sc_b(sc, probe, t0, t1, dt=1, datarate="srvy"):

    sc_list = ["ace", "dscovr", "wind", "themis", "mms", "cluster"]
    Bobj_list = [
        pyspedas.ace.mfi,
        pyspedas.dscovr.mag,
        pyspedas.wind.mfi,
        pyspedas.themis.fgm,
        pyspedas.mms.fgm,
        pyspedas.cluster.fgm,
    ]

    Bobj = Bobj_list[sc_list.index(sc)]

    dtp = "h0"
    if sc in ["ace", "dscovr", "wind"]:
        dtp = ["h3", "h0", "h2"][["ace", "dscovr", "wind"].index(sc)]

    time_arr, B = load_msh_sc_data(
        Bobj,
        sc,
        probe,
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datarate=datarate,
        datatype=dtp,
    )

    fig, ax_list = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for idx in range(3):
        ax = ax_list[idx]
        ax.plot(time_arr, B[idx])
        ax.grid()
        ax.set_ylabel(["Bx", "By", "Bz"][idx])
    ax_list[0].set_title("{} {}".format(sc.upper(), probe.upper()))

    fig.savefig(wrkdir_DNR + "Figs/satellite/{}{}_B.png".format(sc, probe), dpi=150)


def plot_all_sc(
    scs_to_plot=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], plot_planes=True
):

    sc_mva_pos_file_name = wrkdir_DNR + "SC_time_pos_MVA.csv"
    minvec_mva = np.loadtxt(
        sc_mva_pos_file_name, dtype=float, delimiter=";", skiprows=1, usecols=[2, 3, 4]
    )
    maxvec_mva = np.loadtxt(
        sc_mva_pos_file_name,
        dtype=float,
        delimiter=";",
        skiprows=1,
        usecols=[8, 9, 10],
    )
    sc_pos = np.loadtxt(
        sc_mva_pos_file_name, dtype=float, delimiter=";", skiprows=1, usecols=[5, 6, 7]
    )
    sc_name = np.loadtxt(
        sc_mva_pos_file_name, dtype=str, delimiter=";", skiprows=1, usecols=0
    )
    sc_markers = ["x", "x", "x", "^", "^", "^", "^", "^", "o", "o", "o", "o", "*", "*"]
    sc_colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[3],
        CB_color_cycle[4],
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[3],
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]

    fig, ax_list = plt.subplots(
        2, 1, figsize=(12, 12), constrained_layout=True, sharex=True, sharey=True
    )

    x_bs, y_bs = BS_xy()
    x_bs2, z_bs = BS_xz()
    x_mp, y_mp = MP_xy()
    z_mp = y_mp

    for idx in range(2):
        for idx2 in range(sc_name.size):
            if idx2 not in scs_to_plot:
                continue
            ax_list[idx].plot(
                sc_pos[idx2, 0],
                sc_pos[idx2, idx + 1],
                sc_markers[idx2],
                label=sc_name[idx2],
                color=sc_colors[idx2],
                markersize=10,
            )
            ortho_vector = np.cross(minvec_mva[idx2], [[0, 0, 1], [0, 1, 0]][idx])
            ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)
            if plot_planes:
                ax_list[idx].plot(
                    [
                        sc_pos[idx2, 0] - 5 * ortho_vector[0],
                        sc_pos[idx2, 0] + 5 * ortho_vector[0],
                    ],
                    [
                        sc_pos[idx2, idx + 1] - 5 * ortho_vector[idx + 1],
                        sc_pos[idx2, idx + 1] + 5 * ortho_vector[idx + 1],
                    ],
                    color=sc_colors[idx2],
                )
            # ax_list[idx].quiver(
            #     sc_pos[idx2, 0],
            #     sc_pos[idx2, idx + 1],
            #     minvec_mva[idx2, 0] * 20,
            #     minvec_mva[idx2, idx + 1] * 20,
            #     scale_units="xy",
            #     angles="xy",
            #     scale=1,
            #     color=sc_colors[idx2],
            # )
        ax_list[idx].set_ylabel(["Y [RE]", "Z [RE]"][idx])
        ax_list[idx].grid()
        ax_list[idx].plot([x_bs, x_bs2][idx], [y_bs, z_bs][idx], color="k", zorder=0)
        ax_list[idx].plot(x_mp, [y_mp, z_mp][idx], color="k", zorder=0)
        ax_list[idx].plot(
            np.cos(np.arange(0, 2 * np.pi, 0.5)),
            np.sin(np.arange(0, 2 * np.pi, 0.5)),
            color="k",
            zorder=0,
        )
        ax_list[idx].set_aspect("equal")
    ax_list[-1].set_xlabel("X [RE]")
    ax_list[-1].legend()
    ax_list[-1].set_xlim(0, None)

    fig.savefig(
        wrkdir_DNR + "Figs/satellite/sc_pos_{}.png".format(scs_to_plot), dpi=150
    )
    plt.close(fig)


def thd_mms1_c4_timing(t0, t1, dt=1, mva=False):

    thd_time, thd_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "d", "B", t0, t1, intpol=True, dt=dt
    )
    mms1_time, mms1_B = load_msh_sc_data(
        pyspedas.mms.fgm, "mms", "1", "B", t0, t1, intpol=True, dt=dt, datarate="srvy"
    )
    c4_time, c4_B = load_msh_sc_data(
        pyspedas.cluster.fgm, "cluster", "4", "B", t0, t1, intpol=True, dt=dt
    )

    thd_time2, thd_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "rho", t0, t1, intpol=True, dt=dt
    )
    mms1_time2, mms1_rho = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "rho", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    c4_time2, c4_rho = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "rho", t0, t1, intpol=True, dt=dt
    )

    dummy, thd_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, mms1_v = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "v", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    dummy, c4_v = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "v", t0, t1, intpol=True, dt=dt
    )

    thd_vmag = np.linalg.norm(thd_v, axis=0)
    mms1_vmag = np.linalg.norm(mms1_v, axis=0)
    c4_vmag = np.linalg.norm(c4_v, axis=0)

    thd_pdyn = m_p * thd_rho * 1e6 * thd_vmag * thd_vmag * 1e6 / 1e-9
    mms1_pdyn = m_p * mms1_rho * 1e6 * mms1_vmag * mms1_vmag * 1e6 / 1e-9
    c4_pdyn = m_p * c4_rho * 1e6 * c4_vmag * c4_vmag * 1e6 / 1e-9

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/c4_mms1_thd_pos_2022-03-27_21:00:00_21:30:00_numpy.txt",
        dtype="str",
    ).T
    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * r_e * 1e-3
    sc_y = pos_data[5].astype(float) * r_e * 1e-3
    sc_z = pos_data[6].astype(float) * r_e * 1e-3

    sc_coords = np.array([sc_x, sc_y, sc_z]).T

    thd_pos = sc_coords[sc_name == "themisd", :][:-1]
    mms1_pos = sc_coords[sc_name == "mms1", :]
    c4_pos = sc_coords[sc_name == "cluster4", :]

    sc_rel_pos = [
        np.nanmean(mms1_pos - thd_pos, axis=0),
        np.nanmean(c4_pos - thd_pos, axis=0),
    ]
    print(sc_rel_pos)

    if mva:
        Bdata_thd = deepcopy(thd_B[0:3])
        Bdata_mms1 = deepcopy(mms1_B[0:3])
        Bdata_c4 = deepcopy(c4_B[0:3])

        vdata_thd = deepcopy(thd_v[0:3])
        vdata_mms1 = deepcopy(mms1_v[0:3])
        vdata_c4 = deepcopy(c4_v[0:3])

        eigenvecs_thd = MVA(Bdata_thd)
        eigenvecs_mms1 = MVA(Bdata_mms1)
        eigenvecs_c4 = MVA(Bdata_c4)

        print("THD Minimum Variance direction: {}".format(eigenvecs_thd[0]))
        print("MMS1 Minimum Variance direction: {}".format(eigenvecs_mms1[0]))
        print("C4 Minimum Variance direction: {}".format(eigenvecs_c4[0]))

        for idx in range(3):
            thd_B[idx] = np.dot(Bdata_thd.T, eigenvecs_thd[idx])
            thd_v[idx] = np.dot(vdata_thd.T, eigenvecs_thd[idx])

            mms1_B[idx] = np.dot(Bdata_mms1.T, eigenvecs_mms1[idx])
            mms1_v[idx] = np.dot(vdata_mms1.T, eigenvecs_mms1[idx])

            c4_B[idx] = np.dot(Bdata_c4.T, eigenvecs_c4[idx])
            c4_v[idx] = np.dot(vdata_c4.T, eigenvecs_c4[idx])

    labs = ["Bx:", "By:", "Bz:"]
    labs_v = ["Vx:", "Vy:", "Vz:"]
    if mva:
        labs = ["BN:", "BM:", "BL:"]
        labs_v = ["VN:", "VM:", "VL:"]

    print("\n")

    for idx in range(3):
        print(labs[idx])
        timing_analysis_arb(
            [thd_time, mms1_time, c4_time],
            [thd_B[idx], mms1_B[idx], c4_B[idx]],
            sc_rel_pos,
            t0,
            t1,
        )
        print("\n")
        print(labs_v[idx])
        timing_analysis_arb(
            [thd_time2, mms1_time2, c4_time2],
            [thd_v[idx], mms1_v[idx], c4_v[idx]],
            sc_rel_pos,
            t0,
            t1,
        )
        print("\n")

    print("Pdyn:")
    timing_analysis_arb(
        [thd_time2, mms1_time2, c4_time2],
        [thd_pdyn, mms1_pdyn, c4_pdyn],
        sc_rel_pos,
        t0,
        t1,
    )
    print("\n")


def diag_themis(t0, t1, dt=1, grain=1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    probe_names = ["a", "d", "e"]

    sc_B = [
        load_msh_sc_data(
            pyspedas.themis.fgm,
            "themis",
            probe_names[probe],
            "B",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_rho = [
        load_msh_sc_data(
            pyspedas.themis.mom,
            "themis",
            probe_names[probe],
            "rho",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_v = [
        load_msh_sc_data(
            pyspedas.themis.mom,
            "themis",
            probe_names[probe],
            "v",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_pos = [
        load_msh_sc_data(
            pyspedas.themis.state,
            "themis",
            probe_names[probe],
            "pos",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="srvy",
        )
        for probe in range(3)
    ]

    rel_pos = [
        np.nanmean(sc_pos[idx][1] - sc_pos[0][1], axis=-1).T for idx in range(1, 3)
    ]

    time_arr = sc_B[0][0]

    data_arr = np.empty((3, 10, time_arr.size), dtype=float)
    for idx in range(3):
        data_arr[idx, :, :] = [
            sc_B[idx][1][0],
            sc_B[idx][1][1],
            sc_B[idx][1][2],
            np.linalg.norm(sc_B[idx][1][:3], axis=0),
            sc_v[idx][1][0],
            sc_v[idx][1][1],
            sc_v[idx][1][2],
            np.linalg.norm(sc_v[idx][1][:3], axis=0),
            sc_rho[idx][1],
            m_p
            * sc_rho[idx][1]
            * 1e6
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * 1e6
            / 1e-9,
        ]

    fig, ax_list = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for idx in range(1, 3):
        ax_list[0].plot(
            time_arr,
            sc_pos[idx][1][0] - sc_pos[0][1][0],
            color=CB_color_cycle[idx],
            label="TH{}-THA".format(probe_names[idx].upper()),
        )
        ax_list[1].plot(
            time_arr,
            sc_pos[idx][1][1] - sc_pos[0][1][1],
            color=CB_color_cycle[idx],
            label="TH{}-THA".format(probe_names[idx].upper()),
        )
        ax_list[2].plot(
            time_arr,
            sc_pos[idx][1][2] - sc_pos[0][1][2],
            color=CB_color_cycle[idx],
            label="TH{}-THA".format(probe_names[idx].upper()),
        )
    for idx in range(3):
        ax_list[idx].grid()
        ax_list[idx].legend()

    fig.savefig(wrkdir_DNR + "Figs/satellite/themis_diag_pos.png", dpi=150)
    plt.close(fig)

    window_center = np.arange(0, time_arr.size, grain, dtype=int)
    window_halfwidth = np.arange(10, int(time_arr.size / 2), grain, dtype=int)
    window_size = (window_halfwidth * 2 * dt).astype(int)
    print(
        "Window center size: {}, window halfwidth size: {}, Time arr grain size: {}".format(
            window_center.size, window_halfwidth.size, time_arr[0::grain].size
        )
    )

    diag_data = np.empty((4, window_center.size, window_halfwidth.size), dtype=float)
    labs = ["Bx:", "By:", "Bz:", "Bt:", "Vx:", "Vy:", "Vz:", "Vt:", "rho:", "Pdyn:"]
    idcs = [2, 5, 8, 9]

    for idx2 in range(window_center.size):
        for idx3 in range(window_halfwidth.size):
            start_id = max(window_center[idx2] - window_halfwidth[idx3], 0)
            stop_id = min(
                window_center[idx2] + window_halfwidth[idx3] + 1, time_arr.size
            )
            for idx1 in range(len(idcs)):
                print(
                    "Window center: {}, window halfwidth: {}, start id: {}, stop id: {}".format(
                        window_center[idx2], window_halfwidth[idx3], start_id, stop_id
                    )
                )
                var_id = idcs[idx1]
                res = timing_analysis_arb(
                    [
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                    ],
                    [
                        data_arr[0, var_id, start_id:stop_id],
                        data_arr[1, var_id, start_id:stop_id],
                        data_arr[2, var_id, start_id:stop_id],
                    ],
                    rel_pos,
                    prnt=False,
                )
                diag_data[idx1, idx2, idx3] = np.min(res["cross_corr_values"])

    fig, ax_list = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
    ims = []
    cbs = []
    for idx in range(4):
        im = ax_list[idx].pcolormesh(
            time_arr[0::grain],
            window_size,
            diag_data[idx].T,
            shading="gouraud",
            cmap="hot_desaturated",
            vmin=0,
            vmax=1,
        )
        ims.append(im)
        cbs.append(plt.colorbar(ims[-1], ax=ax_list[idx]))
        ax_list[idx].set_title(labs[idcs[idx]])
        ax_list[idx].set_ylabel("Window width [s]")
    ax_list[-1].set_xlabel("Window center")

    fig.savefig(wrkdir_DNR + "Figs/satellite/themis_diag_corr.png", dpi=150)
    plt.close(fig)


def plot_themis(t0, t1, mva=False, dt=1, peakonly=False):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    probe_names = ["a", "d", "e"]

    sc_B = [
        load_msh_sc_data(
            pyspedas.themis.fgm,
            "themis",
            probe_names[probe],
            "B",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_rho = [
        load_msh_sc_data(
            pyspedas.themis.mom,
            "themis",
            probe_names[probe],
            "rho",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_v = [
        load_msh_sc_data(
            pyspedas.themis.mom,
            "themis",
            probe_names[probe],
            "v",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(3)
    ]
    sc_pos = [
        load_msh_sc_data(
            pyspedas.themis.state,
            "themis",
            probe_names[probe],
            "pos",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="srvy",
        )
        for probe in range(3)
    ]

    rel_pos = [
        np.nanmean(sc_pos[idx][1] - sc_pos[0][1], axis=-1).T for idx in range(1, 3)
    ]

    time_arr = sc_B[0][0]

    data_arr = np.empty((3, 10, time_arr.size), dtype=float)
    for idx in range(3):
        data_arr[idx, :, :] = [
            sc_B[idx][1][0],
            sc_B[idx][1][1],
            sc_B[idx][1][2],
            np.linalg.norm(sc_B[idx][1][:3], axis=0),
            sc_v[idx][1][0],
            sc_v[idx][1][1],
            sc_v[idx][1][2],
            np.linalg.norm(sc_v[idx][1][:3], axis=0),
            sc_rho[idx][1],
            m_p
            * sc_rho[idx][1]
            * 1e6
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * 1e6
            / 1e-9,
        ]

    if mva:
        Bdata = [deepcopy(data_arr[idx, 0:3, :]) for idx in range(3)]
        vdata = [deepcopy(data_arr[idx, 4:7, :]) for idx in range(3)]
        eigenvecs = [MVA(Bdata[idx]) for idx in range(3)]
        for prob in range(3):
            print(
                "MMS{} Minimum Variance direction: {}".format(
                    probe_names[prob], eigenvecs[prob][0]
                )
            )
            for idx in range(3):
                data_arr[prob, idx, :] = np.dot(Bdata[prob].T, eigenvecs[prob][idx])
                data_arr[prob, idx + 4, :] = np.dot(vdata[prob].T, eigenvecs[prob][idx])

    t_pdmax = [time_arr[np.argmax(data_arr[idx, 9])] for idx in range(3)]

    # rel_pos = [
    #     sc_pos[idx][1].T[np.argmax(data_arr[idx, 9])]
    #     - sc_pos[0][1].T[np.argmax(data_arr[0, 9])]
    #     for idx in range(1, 4)
    # ]

    panel_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3]
    panel_labs = ["B [nT]", "V [km/s]", "n [1/cm3]", "Pdyn [nPa]"]
    ylabels_all = [
        "Bx [nT]",
        "By [nT]",
        "Bz [nT]",
        "Bt [nT]",
        "vx [km/s]",
        "vy [km/s]",
        "vz [km/s]",
        "vt [km/s]",
        "n [1/cm3]",
        "Pdyn [nPa]",
    ]
    if mva:
        ylabels_all = [
            "BN [nT]",
            "BM [nT]",
            "BL [nT]",
            "Bt [nT]",
            "vN [km/s]",
            "vM [km/s]",
            "vL [km/s]",
            "vt [km/s]",
            "n [1/cm3]",
            "Pdyn [nPa]",
        ]
    sc_labs = ["THA", "THD", "THE"]
    colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        "k",
    ]
    plot_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    line_label = [
        "x",
        "y",
        "z",
        "mag",
        "x",
        "y",
        "z",
        "mag",
        None,
        None,
    ]
    ylims = [
        (-40, 60),
        (-400, 500),
        (5, 35),
        (0, 10),
    ]
    ylims_full = [
        (-20, 30),
        (-20, 40),
        (-40, 50),
        (0, 60),
        (-400, 0),
        (-200, 100),
        (-100, 300),
        (0, 500),
        (5, 35),
        (0, 10),
    ]

    fig, ax_list = plt.subplots(
        10, 3, figsize=(24, 24), sharey="row", constrained_layout=True
    )

    for idx in range(3):
        for idx2 in range(len(panel_id)):
            print("Plotting {} {}".format(sc_labs[idx], panel_labs[panel_id[idx2]]))
            # ax = ax_list[panel_id[idx2], idx]
            # if not plot_legend[idx2]:
            #     ax.grid()
            # ax.plot(
            #     time_arr[idx, panel_id[idx2]],
            #     data_arr[idx, idx2],
            #     color=colors[idx2],
            #     label=line_label[idx2],
            #     alpha=0.5,
            # )
            # if plot_legend[idx2] and idx == 2:
            #     ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
            ax = ax_list[idx2, idx]
            ax.grid()
            ax.plot(
                time_arr,
                data_arr[idx, idx2],
            )
            ax.label_outer()
            ax.set_xlim(t0plot, t1plot)
            ax.axvline(t_pdmax[idx], linestyle="dashed")

    print("Times of Pdynmax: {}".format(t_pdmax))

    for idx in range(3):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
    # for idx in range(len(panel_labs)):
    #     ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
    #     ax_list[idx, 0].set_ylim(ylims[idx][0], ylims[idx][1])
    for idx in range(len(ylabels_all)):
        ax_list[idx, 0].set_ylabel(ylabels_all[idx], labelpad=10, fontsize=20)
        # ax_list[idx, 0].set_ylim(ylims_full[idx][0], ylims_full[idx][1])

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir
        + "themis_ade_t0{}_t1{}_mva{}_peak{}.png".format(t0plot, t1plot, mva, peakonly)
    )
    plt.close(fig)

    print(rel_pos)

    labs = [
        "Bx:",
        "By:",
        "Bz:",
        "Bt:",
        "Vx:",
        "Vy:",
        "Vz:",
        "Vt:",
        "rho:",
        "Pdyn:",
    ]
    grads = [False, False, False, False, False, False, False, False, False, False]
    labs_v = ["Vx:", "Vy:", "Vz:"]
    if mva:
        labs = [
            "BN:",
            "BM:",
            "BL:",
            "Bt:",
            "VN:",
            "VM:",
            "VL:",
            "Vt:",
            "rho:",
            "Pdyn:",
        ]
        labs_v = ["Vmin:", "Vmed:", "Vmax:"]

    print("\n")

    timing_res = []

    for idx in range(10):
        print(labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr],
            [
                data_arr[0][idx],
                data_arr[1][idx],
                data_arr[2][idx],
            ],
            rel_pos,
            t0,
            t1,
            peakonly=peakonly,
            gradient=grads[idx],
        )
        timing_res.append(res)
        print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = labs
    for idx in range(len(labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )
    if mva:
        rowLabels += sc_labs
        for idx in range(len(sc_labs)):
            cellText.append(
                [
                    str(
                        (
                            round(eigenvecs[idx][0][0], ndigits=2),
                            round(eigenvecs[idx][0][1], ndigits=2),
                            round(eigenvecs[idx][0][2], ndigits=2),
                        )
                    ),
                    "",
                    "",
                ]
            )

    # for idx in range(len(cellText)):
    #     for idx2 in range(len(cellText[0])):
    #         cellText[idx][idx2] = cellText[idx][idx2][:5]

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        + "themis_ade_t0{}_t1{}_mva{}_peak{}_table.png".format(
            t0plot, t1plot, mva, peakonly
        )
    )
    plt.close(fig)


def mms_plot_vdf(
    t0,
    t1,
    probe="1",
    fmin=1e-24,
    fmax=1e-20,
    fcut=1e-24,
    vlim=1500,
):

    # t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    # t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    outdir = wrkdir_DNR + "Figs/satellite/VDF/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    for s in np.arange(t0, t1):
        ct = "2022-03-27/21:21:{}".format(str(s).zfill(2))
        slice = pyspedas.mms_part_slice2d(
            time=ct,
            probe=probe,
            instrument="fpi",
            species="i",
            data_rate="brst",
            mag_data_rate="brst",
            rotation="bv",
            slice_norm=np.array([0, 0, 1]),
            interpolation="2d",
            return_slice=True,
            window=1,
            center_time=True,
        )
        data, xmesh, ymesh = slice["data"], slice["xgrid"], slice["ygrid"]
        data[data < fcut] = np.nan
        data[data > 1] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        im = ax.pcolormesh(
            xmesh, ymesh, data, norm="log", cmap="batlow", vmin=fmin, vmax=fmax
        )
        cb = plt.colorbar(im, ax=ax)
        ax.set_xlabel("$v_B$ [km/s]")
        ax.set_xlim(-vlim, vlim)
        ax.set_ylabel("$v_{B\\times v}$ [km/s]")
        ax.set_ylim(-vlim, vlim)
        ax.grid()
        fig.savefig(outdir + "{}.png".format(ct.replace("/", "_")))
        plt.close(fig)


def diag_mms(t0, t1, dt=0.1, grain=1, ij=None, bv=False):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    sc_B = [
        load_msh_sc_data(
            pyspedas.mms.fgm,
            "mms",
            "{}".format(probe),
            "B",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(1, 5)
    ]
    sc_rho = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "rho",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(1, 5)
    ]
    sc_v = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "v",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
        )
        for probe in range(1, 5)
    ]
    sc_pos = [
        load_msh_sc_data(
            pyspedas.mms.mec,
            "mms",
            "{}".format(probe),
            "pos",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="srvy",
        )
        for probe in range(1, 5)
    ]

    rel_pos = [
        np.nanmean(sc_pos[idx][1] - sc_pos[0][1], axis=-1).T for idx in range(1, 4)
    ]

    time_arr = sc_B[0][0]

    data_arr = np.empty((4, 10, time_arr.size), dtype=float)
    for idx in range(4):
        data_arr[idx, :, :] = [
            sc_B[idx][1][0],
            sc_B[idx][1][1],
            sc_B[idx][1][2],
            np.linalg.norm(sc_B[idx][1][:3], axis=0),
            sc_v[idx][1][0],
            sc_v[idx][1][1],
            sc_v[idx][1][2],
            np.linalg.norm(sc_v[idx][1][:3], axis=0),
            sc_rho[idx][1],
            m_p
            * sc_rho[idx][1]
            * 1e6
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * 1e6
            / 1e-9,
        ]

    fig, ax_list = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for idx in range(1, 4):
        ax_list[0].plot(
            time_arr,
            sc_pos[idx][1][0] - sc_pos[0][1][0],
            color=CB_color_cycle[idx],
            label="MMS{}-MMS1".format(idx + 1),
        )
        ax_list[1].plot(
            time_arr,
            sc_pos[idx][1][1] - sc_pos[0][1][1],
            color=CB_color_cycle[idx],
            label="MMS{}-MMS1".format(idx + 1),
        )
        ax_list[2].plot(
            time_arr,
            sc_pos[idx][1][2] - sc_pos[0][1][2],
            color=CB_color_cycle[idx],
            label="MMS{}-MMS1".format(idx + 1),
        )
    for idx in range(3):
        ax_list[idx].grid()
        ax_list[idx].legend()

    fig.savefig(wrkdir_DNR + "Figs/satellite/mms_diag_pos.png", dpi=150)
    plt.close(fig)

    window_center = np.arange(0, time_arr.size, grain, dtype=int)
    window_halfwidth = np.arange(10, int(time_arr.size / 2), grain, dtype=int)
    window_size = (window_halfwidth * 2 * dt).astype(int)
    print(
        "Window center size: {}, window halfwidth size: {}, Time arr grain size: {}".format(
            window_center.size, window_halfwidth.size, time_arr[0::grain].size
        )
    )

    diag_data = np.empty((4, window_center.size, window_halfwidth.size), dtype=float)
    diag_vec_data = np.empty(
        (4, window_center.size, window_halfwidth.size, 3), dtype=float
    )
    labs = ["Bx:", "By:", "Bz:", "Bt:", "Vx:", "Vy:", "Vz:", "Vt:", "rho:", "Pdyn:"]
    idcs = [2, 5, 8, 9]

    for idx2 in range(window_center.size):
        for idx3 in range(window_halfwidth.size):
            start_id = max(window_center[idx2] - window_halfwidth[idx3], 0)
            stop_id = min(
                window_center[idx2] + window_halfwidth[idx3] + 1, time_arr.size
            )
            for idx1 in range(len(idcs)):
                print(
                    "Window center: {}, window halfwidth: {}, start id: {}, stop id: {}".format(
                        window_center[idx2], window_halfwidth[idx3], start_id, stop_id
                    )
                )
                if bv:
                    vbulk = np.nanmean(data_arr[0, 4:7, start_id:stop_id], axis=-1)
                else:
                    vbulk = np.array([0.000001, 0.000001, 0.000001])
                var_id = idcs[idx1]
                res = timing_analysis_arb(
                    [
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                    ],
                    [
                        data_arr[0, var_id, start_id:stop_id],
                        data_arr[1, var_id, start_id:stop_id],
                        data_arr[2, var_id, start_id:stop_id],
                        data_arr[3, var_id, start_id:stop_id],
                    ],
                    rel_pos,
                    prnt=False,
                    bulkv=vbulk,
                )
                diag_data[idx1, idx2, idx3] = np.min(res["cross_corr_values"])
                diag_vec_data[idx1, idx2, idx3, :] = (
                    np.array(res["wave_velocity_relative2sc"]).flatten()
                    * np.sign(np.array(res["wave_velocity_relative2sc"]).flatten()[0])
                    / np.linalg.norm(res["wave_velocity_relative2sc"])
                )
                # diag_vec_data[idx1, idx2, idx3, :] = np.array(
                #     res["wave_vector"]
                # ).flatten() * np.sign(np.array(res["wave_vector"]).flatten()[0])

    fig, ax_list = plt.subplots(4, 4, figsize=(32, 12), constrained_layout=True)
    ims = []
    cbs = []
    for idx in range(4):
        im = ax_list[idx, 0].pcolormesh(
            time_arr[0::grain],
            window_size,
            diag_data[idx].T,
            shading="gouraud",
            cmap="hot_desaturated",
            vmin=0.9,
            vmax=1,
        )
        ims.append(im)
        cbs.append(plt.colorbar(ims[-1], ax=ax_list[idx, 0]))
        ax_list[idx, 0].set_title(labs[idcs[idx]])
        ax_list[idx, 0].set_ylabel("Window width [s]")
        for idx2 in range(3):
            im = ax_list[idx, idx2 + 1].pcolormesh(
                time_arr[0::grain],
                window_size,
                diag_vec_data[idx, :, :, idx2].T,
                shading="gouraud",
                cmap="vik",
                vmin=-1,
                vmax=1,
            )
            ims.append(im)
            if idx2 == 2:
                cbs.append(plt.colorbar(ims[-1], ax=ax_list[idx, idx2 + 1]))
            ax_list[0, idx2 + 1].set_title(["$n_x$", "$n_y$", "$n_z$"][idx2])
    # for idx in range(4):
    #     ax_list[-1, idx].set_xlabel("Window center")

    fig.savefig(wrkdir_DNR + "Figs/satellite/mms_diag_corr.png", dpi=150)
    plt.close(fig)

    print("\n")

    for idx in range(4):
        indcs = np.where(diag_data[idx] == np.max(diag_data[idx]))
        print(indcs)

        if not ij:
            if indcs[0].size == 1:
                i, j = np.array(indcs).flatten()
            else:
                i, j = np.array(indcs).T[0]
        else:
            i, j = ij
        print(
            "{} n vector and pos: {} {} {}, i = {}, j = {}".format(
                labs[idcs[idx]],
                diag_vec_data[idx, i, j, :],
                time_arr[0::grain][i],
                window_size[j],
                i,
                j,
            )
        )

    # return (diag_vec_data[j, i, :], time_arr[0::grain][j], window_size[i])


def mms_tension_vel(
    t0,
    t1,
    dt=0.1,
    filt=None,
    species="i",
    lpfilt=None,
    normalise=False,
    dbdt=False,
):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    species_list = [species, species, species, "i"]

    sc_B = [
        load_msh_sc_data(
            pyspedas.mms.fgm,
            "mms",
            "{}".format(probe),
            "B",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
        )
        for probe in range(1, 5)
    ]

    sc_v = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "v",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]

    sc_pos = [
        load_msh_sc_data(
            pyspedas.mms.mec,
            "mms",
            "{}".format(probe),
            "pos",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="srvy",
            filt=filt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]

    time_arr = sc_B[0][0]

    data_arr = np.empty((4, 9, time_arr.size), dtype=float)

    for idx in range(4):
        data_arr[idx, :, :] = [
            sc_pos[idx][1][0] * 1e3,
            sc_pos[idx][1][1] * 1e3,
            sc_pos[idx][1][2] * 1e3,
            sc_B[idx][1][0] * 1e-9,
            sc_B[idx][1][1] * 1e-9,
            sc_B[idx][1][2] * 1e-9,
            sc_v[idx][1][0],
            sc_v[idx][1][1],
            sc_v[idx][1][2],
        ]

    outdata_arr = np.empty((2, 3, time_arr.size), dtype=float)

    for idx in range(time_arr.size):
        # outdata_arr[0, :, idx] = (
        #     tetra_mag_tension(data_arr[:, [0, 1, 2], idx], data_arr[:, [3, 4, 5], idx])
        #     / mu0
        # )
        outdata_arr[0, :, idx] = tetra_linear_interp(
            data_arr[:, [0, 1, 2], idx], data_arr[:, [3, 4, 5], idx]
        )
        outdata_arr[1, :, idx] = tetra_linear_interp(
            data_arr[:, [0, 1, 2], idx], data_arr[:, [6, 7, 8], idx]
        )
        if normalise:
            outdata_arr[0, :, idx] /= np.linalg.norm(outdata_arr[0, :, idx])
            outdata_arr[1, :, idx] /= np.linalg.norm(outdata_arr[1, :, idx])

    if dbdt:
        outdata_arr[0] = np.gradient(outdata_arr[0], dt, axis=-1)

    fig, ax_list = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    complabels = ["x", "y", "z"]

    for idx in range(3):
        for idx2 in range(2):
            ax_list[idx2].plot(
                time_arr,
                outdata_arr[idx2, idx, :],
                label=complabels[idx],
                color=CB_color_cycle[idx],
            )

    ax_list[0].set_title("MMS1-4")
    if normalise:
        # ax_list[0].set_ylabel("$\\hat{(\mathbf{B}\\cdot\\nabla)\mathbf{B}}$")
        ax_list[0].set_ylabel("$\\hat{B}$")
        ax_list[1].set_ylabel("$\\hat{v}$")
    else:
        # ax_list[0].set_ylabel("$(\mathbf{B}\\cdot\\nabla)\mathbf{B}/\\mu_0$")
        if dbdt:
            ax_list[0].set_ylabel("$dB/dt$")
        else:
            ax_list[0].set_ylabel("$B$")
        ax_list[1].set_ylabel("$v$")
    ax_list[0].set_xlim(t0plot, t1plot)
    ax_list[1].legend()
    ax_list[1].set_xlim(t0plot, t1plot)
    for ax in ax_list:
        ax.grid()

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir
        + "mms_tension_velocity_t0{}_t1{}_species{}_norm{}.png".format(
            t0plot, t1plot, species, normalise
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.psd(outdata_arr[0, 0, :], Fs=int(1 / dt))
    ax.set_xscale("log")
    ax.set_ylim(-400, -150)
    fig.savefig(
        outdir
        + "mms_B_psd_t0{}_t1{}_lpfilt{}_filt{}.png".format(t0plot, t1plot, lpfilt, filt)
    )
    plt.close(fig)

    rel_pos = [
        np.nanmean(sc_pos[idx][1] - sc_pos[0][1], axis=-1).T for idx in range(1, 4)
    ]

    timing = timing_analysis_arb(
        [
            time_arr,
            time_arr,
            time_arr,
            time_arr,
        ],
        [
            data_arr[0, 5, :],
            data_arr[1, 5, :],
            data_arr[2, 5, :],
            data_arr[3, 5, :],
        ],
        rel_pos,
        prnt=False,
        bulkv=np.array([1e-7, 1e-7, 1e-7]),
    )

    wave_vector = np.array(timing["wave_vector"]).flatten()
    print(timing["cross_corr_values"])

    vn = np.array(
        [np.dot(outdata_arr[1, :, idx], wave_vector) for idx in range(time_arr.size)]
    )
    vt = np.array(
        [
            np.sqrt(np.linalg.norm(outdata_arr[1, :, idx]) ** 2 - vn[idx] ** 2)
            for idx in range(time_arr.size)
        ]
    )

    vpar = np.array(
        [
            np.dot(outdata_arr[1, :, idx], outdata_arr[0, :, idx])
            / np.linalg.norm(outdata_arr[0, :, idx])
            for idx in range(time_arr.size)
        ]
    )
    vperp = np.array(
        [
            np.sqrt(np.linalg.norm(outdata_arr[1, :, idx]) ** 2 - vpar[idx] ** 2)
            for idx in range(time_arr.size)
        ]
    )

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax_list[0].plot(time_arr, vn, color=CB_color_cycle[0], label="$v_n$")
    ax_list[0].plot(time_arr, vt, color=CB_color_cycle[1], label="$v_t$")
    ax_list[0].legend()
    ax_list[0].set_xlim(t0plot, t1plot)
    ax_list[0].grid()

    ax_list[1].plot(time_arr, vpar, color=CB_color_cycle[0], label="$v_\\parallel$")
    ax_list[1].plot(time_arr, vperp, color=CB_color_cycle[1], label="$v_\\perp$")
    ax_list[1].legend()
    ax_list[1].set_xlim(t0plot, t1plot)
    ax_list[1].grid()

    fig.savefig(
        outdir
        + "mms_v_par_perp_t0{}_t1{}_lpfilt{}_filt{}.png".format(
            t0plot, t1plot, lpfilt, filt
        )
    )
    plt.close(fig)


def tetra_kvec(r):

    # print(r.shape)
    # print(r[0])

    k0 = np.cross((r[2] - r[1]), (r[3] - r[1])) / np.dot(
        (r[1] - r[0]), np.cross((r[2] - r[1]), (r[3] - r[1]))
    )
    k1 = np.cross((r[3] - r[2]), (r[0] - r[2])) / np.dot(
        (r[2] - r[1]), np.cross((r[3] - r[2]), (r[0] - r[2]))
    )
    k2 = np.cross((r[0] - r[3]), (r[1] - r[3])) / np.dot(
        (r[3] - r[2]), np.cross((r[0] - r[3]), (r[1] - r[3]))
    )
    k3 = np.cross((r[1] - r[0]), (r[2] - r[0])) / np.dot(
        (r[0] - r[3]), np.cross((r[1] - r[0]), (r[2] - r[0]))
    )

    return (k0, k1, k2, k3)


def tetra_linear_gradient(r, F):

    # print(r.shape)
    # print(F.shape)

    k = tetra_kvec(r)

    return (
        np.outer(k[0], F[0])
        + np.outer(k[1], F[1])
        + np.outer(k[2], F[2])
        + np.outer(k[3], F[3])
    )


def tetra_linear_interp(r, F):

    # print(r.shape)
    # print(F.shape)

    k = tetra_kvec(r)
    # print(k)

    rb = 0.25 * (r[0] + r[1] + r[2] + r[3])

    mu = [1 - np.dot(k[i], (rb - r[i])) for i in range(4)]
    # print(mu)

    return mu[0] * F[0] + mu[1] * F[1] + mu[2] * F[2] + mu[3] * F[3]


def tetra_mag_tension(r, B):

    # print(r.shape)
    # print(B.shape)

    B_jacob = tetra_linear_gradient(r, B)
    B_interp = tetra_linear_interp(r, B)

    return B_interp @ B_jacob


def plot_mms(
    t0, t1, mva=False, dt=0.1, peakonly=False, filt=None, species="i", lpfilt=None
):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    species_list = [species, species, species, "i"]

    sc_B = [
        load_msh_sc_data(
            pyspedas.mms.fgm,
            "mms",
            "{}".format(probe),
            "B",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
        )
        for probe in range(1, 5)
    ]
    sc_rho = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "rho",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]
    sc_v = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "v",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]
    sc_pos = [
        load_msh_sc_data(
            pyspedas.mms.mec,
            "mms",
            "{}".format(probe),
            "pos",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="srvy",
            filt=filt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]
    sc_tperp = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "Tperp",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]
    sc_tpara = [
        load_msh_sc_data(
            pyspedas.mms.fpi,
            "mms",
            "{}".format(probe),
            "Tpar",
            t0,
            t1,
            intpol=True,
            dt=dt,
            datarate="brst",
            filt=filt,
            lpfilt=lpfilt,
            species=species_list[probe - 1],
        )
        for probe in range(1, 5)
    ]

    rel_pos = [
        np.nanmean(sc_pos[idx][1] - sc_pos[0][1], axis=-1).T for idx in range(1, 4)
    ]

    time_arr = sc_B[0][0]

    data_arr = np.empty((4, 13, time_arr.size), dtype=float)
    for idx in range(4):
        data_arr[idx, :, :] = [
            sc_B[idx][1][0],
            sc_B[idx][1][1],
            sc_B[idx][1][2],
            np.linalg.norm(sc_B[idx][1][:3], axis=0),
            sc_v[idx][1][0],
            sc_v[idx][1][1],
            sc_v[idx][1][2],
            np.linalg.norm(sc_v[idx][1][:3], axis=0),
            sc_rho[idx][1],
            m_p * sc_rho[idx][1] * 1e6 * sc_v[idx][1][0] * sc_v[idx][1][0] * 1e6 / 1e-9,
            m_p
            * sc_rho[idx][1]
            * 1e6
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * np.linalg.norm(sc_v[idx][1][:3], axis=0)
            * 1e6
            / 1e-9,
            sc_tperp[idx][1] * 0.011606,
            sc_tpara[idx][1] * 0.011606,
        ]

    if mva:
        Bdata = [deepcopy(data_arr[idx, 0:3, :]) for idx in range(4)]
        vdata = [deepcopy(data_arr[idx, 4:7, :]) for idx in range(4)]
        eigenvecs = [MVA(Bdata[idx]) for idx in range(4)]
        for prob in range(4):
            print(
                "MMS{} Minimum Variance direction: {}".format(
                    prob + 1, eigenvecs[prob][0]
                )
            )
            for idx in range(3):
                data_arr[prob, idx, :] = np.dot(Bdata[prob].T, eigenvecs[prob][idx])
                data_arr[prob, idx + 4, :] = np.dot(vdata[prob].T, eigenvecs[prob][idx])

    t_pdmax = [time_arr[np.argmax(data_arr[idx, 9])] for idx in range(4)]

    # rel_pos = [
    #     sc_pos[idx][1].T[np.argmax(data_arr[idx, 9])]
    #     - sc_pos[0][1].T[np.argmax(data_arr[0, 9])]
    #     for idx in range(1, 4)
    # ]

    panel_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4]
    panel_labs = ["B [nT]", "V [km/s]", "n [1/cm3]", "Pdyn [nPa]", "T [MK]"]
    ylabels_all = [
        "Bx [nT]",
        "By [nT]",
        "Bz [nT]",
        "B [nT]",
        "vx [km/s]",
        "vy [km/s]",
        "vz [km/s]",
        "v [km/s]",
        "n [1/cm3]",
        "Pdynx [nPa]",
        "Pdyn [nPa]",
        "TPerp [MK]",
        "TPara [MK]",
    ]
    if mva:
        ylabels_all = [
            "BN [nT]",
            "BM [nT]",
            "BL [nT]",
            "Bt [nT]",
            "vN [km/s]",
            "vM [km/s]",
            "vL [km/s]",
            "vt [km/s]",
            "n [1/cm3]",
            "Pdynx [nPa]",
            "Pdyn [nPa]",
            "TPerp [MK]",
            "TPara [MK]",
        ]
    sc_labs = ["MMS1", "MMS2", "MMS3", "MMS4"]
    colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]
    plot_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    pan1_leg = [True, True, False, True, True]
    line_label = [
        "x",
        "y",
        "z",
        "mag",
        "x",
        "y",
        "z",
        "mag",
        None,
        "x",
        "mag",
        "Tperp",
        "Tpar",
    ]
    ylims = [
        (-40, 60),
        (-400, 500),
        (5, 35),
        (0, 10),
    ]
    ylims_full = [
        (-20, 30),
        (-20, 40),
        (-40, 50),
        (0, 60),
        (-400, 0),
        (-200, 100),
        (-100, 300),
        (0, 500),
        (5, 35),
        (0, 10),
        (0, 10),
    ]

    fig, ax_list = plt.subplots(
        len(panel_labs), 4, figsize=(24, 18), sharey="row", constrained_layout=True
    )

    for idx in range(4):
        for idx2 in range(len(panel_id)):
            print("Plotting {} {}".format(sc_labs[idx], panel_labs[panel_id[idx2]]))
            # ax = ax_list[panel_id[idx2], idx]
            # if not plot_legend[idx2]:
            #     ax.grid()
            # ax.plot(
            #     time_arr[idx, panel_id[idx2]],
            #     data_arr[idx, idx2],
            #     color=colors[idx2],
            #     label=line_label[idx2],
            #     alpha=0.5,
            # )
            # if plot_legend[idx2] and idx == 2:
            #     ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
            ax = ax_list[panel_id[idx2], idx]
            ax.plot(
                time_arr,
                data_arr[idx, idx2],
                color=colors[idx2],
                label=line_label[idx2],
                alpha=0.8,
            )
            ax.label_outer()
            ax.set_xlim(t0plot, t1plot)
            # if plot_legend[idx2]:
            #     ax.legend()
            # ax.axvline(t_pdmax[idx], linestyle="dashed")

    print("Times of Pdynmax: {}".format(t_pdmax))

    for idx in range(4):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
    # for idx in range(len(panel_labs)):
    #     ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
    #     ax_list[idx, 0].set_ylim(ylims[idx][0], ylims[idx][1])
    for idx in range(len(panel_labs)):
        ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
        # ax_list[idx, 0].set_ylim(ylims_full[idx][0], ylims_full[idx][1])
        if pan1_leg[idx]:
            ax_list[idx, 0].legend()
    for ax in ax_list.flatten():
        ax.grid()

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir
        + "mms_all_t0{}_t1{}_mva{}_peak{}_species{}.png".format(
            t0plot, t1plot, mva, peakonly, species
        )
    )
    plt.close(fig)

    print(rel_pos)

    labs = [
        "Bx:",
        "By:",
        "Bz:",
        "Bt:",
        "Vx:",
        "Vy:",
        "Vz:",
        "Vt:",
        "rho:",
        "Pdyn:",
    ]
    grads = [True, True, True, True, True, True, True, True, True, False]
    labs_v = ["Vx:", "Vy:", "Vz:"]
    if mva:
        labs = [
            "BN:",
            "BM:",
            "BL:",
            "Bt:",
            "VN:",
            "VM:",
            "VL:",
            "Vt:",
            "rho:",
            "Pdyn:",
        ]
        labs_v = ["Vmin:", "Vmed:", "Vmax:"]

    print("\n")

    timing_res = []

    for idx in range(10):
        print(labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr, time_arr],
            [
                data_arr[0][idx],
                data_arr[1][idx],
                data_arr[2][idx],
                data_arr[3][idx],
            ],
            rel_pos,
            t0,
            t1,
            peakonly=peakonly,
            gradient=grads[idx],
        )
        timing_res.append(res)
        print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = labs
    for idx in range(len(labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )
    if mva:
        rowLabels += sc_labs
        for idx in range(len(sc_labs)):
            cellText.append(
                [
                    str(
                        (
                            round(eigenvecs[idx][0][0], ndigits=2),
                            round(eigenvecs[idx][0][1], ndigits=2),
                            round(eigenvecs[idx][0][2], ndigits=2),
                        )
                    ),
                    "",
                    "",
                ]
            )

    # for idx in range(len(cellText)):
    #     for idx2 in range(len(cellText[0])):
    #         cellText[idx][idx2] = cellText[idx][idx2][:5]

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        + "mms_all_t0{}_t1{}_mva{}_peak{}_species{}_table.png".format(
            t0plot, t1plot, mva, peakonly, species
        )
    )
    plt.close(fig)


def diag_sc_mva(
    sc, probe, t0, t1, dt=1, grain=1, datarate="srvy", cutoff=0.9, maxwidth=None
):

    sc_list = ["ace", "dscovr", "wind", "themis", "mms", "cluster"]
    Bobj_list = [
        pyspedas.ace.mfi,
        pyspedas.dscovr.mag,
        pyspedas.wind.mfi,
        pyspedas.themis.fgm,
        pyspedas.mms.fgm,
        pyspedas.cluster.fgm,
    ]

    Bobj = Bobj_list[sc_list.index(sc)]

    dtp = "h0"
    if sc in ["ace", "dscovr", "wind"]:
        dtp = ["h3", "h0", "h2"][["ace", "dscovr", "wind"].index(sc)]

    time_arr, B = load_msh_sc_data(
        Bobj,
        sc,
        probe,
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datarate=datarate,
        datatype=dtp,
    )

    window_center = np.arange(0, time_arr.size, grain, dtype=int)
    window_halfwidth = np.arange(10, int(time_arr.size / 2), grain, dtype=int)
    window_size = (window_halfwidth * 2 * dt).astype(int)
    print(
        "Window center size: {}, window halfwidth size: {}, Time arr grain size: {}".format(
            window_center.size, window_halfwidth.size, time_arr[0::grain].size
        )
    )
    diag_data = np.empty((window_center.size, window_halfwidth.size), dtype=float)
    diag_vec_data = np.empty(
        (window_center.size, window_halfwidth.size, 3), dtype=float
    )
    diag_maxvec_data = np.empty(
        (window_center.size, window_halfwidth.size, 3), dtype=float
    )
    diag2_data = np.empty((window_center.size, window_halfwidth.size), dtype=float)

    for idx2 in range(window_center.size):
        for idx3 in range(window_halfwidth.size):
            start_id = max(window_center[idx2] - window_halfwidth[idx3], 0)
            stop_id = min(
                window_center[idx2] + window_halfwidth[idx3] + 1, time_arr.size
            )
            print(
                "Window center: {}, window halfwidth: {}, start id: {}, stop id: {}".format(
                    window_center[idx2], window_halfwidth[idx3], start_id, stop_id
                )
            )
            eigvals, eigvecs = MVA(B[:, start_id:stop_id], eigvals=True, prnt=False)
            diag_data[idx2, idx3] = eigvals[2] - eigvals[0]
            diag_vec_data[idx2, idx3, :] = eigvecs[0] * np.sign(eigvecs[0][0])
            diag_maxvec_data[idx2, idx3, :] = eigvecs[2]
            diag2_data[idx2, idx3] = eigvals[2] - eigvals[1]

    fig, ax = plt.subplots(5, 1, figsize=(8, 15), constrained_layout=True)
    im = ax[0].pcolormesh(
        time_arr[0::grain],
        window_size,
        diag_data.T,
        shading="gouraud",
        cmap="hot_desaturated",
    )
    mva_cutoff = cutoff * np.max(diag_data)
    ax[0].contour(
        time_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
    )
    plt.colorbar(im, ax=ax[0])
    ax[0].set_ylabel("Window width [s]")
    ax[0].set_title("$\\lambda_3-\\lambda_1$")

    im = ax[1].pcolormesh(
        time_arr[0::grain],
        window_size,
        diag2_data.T,
        shading="gouraud",
        cmap="hot_desaturated",
    )
    ax[1].contour(
        time_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
    )
    plt.colorbar(im, ax=ax[1])
    ax[1].set_ylabel("Window width [s]")
    ax[1].set_title("$\\lambda_3-\\lambda_2$")

    for idx in range(3):
        im = ax[idx + 2].pcolormesh(
            time_arr[0::grain],
            window_size,
            diag_vec_data[:, :, idx].T,
            shading="gouraud",
            cmap="vik",
            vmin=-1,
            vmax=1,
        )
        ax[idx + 2].contour(
            time_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
        )
        plt.colorbar(im, ax=ax[idx + 2])
        ax[idx + 2].set_ylabel("Window width [s]")
        ax[idx + 2].set_title(["$n_x$", "$n_y$", "$n_z$"][idx])

    ax[-1].set_xlabel("Window center")
    fig.savefig(
        wrkdir_DNR + "Figs/satellite/{}{}_diag_mva.png".format(sc, probe), dpi=150
    )
    plt.close(fig)

    CmeshCW, WmeshCW = np.meshgrid(window_center, window_size)

    if maxwidth:
        indcs = np.where(np.logical_and(WmeshCW < maxwidth, diag_data.T >= mva_cutoff))
    else:
        indcs = np.where(diag_data.T == np.max(diag_data.T))
    if indcs[0].size == 1:
        i, j = np.array(indcs).flatten()
    else:
        i, j = np.array(indcs).T[0]

    print("\nMaxvec: {}\n".format(diag_maxvec_data[j, i, :]))

    return (diag_vec_data[j, i, :], time_arr[0::grain][j], window_size[i])


def diag_thd_mms1_c4(t0, t1, dt=1, sc_order=[0, 1, 2], grain=1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/c4_mms1_thd_pos_2022-03-27_21:00:00_21:30:00_numpy.txt",
        dtype="str",
    ).T
    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * r_e * 1e-3
    sc_y = pos_data[5].astype(float) * r_e * 1e-3
    sc_z = pos_data[6].astype(float) * r_e * 1e-3

    sc_coords = np.array([sc_x, sc_y, sc_z]).T

    thd_pos = sc_coords[sc_name == "themisd", :][:-1]
    mms1_pos = sc_coords[sc_name == "mms1", :]
    c4_pos = sc_coords[sc_name == "cluster4", :]

    sc_poses = [thd_pos, mms1_pos, c4_pos]

    sc_rel_pos = [
        np.nanmean(sc_poses[sc_order[1]] - sc_poses[sc_order[0]], axis=0),
        np.nanmean(sc_poses[sc_order[2]] - sc_poses[sc_order[0]], axis=0),
    ]
    print(sc_rel_pos)

    thd_time, thd_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "d", "B", t0, t1, intpol=True, dt=dt
    )
    mms1_time, mms1_B = load_msh_sc_data(
        pyspedas.mms.fgm, "mms", "1", "B", t0, t1, intpol=True, dt=dt, datarate="srvy"
    )
    c4_time, c4_B = load_msh_sc_data(
        pyspedas.cluster.fgm, "cluster", "4", "B", t0, t1, intpol=True, dt=dt
    )

    thd_time2, thd_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "rho", t0, t1, intpol=True, dt=dt
    )
    mms1_time2, mms1_rho = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "rho", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    c4_time2, c4_rho = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "rho", t0, t1, intpol=True, dt=dt
    )

    dummy, thd_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, mms1_v = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "v", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    dummy, c4_v = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "v", t0, t1, intpol=True, dt=dt
    )

    thd_vmag = np.linalg.norm(thd_v, axis=0)
    mms1_vmag = np.linalg.norm(mms1_v, axis=0)
    c4_vmag = np.linalg.norm(c4_v, axis=0)

    thd_Bmag = np.linalg.norm(thd_B[0:3], axis=0)
    mms1_Bmag = np.linalg.norm(mms1_B[0:3], axis=0)
    c4_Bmag = np.linalg.norm(c4_B[0:3], axis=0)

    thd_pdyn = m_p * thd_rho * 1e6 * thd_vmag * thd_vmag * 1e6 / 1e-9
    mms1_pdyn = m_p * mms1_rho * 1e6 * mms1_vmag * mms1_vmag * 1e6 / 1e-9
    c4_pdyn = m_p * c4_rho * 1e6 * c4_vmag * c4_vmag * 1e6 / 1e-9

    time_arr = thd_time

    data_arr = np.empty((3, 10, time_arr.size), dtype=float)

    data_arr[0, :, :] = [
        thd_B[0],
        thd_B[1],
        thd_B[2],
        thd_Bmag,
        thd_v[0],
        thd_v[1],
        thd_v[2],
        thd_vmag,
        thd_rho,
        thd_pdyn,
    ]
    data_arr[1, :, :] = [
        mms1_B[0],
        mms1_B[1],
        mms1_B[2],
        mms1_Bmag,
        mms1_v[0],
        mms1_v[1],
        mms1_v[2],
        mms1_vmag,
        mms1_rho,
        mms1_pdyn,
    ]
    data_arr[2, :, :] = [
        c4_B[0],
        c4_B[1],
        c4_B[2],
        c4_Bmag,
        c4_v[0],
        c4_v[1],
        c4_v[2],
        c4_vmag,
        c4_rho,
        c4_pdyn,
    ]
    t_pdmax = [time_arr[np.argmax(data_arr[idx, 9, :])] for idx in range(3)]

    sc_labs = ["THD", "MMS1", "C4"]

    window_center = np.arange(0, time_arr.size, grain, dtype=int)
    window_halfwidth = np.arange(10, int(time_arr.size / 2), grain, dtype=int)
    window_size = (window_halfwidth * 2 * dt).astype(int)
    print(
        "Window center size: {}, window halfwidth size: {}, Time arr grain size: {}".format(
            window_center.size, window_halfwidth.size, time_arr[0::grain].size
        )
    )
    idcs = [0, 1, 2, 4, 9]
    diag_data = np.empty(
        (len(idcs), window_center.size, window_halfwidth.size), dtype=float
    )
    labs = ["Bx:", "By:", "Bz:", "Bt:", "Vx:", "Vy:", "Vz:", "Vt:", "rho:", "Pdyn:"]

    for idx2 in range(window_center.size):
        for idx3 in range(window_halfwidth.size):
            start_id = max(window_center[idx2] - window_halfwidth[idx3], 0)
            stop_id = min(
                window_center[idx2] + window_halfwidth[idx3] + 1, time_arr.size
            )
            for idx1 in range(len(idcs)):
                print(
                    "Window center: {}, window halfwidth: {}, start id: {}, stop id: {}".format(
                        window_center[idx2], window_halfwidth[idx3], start_id, stop_id
                    )
                )
                var_id = idcs[idx1]
                res = timing_analysis_arb(
                    [
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                        time_arr[start_id:stop_id],
                    ],
                    [
                        data_arr[sc_order[0], var_id, start_id:stop_id],
                        data_arr[sc_order[1], var_id, start_id:stop_id],
                        data_arr[sc_order[2], var_id, start_id:stop_id],
                    ],
                    sc_rel_pos,
                    t0,
                    t1,
                    prnt=False,
                )
                diag_data[idx1, idx2, idx3] = np.min(res["cross_corr_values"])

    fig, ax_list = plt.subplots(
        len(idcs), 1, figsize=(8, 3 * len(idcs)), constrained_layout=True
    )
    ims = []
    cbs = []
    for idx in range(len(idcs)):
        im = ax_list[idx].pcolormesh(
            time_arr[0::grain],
            window_size,
            diag_data[idx].T,
            shading="gouraud",
            cmap="hot_desaturated",
            vmin=0,
            vmax=1,
        )
        ims.append(im)
        cbs.append(plt.colorbar(ims[-1], ax=ax_list[idx]))
        ax_list[idx].set_title(labs[idcs[idx]])
        ax_list[idx].set_ylabel("Window width [s]")
    ax_list[-1].set_xlabel("Window center")

    fig.savefig(wrkdir_DNR + "Figs/satellite/thd_mms1_c4_diag_corr.png", dpi=150)
    plt.close(fig)


def plot_thd_the_tha_mms1(t0, t1, dt=1, mva=False, sc_order=[0, 1, 2, 3]):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    # pos_data = np.loadtxt(
    #     wrkdir_DNR
    #     + "satellites/c4_mms1_thd_pos_2022-03-27_21:00:00_21:30:00_numpy.txt",
    #     dtype="str",
    # ).T
    # sc_name = pos_data[3]
    # sc_x = pos_data[4].astype(float) * r_e * 1e-3
    # sc_y = pos_data[5].astype(float) * r_e * 1e-3
    # sc_z = pos_data[6].astype(float) * r_e * 1e-3

    # sc_coords = np.array([sc_x, sc_y, sc_z]).T

    # thd_pos = sc_coords[sc_name == "themisd", :][:-1]
    # mms1_pos = sc_coords[sc_name == "mms1", :]
    # c4_pos = sc_coords[sc_name == "cluster4", :]

    dummy_time, thd_pos = load_msh_sc_data(
        pyspedas.themis.state, "themis", "d", "pos", t0, t1, intpol=True, dt=dt
    )
    # thd_pos = np.nanmean(thd_pos,axis=-1)

    dummy_time, the_pos = load_msh_sc_data(
        pyspedas.themis.state, "themis", "e", "pos", t0, t1, intpol=True, dt=dt
    )
    # the_pos = np.nanmean(the_pos,axis=-1)

    dummy_time, tha_pos = load_msh_sc_data(
        pyspedas.themis.state, "themis", "a", "pos", t0, t1, intpol=True, dt=dt
    )
    # tha_pos = np.nanmean(tha_pos,axis=-1)

    dummy_time, mms1_pos = load_msh_sc_data(
        pyspedas.mms.mec, "mms", "1", "pos", t0, t1, intpol=True, dt=dt, datarate="srvy"
    )
    # tha_pos = np.nanmean(tha_pos,axis=-1)

    sc_poses = [thd_pos, the_pos, tha_pos, mms1_pos]

    sc_rel_pos = [
        np.nanmean(sc_poses[sc_order[1]] - sc_poses[sc_order[0]], axis=-1),
        np.nanmean(sc_poses[sc_order[2]] - sc_poses[sc_order[0]], axis=-1),
        np.nanmean(sc_poses[sc_order[3]] - sc_poses[sc_order[0]], axis=-1),
    ]
    print(sc_rel_pos)

    thd_time, thd_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "d", "B", t0, t1, intpol=True, dt=dt
    )
    the_time, the_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "e", "B", t0, t1, intpol=True, dt=dt
    )
    tha_time, tha_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "a", "B", t0, t1, intpol=True, dt=dt
    )
    mms1_time, mms1_B = load_msh_sc_data(
        pyspedas.mms.fgm, "mms", "1", "B", t0, t1, intpol=True, dt=dt, datarate="srvy"
    )

    thd_time2, thd_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "rho", t0, t1, intpol=True, dt=dt
    )
    the_time2, the_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "e", "rho", t0, t1, intpol=True, dt=dt
    )
    tha_time2, tha_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "a", "rho", t0, t1, intpol=True, dt=dt
    )
    mms1_time2, mms1_rho = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "rho", t0, t1, intpol=True, dt=dt, datarate="fast"
    )

    dummy, thd_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, the_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "e", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, tha_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "a", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, mms1_v = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "v", t0, t1, intpol=True, dt=dt, datarate="fast"
    )

    thd_vmag = np.linalg.norm(thd_v, axis=0)
    the_vmag = np.linalg.norm(the_v, axis=0)
    tha_vmag = np.linalg.norm(tha_v, axis=0)
    mms1_vmag = np.linalg.norm(mms1_v, axis=0)

    thd_Bmag = np.linalg.norm(thd_B[0:3], axis=0)
    the_Bmag = np.linalg.norm(the_B[0:3], axis=0)
    tha_Bmag = np.linalg.norm(tha_B[0:3], axis=0)
    mms1_Bmag = np.linalg.norm(mms1_B[0:3], axis=0)

    thd_pdyn = m_p * thd_rho * 1e6 * thd_vmag * thd_vmag * 1e6 / 1e-9
    the_pdyn = m_p * the_rho * 1e6 * the_vmag * the_vmag * 1e6 / 1e-9
    tha_pdyn = m_p * tha_rho * 1e6 * tha_vmag * tha_vmag * 1e6 / 1e-9
    mms1_pdyn = m_p * mms1_rho * 1e6 * mms1_vmag * mms1_vmag * 1e6 / 1e-9

    time_arr = thd_time

    data_arr = np.empty((4, 10, time_arr.size), dtype=float)

    data_arr[0, :, :] = [
        thd_B[0],
        thd_B[1],
        thd_B[2],
        thd_Bmag,
        thd_v[0],
        thd_v[1],
        thd_v[2],
        thd_vmag,
        thd_rho,
        thd_pdyn,
    ]
    data_arr[1, :, :] = [
        the_B[0],
        the_B[1],
        the_B[2],
        the_Bmag,
        the_v[0],
        the_v[1],
        the_v[2],
        the_vmag,
        the_rho,
        the_pdyn,
    ]
    data_arr[2, :, :] = [
        tha_B[0],
        tha_B[1],
        tha_B[2],
        tha_Bmag,
        tha_v[0],
        tha_v[1],
        tha_v[2],
        tha_vmag,
        tha_rho,
        tha_pdyn,
    ]
    data_arr[3, :, :] = [
        mms1_B[0],
        mms1_B[1],
        mms1_B[2],
        mms1_Bmag,
        mms1_v[0],
        mms1_v[1],
        mms1_v[2],
        mms1_vmag,
        mms1_rho,
        mms1_pdyn,
    ]

    t_pdmax = [time_arr[np.argmax(data_arr[idx, 9, :])] for idx in range(4)]

    sc_labs = ["THD", "THE", "THA", "MMS1"]
    if mva:
        Bdata = [deepcopy(data_arr[idx, 0:3, :]) for idx in range(4)]
        vdata = [deepcopy(data_arr[idx, 4:7, :]) for idx in range(4)]
        eigenvecs = [MVA(Bdata[idx]) for idx in range(4)]
        for prob in range(4):
            print(
                "{} Minimum Variance direction: {}".format(
                    sc_labs[prob], eigenvecs[prob][0]
                )
            )
            for idx in range(3):
                data_arr[prob, idx, :] = np.dot(Bdata[prob].T, eigenvecs[prob][idx])
                data_arr[prob, idx + 4, :] = np.dot(vdata[prob].T, eigenvecs[prob][idx])

    panel_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3]
    panel_labs = ["B [nT]", "V [km/s]", "n [1/cm3]", "Pdyn [nPa]"]
    ylabels_all = [
        "Bx [nT]",
        "By [nT]",
        "Bz [nT]",
        "Bt [nT]",
        "vx [km/s]",
        "vy [km/s]",
        "vz [km/s]",
        "vt [km/s]",
        "n [1/cm3]",
        "Pdyn [nPa]",
    ]
    if mva:
        ylabels_all = [
            "BN [nT]",
            "BM [nT]",
            "BL [nT]",
            "Bt [nT]",
            "vN [km/s]",
            "vM [km/s]",
            "vL [km/s]",
            "vt [km/s]",
            "n [1/cm3]",
            "Pdyn [nPa]",
        ]

    colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        "k",
    ]
    plot_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    line_label = [
        "x",
        "y",
        "z",
        "mag",
        "x",
        "y",
        "z",
        "mag",
        None,
        None,
    ]
    ylims = [
        (-40, 60),
        (-400, 500),
        (5, 35),
        (0, 10),
    ]
    ylims_full = [
        (-20, 30),
        (-20, 40),
        (-40, 50),
        (0, 60),
        (-400, 0),
        (-200, 100),
        (-100, 300),
        (0, 500),
        (5, 35),
        (0, 10),
    ]

    fig, ax_list = plt.subplots(
        10, 4, figsize=(24, 24), sharey="row", constrained_layout=True
    )

    for idx in range(4):
        for idx2 in range(len(panel_id)):
            print("Plotting {} {}".format(sc_labs[idx], panel_labs[panel_id[idx2]]))
            # ax = ax_list[panel_id[idx2], idx]
            # if not plot_legend[idx2]:
            #     ax.grid()
            # ax.plot(
            #     time_arr[idx, panel_id[idx2]],
            #     data_arr[idx, idx2],
            #     color=colors[idx2],
            #     label=line_label[idx2],
            #     alpha=0.5,
            # )
            # if plot_legend[idx2] and idx == 2:
            #     ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
            ax = ax_list[idx2, idx]
            ax.grid()
            ax.plot(
                time_arr,
                data_arr[idx, idx2],
            )
            ax.label_outer()
            ax.set_xlim(t0plot, t1plot)
            ax.axvline(t_pdmax[idx], linestyle="dashed")

    print("Times of Pdynmax: {}".format(t_pdmax))

    for idx in range(4):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
    # for idx in range(len(panel_labs)):
    #     ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
    #     ax_list[idx, 0].set_ylim(ylims[idx][0], ylims[idx][1])
    for idx in range(len(ylabels_all)):
        ax_list[idx, 0].set_ylabel(ylabels_all[idx], labelpad=10, fontsize=20)
        # ax_list[idx, 0].set_ylim(ylims_full[idx][0], ylims_full[idx][1])

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir + "thd_the_tha_mms1_t0{}_t1{}_mva{}.png".format(t0plot, t1plot, mva)
    )
    plt.close(fig)

    labs = ["Bx:", "By:", "Bz:", "Bt:", "Vx:", "Vy:", "Vz:", "Vt:", "rho", "Pdyn:"]
    # labs_v = ["Vx:", "Vy:", "Vz:"]
    if mva:
        # labs = ["Bmin:", "Bmed:", "Bmax:"]
        # labs_v = ["Vmin:", "Vmed:", "Vmax:"]
        labs = [
            "BN:",
            "BM:",
            "BL:",
            "Bt:",
            "VN:",
            "VM:",
            "VL:",
            "Vt:",
            "rho",
            "Pdyn:",
        ]

    print("\n")

    timing_res = []

    for idx in range(10):
        print(labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr, time_arr],
            [
                data_arr[sc_order[0], idx, :],
                data_arr[sc_order[1], idx, :],
                data_arr[sc_order[2], idx, :],
                data_arr[sc_order[3], idx, :],
            ],
            sc_rel_pos,
            t0,
            t1,
        )
        timing_res.append(res)
        print("\n")
    #     print("\n")
    #     print(labs_v[idx])
    #     timing_analysis_arb(
    #         [time_arr, time_arr, time_arr],
    #         [data_arr[0, 4 + idx, :], data_arr[1, 4 + idx, :], data_arr[2, 4 + idx, :]],
    #         sc_rel_pos,
    #         t0,
    #         t1,
    #     )
    #     print("\n")

    # print("rho:")
    # timing_analysis_arb(
    #     [time_arr, time_arr, time_arr],
    #     [data_arr[0, 8, :], data_arr[1, 8, :], data_arr[2, 8, :]],
    #     sc_rel_pos,
    #     t0,
    #     t1,
    # )
    # print("\n")

    # print("Pdyn:")
    # timing_analysis_arb(
    #     [time_arr, time_arr, time_arr],
    #     [data_arr[0, 9, :], data_arr[1, 9, :], data_arr[2, 9, :]],
    #     sc_rel_pos,
    #     t0,
    #     t1,
    # )
    # print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = labs
    for idx in range(len(labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )
    if mva:
        rowLabels += sc_labs
        for idx in range(len(sc_labs)):
            cellText.append(
                [
                    str(
                        (
                            round(eigenvecs[idx][0][0], ndigits=2),
                            round(eigenvecs[idx][0][1], ndigits=2),
                            round(eigenvecs[idx][0][2], ndigits=2),
                        )
                    ),
                    "",
                    "",
                ]
            )

    # for idx in range(len(cellText)):
    #     for idx2 in range(len(cellText[0])):
    #         cellText[idx][idx2] = cellText[idx][idx2][:5]

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        + "thd_the_tha_mms1_t0{}_t1{}_mva{}_table.png".format(t0plot, t1plot, mva)
    )
    plt.close(fig)


def plot_thd_mms1_c4(t0, t1, dt=1, mva=False, sc_order=[0, 1, 2]):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/c4_mms1_thd_pos_2022-03-27_21:00:00_21:30:00_numpy.txt",
        dtype="str",
    ).T
    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * r_e * 1e-3
    sc_y = pos_data[5].astype(float) * r_e * 1e-3
    sc_z = pos_data[6].astype(float) * r_e * 1e-3

    sc_coords = np.array([sc_x, sc_y, sc_z]).T

    thd_pos = sc_coords[sc_name == "themisd", :][:-1]
    mms1_pos = sc_coords[sc_name == "mms1", :]
    c4_pos = sc_coords[sc_name == "cluster4", :]

    sc_poses = [thd_pos, mms1_pos, c4_pos]

    sc_rel_pos = [
        np.nanmean(sc_poses[sc_order[1]] - sc_poses[sc_order[0]], axis=0),
        np.nanmean(sc_poses[sc_order[2]] - sc_poses[sc_order[0]], axis=0),
    ]
    print(sc_rel_pos)

    thd_time, thd_B = load_msh_sc_data(
        pyspedas.themis.fgm, "themis", "d", "B", t0, t1, intpol=True, dt=dt
    )
    mms1_time, mms1_B = load_msh_sc_data(
        pyspedas.mms.fgm, "mms", "1", "B", t0, t1, intpol=True, dt=dt, datarate="srvy"
    )
    c4_time, c4_B = load_msh_sc_data(
        pyspedas.cluster.fgm, "cluster", "4", "B", t0, t1, intpol=True, dt=dt
    )

    thd_time2, thd_rho = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "rho", t0, t1, intpol=True, dt=dt
    )
    mms1_time2, mms1_rho = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "rho", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    c4_time2, c4_rho = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "rho", t0, t1, intpol=True, dt=dt
    )

    dummy, thd_v = load_msh_sc_data(
        pyspedas.themis.mom, "themis", "d", "v", t0, t1, intpol=True, dt=dt
    )
    dummy, mms1_v = load_msh_sc_data(
        pyspedas.mms.fpi, "mms", "1", "v", t0, t1, intpol=True, dt=dt, datarate="fast"
    )
    dummy, c4_v = load_msh_sc_data(
        pyspedas.cluster.cis, "cluster", "4", "v", t0, t1, intpol=True, dt=dt
    )

    thd_vmag = np.linalg.norm(thd_v, axis=0)
    mms1_vmag = np.linalg.norm(mms1_v, axis=0)
    c4_vmag = np.linalg.norm(c4_v, axis=0)

    thd_Bmag = np.linalg.norm(thd_B[0:3], axis=0)
    mms1_Bmag = np.linalg.norm(mms1_B[0:3], axis=0)
    c4_Bmag = np.linalg.norm(c4_B[0:3], axis=0)

    thd_pdyn = m_p * thd_rho * 1e6 * thd_vmag * thd_vmag * 1e6 / 1e-9
    mms1_pdyn = m_p * mms1_rho * 1e6 * mms1_vmag * mms1_vmag * 1e6 / 1e-9
    c4_pdyn = m_p * c4_rho * 1e6 * c4_vmag * c4_vmag * 1e6 / 1e-9

    time_arr = thd_time

    data_arr = np.empty((3, 10, time_arr.size), dtype=float)

    data_arr[0, :, :] = [
        thd_B[0],
        thd_B[1],
        thd_B[2],
        thd_Bmag,
        thd_v[0],
        thd_v[1],
        thd_v[2],
        thd_vmag,
        thd_rho,
        thd_pdyn,
    ]
    data_arr[1, :, :] = [
        mms1_B[0],
        mms1_B[1],
        mms1_B[2],
        mms1_Bmag,
        mms1_v[0],
        mms1_v[1],
        mms1_v[2],
        mms1_vmag,
        mms1_rho,
        mms1_pdyn,
    ]
    data_arr[2, :, :] = [
        c4_B[0],
        c4_B[1],
        c4_B[2],
        c4_Bmag,
        c4_v[0],
        c4_v[1],
        c4_v[2],
        c4_vmag,
        c4_rho,
        c4_pdyn,
    ]
    t_pdmax = [time_arr[np.argmax(data_arr[idx, 9, :])] for idx in range(3)]

    sc_labs = ["THD", "MMS1", "C4"]
    if mva:
        Bdata = [deepcopy(data_arr[idx, 0:3, :]) for idx in range(3)]
        vdata = [deepcopy(data_arr[idx, 4:7, :]) for idx in range(3)]
        eigenvecs = [MVA(Bdata[idx]) for idx in range(3)]
        for prob in range(3):
            print(
                "{} Minimum Variance direction: {}".format(
                    sc_labs[prob], eigenvecs[prob][0]
                )
            )
            for idx in range(3):
                data_arr[prob, idx, :] = np.dot(Bdata[prob].T, eigenvecs[prob][idx])
                data_arr[prob, idx + 4, :] = np.dot(vdata[prob].T, eigenvecs[prob][idx])

    panel_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3]
    panel_labs = ["B [nT]", "V [km/s]", "n [1/cm3]", "Pdyn [nPa]"]
    ylabels_all = [
        "Bx [nT]",
        "By [nT]",
        "Bz [nT]",
        "Bt [nT]",
        "vx [km/s]",
        "vy [km/s]",
        "vz [km/s]",
        "vt [km/s]",
        "n [1/cm3]",
        "Pdyn [nPa]",
    ]
    if mva:
        ylabels_all = [
            "BN [nT]",
            "BM [nT]",
            "BL [nT]",
            "Bt [nT]",
            "vN [km/s]",
            "vM [km/s]",
            "vL [km/s]",
            "vt [km/s]",
            "n [1/cm3]",
            "Pdyn [nPa]",
        ]

    colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        "k",
    ]
    plot_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    line_label = [
        "x",
        "y",
        "z",
        "mag",
        "x",
        "y",
        "z",
        "mag",
        None,
        None,
    ]
    ylims = [
        (-40, 60),
        (-400, 500),
        (5, 35),
        (0, 10),
    ]
    ylims_full = [
        (-20, 30),
        (-20, 40),
        (-40, 50),
        (0, 60),
        (-400, 0),
        (-200, 100),
        (-100, 300),
        (0, 500),
        (5, 35),
        (0, 10),
    ]

    fig, ax_list = plt.subplots(
        10, 3, figsize=(18, 24), sharey="row", constrained_layout=True
    )

    for idx in range(3):
        for idx2 in range(len(panel_id)):
            print("Plotting {} {}".format(sc_labs[idx], panel_labs[panel_id[idx2]]))
            # ax = ax_list[panel_id[idx2], idx]
            # if not plot_legend[idx2]:
            #     ax.grid()
            # ax.plot(
            #     time_arr[idx, panel_id[idx2]],
            #     data_arr[idx, idx2],
            #     color=colors[idx2],
            #     label=line_label[idx2],
            #     alpha=0.5,
            # )
            # if plot_legend[idx2] and idx == 2:
            #     ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
            ax = ax_list[idx2, idx]
            ax.grid()
            ax.plot(
                time_arr,
                data_arr[idx, idx2],
            )
            ax.label_outer()
            ax.set_xlim(t0plot, t1plot)
            ax.axvline(t_pdmax[idx], linestyle="dashed")

    print("Times of Pdynmax: {}".format(t_pdmax))

    for idx in range(3):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
    # for idx in range(len(panel_labs)):
    #     ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
    #     ax_list[idx, 0].set_ylim(ylims[idx][0], ylims[idx][1])
    for idx in range(len(ylabels_all)):
        ax_list[idx, 0].set_ylabel(ylabels_all[idx], labelpad=10, fontsize=20)
        # ax_list[idx, 0].set_ylim(ylims_full[idx][0], ylims_full[idx][1])

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(outdir + "thd_mms1_c4_t0{}_t1{}_mva{}.png".format(t0plot, t1plot, mva))
    plt.close(fig)

    labs = ["Bx:", "By:", "Bz:", "Bt:", "Vx:", "Vy:", "Vz:", "Vt:", "rho", "Pdyn:"]
    # labs_v = ["Vx:", "Vy:", "Vz:"]
    if mva:
        # labs = ["Bmin:", "Bmed:", "Bmax:"]
        # labs_v = ["Vmin:", "Vmed:", "Vmax:"]
        labs = [
            "BN:",
            "BM:",
            "BL:",
            "Bt:",
            "VN:",
            "VM:",
            "VL:",
            "Vt:",
            "rho",
            "Pdyn:",
        ]

    print("\n")

    timing_res = []

    for idx in range(10):
        print(labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr],
            [
                data_arr[sc_order[0], idx, :],
                data_arr[sc_order[1], idx, :],
                data_arr[sc_order[2], idx, :],
            ],
            sc_rel_pos,
            t0,
            t1,
        )
        timing_res.append(res)
        print("\n")
    #     print("\n")
    #     print(labs_v[idx])
    #     timing_analysis_arb(
    #         [time_arr, time_arr, time_arr],
    #         [data_arr[0, 4 + idx, :], data_arr[1, 4 + idx, :], data_arr[2, 4 + idx, :]],
    #         sc_rel_pos,
    #         t0,
    #         t1,
    #     )
    #     print("\n")

    # print("rho:")
    # timing_analysis_arb(
    #     [time_arr, time_arr, time_arr],
    #     [data_arr[0, 8, :], data_arr[1, 8, :], data_arr[2, 8, :]],
    #     sc_rel_pos,
    #     t0,
    #     t1,
    # )
    # print("\n")

    # print("Pdyn:")
    # timing_analysis_arb(
    #     [time_arr, time_arr, time_arr],
    #     [data_arr[0, 9, :], data_arr[1, 9, :], data_arr[2, 9, :]],
    #     sc_rel_pos,
    #     t0,
    #     t1,
    # )
    # print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = labs
    for idx in range(len(labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )
    if mva:
        rowLabels += sc_labs
        for idx in range(len(sc_labs)):
            cellText.append(
                [
                    str(
                        (
                            round(eigenvecs[idx][0][0], ndigits=2),
                            round(eigenvecs[idx][0][1], ndigits=2),
                            round(eigenvecs[idx][0][2], ndigits=2),
                        )
                    ),
                    "",
                    "",
                ]
            )

    # for idx in range(len(cellText)):
    #     for idx2 in range(len(cellText[0])):
    #         cellText[idx][idx2] = cellText[idx][idx2][:5]

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        outdir + "thd_mms1_c4_t0{}_t1{}_mva{}_table.png".format(t0plot, t1plot, mva)
    )
    plt.close(fig)


def sw_timing(t0, t1, dt=1, sc_order=[0, 1, 2, 3], filt=None):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    sc_labs = ["ACE", "DSCOVR", "Wind", "THB"]

    ace_time, ace_B = load_msh_sc_data(
        pyspedas.ace.mfi,
        "ace",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h3",
        filt=filt,
    )
    dscovr_time, dscovr_B = load_msh_sc_data(
        pyspedas.dscovr.mag,
        "dscovr",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h0",
        filt=filt,
    )
    wind_time, wind_B = load_msh_sc_data(
        pyspedas.wind.mfi,
        "wind",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h2",
        filt=filt,
    )
    thb_time, thb_B = load_msh_sc_data(
        pyspedas.themis.fgm,
        "themis",
        "b",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        filt=filt,
    )
    sc_B = [ace_B, dscovr_B, wind_B, thb_B]
    time_arr = ace_time

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/ace_dscovr_wind_pos_2022-03-27_19:00:00_21:00:00_numpy.txt",
        dtype="str",
    ).T
    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * r_e * 1e-3
    sc_y = pos_data[5].astype(float) * r_e * 1e-3
    sc_z = pos_data[6].astype(float) * r_e * 1e-3

    sc_coords = np.array([sc_x, sc_y, sc_z]).T

    ace_pos = sc_coords[sc_name == "ace", :][:-1]
    dscovr_pos = sc_coords[sc_name == "dscovr", :]
    wind_pos = sc_coords[sc_name == "wind", :][:-1]
    thb_pos = np.array([225214.85, -311541.9, -32746.94])

    sc_poses = [ace_pos, dscovr_pos, wind_pos, thb_pos]

    sc_rel_pos = [
        np.nanmean(sc_poses[sc_order[1]] - sc_poses[sc_order[0]], axis=0),
        np.nanmean(sc_poses[sc_order[2]] - sc_poses[sc_order[0]], axis=0),
        np.nanmean(sc_poses[sc_order[3]] - sc_poses[sc_order[0]], axis=0),
    ]
    print(sc_rel_pos)

    title_labs = ["Bx", "By", "Bz", "Bmag"]

    data_arr = np.zeros((4, 4, time_arr.size), dtype=float)

    for idx in range(4):
        data_arr[idx, :, :] = [
            sc_B[idx][0],
            sc_B[idx][1],
            sc_B[idx][2],
            np.linalg.norm(sc_B[idx][:3], axis=0),
        ]

    timing_res = []

    for idx in range(4):
        print(title_labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr, time_arr],
            [
                data_arr[sc_order[0], idx, :],
                data_arr[sc_order[1], idx, :],
                data_arr[sc_order[2], idx, :],
                data_arr[sc_order[3], idx, :],
            ],
            sc_rel_pos,
            t0,
            t1,
        )
        timing_res.append(res)
        print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = title_labs
    for idx in range(len(title_labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        wrkdir_DNR
        + "Figs/satellite/"
        + "ace_dscovr_wind_thb_t0{}_t1{}_table.png".format(t0plot, t1plot)
    )
    plt.close(fig)


def plot_ace_dscovr_wind(
    t0, t1, dt=1, sc_order=[0, 1, 2], mva=False, filt=None, rotate=[None, None, None]
):

    if type(rotate) is not np.ndarray:
        rotate = np.array(rotate)

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    sc_labs = ["ACE", "DSCOVR", "Wind"]

    ace_time, ace_B = load_msh_sc_data(
        pyspedas.ace.mfi,
        "ace",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h3",
        filt=filt,
    )
    omnidata = pyspedas.omni.data(trange=[t0, t1], notplot=True, time_clip=True)
    dummy_t, v_arr = intpol_data(
        omnidata["flow_speed"]["x"], omnidata["flow_speed"]["y"], t0, t1, dt
    )
    sw_speed = np.nanmean(v_arr)
    dscovr_time, dscovr_B = load_msh_sc_data(
        pyspedas.dscovr.mag,
        "dscovr",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h0",
        filt=filt,
    )
    wind_time, wind_B = load_msh_sc_data(
        pyspedas.wind.mfi,
        "wind",
        "1",
        "B",
        t0,
        t1,
        intpol=True,
        dt=dt,
        datatype="h2",
        filt=filt,
    )

    sc_B = [ace_B, dscovr_B, wind_B]

    time_arr = ace_time

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/ace_dscovr_wind_pos_2022-03-27_19:00:00_21:00:00_numpy.txt",
        dtype="str",
    ).T
    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * r_e * 1e-3
    sc_y = pos_data[5].astype(float) * r_e * 1e-3
    sc_z = pos_data[6].astype(float) * r_e * 1e-3

    sc_coords = np.array([sc_x, sc_y, sc_z]).T

    ace_pos = sc_coords[sc_name == "ace", :][:-1]
    dscovr_pos = sc_coords[sc_name == "dscovr", :]
    wind_pos = sc_coords[sc_name == "wind", :][:-1]

    sc_poses = [ace_pos, dscovr_pos, wind_pos]

    sc_rel_pos = [
        np.nanmean(sc_poses[sc_order[1]] - sc_poses[sc_order[0]], axis=0),
        np.nanmean(sc_poses[sc_order[2]] - sc_poses[sc_order[0]], axis=0),
    ]
    print(sc_rel_pos)

    data_arr = np.zeros((3, 4, time_arr.size), dtype=float)

    for idx in range(3):
        data_arr[idx, :, :] = [
            sc_B[idx][0],
            sc_B[idx][1],
            sc_B[idx][2],
            np.linalg.norm(sc_B[idx], axis=0),
        ]

    # print("NaNs detected:", np.isnan(data_arr).any())
    # print(data_arr)

    if mva:
        Bdata = [deepcopy(data_arr[idx, 0:3, :]) for idx in range(3)]
        eigenvecs = [MVA(Bdata[idx]) for idx in range(3)]
        for prob in range(3):
            print(
                "{} Minimum Variance direction: {}".format(
                    sc_labs[prob], eigenvecs[prob][0]
                )
            )
            for idx in range(3):
                data_arr[prob, idx, :] = np.dot(Bdata[prob].T, eigenvecs[prob][idx])

    title_labs = ["Bx", "By", "Bz", "Bmag"]
    if mva:
        title_labs = ["BN", "BM", "BL", "Bmag"]

    # ace_clock, dscovr_clock, wind_clock = [
    #     np.rad2deg(np.arctan2(B[2], B[1])) for B in [ace_B, dscovr_B, wind_B]
    # ]
    # ace_cone, dscovr_cone, wind_cone = [
    #     np.rad2deg(np.arctan2(np.sqrt(B[2] ** 2 + B[1] ** 2), B[0]))
    #     for B in [ace_B, dscovr_B, wind_B]
    # ]
    # ace_Bmag, dscovr_Bmag, wind_Bmag = [
    #     np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2) for B in [ace_B, dscovr_B, wind_B]
    # ]

    fig, ax_list = plt.subplots(
        4, 3, figsize=(18, 12), constrained_layout=True, sharey="row"
    )

    for idx in range(3):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
        for idx2 in range(4):
            ax = ax_list[idx2, idx]
            ax.plot(time_arr, data_arr[idx, idx2])
            ax.set_xlim(t0plot, t1plot)
            ax.grid()

    for ax in ax_list.flatten():
        ax.label_outer()

    for idx in range(4):
        ax_list[idx, 0].set_ylabel(title_labs[idx], labelpad=10, fontsize=20)

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir + "ace_dscovr_wind_t0{}_t1{}_mva{}.png".format(t0plot, t1plot, mva)
    )
    plt.close(fig)

    timing_res = []

    for idx in range(4):
        print(title_labs[idx])
        res = timing_analysis_arb(
            [time_arr, time_arr, time_arr],
            [
                data_arr[sc_order[0], idx, :],
                data_arr[sc_order[1], idx, :],
                data_arr[sc_order[2], idx, :],
            ],
            sc_rel_pos,
            t0,
            t1,
            sw_speed=sw_speed,
        )
        timing_res.append(res)
        print("\n")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    cellText = []
    colLabels = ["n", "v", "c"]
    rowLabels = title_labs
    for idx in range(len(title_labs)):
        res = timing_res[idx]
        cellText.append(
            [
                str(
                    (
                        round(res["wave_vector"][0][0], ndigits=2),
                        round(res["wave_vector"][1][0], ndigits=2),
                        round(res["wave_vector"][2][0], ndigits=2),
                    )
                ),
                str(round(res["wave_velocity_sc_frame"], ndigits=2)),
                str(round(np.min(res["cross_corr_values"]), ndigits=2)),
            ]
        )
    if mva:
        rowLabels += sc_labs
        for idx in range(len(sc_labs)):
            cellText.append(
                [
                    str(
                        (
                            round(eigenvecs[idx][0][0], ndigits=2),
                            round(eigenvecs[idx][0][1], ndigits=2),
                            round(eigenvecs[idx][0][2], ndigits=2),
                        )
                    ),
                    "",
                    "",
                ]
            )

    # for idx in range(len(cellText)):
    #     for idx2 in range(len(cellText[0])):
    #         cellText[idx][idx2] = cellText[idx][idx2][:5]

    ax.table(
        cellText=cellText,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
    )

    fig.tight_layout()

    fig.savefig(
        outdir + "ace_dscovr_wind_t0{}_t1{}_mva{}_table.png".format(t0plot, t1plot, mva)
    )
    plt.close(fig)
    if rotate.any():
        fig, ax_list = plt.subplots(
            1, 3, figsize=(18, 6), constrained_layout=True, sharey=True
        )
        for idx in range(3):
            ax = ax_list[idx]
            ax.plot(time_arr, np.dot(data_arr[idx, 0:3, :].T, rotate))
            ax.set_title(sc_labs[idx])
            ax.grid()
            ax.set_xlim(t0plot, t1plot)
        ax_list[0].set_ylabel("BN")

        fig.savefig(
            outdir
            + "ace_dscovr_wind_t0{}_t1{}_rot{}.png".format(t0plot, t1plot, rotate)
        )
        plt.close(fig)


def timing_analysis_arb(
    sc_times,
    sc_data,
    sc_rel_pos,
    t0=None,
    t1=None,
    peakonly=False,
    gradient=False,
    prnt=True,
    bulkv=np.array([None, None, None]),
    sw_speed=None,
):
    # Adapted from code created by Lucile Turc

    # Inputs:
    # data is a list of dictionaries containing the virtual spacecraft time series
    # ind_sc selects one virtual spacecraft from the list of dictionaries
    # min_time and max_time indicate between which time steps we perform the timing analysis
    # var4analysis is a dictionary key which selects the time series on which timing analysis is performed
    # Output:
    # results is a dictionary containing the results of the timing analysis

    Re = 6371.0  # Earth radius in km
    time_difference = []
    # dt = 0.5  # Time step dt = 0.5 in all Vlasiator runs

    # ******************************************************************************#
    # Calculate the correlation function between two times series
    # ******************************************************************************#
    cross_corr_values = []

    # print(min_time,max_time)

    # ref_sc = ind_sc[0]

    # t0plot = (
    #     datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    #     .replace(tzinfo=timezone.utc)
    #     .timestamp()
    # )
    # t1plot = (
    #     datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")
    #     .replace(tzinfo=timezone.utc)
    #     .timestamp()
    # )

    sc_times_new = []
    # print(sc_times[0].size)

    for idx in range(len(sc_times)):
        sc_times_new.append(
            np.array(
                [t.replace(tzinfo=timezone.utc).timestamp() for t in sc_times[idx]]
            )
        )

    sc_pos_rel = deepcopy(sc_rel_pos)

    dt = sc_times_new[0][1] - sc_times_new[0][0]

    if gradient:
        for idx in range(len(sc_data)):
            sc_data[idx] = np.gradient(sc_data[idx])

    rel_sc_norm = (sc_data[0] - np.mean(sc_data[0])) / (
        np.std(sc_data[0], ddof=1) * sc_data[0].size
    )

    if peakonly:
        reftime = sc_times_new[0][np.argmax(np.abs(sc_data[0]))]
        for idx in range(1, len(sc_data)):
            time_difference.append(
                sc_times_new[idx][np.argmax(np.abs(sc_data[idx]))] - reftime
            )
            cross_corr_values.append(1)
    else:
        for sc in sc_data[1:]:
            sc_norm = (sc - np.mean(sc)) / np.std(sc, ddof=1)
            c = np.correlate(sc_norm, rel_sc_norm, "full")
            offset = np.argmax(c)
            alpha = c[offset - 1]
            beta = c[offset]
            gamma = c[offset + 1]
            offset2 = (
                offset
                - len(c) / 2.0
                + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            )
            if prnt:
                print("offset", offset, offset2)
            cross_corr_values.append(np.max(c))
            # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
            time_difference.append(offset2 * dt)

    if sw_speed:
        time_difference.append(100)
        sc_pos_rel.append(np.array([-sw_speed * 100, 0, 0]))
    print(sc_pos_rel)
    # # ******************************************************************************#

    time_difference = np.array(time_difference)

    matrix_positions = np.zeros((3, 3))

    for idx in range(len(sc_pos_rel)):
        matrix_positions[idx] = sc_pos_rel[idx]

    # We now invert the matrix of spacecraft relative positions and multiply it with the time lags in order to solve the system
    # of equations for the wave vector
    # The vector obtained from this operation is the wave vector divided by the phase velocity in the spacecraft frame

    result = np.dot(
        np.linalg.inv(matrix_positions[0 : len(sc_pos_rel), 0 : len(sc_pos_rel)]),
        time_difference[0 : len(sc_pos_rel)],
    )
    result.shape = (len(sc_pos_rel), 1)

    norm_result = np.linalg.norm(result)

    wave_velocity_sc_frame = 1.0 / norm_result

    wave_vector = np.zeros((3, 1))
    wave_vector[0 : len(sc_pos_rel)] = result / norm_result

    results = {}
    results["wave_vector"] = wave_vector
    results["wave_velocity_sc_frame"] = wave_velocity_sc_frame
    results["cross_corr_values"] = cross_corr_values

    predicted_time_lags = (
        np.array([np.dot(wave_vector.flatten(), distance) for distance in sc_pos_rel])
        / wave_velocity_sc_frame
    )
    if bulkv.any():
        Vpl = wave_velocity_sc_frame - np.dot(bulkv, wave_vector)
        wave_velocity_relative2sc = (
            bulkv.reshape((3, 1))
            + (wave_velocity_sc_frame - np.dot(bulkv, wave_vector)) * wave_vector
        )
        wave_velocity_relative2sc.shape = 3
        if prnt:
            print("Wave velocity relative to spacecraft ", wave_velocity_relative2sc)

        results["wave_velocity_plasma_frame"] = Vpl
        results["wave_velocity_relative2sc"] = wave_velocity_relative2sc
    if prnt:
        print("Predicted time lags", predicted_time_lags)
        print("Time differences: ", time_difference)
        print("Correlation coefficients: ", cross_corr_values)
        print("Wave phase velocity ", wave_velocity_sc_frame)
        print("Wave vector ", wave_vector)
        print(result)
        print("Timing analysis")
        print(matrix_positions)

    return results
