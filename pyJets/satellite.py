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

# import scipy
# import scipy.linalg
from scipy.linalg import eig
from scipy.fft import rfft2
from scipy.signal import butter, sosfilt, cwt, morlet2
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

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


def load_msh_sc_data(sc, probe, var, t0, t1, intpol=True, dt=1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    vars_list = ["B", "rho", "v"]
    sc_list = ["mms", "themis", "cluster"]
    moms_list = ["fpi", "mom", "cis"]
    sc_var_names = [
        [
            "mms{}_fgm_b_gse_srvy_l2".format(probe),
            "mms{}_dis_numberdensity_fast".format(probe),
            "mms{}_dis_bulkv_gse_fast".format(probe),
        ],
        [
            "th{}_fgs_gse".format(probe),
            "th{}_peim_density".format(probe),
            "th{}_peim_velocity_gse".format(probe),
        ],
        [
            "B_xyz_gse__C{}_UP_FGM".format(probe),
            "N_p__C{}_PP_CIS".format(probe),
            "V_p_xyz_gse__C{}_PP_CIS".format(probe),
        ],
    ]

    sc_obj = [
        pyspedas.mms,
        pyspedas.themis,
        pyspedas.cluster,
    ][sc_list.index(sc)]

    if var == "B":
        sc_ins_obj = sc_obj.fgm
    elif sc == "themis":
        sc_ins_obj = sc_obj.mom
    elif sc == "mms":
        sc_ins_obj = sc_obj.fpi
    elif sc == "cluster":
        sc_ins_obj = sc_obj.cis

    sc_data = sc_ins_obj(trange=[t0, t1], probe=probe, notplot=True, time_clip=True)[
        sc_var_names[sc_list.index(sc)][vars_list.index(var)]
    ]
    time_list = np.array(sc_data["x"])
    data_list = np.array(sc_data["y"]).T

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
                [np.interp(newtime, time_list, data) for data in data_list]
            )
        else:
            newdata = np.interp(newtime, time_list, data_list)
        newtime = np.array([datetime.utcfromtimestamp(t) for t in newtime])

        return (newtime, newdata)
    else:
        if type(time_list[0]) != datetime:
            time_list = np.array([datetime.utcfromtimestamp(t) for t in time_list])
        return (time_list, data_list)


def thd_mms1_c4_timing(t0, t1):

    thd_time, thd_B = load_msh_sc_data("themis", "d", "B", t0, t1, intpol=True, dt=5)
    mms1_time, mms1_B = load_msh_sc_data("mms", "1", "B", t0, t1, intpol=True, dt=5)
    c4_time, c4_B = load_msh_sc_data("cluster", "4", "B", t0, t1, intpol=True, dt=5)

    dummy, thd_rho = load_msh_sc_data("themis", "d", "rho", t0, t1, intpol=True, dt=5)
    dummy, mms1_rho = load_msh_sc_data("mms", "1", "rho", t0, t1, intpol=True, dt=5)
    dummy, c4_rho = load_msh_sc_data("cluster", "4", "rho", t0, t1, intpol=True, dt=5)

    dummy, thd_v = load_msh_sc_data("themis", "d", "v", t0, t1, intpol=True, dt=5)
    dummy, mms1_v = load_msh_sc_data("mms", "1", "v", t0, t1, intpol=True, dt=5)
    dummy, c4_v = load_msh_sc_data("cluster", "4", "v", t0, t1, intpol=True, dt=5)

    thd_vmag = np.linalg.norm(thd_v, axis=-1)
    mms1_vmag = np.linalg.norm(mms1_v, axis=-1)
    c4_vmag = np.linalg.norm(c4_v, axis=-1)

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

    labs = ["Bx:", "By:", "Bz:"]

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

    print("Pdyn:")
    timing_analysis_arb(
        [thd_time, mms1_time, c4_time],
        [thd_pdyn, mms1_pdyn, c4_pdyn],
        sc_rel_pos,
        t0,
        t1,
    )
    print("\n")


def plot_thd_mms1_c4(t0, t1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    thd_mag = pyspedas.themis.fgm(trange=[t0, t1], notplot=True, probe="d")
    thd_mom = pyspedas.themis.mom(trange=[t0, t1], notplot=True, probe="d")

    mms1_mag = pyspedas.mms.fgm(trange=[t0, t1], notplot=True, probe="1")
    mms1_mom = pyspedas.mms.fpi(trange=[t0, t1], notplot=True, probe="1")

    c4_mag = pyspedas.cluster.fgm(trange=[t0, t1], notplot=True, probe="4")
    c4_mom = pyspedas.cluster.cis(trange=[t0, t1], notplot=True, probe="4")

    time_arr = np.empty((3, 4), dtype=object)
    data_arr = np.empty((3, 10), dtype=object)

    time_arr[0, :] = [
        [datetime.utcfromtimestamp(t) for t in thd_mag["thd_fgs_gse"]["x"]],
        [datetime.utcfromtimestamp(t) for t in thd_mom["thd_peim_density"]["x"]],
        [datetime.utcfromtimestamp(t) for t in thd_mom["thd_peim_density"]["x"]],
        [datetime.utcfromtimestamp(t) for t in thd_mom["thd_peim_density"]["x"]],
    ]
    data_arr[0, :] = [
        thd_mag["thd_fgs_gse"]["y"].T[0],
        thd_mag["thd_fgs_gse"]["y"].T[1],
        thd_mag["thd_fgs_gse"]["y"].T[2],
        thd_mag["thd_fgs_btotal"]["y"],
        thd_mom["thd_peim_velocity_gse"]["y"].T[0],
        thd_mom["thd_peim_velocity_gse"]["y"].T[1],
        thd_mom["thd_peim_velocity_gse"]["y"].T[2],
        np.linalg.norm(thd_mom["thd_peim_velocity_gse"]["y"], axis=-1),
        thd_mom["thd_peim_density"]["y"],
        m_p
        * thd_mom["thd_peim_density"]["y"]
        * 1e6
        * np.linalg.norm(thd_mom["thd_peim_velocity_gse"]["y"], axis=-1)
        * np.linalg.norm(thd_mom["thd_peim_velocity_gse"]["y"], axis=-1)
        * 1e6
        / 1e-9,
    ]

    time_arr[1, :] = [
        mms1_mag["mms1_fgm_b_gse_srvy_l2"]["x"],
        mms1_mom["mms1_dis_numberdensity_fast"]["x"],
        mms1_mom["mms1_dis_numberdensity_fast"]["x"],
        mms1_mom["mms1_dis_numberdensity_fast"]["x"],
    ]
    data_arr[1, :] = [
        mms1_mag["mms1_fgm_b_gse_srvy_l2"]["y"].T[0],
        mms1_mag["mms1_fgm_b_gse_srvy_l2"]["y"].T[1],
        mms1_mag["mms1_fgm_b_gse_srvy_l2"]["y"].T[2],
        np.linalg.norm(mms1_mag["mms1_fgm_b_gse_srvy_l2"]["y"], axis=-1),
        mms1_mom["mms1_dis_bulkv_gse_fast"]["y"].T[0],
        mms1_mom["mms1_dis_bulkv_gse_fast"]["y"].T[1],
        mms1_mom["mms1_dis_bulkv_gse_fast"]["y"].T[2],
        np.linalg.norm(mms1_mom["mms1_dis_bulkv_gse_fast"]["y"], axis=-1),
        mms1_mom["mms1_dis_numberdensity_fast"]["y"],
        m_p
        * mms1_mom["mms1_dis_numberdensity_fast"]["y"]
        * 1e6
        * np.linalg.norm(mms1_mom["mms1_dis_bulkv_gse_fast"]["y"], axis=-1)
        * np.linalg.norm(mms1_mom["mms1_dis_bulkv_gse_fast"]["y"], axis=-1)
        * 1e6
        / 1e-9,
    ]

    time_arr[2, :] = [
        c4_mag["B_xyz_gse__C4_UP_FGM"]["x"],
        c4_mom["N_p__C4_PP_CIS"]["x"],
        c4_mom["N_p__C4_PP_CIS"]["x"],
        c4_mom["N_p__C4_PP_CIS"]["x"],
    ]
    data_arr[2, :] = [
        c4_mag["B_xyz_gse__C4_UP_FGM"]["y"].T[0],
        c4_mag["B_xyz_gse__C4_UP_FGM"]["y"].T[1],
        c4_mag["B_xyz_gse__C4_UP_FGM"]["y"].T[2],
        np.linalg.norm(c4_mag["B_xyz_gse__C4_UP_FGM"]["y"], axis=-1),
        c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"].T[0],
        c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"].T[1],
        c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"].T[2],
        np.linalg.norm(c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"], axis=-1),
        c4_mom["N_p__C4_PP_CIS"]["y"],
        m_p
        * c4_mom["N_p__C4_PP_CIS"]["y"]
        * 1e6
        * np.linalg.norm(c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"], axis=-1)
        * np.linalg.norm(c4_mom["V_p_xyz_gse__C4_PP_CIS"]["y"], axis=-1)
        * 1e6
        / 1e-9,
    ]

    time_masks = [
        np.logical_and(
            np.array(time_arr[idx, -1]) >= t0plot, np.array(time_arr[idx, -1]) <= t1plot
        )
        for idx in range(3)
    ]
    print(time_masks)

    t_pdmax = [
        np.array(time_arr[idx, -1])[time_masks[idx]][
            np.argmax(np.array(data_arr[idx, -1])[time_masks[idx]])
        ]
        for idx in range(3)
    ]

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
    sc_labs = ["THD", "MMS1", "C4"]
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
                time_arr[idx, panel_id[idx2]],
                data_arr[idx, idx2],
            )
            ax.label_outer()
            ax.set_xlim(t0plot, t1plot)
            ax.axvline(t_pdmax[idx], linestyle="dashed")

    for idx in range(3):
        ax_list[0, idx].set_title(sc_labs[idx], pad=10, fontsize=20)
    # for idx in range(len(panel_labs)):
    #     ax_list[idx, 0].set_ylabel(panel_labs[idx], labelpad=10, fontsize=20)
    #     ax_list[idx, 0].set_ylim(ylims[idx][0], ylims[idx][1])
    for idx in range(len(ylabels_all)):
        ax_list[idx, 0].set_ylabel(ylabels_all[idx], labelpad=10, fontsize=20)
        ax_list[idx, 0].set_ylim(ylims_full[idx][0], ylims_full[idx][1])

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(outdir + "thd_mms1_c4_t0{}_t1{}.png".format(t0plot, t1plot))
    plt.close(fig)


def plot_ace_dscovr_wind(t0, t1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    ace_data = pyspedas.ace.mfi(trange=[t0, t1], notplot=True, time_clip=True)
    dscovr_data = pyspedas.dscovr.mag(trange=[t0, t1], notplot=True, time_clip=True)
    wind_data = pyspedas.wind.mfi(trange=[t0, t1], notplot=True, time_clip=True)

    ace_t = ace_data["BGSEc"]["x"]
    ace_B = ace_data["BGSEc"]["y"].T

    dscovr_t = dscovr_data["dsc_h0_mag_B1GSE"]["x"]
    dscovr_B = dscovr_data["dsc_h0_mag_B1GSE"]["y"].T

    wind_t = wind_data["BGSE"]["x"]
    wind_B = wind_data["BGSE"]["y"].T

    ace_clock, dscovr_clock, wind_clock = [
        np.rad2deg(np.arctan2(B[2], B[1])) for B in [ace_B, dscovr_B, wind_B]
    ]
    ace_cone, dscovr_cone, wind_cone = [
        np.rad2deg(np.arctan2(np.sqrt(B[2] ** 2 + B[1] ** 2), B[0]))
        for B in [ace_B, dscovr_B, wind_B]
    ]
    ace_Bmag, dscovr_Bmag, wind_Bmag = [
        np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2) for B in [ace_B, dscovr_B, wind_B]
    ]

    time_list = [ace_t, dscovr_t, wind_t]
    data_list = [
        [ace_B[0], ace_B[1], ace_B[2], ace_Bmag, ace_clock, ace_cone],
        [dscovr_B[0], dscovr_B[1], dscovr_B[2], dscovr_Bmag, dscovr_clock, dscovr_cone],
        [wind_B[0], wind_B[1], wind_B[2], wind_Bmag, wind_clock, wind_cone],
    ]
    title_labs = ["Bx", "By", "Bz", "Bmag", "Clock", "Cone"]
    ylabs = ["ACE", "DSCOVR", "Wind"]

    fig, ax_list = plt.subplots(
        6, 3, figsize=(18, 18), constrained_layout=True, sharey="row"
    )

    for idx in range(3):
        ax_list[0, idx].set_title(ylabs[idx], pad=10, fontsize=20)
        for idx2 in range(6):
            ax = ax_list[idx2, idx]
            if idx == 0:
                ax.set_ylabel(title_labs[idx2], labelpad=10, fontsize=20)
            if idx == 2:
                ax.plot(time_list[idx], data_list[idx][idx2])
            else:
                ax.plot(
                    time_list[idx],
                    uniform_filter1d(interpolate_nans(data_list[idx][idx2]), size=60),
                )
            # ax.plot(time_list[idx], data_list[idx][idx2])
            ax.set_xlim(t0plot, t1plot)
            ax.grid()

    for ax in ax_list.flatten():
        ax.label_outer()

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(outdir + "ace_dscovr_wind_t0{}_t1{}.png".format(t0plot, t1plot))
    plt.close(fig)

    print("\n")
    print("Bx: ")
    timing_analysis_ace_dscovr_wind(
        ace_t, dscovr_t, wind_t, ace_B[0], dscovr_B[0], wind_B[0], t0, t1
    )
    print("\n")

    print("By: ")
    timing_analysis_ace_dscovr_wind(
        ace_t, dscovr_t, wind_t, ace_B[1], dscovr_B[1], wind_B[1], t0, t1
    )
    print("\n")

    print("Bz: ")
    timing_analysis_ace_dscovr_wind(
        ace_t, dscovr_t, wind_t, ace_B[2], dscovr_B[2], wind_B[2], t0, t1
    )


def timing_analysis_ace_dscovr_wind(
    ace_time, dscovr_time, wind_time, ace_data, dscovr_data, wind_data, t0, t1
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

    t0plot = (
        datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    t1plot = (
        datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    pos_data = np.loadtxt(
        wrkdir_DNR
        + "satellites/ace_dscovr_wind_pos_2022-03-27_19:00:00_21:00:00_numpy.txt",
        dtype=str,
    ).T

    sc_name = pos_data[3]
    sc_x = pos_data[4].astype(float) * Re
    sc_y = pos_data[5].astype(float) * Re
    sc_z = pos_data[6].astype(float) * Re

    ace_pos = np.array(
        [
            np.nanmean(sc_x[sc_name == "ace"]),
            np.nanmean(sc_y[sc_name == "ace"]),
            np.nanmean(sc_z[sc_name == "ace"]),
        ]
    )
    dscovr_pos = np.array(
        [
            np.nanmean(sc_x[sc_name == "dscovr"]),
            np.nanmean(sc_y[sc_name == "dscovr"]),
            np.nanmean(sc_z[sc_name == "dscovr"]),
        ]
    )
    wind_pos = np.array(
        [
            np.nanmean(sc_x[sc_name == "wind"]),
            np.nanmean(sc_y[sc_name == "wind"]),
            np.nanmean(sc_z[sc_name == "wind"]),
        ]
    )

    uni_time = np.array(
        [
            wind_time[idx].timestamp()
            for idx in range(wind_time.size)
            if not np.isnan(wind_data[idx])
        ]
    )
    uni_time = uni_time[np.logical_and(uni_time >= t0plot, uni_time <= t1plot)]
    ace_time_unix = np.array(
        [
            ace_time[idx].replace(tzinfo=timezone.utc).timestamp()
            for idx in range(ace_time.size)
        ]
    )
    dscovr_time_unix = np.array(
        [
            dscovr_time[idx].replace(tzinfo=timezone.utc).timestamp()
            for idx in range(dscovr_time.size)
        ]
    )
    wind_time_unix = np.array(
        [
            wind_time[idx].replace(tzinfo=timezone.utc).timestamp()
            for idx in range(wind_time.size)
        ]
    )

    dt = uni_time[1] - uni_time[0]

    wind_data_clean = np.interp(
        uni_time, wind_time_unix[~np.isnan(wind_data)], wind_data[~np.isnan(wind_data)]
    )
    ace_data_clean = np.interp(
        uni_time, ace_time_unix[~np.isnan(ace_data)], ace_data[~np.isnan(ace_data)]
    )
    dscovr_data_clean = np.interp(
        uni_time,
        dscovr_time_unix[~np.isnan(dscovr_data)],
        dscovr_data[~np.isnan(dscovr_data)],
    )

    ace_norm = (ace_data_clean - np.mean(ace_data_clean)) / (
        np.std(ace_data_clean, ddof=1)  # * ace_data_clean.size
    )
    dscovr_norm = (dscovr_data_clean - np.mean(dscovr_data_clean)) / (
        np.std(dscovr_data_clean, ddof=1)  # * dscovr_data_clean.size
    )
    wind_norm = (wind_data_clean - np.mean(wind_data_clean)) / (
        np.std(wind_data_clean, ddof=1) * wind_data_clean.size
    )

    c = np.correlate(ace_norm, wind_norm, "full")
    offset = np.argmax(c)
    alpha = c[offset - 1]
    beta = c[offset]
    gamma = c[offset + 1]
    offset2 = offset - len(c) / 2.0 + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    print("offset", offset, offset2)
    cross_corr_values.append(np.max(c))
    # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
    time_difference.append(offset2 * dt)

    c = np.correlate(dscovr_norm, wind_norm, "full")
    offset = np.argmax(c)
    alpha = c[offset - 1]
    beta = c[offset]
    gamma = c[offset + 1]
    offset2 = offset - len(c) / 2.0 + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    print("offset", offset, offset2)
    cross_corr_values.append(np.max(c))
    # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
    time_difference.append(offset2 * dt)

    # # ******************************************************************************#

    time_difference = np.array(time_difference)
    print("Time differences: ", time_difference)

    matrix_positions = np.zeros((3, 3))

    matrix_positions[0] = ace_pos - wind_pos
    matrix_positions[1] = dscovr_pos - wind_pos
    matrix_positions[2] = ace_pos - dscovr_pos

    print("Timing analysis")
    print(matrix_positions)
    # We now invert the matrix of spacecraft relative positions and multiply it with the time lags in order to solve the system
    # of equations for the wave vector
    # The vector obtained from this operation is the wave vector divided by the phase velocity in the spacecraft frame

    result = np.dot(np.linalg.inv(matrix_positions[0:2, 0:2]), time_difference[0:2])
    result.shape = (2, 1)

    norm_result = np.linalg.norm(result)

    wave_velocity_sc_frame = 1.0 / norm_result

    print(result)

    wave_vector = np.zeros((3, 1))
    wave_vector[0:2] = result / norm_result

    print("Wave phase velocity ", wave_velocity_sc_frame)
    print("Wave vector ", wave_vector)

    results = {}
    results["wave_vector"] = wave_vector
    results["wave_velocity_sc_frame"] = wave_velocity_sc_frame
    results["cross_corr_values"] = cross_corr_values
    print("Correlation coefficients: ", cross_corr_values)

    return results


def timing_analysis_arb(sc_times, sc_data, sc_rel_pos, t0, t1):
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

    t0plot = (
        datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    t1plot = (
        datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    sc_times_new = []

    for idx in range(len(sc_times)):
        sc_times_new.append(
            np.array(
                [t.replace(tzinfo=timezone.utc).timestamp() for t in sc_times[idx]]
            )
        )

    dt = sc_times_new[0][1] - sc_times_new[0][0]

    rel_sc_norm = (sc_data[0] - np.mean(sc_data[0])) / (
        np.std(sc_data[0], ddof=1) * sc_data[0].size
    )

    for sc in sc_data[1:]:
        sc_norm = (sc - np.mean(sc)) / np.std(sc, ddof=1)
        c = np.correlate(sc_norm, rel_sc_norm, "full")
        offset = np.argmax(c)
        alpha = c[offset - 1]
        beta = c[offset]
        gamma = c[offset + 1]
        offset2 = (
            offset - len(c) / 2.0 + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        )
        print("offset", offset, offset2)
        cross_corr_values.append(np.max(c))
        # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
        time_difference.append(offset2 * dt)

    # # ******************************************************************************#

    time_difference = np.array(time_difference)
    print("Time differences: ", time_difference)

    matrix_positions = np.zeros((3, 3))

    for idx in range(len(sc_times) - 1):
        matrix_positions[idx] = sc_rel_pos[idx]

    print("Timing analysis")
    print(matrix_positions)
    # We now invert the matrix of spacecraft relative positions and multiply it with the time lags in order to solve the system
    # of equations for the wave vector
    # The vector obtained from this operation is the wave vector divided by the phase velocity in the spacecraft frame

    result = np.dot(
        np.linalg.inv(matrix_positions[0 : len(sc_rel_pos), 0 : len(sc_rel_pos)]),
        time_difference[0 : len(sc_rel_pos)],
    )
    result.shape = (len(sc_rel_pos), 1)

    norm_result = np.linalg.norm(result)

    wave_velocity_sc_frame = 1.0 / norm_result

    print(result)

    wave_vector = np.zeros((3, 1))
    wave_vector[0 : len(sc_rel_pos)] = result / norm_result

    print("Wave phase velocity ", wave_velocity_sc_frame)
    print("Wave vector ", wave_vector)

    results = {}
    results["wave_vector"] = wave_vector
    results["wave_velocity_sc_frame"] = wave_velocity_sc_frame
    results["cross_corr_values"] = cross_corr_values
    print("Correlation coefficients: ", cross_corr_values)

    return results
