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

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S").timestamp()
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S").timestamp()

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
        [ace_time[idx].timestamp() for idx in range(ace_time.size)]
    )
    dscovr_time_unix = np.array(
        [dscovr_time[idx].timestamp() for idx in range(dscovr_time.size)]
    )
    wind_time_unix = np.array(
        [wind_time[idx].timestamp() for idx in range(wind_time.size)]
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

    # c = np.correlate(ace_norm,dscovr_norm,"full")
    # offset = np.argmax(c)
    # alpha = c[offset - 1]
    # beta = c[offset]
    # gamma = c[offset + 1]
    # offset2 = (
    #         offset - len(c) / 2.0 + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    #     )
    # print("offset", offset, offset2)
    # cross_corr_values.append(np.max(c))
    # # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
    # time_difference.append(offset2 * dt)

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

    # for isc in ind_sc[1:]:
    #     # To obtain a normalized crosscorrelation:
    #     a = (data[ref_sc, var_ind] - np.mean(data[ref_sc, var_ind])) / (
    #         np.std(data[ref_sc, var_ind], ddof=1) * len(data[ref_sc, var_ind])
    #     )

    #     b = (data[isc, var_ind] - np.mean(data[isc, var_ind])) / (
    #         np.std(data[isc, var_ind], ddof=1)
    #     )

    #     c = np.correlate(b, a, "full")

    #     # This gives the offset with the same time resolution as in the Vlasiator data set, so it as a 0.5 s error in the estimate
    #     offset = np.argmax(c)

    #     alpha = c[offset - 1]
    #     beta = c[offset]
    #     gamma = c[offset + 1]

    #     # Here we fit the data points at and surrounding the selected offset time with a parabola, to improve on the time resolution
    #     # It is this offset value that is then used in the remainder of the script
    #     offset2 = (
    #         offset - len(c) / 2.0 + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    #     )

    #     print("offset", offset, offset2)

    #     cross_corr_values.append(np.max(c))

    #     # Offset being given as an index in the time array, we multiply it by the time step dt to obtain the actual time lag in s.
    #     time_difference.append(offset2 * dt)

    # # ******************************************************************************#

    time_difference = np.array(time_difference)
    print("Time differences: ", time_difference)

    matrix_positions = np.zeros((3, 3))

    matrix_positions[0] = ace_pos - wind_pos
    matrix_positions[1] = dscovr_pos - wind_pos
    matrix_positions[2] = ace_pos - dscovr_pos

    # pos_ref_sc = np.zeros((3, 3))

    # pos_ref_sc[0, :] = [
    #     pos_jonas[ref_sc, 0],
    #     pos_jonas[ref_sc, 1],
    #     pos_jonas[ref_sc, 2],
    # ]
    # pos_ref_sc[1, :] = [
    #     pos_jonas[ref_sc, 0],
    #     pos_jonas[ref_sc, 1],
    #     pos_jonas[ref_sc, 2],
    # ]
    # pos_ref_sc[2, :] = [
    #     pos_jonas[ind_sc[1], 0],
    #     pos_jonas[ind_sc[1], 1],
    #     pos_jonas[ind_sc[1], 2],
    # ]

    # pos_ref_sc[0] = pos_jonas[ref_sc]
    # pos_ref_sc[1] = pos_jonas[ref_sc]
    # pos_ref_sc[2] = pos_jonas[ind_sc[1]]

    # pos_ref_sc = pos_ref_sc * Re

    # ind = 0
    # for isc in ind_sc[1:]:
    #     # Position of the second spacecraft from the considered pair
    #     # R2 = [pos_jonas[isc, 0], pos_jonas[isc, 1], pos_jonas[isc, 2]]
    #     R2 = pos_jonas[isc]
    #     # R2 = np.array(R2) * Re
    #     R2 = R2 * Re

    #     matrix_positions[ind] = R2 - pos_ref_sc[ind]

    #     ind = ind + 1

    # matrix_positions[2, :] = R2 - pos_ref_sc[2, :]

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

    # V_bulk = np.array(
    #     [
    #         np.mean(data[ref_sc, -7][data[ref_sc, 5] >= 3]),
    #         np.mean(data[ref_sc, -6][data[ref_sc, 5] >= 3]),
    #         np.mean(data[ref_sc, -5][data[ref_sc, 5] >= 3]),
    #     ]
    # )
    # V_A = np.array(
    #     [
    #         np.mean(data[ref_sc, -4][data[ref_sc, 5] >= 3]),
    #         np.mean(data[ref_sc, -3][data[ref_sc, 5] >= 3]),
    #         np.mean(data[ref_sc, -2][data[ref_sc, 5] >= 3]),
    #     ]
    # )
    # results["alfven_velocity"] = V_A
    # print("Bulk velocity: ", V_bulk, np.linalg.norm(V_bulk))
    # Vpl = wave_velocity_sc_frame - np.dot(V_bulk, wave_vector)

    # if "proton/V.x" in data:
    #     V_bulk = np.array(
    #         [
    #             np.mean(data["proton/V.x"][min_time:max_time, ref_sc]),
    #             np.mean(data["proton/V.y"][min_time:max_time, ref_sc]),
    #             np.mean(data["proton/V.z"][min_time:max_time, ref_sc]),
    #         ]
    #     )
    #

    # elif "V.x" in data:
    #     V_bulk = np.array(
    #         [
    #             np.mean(data["V.x"][min_time:max_time, ref_sc]),
    #             np.mean(data["V.y"][min_time:max_time, ref_sc]),
    #             np.mean(data["V.z"][min_time:max_time, ref_sc]),
    #         ]
    #     )
    #     Vpl = wave_velocity_sc_frame - np.dot(V_bulk, wave_vector)
    # else:
    #     print("No bulk velocity found - wave velocity only in simulation frame")

    # if "Vpl" in locals():
    #     print("Wave phase velocity in plasma frame ", Vpl)

    #     wave_velocity_relative2sc = (
    #         V_bulk.reshape((3, 1))
    #         + (wave_velocity_sc_frame - np.dot(V_bulk, wave_vector)) * wave_vector
    #     )
    #     wave_velocity_relative2sc.shape = 3
    #     print("Wave velocity relative to spacecraft ", wave_velocity_relative2sc)

    #     results["wave_velocity_plasma_frame"] = Vpl
    #     results["wave_velocity_relative2sc"] = wave_velocity_relative2sc

    #     results["bulk_velocity"] = V_bulk

    return results
