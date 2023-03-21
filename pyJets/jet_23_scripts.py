# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
import jet_aux as jx
from pyJets.jet_aux import CB_color_cycle
import pytools as pt
import os

# import scipy
# import scipy.linalg
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

import plot_contours as pc
import jet_analyser as ja
import jet_io as jio
import jet_jh2020 as jh20
from papu_2 import get_fcs_jets, get_non_jets

# mpl.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# plt.rcParams.update(params)

r_e = 6.371e6
m_p = 1.672621898e-27
mu0 = 1.25663706212e-06
kb = 1.380649e-23

wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"


def ani_timeseries():

    jetid = 596
    runid = "ABC"
    kind = ""

    global ax, x0, y0, pdmax, bulkpath, jetid_g, axr0, axr1, axr2, axr3, axr4, fnr0_g, pm_g, ax_ylabels, vmaxs, vmins, t0
    global runid_g, sj_ids_g, non_ids_g, kind_g, Blines_g
    kind_g = kind
    jetid_g = jetid
    runid_g = runid
    Blines_g = False
    runids = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw
    bulkpath = jx.find_bulkpath(runid)
    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]

    sj_ids_g = get_fcs_jets(runid)
    non_ids_g = get_non_jets(runid)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    spec = fig.add_gridspec(5, 10)
    ax = fig.add_subplot(spec[:, :5])
    axr0 = fig.add_subplot(spec[0, 5:])
    axr1 = fig.add_subplot(spec[1, 5:])
    axr2 = fig.add_subplot(spec[2, 5:])
    axr3 = fig.add_subplot(spec[3, 5:])
    axr4 = fig.add_subplot(spec[4, 5:])

    ax_ylabels = [
        "$n~[n_{sw}]$",
        "$v~[v_{sw}]$",
        "$P_{dyn}$\n$[P_{dyn,sw}]$",
        "$B~[B_{IMF}]$",
        "$T~[T_{sw}]$",
    ]
    vmins = [0, -1, 0, -4, 0]
    vmaxs = [6, 1, 3, 4, 25]

    t0 = 475
    fnr0 = int(t0 * 2)
    fnr0_g = fnr0
    pm_g = 5

    global ts_t_arr, ts_v_arrs, ts_v_vars, ts_v_ops, var_ax_idx, ts_v_norm, ts_v_colors, ts_v_labels
    # ts_t_arr = []
    ts_t_arr = np.empty(np.arange(fnr0 - pm_g, fnr0 + pm_g + 0.1, 1).size)
    ts_t_arr.fill(np.nan)
    # ts_v_arrs = [[], [], [], [], [], [], [], [], [], [], [], []]
    ts_v_arrs = np.empty((12, np.arange(fnr0 - pm_g, fnr0 + pm_g + 0.1, 1).size))
    ts_v_arrs.fill(np.nan)
    ts_v_norm = [
        rho_sw,
        v_sw,
        v_sw,
        v_sw,
        v_sw,
        Pdyn_sw,
        B_sw,
        B_sw,
        B_sw,
        B_sw,
        T_sw,
        T_sw,
    ]
    var_ax_idx = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4]
    ts_v_vars = [
        "rho",
        "v",
        "v",
        "v",
        "v",
        "Pdyn",
        "B",
        "B",
        "B",
        "B",
        "TParallel",
        "TPerpendicular",
    ]
    ts_v_ops = [
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "pass",
    ]
    ts_v_colors = [
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]
    ts_v_labels = ["", "x", "y", "z", "tot", "", "x", "y", "z", "tot", "par", "perp"]

    props = jio.PropReader(str(jetid).zfill(5), runid)
    global lines
    lines = []

    x0 = 10.5
    y0 = -2.4

    for idx, a in enumerate([axr0, axr1, axr2, axr3, axr4]):
        a.set_ylabel(ax_ylabels[idx], labelpad=10, fontsize=20)
        a.set_xlim(t0 - pm_g / 2.0, t0 + pm_g / 2.0)
        a.set_ylim(vmins[idx], vmaxs[idx])
        a.tick_params(labelsize=16)
        a.grid()
        if idx < 4:
            a.xaxis.set_ticklabels([])
        else:
            a.set_xlabel("Simulation time [s]", labelpad=10, fontsize=20)

    for idx2, ax_idx in enumerate(var_ax_idx):
        a = [axr0, axr1, axr2, axr3, axr4][ax_idx]
        lines.append(
            a.plot(
                ts_t_arr,
                ts_v_arrs[idx2],
                color=ts_v_colors[idx2],
                label=ts_v_labels[idx2],
            )
        )

    ani = FuncAnimation(
        fig,
        jet_ts_update,
        frames=np.arange(fnr0 - pm_g, fnr0 + pm_g + 0.1, 1),
        blit=False,
    )
    ani.save(
        wrkdir_DNR + "papu22/ripple_jet.mp4",
        fps=5,
        dpi=150,
        bitrate=1000,
    )
    # print("Saved animation of jet {} in run {}".format(jetid, runid))
    plt.close(fig)


def jet_ts_update(fnr):
    idx3 = int(fnr - (fnr0_g - pm_g))
    print("t = {}s".format(float(fnr) / 2.0))
    ax.clear()
    fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))
    global filenr_g
    filenr_g = fnr
    pt.plot.plot_colormap(
        axes=ax,
        filename=bulkpath + fname,
        var="Pdyn",
        vmin=0,
        vmax=3.5,
        vscale=1e9,
        # cbtitle="$\\rho_{st}/\\rho_{th}",
        usesci=0,
        scale=2,
        title="",
        boxre=[x0 - 2, x0 + 2, y0 - 2, y0 + 2],
        internalcb=True,
        lin=1,
        colormap="Blues_r",
        tickinterval=1.0,
        external=ext_jet,
        # expression=expr_rhoratio,
        pass_vars=[
            "RhoNonBackstream",
            "RhoBackstream",
            "PTensorNonBackstreamDiagonal",
            "B",
            "v",
            "rho",
            "core_heating",
            "CellID",
            "Mmsx",
            "Pdyn",
        ],
    )
    ax.set_title(
        "Run: {} t = {}s".format(runid_g, float(fnr) / 2.0),
        pad=10,
        fontsize=20,
    )
    ax.axhline(y0, linestyle="dashed", linewidth=0.6, color="k")
    ax.axvline(x0, linestyle="dashed", linewidth=0.6, color="k")

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))
    )
    ts_t_arr[idx3] = float(fnr) / 2.0
    for idx in range(len(ts_v_ops)):
        val = (
            vlsvobj.read_interpolated_variable(
                ts_v_vars[idx], [x0 * r_e, y0 * r_e, 0], operator=ts_v_ops[idx]
            )
            / ts_v_norm[idx]
        )
        # ts_v_arrs[idx].append(val)
        ts_v_arrs[idx][idx3] = val

    for idx2, ax_idx in enumerate(var_ax_idx):
        # print(ax_idx)
        a = [axr0, axr1, axr2, axr3, axr4][ax_idx]
        lines[idx2][0].set_xdata(ts_t_arr)
        lines[idx2][0].set_ydata(ts_v_arrs[idx2])
        # a.clear()
        # a.plot(
        #     ts_t_arr, ts_v_arrs[idx2], color=ts_v_colors[idx2], label=ts_v_labels[idx2]
        # )
        # a.set_ylabel(ax_ylabels[var_ax_idx[idx2]], labelpad=10, fontsize=20)
        # a.set_xlim(t0 - pm_g / 2.0, t0 + pm_g / 2.0)
        # a.set_ylim(vmins[var_ax_idx[idx2]], vmaxs[var_ax_idx[idx2]])
        # a.tick_params(labelsize=16)
        # if ax_idx < 4:
        #     a.xaxis.set_ticklabels([])
        # else:
        #     a.set_xlabel("Simulation time [s]", labelpad=10, fontsize=20)

    axr1.legend()
    axr3.legend()
    axr4.legend()

    # plt.tight_layout()


def ext_jet(ax, XmeshXY, YmeshXY, pass_maps):
    B = pass_maps["B"]
    rho = pass_maps["rho"]
    cellids = pass_maps["CellID"]
    mmsx = pass_maps["Mmsx"]
    core_heating = pass_maps["core_heating"]
    Bmag = np.linalg.norm(B, axis=-1)
    Pdyn = pass_maps["Pdyn"]

    try:
        slams_cells = np.loadtxt(
            "/wrk-vakka/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(
                runid_g, int(filenr_g)
            )
        ).astype(int)
    except:
        slams_cells = []
    try:
        jet_cells = np.loadtxt(
            "/wrk-vakka/users/jesuni/working/jets/Masks/{}/{}.mask".format(
                runid_g, int(filenr_g)
            )
        ).astype(int)
    except:
        jet_cells = []

    sj_jetobs = [
        jio.PropReader(str(int(sj_id)).zfill(5), runid_g, transient="jet")
        for sj_id in sj_ids_g
    ]
    non_sjobs = [
        jio.PropReader(str(int(non_id)).zfill(5), runid_g, transient="jet")
        for non_id in non_ids_g
    ]

    sj_xlist = []
    sj_ylist = []
    non_xlist = []
    non_ylist = []

    for jetobj in sj_jetobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            sj_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
            sj_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))
    for jetobj in non_sjobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
            non_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))

    slams_mask = np.in1d(cellids, slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask, cellids.shape)

    jet_mask = np.in1d(cellids, jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask, cellids.shape)

    ch_mask = (core_heating > 3 * T_sw).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2 * rho_sw).astype(int)

    cav_shfa_mask = (Bmag < 0.8 * B_sw).astype(int)
    cav_shfa_mask[rho >= 0.8 * rho_sw] = 0

    diamag_mask = (Pdyn >= 1.2 * Pdyn_sw).astype(int)
    diamag_mask[Bmag > B_sw] = 0

    CB_color_cycle = jx.CB_color_cycle

    start_points = np.array(
        [np.ones(20) * x0 + 0.5, np.linspace(y0 - 0.9, y0 + 0.9, 20)]
    ).T
    # start_points = np.array([np.linspace(x0 - 0.9, x0 + 0.9, 10), np.ones(10) * y0]).T

    if Blines_g:
        stream = ax.streamplot(
            XmeshXY,
            YmeshXY,
            B[:, :, 0],
            B[:, :, 1],
            # arrowstyle="-",
            # broken_streamlines=False,
            color="k",
            linewidth=0.6,
            # minlength=4,
            density=35,
            start_points=start_points,
        )

    jet_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        jet_mask,
        [0.5],
        linewidths=2,
        colors=CB_color_cycle[2],
        linestyles=["solid"],
    )

    ch_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        ch_mask,
        [0.5],
        linewidths=2,
        colors=CB_color_cycle[1],
        linestyles=["solid"],
    )

    slams_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        slams_mask,
        [0.5],
        linewidths=2,
        colors=CB_color_cycle[0],
        linestyles=["solid"],
    )

    rho_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        rho_mask,
        [0.5],
        linewidths=2,
        colors=CB_color_cycle[3],
        linestyles=["solid"],
    )

    mach_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        mach_mask,
        [0.5],
        linewidths=2,
        colors=CB_color_cycle[4],
        linestyles=["solid"],
    )

    (non_pos,) = ax.plot(
        non_xlist,
        non_ylist,
        "o",
        color="black",
        markersize=10,
        markeredgecolor="white",
        fillstyle="full",
        mew=1,
        label="Non-FCS-jet",
    )
    (sj_pos,) = ax.plot(
        sj_xlist,
        sj_ylist,
        "o",
        color="red",
        markersize=10,
        markeredgecolor="white",
        fillstyle="full",
        mew=1,
        label="FCS-jet",
    )

    itr_jumbled = [3, 1, 4, 2, 0]

    # proxy = [
    #     plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy = [
    #     mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy_labs = (
    #         "$n=2n_\mathrm{sw}$",
    #         "$T_\mathrm{core}=3T_\mathrm{sw}$",
    #         "$M_{\mathrm{MS},x}=1$",
    #         "Jet",
    #         "FCS",
    #         "Non-FCS jet",
    #         "FCS-jet"
    #     )
    proxy = [
        mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
        for itr in range(3)
    ]
    proxy_labs = [
        "$n=2n_\mathrm{sw}$",
        "$T_\mathrm{core}=3T_\mathrm{sw}$",
        "$M_{\mathrm{MS},x}=1$",
    ]

    xmin, xmax, ymin, ymax = (
        np.min(XmeshXY),
        np.max(XmeshXY),
        np.min(YmeshXY),
        np.max(YmeshXY),
    )

    if ~(jet_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[3]]))
        proxy_labs.append("Jet")
    if ~(slams_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[4]]))
        proxy_labs.append("FCS")
    if np.logical_and(
        np.logical_and(non_xlist >= xmin, non_xlist <= xmax),
        np.logical_and(non_ylist >= ymin, non_ylist <= ymax),
    ).any():
        proxy.append(non_pos)
        proxy_labs.append("Non-FCS jet")
    if np.logical_and(
        np.logical_and(sj_xlist >= xmin, sj_xlist <= xmax),
        np.logical_and(sj_ylist >= ymin, sj_ylist <= ymax),
    ).any():
        proxy.append(sj_pos)
        proxy_labs.append("FCS-jet")

    ax.legend(
        proxy,
        proxy_labs,
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="lower left",
        fontsize=14,
    )
